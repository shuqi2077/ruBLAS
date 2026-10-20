use crate::kernel_ir::definition::MatmulSetupError;
use ruda_core::{device::Device, ir::{ElemType, FloatKind, features::MmaConfig}, tensor::{DType, Shape}};
use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::{dsl::{Runtime, calculate_ruda_count_elemwise, prelude::*}, tensor::{RudaTensor, allocation::empty_device_contiguous_dtype, contiguous::into_contiguous}};

/// Postprocessing performed in FP32 inside the GEMM kernel, before output rounding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GemmActivation {
    None,
    Relu,
    /// The tanh GELU formula used by cuBLASLt GELU epilogues.
    Gelu,
}

/// Auto selects a supported cooperative matrix path; only unsupported setup
/// falls back to scalar. TensorCore never silently falls back after a request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FusedGemmStrategy { Auto, Scalar, TensorCore }

fn invalid(message: &str) -> MatmulSetupError {
    MatmulSetupError::InvalidConfig(Box::new(message.to_string()))
}

/// `activation(lhs[M,K] @ rhs[K,N] + bias[N])`, with matching FP16/BF16
/// inputs, FP32 accumulation, and FP16/BF16 or FP32 output. Bias is packed
/// internally and can use input storage or FP32. Strided/transposed operands
/// are accepted via device-contiguous copies. No GEMM result or pre-activation
/// matrix is materialized: bias and activation are part of the GEMM store.
/// This is a row-major column-bias API, not a cuBLASLt ABI replacement.
pub fn matmul_bias_activation<R: Runtime>(
    lhs: RudaTensor<R>, rhs: RudaTensor<R>, bias: RudaTensor<R>,
    activation: GemmActivation, strategy: FusedGemmStrategy, out_dtype: DType,
) -> Result<RudaTensor<R>, MatmulSetupError> {
    if lhs.meta.num_dims() != 2 || rhs.meta.num_dims() != 2 {
        return Err(invalid("fused GEMM requires [M,K] and [K,N]"));
    }
    let (m,k,n) = (lhs.meta.shape()[0], lhs.meta.shape()[1], rhs.meta.shape()[1]);
    if rhs.meta.shape()[0] != k || bias.meta.shape()[..] != [n]
        || !matches!(lhs.dtype, DType::F16 | DType::BF16) || lhs.dtype != rhs.dtype
        || (bias.dtype != lhs.dtype && bias.dtype != DType::F32)
        || (out_dtype != lhs.dtype && out_dtype != DType::F32)
    { return Err(invalid("fused GEMM input/bias/output shape or dtype mismatch")); }
    for t in [&lhs, &rhs, &bias] {
        if t.qparams.is_some() || t.device.to_id() != lhs.device.to_id()
            || !t.client.same_execution_queue(&lhs.client)
            || t.meta.shape().iter().try_fold(1usize, |a,&b| a.checked_mul(b)).is_none_or(|x| x > u32::MAX as usize)
        { return Err(invalid("fused GEMM requires unquantized same-device/queue tensors with U32 indexing")); }
    }
    if m.checked_mul(n).is_none_or(|x| x > u32::MAX as usize) {
        return Err(invalid("fused GEMM output exceeds U32 indexing"));
    }
    let cfg = MmaConfig { a_type: lhs.dtype.into(), b_type: lhs.dtype.into(),
        cd_type: ElemType::Float(FloatKind::F32).into(), m:16, n:16, k:16 };
    let hw = &lhs.client.properties().hardware;
    let cooperative = hw.plane_size_min == 32 && hw.plane_size_max == 32
        && hw.max_shared_memory_size >= 2048 && lhs.client.features().matmul.cmma.contains(&cfg)
        && n.div_ceil(16) <= hw.max_ruda_count.0 as usize
        && m.div_ceil(16) <= hw.max_ruda_count.1 as usize;
    if strategy == FusedGemmStrategy::TensorCore && !cooperative {
        return Err(invalid("requested fused GEMM 16x16x16 Tensor Core configuration is unavailable"));
    }
    let out = empty_device_contiguous_dtype(lhs.client.clone(), lhs.device.clone(), Shape::from([m,n]), out_dtype);
    if m == 0 || n == 0 { return Ok(out); }
    let lhs = into_contiguous(lhs); let rhs = into_contiguous(rhs); let bias = into_contiguous(bias);
    let code = match activation { GemmActivation::None => 0, GemmActivation::Relu => 1, GemmActivation::Gelu => 2 };
    if cooperative && strategy != FusedGemmStrategy::Scalar {
        cooperative_kernel::launch::<R>(&lhs.client,
            RudaCount::Static(n.div_ceil(16) as u32, m.div_ceil(16) as u32, 1), RudaDim::new_1d(32),
            lhs.clone().into_array_arg(), rhs.into_array_arg(), bias.clone().into_array_arg(), out.clone().into_array_arg(),
            m as u32, n as u32, k as u32, code, lhs.dtype.into(), bias.dtype.into(), out_dtype.into());
    } else {
        let dim = RudaDim::new(lhs.client.properties(), m*n);
        scalar_kernel::launch::<R>(&lhs.client, calculate_ruda_count_elemwise(&lhs.client, m*n, dim), dim,
            lhs.clone().into_array_arg(), rhs.into_array_arg(), bias.clone().into_array_arg(), out.clone().into_array_arg(),
            n as u32, k as u32, code, lhs.dtype.into(), bias.dtype.into(), out_dtype.into());
    }
    Ok(out)
}

#[ruda]
fn activate(x: f32, #[comptime] activation: u32) -> f32 {
    if comptime!(activation == 1) {
        // Comparisons preserve NaN instead of turning it into a zero.
        if x < 0.0 { f32::new(0.0f32) } else { x }
    } else if comptime!(activation == 2) {
        0.5 * x * (1.0 + (0.7978845608028654 * (x + 0.044715 * x*x*x)).tanh())
    } else { x }
}

#[ruda(launch)]
fn scalar_kernel<F: Float, B: Float, O: Float>(
    lhs: &Array<F>, rhs: &Array<F>, bias: &Array<B>, out: &mut Array<O>,
    columns: u32, inner: u32, #[comptime] activation: u32,
    #[define(F)] _input: StorageType, #[define(B)] _bias: StorageType, #[define(O)] _output: StorageType,
) {
    let p = ABSOLUTE_POS;
    if p >= out.len() { terminate!(); }
    let row = p / columns as usize; let col = p % columns as usize;
    let mut sum = 0.0f32;
    for kk in 0..inner as usize {
        sum = fma(f32::cast_from(lhs[row*inner as usize+kk]), f32::cast_from(rhs[kk*columns as usize+col]), sum);
    }
    out[p] = O::cast_from(activate(sum + f32::cast_from(bias[col]), activation));
}

#[ruda(launch)]
fn cooperative_kernel<F: Float, B: Float, O: Float>(
    lhs: &Array<F>, rhs: &Array<F>, bias: &Array<B>, out: &mut Array<O>,
    rows: u32, columns: u32, inner: u32, #[comptime] activation: u32,
    #[define(F)] _input: StorageType, #[define(B)] _bias: StorageType, #[define(O)] _output: StorageType,
) {
    let row_base = RUDA_POS_Y as usize * 16;
    let col_base = RUDA_POS_X as usize * 16;
    let lane = UNIT_POS as usize;
    let mut left = SharedMemory::<F>::new_aligned(256usize,32usize);
    let mut right = SharedMemory::<F>::new_aligned(256usize,32usize);
    let mut result = SharedMemory::<f32>::new_aligned(256usize,32usize);
    let acc = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator,
        16usize,16usize,16usize,cmma::MatrixLayout::Undefined,0.0);
    let mut base_k = 0usize;
    while base_k < inner as usize {
        #[unroll]
        for i in 0usize..8usize {
            let t = lane+i*32;
            let row = row_base+t/16; let col = col_base+t%16;
            let kk_a = base_k+t%16; let kk_b = base_k+t/16;
            let mut a = F::cast_from(0.0f32); let mut b = F::cast_from(0.0f32);
            if row < rows as usize && kk_a < inner as usize { a = lhs[row*inner as usize+kk_a]; }
            if col < columns as usize && kk_b < inner as usize { b = rhs[kk_b*columns as usize+col]; }
            left[t] = a; right[t] = b;
        }
        sync_ruda();
        let a = cmma::Matrix::<F>::from_slice(cmma::MatrixIdent::A,
            16usize,16usize,16usize,cmma::MatrixLayout::RowMajor,&left.to_slice(),16);
        let b = cmma::Matrix::<F>::from_slice(cmma::MatrixIdent::B,
            16usize,16usize,16usize,cmma::MatrixLayout::RowMajor,&right.to_slice(),16);
        cmma::execute::<F,F,f32,f32, _>(&a,&b,&acc,&acc);
        sync_ruda();
        if inner as usize-base_k <= 16 { base_k=inner as usize; } else { base_k+=16; }
    }
    cmma::store(&mut result.to_slice_mut(), &acc, 16, cmma::MatrixLayout::RowMajor);
    sync_ruda();
    #[unroll]
    for i in 0usize..8usize {
        let t=lane+i*32; let row=row_base+t/16; let col=col_base+t%16;
        if row < rows as usize && col < columns as usize {
            out[row*columns as usize+col] = O::cast_from(activate(result[t]+f32::cast_from(bias[col]), activation));
        }
    }
}
