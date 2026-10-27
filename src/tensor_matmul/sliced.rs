use ruda_core::tensor::{DType, Shape};
use ruda_kernel::{dsl as kernel_dsl, dsl::prelude::*, tensor::{RudaTensor,
    allocation::empty_device_contiguous_dtype, permutation::swap_dims}};
use crate::kernel_ir::{definition::MatmulSetupError, launch::Strategy};
use super::{F32MathMode, launch_matmul_with_precision};

/// A100 Qwen K/V projection: first sliced accumulator plus bias, then second.
pub fn matmul_bf16_sliced_k2_bias<R: Runtime>(a: RudaTensor<R>, b: RudaTensor<R>,
    bias: RudaTensor<R>, out: RudaTensor<R>) -> Result<RudaTensor<R>, MatmulSetupError> {
    if a.dtype != DType::BF16 || b.dtype != DType::BF16 || bias.dtype != DType::BF16 || out.dtype != DType::BF16
        || a.qparams.is_some() || b.qparams.is_some() || bias.qparams.is_some() || out.qparams.is_some()
        || a.meta.num_dims()!=2 || b.meta.num_dims()!=2 || bias.meta.num_dims()!=2 || out.meta.num_dims()!=2
        || a.meta.shape()[..]!=[512,3584] || b.meta.shape()[..]!=[3584,512]
        || bias.meta.shape()[..]!=[512,512]
        || out.meta.shape()[..]!=[512,512] || out.meta.strides()[..]!=[512,1]
        || !a.client.same_execution_queue(&b.client) || !a.client.same_execution_queue(&bias.client)
        || !a.client.same_execution_queue(&out.client)
    {
        return Err(MatmulSetupError::InvalidConfig(Box::new("unsupported BF16 sliced-K2 projection".to_string())));
    }
    let left = empty_device_contiguous_dtype(a.client.clone(),a.device.clone(),Shape::from([2,512,1792]),DType::BF16);
    let right = empty_device_contiguous_dtype(a.client.clone(),a.device.clone(),Shape::from([2,512,1792]),DType::BF16);
    for (input,output,rhs) in [(a.clone(),left.clone(),false),(b,right.clone(),true)] {
        pack::launch::<R>(&a.client,RudaCount::Static((2*512*1792/128) as u32,1,1),RudaDim::new_1d(128),
            input.into_tensor_arg(),output.into_array_arg(),rhs,DType::BF16.into());
    }
    let partial = empty_device_contiguous_dtype(a.client.clone(),a.device.clone(),Shape::from([2,512,512]),DType::F32);
    let right = swap_dims(right, 1, 2);
    if !super::matmul_bf16_training(&left, &right, &partial) {
        launch_matmul_with_precision(&Strategy::DoubleAsyncCyclicCmma(Default::default()),
            left, right, partial.clone(), F32MathMode::Strict)?;
    }
    reduce::launch::<R>(&a.client,RudaCount::Static((512*512/128) as u32,1,1),RudaDim::new_1d(128),
        partial.into_array_arg(),bias.into_tensor_arg(),out.clone().into_array_arg(),DType::BF16.into());
    Ok(out)
}

/// Ordered three-part split-K with BF16 writeback between partial sums.
pub fn matmul_bf16_serial_k3<R: Runtime>(a: RudaTensor<R>, b: RudaTensor<R>,
    out: RudaTensor<R>) -> Result<RudaTensor<R>, MatmulSetupError> {
    split_k(a, b, out, 3, false)
}

/// Round each split-K partial to BF16 before its ordered FP32 reduction.
pub fn matmul_bf16_rounded_parallel_k<R: Runtime>(a: RudaTensor<R>, b: RudaTensor<R>,
    out: RudaTensor<R>, parts: usize) -> Result<RudaTensor<R>, MatmulSetupError> {
    split_k(a, b, out, parts, true)
}

fn split_k<R: Runtime>(a: RudaTensor<R>, b: RudaTensor<R>, out: RudaTensor<R>,
    parts: usize, rounded_partials: bool) -> Result<RudaTensor<R>, MatmulSetupError> {
    if a.dtype != DType::BF16 || b.dtype != DType::BF16 || out.dtype != DType::BF16
        || a.qparams.is_some() || b.qparams.is_some() || out.qparams.is_some()
        || a.meta.num_dims()!=2 || b.meta.num_dims()!=2 || out.meta.num_dims()!=2
        || !matches!(parts, 3 | 5) || (!rounded_partials && parts != 3)
        || b.meta.shape()[0]!=a.meta.shape()[1]
        || out.meta.shape()[..]!=[a.meta.shape()[0],b.meta.shape()[1]]
        || out.meta.strides()[..]!=[b.meta.shape()[1],1]
        || a.meta.strides()[1]!=1 || (b.meta.strides()[0]!=1 && b.meta.strides()[1]!=1)
        || !a.client.same_execution_queue(&b.client) || !a.client.same_execution_queue(&out.client)
    {
        return Err(MatmulSetupError::InvalidConfig(Box::new("unsupported BF16 split-K projection".to_string())));
    }
    let (m, k, n) = (a.meta.shape()[0], a.meta.shape()[1], b.meta.shape()[1]);
    let block = k.div_ceil(parts*64)*64;
    if m == 0 || n == 0 || k == 0 || k % 64 != 0 || k <= (parts-1)*block
        || m.checked_mul(n).and_then(|v| v.checked_mul(parts)).is_none_or(|v| v > u32::MAX as usize)
    {
        return Err(MatmulSetupError::InvalidConfig(Box::new("invalid BF16 split-K extent".to_string())));
    }
    let partial = empty_device_contiguous_dtype(a.client.clone(),a.device.clone(),Shape::from([parts,m,n]),
        if rounded_partials { DType::BF16 } else { DType::F32 });
    let element_bytes = if rounded_partials { 2 } else { 4 };
    if !super::training_gemm::split_k(&a, &b, &partial, parts, block) {
        for part in 0..parts {
            let start = part*block;
            let len = (k-start).min(block);
            let mut left = a.clone();
            left.meta.shape_mut()[1] = len;
            left.handle.offset_start = Some(a.handle.offset_start.unwrap_or(0)+(start*a.meta.strides()[1]*2) as u64);
            let mut right = b.clone();
            right.meta.shape_mut()[0] = len;
            right.handle.offset_start = Some(b.handle.offset_start.unwrap_or(0)+(start*b.meta.strides()[0]*2) as u64);
            let mut output = partial.clone();
            output.meta.remove(0);
            output.handle.offset_start = Some(partial.handle.offset_start.unwrap_or(0)+(part*m*n*element_bytes) as u64);
            if super::matmul_bf16_training(&left, &right, &output) { continue; }
            match launch_matmul_with_precision(&Strategy::DoubleAsyncCyclicCmma(Default::default()),
                left.clone(),right.clone(),output.clone(),F32MathMode::Strict) {
                Ok(()) => {},
                Err(MatmulSetupError::Unavailable(_) | MatmulSetupError::InvalidConfig(_) | MatmulSetupError::Vectorization(_)) => {
                    launch_matmul_with_precision(&Strategy::Auto,left,right,output,F32MathMode::Strict)?;
                },
                Err(error) => return Err(error),
            }
        }
    }
    if rounded_partials {
        rounded_parallel_reduce::launch::<R>(&a.client,RudaCount::Static((m*n).div_ceil(128) as u32,1,1),RudaDim::new_1d(128),
            partial.into_array_arg(),out.clone().into_array_arg(),parts,DType::BF16.into());
    } else {
        serial_reduce::launch::<R>(&a.client,RudaCount::Static((m*n).div_ceil(128) as u32,1,1),RudaDim::new_1d(128),
            partial.into_array_arg(),out.clone().into_array_arg(),DType::BF16.into());
    }
    Ok(out)
}

/// Parallel LM-head input gradients without copying the frozen vocabulary weight.
pub fn matmul_bf16_head_parallel_k6<R: Runtime>(a: RudaTensor<R>, b: RudaTensor<R>,
    out: RudaTensor<R>) -> Result<RudaTensor<R>, MatmulSetupError> {
    if a.dtype!=DType::BF16 || b.dtype!=DType::BF16 || !matches!(out.dtype,DType::BF16|DType::F32)
        || a.qparams.is_some() || b.qparams.is_some() || out.qparams.is_some()
        || a.meta.num_dims()!=2 || b.meta.num_dims()!=2 || out.meta.num_dims()!=2
        || !matches!(a.meta.shape()[0],31|32) || a.meta.shape()[1]!=152064
        || b.meta.shape()[..]!=[152064,3584] || out.meta.shape()[..]!=[a.meta.shape()[0],3584]
        || a.meta.strides()[..]!=[152064,1] || b.meta.strides()[..]!=[3584,1]
        || out.meta.strides()[..]!=[3584,1]
        || a.handle.offset_start.unwrap_or(0)%16!=0 || b.handle.offset_start.unwrap_or(0)%16!=0
        || !a.client.same_execution_queue(&b.client) || !a.client.same_execution_queue(&out.client)
    {
        return Err(MatmulSetupError::InvalidConfig(Box::new("unsupported BF16 parallel LM-head gradient".to_string())));
    }
    let m = a.meta.shape()[0];
    let mut left = a.clone();
    left.meta.shape_mut()[1] = 25344;
    left.meta.insert(0,6,25344);
    let mut right = b;
    right.meta.shape_mut()[0] = 25344;
    right.meta.insert(0,6,25344*3584);
    let partial = empty_device_contiguous_dtype(a.client.clone(),a.device.clone(),Shape::from([6,m,3584]),DType::F32);
    if !super::matmul_bf16_training(&left, &right, &partial) {
        launch_matmul_with_precision(&Strategy::Auto,left,right,partial.clone(),F32MathMode::Strict)?;
    }
    parallel_reduce::launch::<R>(&a.client,RudaCount::Static((m*3584).div_ceil(128) as u32,1,1),RudaDim::new_1d(128),
        partial.into_array_arg(),out.clone().into_array_arg(),out.dtype.into());
    Ok(out)
}

#[ruda(launch)]
fn parallel_reduce<F: Float>(partial: &Array<f32>, out: &mut Array<F>, #[define(F)] _dtype: StorageType) {
    let pos = ABSOLUTE_POS as usize;
    if pos < out.len() {
        let mut value = partial[pos];
        #[unroll]
        for part in 1usize..6 { value += partial[part*out.len()+pos]; }
        out[pos] = F::cast_from(value);
    }
}

#[ruda(launch)]
fn rounded_parallel_reduce<F: Float>(partial: &Array<F>, out: &mut Array<F>,
    #[comptime] parts: usize, #[define(F)] _dtype: StorageType) {
    let pos = ABSOLUTE_POS as usize;
    if pos < out.len() {
        let mut value = f32::cast_from(partial[pos]);
        #[unroll]
        for part in 1usize..parts { value += f32::cast_from(partial[part*out.len()+pos]); }
        out[pos] = F::cast_from(value);
    }
}

#[ruda(launch)]
fn serial_reduce<F: Float>(partial: &Array<f32>, out: &mut Array<F>, #[define(F)] _dtype: StorageType) {
    let pos = ABSOLUTE_POS as usize;
    if pos < out.len() {
        let first = F::cast_from(partial[pos]);
        let second = F::cast_from(f32::cast_from(first)+partial[out.len()+pos]);
        out[pos] = F::cast_from(f32::cast_from(second)+partial[2*out.len()+pos]);
    }
}

#[ruda(launch)]
fn pack<F: Float>(input: &Tensor<F>, output: &mut Array<F>,
    #[comptime] rhs: bool, #[define(F)] _dtype: StorageType) {
    let pos = ABSOLUTE_POS as usize;
    if pos < output.len() {
        let part = pos / (512*1792);
        let row = (pos % (512*1792)) / 1792;
        let local = pos % 1792;
        let inner = (local/32)*64+part*32+local%32;
        let mut index = row*input.stride(0)+inner*input.stride(1);
        if comptime!(rhs) { index = inner*input.stride(0)+row*input.stride(1); }
        output[pos] = input[index];
    }
}

#[ruda(launch)]
fn reduce<F: Float>(partial: &Array<f32>, bias: &Tensor<F>, out: &mut Array<F>, #[define(F)] _dtype: StorageType) {
    let pos = ABSOLUTE_POS as usize;
    if pos < out.len() {
        let offset = (pos/512)*bias.stride(0)+(pos%512)*bias.stride(1);
        out[pos] = F::cast_from((partial[pos]+f32::cast_from(bias[offset]))+partial[512*512+pos]);
    }
}
