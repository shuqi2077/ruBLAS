use super::{F32MathMode, MatmulStrategy, matmul_with_precision};
use crate::kernel_ir::definition::MatmulSetupError;
use ruda_core::tensor::{DType, calculate_matmul_output};
use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::calculate_ruda_count_elemwise;
use ruda_kernel::dsl::prelude::*;
use ruda_kernel::tensor::{
    RudaTensor, allocation::empty_device_contiguous_dtype, permutation::swap_dims,
};

#[derive(Clone, Copy, Debug)]
pub struct GemmOptions {
    pub alpha: f64,
    pub beta: f64,
    pub transpose_lhs: bool,
    pub transpose_rhs: bool,
    pub f32_math: F32MathMode,
}

impl Default for GemmOptions {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            beta: 0.0,
            transpose_lhs: false,
            transpose_rhs: false,
            f32_math: F32MathMode::Strict,
        }
    }
}

fn invalid(message: &str) -> MatmulSetupError {
    MatmulSetupError::InvalidConfig(Box::new(message.to_string()))
}

pub fn gemm<R: Runtime>(
    mut lhs: RudaTensor<R>,
    mut rhs: RudaTensor<R>,
    c: Option<RudaTensor<R>>,
    options: GemmOptions,
    strategy: MatmulStrategy,
) -> Result<RudaTensor<R>, MatmulSetupError> {
    if !matches!(lhs.dtype, DType::F32 | DType::F64)
        || lhs.dtype != rhs.dtype
        || lhs.qparams.is_some()
        || rhs.qparams.is_some()
    {
        return Err(invalid(
            "GEMM requires matching unquantized F32 or F64 inputs",
        ));
    }
    if !lhs.client.same_execution_queue(&rhs.client) {
        return Err(invalid("GEMM input execution queues differ"));
    }
    let lhs_rank = lhs.meta.num_dims();
    let rhs_rank = rhs.meta.num_dims();
    if lhs_rank < 2 || rhs_rank < 2 {
        return Err(invalid("GEMM inputs require at least two axes"));
    }
    if options.transpose_lhs {
        lhs = swap_dims(lhs, lhs_rank - 2, lhs_rank - 1);
    }
    if options.transpose_rhs {
        rhs = swap_dims(rhs, rhs_rank - 2, rhs_rank - 1);
    }
    let shape = calculate_matmul_output(lhs.meta.shape(), rhs.meta.shape())
        .map_err(|_| invalid("GEMM matrix or batch dimensions are incompatible"))?;
    let dtype = lhs.dtype;
    let alpha = if dtype == DType::F32 {
        options.alpha as f32 as f64
    } else {
        options.alpha
    };
    let beta = if dtype == DType::F32 {
        options.beta as f32 as f64
    } else {
        options.beta
    };
    if let Some(c) = &c {
        if c.dtype != lhs.dtype
            || c.qparams.is_some()
            || c.meta.shape() != &shape
            || !lhs.client.same_execution_queue(&c.client)
        {
            return Err(invalid(
                "GEMM C must match the output shape, dtype and execution queue",
            ));
        }
    } else if beta != 0.0 {
        return Err(invalid("GEMM with nonzero beta requires C"));
    }
    let count = shape
        .iter()
        .try_fold(1usize, |size, &extent| size.checked_mul(extent))
        .filter(|&size| size <= u32::MAX as usize)
        .ok_or_else(|| invalid("GEMM output exceeds device indexing"))?;
    let product_needed = alpha != 0.0 && lhs.meta.shape()[lhs_rank - 1] != 0;
    let product = if product_needed && count > 0 {
        matmul_with_precision(lhs.clone(), rhs, None, strategy, dtype, options.f32_math)?
    } else {
        lhs.clone()
    };
    if product_needed && alpha == 1.0 && beta == 0.0 && count > 0 {
        return Ok(product);
    }
    let output =
        empty_device_contiguous_dtype(lhs.client.clone(), lhs.device.clone(), shape, dtype);
    if count == 0 {
        return Ok(output);
    }
    let c = c.unwrap_or_else(|| product.clone());
    let dim = RudaDim::new(lhs.client.properties(), count);
    let grid = calculate_ruda_count_elemwise(&lhs.client, count, dim);
    match dtype {
        DType::F32 => epilogue::launch::<f32, R>(
            &lhs.client,
            grid,
            dim,
            product.into_tensor_arg(),
            c.into_tensor_arg(),
            output.clone().into_tensor_arg(),
            alpha as f32,
            beta as f32,
            product_needed,
            beta != 0.0,
        ),
        DType::F64 => epilogue::launch::<f64, R>(
            &lhs.client,
            grid,
            dim,
            product.into_tensor_arg(),
            c.into_tensor_arg(),
            output.clone().into_tensor_arg(),
            alpha,
            beta,
            product_needed,
            beta != 0.0,
        ),
        _ => unreachable!(),
    }
    Ok(output)
}

#[ruda(launch)]
fn epilogue<F: Float + RudaElement>(
    product: &Tensor<F>,
    c: &Tensor<F>,
    output: &mut Tensor<F>,
    alpha: F,
    beta: F,
    #[comptime] read_product: bool,
    #[comptime] read_c: bool,
) {
    let pos = ABSOLUTE_POS;
    if pos >= output.len() {
        terminate!();
    }
    let mut product_offset = 0usize;
    let mut c_offset = 0usize;
    let mut remaining = pos;
    let mut axis = output.rank();
    while axis > 0 {
        axis -= 1;
        let coordinate = remaining % output.shape(axis);
        remaining /= output.shape(axis);
        if comptime!(read_product) {
            product_offset += coordinate * product.stride(axis);
        }
        if comptime!(read_c) {
            c_offset += coordinate * c.stride(axis);
        }
    }
    let mut value = F::new(0.0);
    if comptime!(read_product) {
        value = alpha * product[product_offset];
    }
    if comptime!(read_c) {
        value += beta * c[c_offset];
    }
    output[pos] = value;
}
