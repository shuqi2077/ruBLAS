use ruda_core::tensor::DType;
use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::{dsl::prelude::*, tensor::RudaTensor};
use crate::kernel_ir::definition::MatmulSetupError;

#[ruda(launch_unchecked)]
fn warp_validated_ordered<F: Float>(
    lhs: &Tensor<F>, rhs: &Tensor<F>, out: &mut Tensor<F>,
    k: u32, n: u32, lhs_stride: u32, rhs_k_stride: u32, rhs_column_stride: u32, out_stride: u32,
    #[comptime] warps: u32, #[comptime] reference_width: u32,
    #[define(F)] _dtype: StorageType,
) {
    let lane = UNIT_POS_X;
    let group = lane / reference_width;
    let component = lane % reference_width;
    let group_base = group * reference_width;
    let column = (RUDA_POS_X * warps + UNIT_POS_Y) * (32 / reference_width) + group;
    let mut acc = 0.0f32;
    let mut base = 0u32;
    while base < k {
        let position = base + component;
        if column < n && position < k {
            let left = f32::cast_from(lhs[(position * lhs_stride) as usize]);
            let right = f32::cast_from(rhs[(column * rhs_column_stride + position * rhs_k_stride) as usize]);
            acc = fma(left, right, acc);
        }
        base += reference_width;
    }
    if comptime!(reference_width == 1) {
        if column < n { out[(column * out_stride) as usize] = F::cast_from(acc); }
    } else {
        let mut sum = 0.0f32;
        #[unroll]
        for value in 0u32..reference_width { sum += plane_shuffle(acc, group_base + value); }
        if component == 0 && column < n { out[(column * out_stride) as usize] = F::cast_from(sum); }
    }
}

pub(crate) fn launch<R: Runtime>(
    lhs: RudaTensor<R>, rhs: RudaTensor<R>, out: RudaTensor<R>, warps: u32,
) -> Result<(), MatmulSetupError> {
    if !matches!(warps, 4 | 8 | 16) {
        return Err(MatmulSetupError::InvalidConfig(Box::new("unsupported validated GEMV warp count".to_string())));
    }
    let width = crate::tensor_matmul::gemv_fused_ordered_width(&lhs, &rhs, &out)?;
    let rank = lhs.meta.num_dims();
    let k = lhs.meta.shape()[rank - 1] as u32;
    let n = rhs.meta.shape()[rank - 1] as u32;
    let dim = RudaDim::new_2d(32, warps);
    let client = lhs.client.clone();
    let max_dim = client.properties().hardware.max_ruda_dim;
    if dim.x > max_dim.0 || dim.y > max_dim.1 || dim.z > max_dim.2 {
        return Err(MatmulSetupError::InvalidConfig(Box::new("validated GEMV block exceeds device axes".to_string())));
    }
    let lhs_stride = lhs.meta.strides()[rank - 1] as u32;
    let rhs_k_stride = rhs.meta.strides()[rank - 2] as u32;
    let rhs_column_stride = rhs.meta.strides()[rank - 1] as u32;
    let out_stride = out.meta.strides()[rank - 1] as u32;
    unsafe {
        warp_validated_ordered::launch_unchecked::<R>(&client,
            RudaCount::Static(n.div_ceil(warps * (32 / width)), 1, 1), dim,
            lhs.into_tensor_arg(), rhs.into_tensor_arg(), out.into_tensor_arg(),
            k, n, lhs_stride, rhs_k_stride, rhs_column_stride, out_stride,
            warps, width, DType::BF16.into());
    }
    Ok(())
}
