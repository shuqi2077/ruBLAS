use ruda_core::{device::Device, ir::features::Plane, tensor::DType};
use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::tensor_vector_size_parallel;
use ruda_kernel::library::tensor::{MatrixBatchLayout, matrix_batch_layout};
use ruda_kernel::tiling::MatrixLayout;
use ruda_kernel::{dsl::prelude::*, tensor::RudaTensor};

use crate::kernel_ir::definition::MatmulSetupError;

fn invalid(message: &str) -> MatmulSetupError {
    MatmulSetupError::InvalidConfig(Box::new(message.to_string()))
}

fn gemv_search_eligible<R: Runtime>(
    lhs: &RudaTensor<R>,
    rhs: &RudaTensor<R>,
    out: &RudaTensor<R>,
) -> bool {
    let tensors = [lhs, rhs, out];
    for tensor in tensors {
        let shape = tensor.meta.shape();
        if shape.len() < 2
            || tensor.meta.strides().len() != shape.len()
            || shape[..shape.len() - 2].iter().any(|&extent| extent != 1)
            || tensor.dtype != DType::BF16
            || tensor.qparams.is_some()
            || tensor.handle.offset_start.unwrap_or(0) % 2 != 0
            || tensor
                .handle
                .offset_start
                .unwrap_or(0)
                .checked_add(tensor.handle.offset_end.unwrap_or(0))
                .is_none_or(|offsets| offsets > tensor.handle.size())
            || tensor.device.to_id() != lhs.device.to_id()
            || !tensor.client.same_execution_queue(&lhs.client)
        {
            return false;
        }
        let Some(last) = shape.iter().zip(tensor.meta.strides().iter()).try_fold(
            0usize,
            |offset, (&extent, &stride)| {
                offset.checked_add(extent.checked_sub(1)?.checked_mul(stride)?)
            },
        ) else {
            return false;
        };
        if last >= u32::MAX as usize
            || last
                .checked_add(1)
                .and_then(|n| n.checked_mul(2))
                .is_none_or(|bytes| bytes as u64 > tensor.handle.size_in_used())
        {
            return false;
        }
    }
    let a = lhs.meta.shape();
    let b = rhs.meta.shape();
    let c = out.meta.shape();
    let k = a[a.len() - 1];
    let n = b[b.len() - 1];
    let props = &lhs.client.properties().hardware;
    a[a.len() - 2] == 1
        && b[b.len() - 2] == k
        && c[c.len() - 2] == 1
        && c[c.len() - 1] == n
        && k != 0
        && k <= (u32::MAX - 127) as usize
        && n != 0
        && n.div_ceil(4) <= props.max_ruda_count.0 as usize
        && lhs.meta.strides()[a.len() - 1] == 1
        && rhs.meta.strides()[b.len() - 2] == 1
        && rhs.meta.strides()[b.len() - 1] >= k
        && rhs.meta.strides()[b.len() - 1] <= u32::MAX as usize
        && out.meta.strides()[c.len() - 1] != 0
        && out.meta.strides()[c.len() - 1] <= u32::MAX as usize
        && props.plane_size_min == 32
        && props.plane_size_max == 32
        && props.max_units_per_ruda >= 512
        && lhs.client.features().plane.contains(Plane::Ops)
}

#[ruda(launch)]
fn warp_nt_ordered<F: Float>(
    lhs: &Tensor<F>,
    rhs: &Tensor<F>,
    out: &mut Tensor<F>,
    k: u32,
    n: u32,
    rhs_column_stride: u32,
    out_stride: u32,
    #[comptime] warps: u32,
    #[comptime] reference_width: u32,
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
            let left = f32::cast_from(lhs[position as usize]);
            let right = f32::cast_from(rhs[(column * rhs_column_stride + position) as usize]);
            acc = fma(left, right, acc);
        }
        base += reference_width;
    }
    if comptime!(reference_width == 1) {
        if column < n {
            out[(column * out_stride) as usize] = F::cast_from(acc);
        }
    } else {
        let mut sum = 0.0f32;
        #[unroll]
        for value in 0u32..reference_width {
            sum += plane_shuffle(acc, group_base + value);
        }
        if component == 0 && column < n {
            out[(column * out_stride) as usize] = F::cast_from(sum);
        }
    }
}

pub fn gemv_search_ordered_width<R: Runtime>(
    lhs: &RudaTensor<R>,
    rhs: &RudaTensor<R>,
    out: &RudaTensor<R>,
) -> Result<u32, MatmulSetupError> {
    if !gemv_search_eligible(lhs, rhs, out)
        || MatrixLayout::from_shape_and_strides(out.meta.shape(), out.meta.strides(), None).is_err()
        || lhs.meta.shape().len() != rhs.meta.shape().len()
        || lhs.meta.shape().len() != out.meta.shape().len()
        || !matches!(
            matrix_batch_layout(lhs.meta.strides(), None),
            MatrixBatchLayout::Contiguous
        )
        || !matches!(
            matrix_batch_layout(rhs.meta.strides(), None),
            MatrixBatchLayout::MildlyPermuted {
                transposed: true,
                batch_swap: false
            }
        )
    {
        return Err(invalid(
            "ordered NT GEMV requires the naive reference's direct NT bindings",
        ));
    }
    let lhs_width = tensor_vector_size_parallel(
        lhs.client.io_optimized_vector_sizes(lhs.dtype.size()),
        lhs.meta.shape(),
        lhs.meta.strides(),
        lhs.meta.shape().len() - 1,
    );
    let rhs_width = tensor_vector_size_parallel(
        lhs.client.io_optimized_vector_sizes(rhs.dtype.size()),
        rhs.meta.shape(),
        rhs.meta.strides(),
        rhs.meta.shape().len() - 2,
    );
    let width = lhs_width.max(rhs_width);
    if !matches!(width, 1 | 2 | 4 | 8 | 16 | 32) {
        return Err(invalid("naive vector width cannot fit an ordered subwarp"));
    }
    Ok(width as u32)
}

pub fn launch_gemv_search_ordered<R: Runtime>(
    lhs: RudaTensor<R>,
    rhs: RudaTensor<R>,
    out: RudaTensor<R>,
    warps: u32,
) -> Result<(), MatmulSetupError> {
    if !matches!(warps, 4 | 8 | 16) {
        return Err(invalid("unsupported ordered NT GEMV warp count"));
    }
    let reference_width = gemv_search_ordered_width(&lhs, &rhs, &out)?;
    let a = lhs.meta.shape();
    let b = rhs.meta.shape();
    let c = out.meta.shape();
    let k = a[a.len() - 1] as u32;
    let n = b[b.len() - 1] as u32;
    let rhs_column_stride = rhs.meta.strides()[b.len() - 1] as u32;
    let out_stride = out.meta.strides()[c.len() - 1] as u32;
    let columns_per_block = warps * (32 / reference_width);
    warp_nt_ordered::launch::<R>(
        &lhs.client,
        RudaCount::Static(n.div_ceil(columns_per_block), 1, 1),
        RudaDim::new_2d(32, warps),
        lhs.clone().into_tensor_arg(),
        rhs.into_tensor_arg(),
        out.into_tensor_arg(),
        k,
        n,
        rhs_column_stride,
        out_stride,
        warps,
        reference_width,
        DType::BF16.into(),
    );
    Ok(())
}
