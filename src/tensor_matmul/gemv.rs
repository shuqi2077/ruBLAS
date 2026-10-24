use ruda_core::{device::Device, ir::features::Plane, tensor::DType};
use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::{dsl::prelude::*, tensor::RudaTensor};
use crate::kernel_ir::definition::MatmulSetupError;

#[ruda(launch)]
fn warp_gemv<I: Float, O: Float>(
    lhs: &Tensor<I>, rhs: &Tensor<I>, out: &mut Tensor<O>, k: u32, n: u32,
    #[define(I)] _input: StorageType, #[define(O)] _output: StorageType,
) {
    let column = RUDA_POS_X * 4 + UNIT_POS_Y;
    let lane = UNIT_POS_X;
    let mut accumulator = 0.0f32;
    let mut position = lane;
    while position < k {
        if column < n {
            let a = f32::cast_from(lhs[position as usize]);
            let b = f32::cast_from(rhs[(column * (rhs.stride(1) as u32) + position) as usize]);
            accumulator = fma(a, b, accumulator);
        }
        position += 32;
    }
    let sum = plane_sum(accumulator);
    if lane == 0 && column < n { out[column as usize] = O::cast_from(sum); }
}

#[ruda(launch)]
fn warp_gemv_vector<I: Float, O: Float, N: Size>(
    lhs: &Array<Vector<I, N>>, rhs: &Array<Vector<I, N>>, out: &mut Array<O>,
    k_vectors: u32, n: u32, column_stride: u32,
    #[define(I)] _input: StorageType, #[define(O)] _output: StorageType,
) {
    let column = RUDA_POS_X * 4 + UNIT_POS_Y;
    let lane = UNIT_POS_X;
    let mut accumulator = Vector::<f32, N>::new(0.0f32);
    let mut position = lane;
    while position < k_vectors {
        if column < n {
            let a = Vector::<f32, N>::cast_from(lhs[position as usize]);
            let b = Vector::<f32, N>::cast_from(rhs[(column * column_stride + position) as usize]);
            accumulator = fma(a, b, accumulator);
        }
        position += 32;
    }
    let sum = plane_sum(Vector::vector_sum(accumulator));
    if lane == 0 && column < n { out[column as usize] = O::cast_from(sum); }
}

/// Single-row, transposed-weight F16/BF16 GEMV with F32 accumulation.
pub fn launch_gemv_low_precision<R: Runtime>(
    lhs: RudaTensor<R>, rhs: RudaTensor<R>, out: RudaTensor<R>,
) -> Result<(), MatmulSetupError> {
    let invalid = || MatmulSetupError::InvalidConfig(Box::new(
        "GEMV requires disjoint bounded singleton-row F16/BF16 inputs and contiguous output".to_string()));
    if !matches!(lhs.dtype, DType::BF16 | DType::F16) || rhs.dtype != lhs.dtype
        || (out.dtype != lhs.dtype && out.dtype != DType::F32)
        || [ &lhs, &rhs, &out ].iter().any(|x| x.meta.num_dims()!=2 || x.qparams.is_some()
            || x.device.to_id()!=lhs.device.to_id() || !x.client.same_execution_queue(&lhs.client))
    { return Err(invalid()); }
    let (k,n) = (lhs.meta.shape()[1],rhs.meta.shape()[1]);
    if lhs.meta.shape()[0]!=1 || rhs.meta.shape()[0]!=k || out.meta.shape()[0]!=1 || out.meta.shape()[1]!=n
        || k==0 || n==0 || k>(u32::MAX-32) as usize || n>u32::MAX as usize
        || lhs.meta.strides()[1]!=1 || rhs.meta.strides()[0]!=1
        || rhs.meta.strides()[1]<k || out.meta.strides()[1]!=1
    { return Err(invalid()); }
    for tensor in [&lhs,&rhs,&out] {
        let bytes=tensor.dtype.size();
        let Some(last)=tensor.meta.shape().iter().zip(tensor.meta.strides().iter())
            .try_fold(0usize, |span,(&size,&stride)| span.checked_add(size.checked_sub(1)?.checked_mul(stride)?))
        else { return Err(invalid()); };
        if last>=u32::MAX as usize || tensor.handle.offset_start.unwrap_or(0)%bytes as u64!=0
            || last.checked_add(1).and_then(|v|v.checked_mul(bytes))
                .is_none_or(|span|span as u64>tensor.handle.size_in_used())
        { return Err(invalid()); }
    }
    let output_id=out.handle.memory.clone().binding().id();
    let output_begin=out.handle.offset_start.unwrap_or(0);
    let Some(output_end)=out.handle.size().checked_sub(out.handle.offset_end.unwrap_or(0))
    else { return Err(invalid()); };
    for input in [&lhs,&rhs] {
        let Some(input_end)=input.handle.size().checked_sub(input.handle.offset_end.unwrap_or(0))
        else { return Err(invalid()); };
        if input.handle.memory.clone().binding().id()==output_id
            && input.handle.offset_start.unwrap_or(0)<output_end && output_begin<input_end
        { return Err(invalid()); }
    }
    let client=lhs.client.clone();
    let hardware=&client.properties().hardware;
    if hardware.plane_size_min!=32 || hardware.plane_size_max!=32
        || !client.features().plane.contains(Plane::Ops) || hardware.max_units_per_ruda<128
        || hardware.max_ruda_dim.0<32 || hardware.max_ruda_dim.1<4
        || n.div_ceil(4)>hardware.max_ruda_count.0 as usize
    { return Err(invalid()); }
    let vector_size = 8usize;
    let aligned = k % vector_size == 0 && rhs.meta.strides()[1] == k
        && [&lhs, &rhs].iter().all(|x| x.handle.offset_start.unwrap_or(0) % (vector_size * x.dtype.size()) as u64 == 0);
    if aligned {
        warp_gemv_vector::launch::<R>(&client,RudaCount::Static(n.div_ceil(4) as u32,1,1),RudaDim::new_2d(32,4),
            vector_size, lhs.clone().into_array_arg(),rhs.clone().into_array_arg(),out.clone().into_array_arg(),
            (k / vector_size) as u32,n as u32,(rhs.meta.strides()[1] / vector_size) as u32,
            lhs.dtype.into(),out.dtype.into());
    } else {
        warp_gemv::launch::<R>(&client,RudaCount::Static(n.div_ceil(4) as u32,1,1),RudaDim::new_2d(32,4),
            lhs.clone().into_tensor_arg(),rhs.into_tensor_arg(),out.clone().into_tensor_arg(),
            k as u32,n as u32,lhs.dtype.into(),out.dtype.into());
    }
    Ok(())
}
