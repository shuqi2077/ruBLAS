use ruda_core::tensor::{DType, Shape};
use ruda_kernel::{dsl as kernel_dsl, dsl::prelude::*, tensor::{RudaTensor, allocation::empty_device_contiguous_dtype}};
use crate::kernel_ir::{definition::MatmulSetupError, launch::Strategy};
use super::{F32MathMode, launch_matmul_with_precision};

// Bound the length of Tensor Core accumulation before cancellation-sensitive
// FP16 outputs are rounded. Partial products and their final sum remain FP32.
pub(super) fn try_launch<R: Runtime>(a: &RudaTensor<R>, b: &RudaTensor<R>, out: &RudaTensor<R>,
    math: F32MathMode) -> Result<bool, MatmulSetupError> {
    if a.dtype != DType::F16 || b.dtype != DType::F16 || !matches!(out.dtype, DType::F16 | DType::F32)
        || a.qparams.is_some() || b.qparams.is_some() || out.qparams.is_some()
        || a.meta.num_dims() != 2 || b.meta.num_dims() != 2 || out.meta.num_dims() != 2
        || !a.client.same_execution_queue(&b.client) || !a.client.same_execution_queue(&out.client)
    { return Ok(false); }
    let (m,k,n) = (a.meta.shape()[0],a.meta.shape()[1],b.meta.shape()[1]);
    if m < 32 || k < 8192 || n < 128 || b.meta.shape()[0] != k
        || out.meta.shape()[..] != [m,n] || out.meta.strides()[..] != [n,1]
        || ![a,b].iter().all(|x| x.meta.strides()[0] == 1 || x.meta.strides()[1] == 1)
    { return Ok(false); }
    let splits = k.div_ceil(2048);
    let Some(elements) = m.checked_mul(n).and_then(|v|v.checked_mul(splits)) else { return Ok(false); };
    if elements > 32 * 1024 * 1024 { return Ok(false); }
    let partial = empty_device_contiguous_dtype(a.client.clone(),a.device.clone(),Shape::from([splits,m,n]),DType::F32);
    for split in 0..splits {
        let start = split * 2048;
        let len = (k-start).min(2048);
        let mut left = a.clone();
        left.meta.shape_mut()[1] = len;
        left.handle.offset_start = Some(a.handle.offset_start.unwrap_or(0) + (start*a.meta.strides()[1]*2) as u64);
        let mut right = b.clone();
        right.meta.shape_mut()[0] = len;
        right.handle.offset_start = Some(b.handle.offset_start.unwrap_or(0) + (start*b.meta.strides()[0]*2) as u64);
        let mut output = partial.clone();
        output.meta.remove(0);
        output.handle.offset_start = Some(partial.handle.offset_start.unwrap_or(0) + (split*m*n*4) as u64);
        match launch_matmul_with_precision(&Strategy::DoubleAsyncCyclicCmma(Default::default()),left.clone(),right.clone(),output.clone(),math) {
            Ok(()) => {},
            Err(MatmulSetupError::Unavailable(_) | MatmulSetupError::InvalidConfig(_) | MatmulSetupError::Vectorization(_)) => {
                launch_matmul_with_precision(&Strategy::Auto,left,right,output,math)?;
            },
            Err(error) => return Err(error),
        }
    }
    reduce::launch::<R>(&a.client,RudaCount::Static((m*n).div_ceil(128) as u32,1,1),RudaDim::new_1d(128),
        partial.into_array_arg(),out.clone().into_array_arg(),splits as u32,out.dtype.into());
    Ok(true)
}

#[ruda(launch)]
fn reduce<F: Float>(partial: &Array<f32>, out: &mut Array<F>, splits: u32, #[define(F)] _dtype: StorageType) {
    let pos = ABSOLUTE_POS as usize;
    if pos < out.len() {
        let mut total = 0.0f32;
        for split in 0usize..splits as usize { total += partial[split*out.len()+pos]; }
        out[pos] = F::cast_from(total);
    }
}
