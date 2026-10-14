use super::{GroupedMatmulError,GroupedStrategy,kernel,regroup};
use ruda_core::{device::Device,tensor::DType};
use ruda_kernel::{dsl::{Runtime,calculate_ruda_count_elemwise,prelude::*},tensor::RudaTensor};

pub struct GroupedKernelPlan<R:Runtime> {
    pub kernels:Vec<PreparedKernel<R>>,
    pub workspace:Vec<RudaTensor<R>>,
}

/// Prepares the same arbitrary-row expert GEMM as the immediate entry point.
/// Counts, offsets and row regrouping are recomputed on the device at replay.
///
/// # Safety
/// Output must not alias any input. Bindings and returned workspace must remain
/// valid, with no concurrent writers, until every replay completes.
#[allow(unsafe_code)]
pub unsafe fn prepare_grouped_matmul_nt<R:Runtime>(input:&RudaTensor<R>,weights:&RudaTensor<R>,
    row_experts:&RudaTensor<R>,output:&RudaTensor<R>,strategy:GroupedStrategy)
    ->Result<GroupedKernelPlan<R>,GroupedMatmulError>
{
    if input.meta.num_dims()!=2 || weights.meta.num_dims()!=3 {
        return Err(GroupedMatmulError("grouped matmul requires [M,K] input and [E,N,K] weights"));
    }
    let (m,k,e,n)=(input.meta.shape()[0],input.meta.shape()[1],weights.meta.shape()[0],weights.meta.shape()[1]);
    if k==0 || n==0 || e==0 && m!=0 || weights.meta.shape()[2]!=k
        || row_experts.meta.shape()[..]!=[m] || row_experts.dtype!=DType::U32
        || output.meta.shape()[..]!=[m,n] || output.dtype!=input.dtype || weights.dtype!=input.dtype
        || !matches!(input.dtype,DType::F32|DType::F16|DType::BF16) {
        return Err(GroupedMatmulError("invalid grouped graph shape/dtype"));
    }
    for t in [input,weights,row_experts,output] {
        if t.qparams.is_some() || t.device.to_id()!=input.device.to_id()
            || !t.client.same_execution_queue(&input.client)
            || t.meta.shape().iter().try_fold(1usize,|n,&d|n.checked_mul(d)).is_none_or(|n|n>u32::MAX as usize)
            || t.meta.num_elements()!=0 && !t.is_contiguous() {
            return Err(GroupedMatmulError("grouped graph requires contiguous non-quantized same-device/same-queue U32-indexed bindings"));
        }
    }
    if strategy!=GroupedStrategy::Scalar {
        if regroup::supported(input,e,n) {return Ok(regroup::prepare(input,weights,row_experts,output));}
        if strategy==GroupedStrategy::TensorCore {
            return Err(GroupedMatmulError("requested grouped 16x16x16 Tensor Core configuration is unavailable"));
        }
    }
    let mut kernels=Vec::new();
    if m!=0 {
        let dim=RudaDim::new(input.client.properties(),m*n);
        kernels.push(kernel::grouped_nt::prepare::<R>(&input.client,
            calculate_ruda_count_elemwise(&input.client,m*n,dim),dim,input.clone().into_array_arg(),
            weights.clone().into_array_arg(),row_experts.clone().into_array_arg(),output.clone().into_array_arg(),
            e as u32,n as u32,k as u32,input.dtype.into()));
    }
    Ok(GroupedKernelPlan {kernels,workspace:Vec::new()})
}
