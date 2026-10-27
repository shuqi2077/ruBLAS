use super::init_matmul_output;
use ruda_kernel::{dsl::Runtime, tensor::{RudaTensor, dequantize::dequantize}};
use ruda_core::tensor::{DType, QTensorPrimitive};
use ruda_core::quant::scheme::QuantLevel;
use crate::kernel_ir::{
        definition::{MatmulElems, MatmulGlobalElems, MatmulSetupError},
        launch::Strategy,
    };
use ruda_kernel::tiling::InputBinding;

#[cfg(feature = "tensor-matmul-autotune")]
use super::matmul_autotune_with_precision;

pub use crate::kernel_ir::definition::F32MathMode;

pub fn launch_matmul_pipelined<R: Runtime>(lhs:RudaTensor<R>,rhs:RudaTensor<R>,out:RudaTensor<R>,
    f32_math:F32MathMode)->Result<(),MatmulSetupError> {
    launch_matmul_pipelined_counted(lhs,rhs,out,f32_math).map(|_|())
}

pub fn launch_matmul_pipelined_counted<R: Runtime>(lhs:RudaTensor<R>,rhs:RudaTensor<R>,out:RudaTensor<R>,
    f32_math:F32MathMode)->Result<usize,MatmulSetupError> {
    let kernels=prepare_matmul_pipelined_inner(lhs,rhs,out,f32_math,None,true)?;let count=kernels.len();
    for kernel in kernels {
        let (task,count,args,origin)=kernel.into_parts();origin.launch(task,count,args);
    }
    Ok(count)
}

pub fn prepare_matmul_pipelined<R: Runtime>(lhs:RudaTensor<R>,rhs:RudaTensor<R>,out:RudaTensor<R>,
    f32_math:F32MathMode)->Result<ruda_kernel::dsl::prelude::PreparedKernel<R>,MatmulSetupError> {
    prepare_matmul_pipelined_inner(lhs,rhs,out,f32_math,None,false).map(|mut kernels|kernels.remove(0))
}

pub fn launch_addmm_pipelined<R: Runtime>(bias:RudaTensor<R>,lhs:RudaTensor<R>,rhs:RudaTensor<R>,out:RudaTensor<R>,
    alpha:f32,beta:f32,f32_math:F32MathMode)->Result<(),MatmulSetupError> {
    launch_addmm_pipelined_counted(bias,lhs,rhs,out,alpha,beta,f32_math).map(|_|())
}

pub fn launch_addmm_pipelined_counted<R: Runtime>(bias:RudaTensor<R>,lhs:RudaTensor<R>,rhs:RudaTensor<R>,out:RudaTensor<R>,
    alpha:f32,beta:f32,f32_math:F32MathMode)->Result<usize,MatmulSetupError> {
    if alpha==0.0 || out.meta.num_dims()!=2 || bias.meta.shape()!=out.meta.shape() || bias.dtype!=out.dtype
        || out.dtype!=lhs.dtype || bias.qparams.is_some() || !bias.client.same_execution_queue(&out.client) {
        return Err(MatmulSetupError::InvalidConfig(Box::new("pipelined addmm requires a matching two-dimensional bias and output")));
    }
    let kernels=prepare_matmul_pipelined_inner(lhs,rhs,out,f32_math,Some((bias,alpha,beta)),true)?;let count=kernels.len();
    for kernel in kernels {
        let (task,count,args,origin)=kernel.into_parts();origin.launch(task,count,args);
    }
    Ok(count)
}

fn prepare_matmul_pipelined_inner<R: Runtime>(lhs:RudaTensor<R>,rhs:RudaTensor<R>,out:RudaTensor<R>,
    f32_math:F32MathMode,epilogue:Option<(RudaTensor<R>,f32,f32)>,allow_split:bool)->Result<Vec<ruda_kernel::dsl::prelude::PreparedKernel<R>>,MatmulSetupError> {
    use crate::kernel_ir::{components::{tile::TileMatmulKind,stage::PartitionBuffering},
        definition::{MatmulProblem,TilingBlueprint,TilingScheme,MatmulGlobalElems},
        routines::{BlueprintStrategy,double_buffering::AsyncCyclicDoubleBufferingAlgorithm,simple::SimpleAlgorithm}};
    use ruda_kernel::{dsl::prelude::AddressType,tiling::{TileSize,PartitionSize,StageSize,
        ruda_count::{RudaCountStrategy,GlobalOrder,HyperrudaBlueprint}}};
    fn canonical<R:Runtime>(tensor:&RudaTensor<R>,transposed:bool)->bool {
        let shape=tensor.meta.shape();let strides=tensor.meta.strides();let rank=shape.len();
        if rank<2 {return false;}
        let (rows,columns)=(shape[rank-2],shape[rank-1]);
        let (row_stride,column_stride)=if transposed {(1,rows)}else {(columns,1)};
        if (rows>1 && strides[rank-2]!=row_stride) || (columns>1 && strides[rank-1]!=column_stride) {return false;}
        let Some(mut span)=rows.checked_mul(columns) else {return false;};
        for axis in (0..rank-2).rev() {
            if shape[axis]>1 && strides[axis]!=span {return false;}
            let Some(next)=span.checked_mul(shape[axis]) else {return false;};span=next;
        }
        true
    }
    if !matches!(lhs.dtype,DType::F16|DType::BF16) || rhs.dtype!=lhs.dtype
        || !matches!(out.dtype,DType::F16|DType::BF16|DType::F32)
        || [&lhs,&rhs,&out].iter().any(|t|t.qparams.is_some() || t.meta.num_dims()<2
            || !t.client.same_execution_queue(&out.client))
        || !canonical(&lhs,false) || !canonical(&out,false)
        || !(canonical(&rhs,false) || canonical(&rhs,true)) {
        return Err(MatmulSetupError::InvalidConfig(Box::new("prepared pipelined matmul requires matching F16/BF16 inputs and one execution queue")));
    }
    let globals=MatmulGlobalElems{f32_math,lhs:lhs.dtype.into(),rhs:rhs.dtype.into(),out:out.dtype.into()};
    let problem=MatmulProblem::from_shapes_and_strides(lhs.meta.shape().clone(),rhs.meta.shape().clone(),out.meta.shape().clone(),
        lhs.meta.strides().clone(),rhs.meta.strides().clone(),out.meta.strides().clone(),globals.clone(),AddressType::U32,None,None)?;
    let types=MatmulElems::from_globals(&globals);
    let client=out.client.clone();
    let mma=ruda_core::ir::features::MmaConfig {
        a_type:globals.lhs,b_type:globals.rhs,
        cd_type:ruda_core::ir::ElemType::Float(ruda_core::ir::FloatKind::F32).into(),m:16,n:8,k:16,
    };
    if client.features().matmul.mma.contains(&mma) && client.features().alignment
        && client.features().copy_async && client.features().matmul.ldmatrix.contains(&globals.lhs)
        && crate::kernel_ir::components::global::read::validate_async_barrier(client.properties()).is_ok()
        && client.properties().hardware.plane_size_min==32 && client.properties().hardware.plane_size_max==32
        && client.properties().hardware.max_units_per_ruda>=128
        && client.properties().hardware.max_ruda_dim.0>=32 && client.properties().hardware.max_ruda_dim.1>=4
        && client.properties().hardware.max_shared_memory_size>=32776
        && problem.m>=64 && problem.n>=64 && problem.k>=32
        && problem.k%32==0
        && problem.rhs_layout==ruda_kernel::tiling::MatrixLayout::ColMajor
        && lhs.meta.shape()[..lhs.meta.num_dims()-2]==out.meta.shape()[..out.meta.num_dims()-2]
        && rhs.meta.shape()[..rhs.meta.num_dims()-2]==out.meta.shape()[..out.meta.num_dims()-2]
        && [&lhs,&rhs].iter().all(|t|t.handle.offset_start.unwrap_or(0)%16==0)
        && [&lhs,&rhs,&out].iter().all(|t|t.meta.num_elements()<=u32::MAX as usize)
        && problem.m.div_ceil(64)<=client.properties().hardware.max_ruda_count.0 as usize
        && problem.n.div_ceil(64)<=client.properties().hardware.max_ruda_count.1 as usize
        && out.meta.num_elements()/(problem.m*problem.n)<=client.properties().hardware.max_ruda_count.2 as usize {
        if allow_split {
            if let Some(kernels)=super::mma_split_k::prepare(lhs.clone(),rhs.clone(),out.clone(),epilogue.clone()) {return Ok(kernels);}
        }
        return Ok(vec![if let Some((bias,alpha,beta))=epilogue {
            super::mma_pipeline::prepare_addmm(lhs,rhs,out,bias,alpha,beta)
        } else {super::mma_pipeline::prepare(lhs,rhs,out)}]);
    }
    if epilogue.is_some() {
        return Err(MatmulSetupError::InvalidConfig(Box::new("pipelined addmm is unavailable for this layout or device")));
    }
    let left=InputBinding::new(lhs.binding(),globals.lhs);let right=InputBinding::new(rhs.binding(),globals.rhs);
    let output=out.binding();
    if client.features().matmul.mma.contains(&mma) && client.features().alignment
        && problem.m>=64 && problem.n>=64 && problem.k>=64
        && problem.m%64==0 && problem.n%64==0 && problem.k%64==0
        && problem.rhs_layout==ruda_kernel::tiling::MatrixLayout::ColMajor {
        use crate::kernel_ir::definition::SwizzleModes;
        use ruda_kernel::tiling::stage::SwizzleMode;
        let large=problem.m%128==0 && problem.n%128==0;
        let scheme=TilingScheme::builder().with_tile_size(TileSize::new(16,8,16))
            .with_partition_size(PartitionSize::new(2,8,4))
            .with_stage_size(StageSize::new(if large {4}else {2},if large {2}else {1},1)).build().unwrap();
        let hyper=HyperrudaBlueprint::builder().global_order(GlobalOrder::SwizzleRow(4))
            .ruda_count_strategy(RudaCountStrategy::FromProblem).build();
        let plan=TilingBlueprint::builder(TileMatmulKind::Mma,scheme,32,&problem)
            .partition_buffering(PartitionBuffering::Double).hyperruda_blueprint(hyper)
            .shared_swizzle(SwizzleModes {lhs:SwizzleMode::B128,rhs:SwizzleMode::B128,..Default::default()}).build();
        return crate::kernel_ir::launch::launch_tiling::prepare_ref::<R,AsyncCyclicDoubleBufferingAlgorithm>(
            &client,left,right,output,&BlueprintStrategy::Forced(plan),&types).map(|kernel|vec![kernel]);
    }
    if problem.m<64 || problem.n<128 || problem.k<128 {
        return crate::kernel_ir::launch::launch_tiling::prepare_ref::<R,SimpleAlgorithm>(
            &client,left,right,output,&BlueprintStrategy::default(),&types).map(|kernel|vec![kernel]);
    }
    let strategy=if problem.lhs_shape.len()==2 && problem.m>=256 && problem.n>=1024 && problem.k>=1024
        && problem.m%128==0 && problem.n%128==0 && problem.k%64==0
        && problem.rhs_strides[0]==1 && problem.rhs_strides[1]==problem.k {
        let scheme=TilingScheme::builder().with_tile_size(TileSize::new(16,16,16))
            .with_partition_size(PartitionSize::new(2,8,4)).with_stage_size(StageSize::new(4,1,1)).build().unwrap();
        let hyper=HyperrudaBlueprint::builder().global_order(GlobalOrder::SwizzleRow(4))
            .ruda_count_strategy(RudaCountStrategy::FromProblem).build();
        let plan=TilingBlueprint::builder(TileMatmulKind::Cmma,scheme,32,&problem)
            .partition_buffering(PartitionBuffering::Double).hyperruda_blueprint(hyper).build();
        BlueprintStrategy::Forced(plan)
    } else {BlueprintStrategy::default()};
    crate::kernel_ir::launch::launch_tiling::prepare_ref::<R,AsyncCyclicDoubleBufferingAlgorithm>(
        &client,left,right,output,&strategy,&types).map(|kernel|vec![kernel])
}

/// The strategy to be used when launching a matmul kernel.
pub enum MatmulStrategy {
    #[cfg(feature = "tensor-matmul-autotune")]
    /// Using autotune to choose the best kernel based on runtime information.
    Autotune,
    /// Ruda implementation of matmul.
    Ruda,
    /// Tensor Core path with the partial K32 tile first; no materialized padding.
    CmmaResidueFirst,
    /// One output element per unit, used by runtimes without plane arithmetic.
    Naive,
}

impl Default for MatmulStrategy {
    fn default() -> Self {
        // if autotune is enabled, default to autotune
        #[cfg(feature = "tensor-matmul-autotune")]
        return MatmulStrategy::Autotune;

        #[cfg(not(feature = "tensor-matmul-autotune"))]
        MatmulStrategy::Ruda
    }
}

/// Launch a matmul kernel using the given strategy.
pub fn matmul<R: Runtime>(
    lhs: RudaTensor<R>,
    rhs: RudaTensor<R>,
    out: Option<RudaTensor<R>>,
    strategy: MatmulStrategy,
    out_dtype: DType,
) -> Result<RudaTensor<R>, MatmulSetupError> {
    matmul_with_precision(lhs, rhs, out, strategy, out_dtype, F32MathMode::Strict)
}

/// Launch matmul with an explicit, per-call F32 arithmetic policy.
pub fn matmul_with_precision<R: Runtime>(
    lhs: RudaTensor<R>,
    rhs: RudaTensor<R>,
    out: Option<RudaTensor<R>>,
    strategy: MatmulStrategy,
    out_dtype: DType,
    f32_math: F32MathMode,
) -> Result<RudaTensor<R>, MatmulSetupError> {
    let use_fast_path = matches!(strategy, MatmulStrategy::Ruda);
    #[cfg(feature = "tensor-matmul-autotune")]
    let use_fast_path = use_fast_path || matches!(strategy, MatmulStrategy::Autotune);
    let out = if use_fast_path && matches!(lhs.dtype, DType::F16 | DType::BF16)
        && lhs.meta.num_elements() > 0 && rhs.meta.num_elements() > 0
    {
        let output = out.unwrap_or_else(|| init_matmul_output(&lhs, &rhs, out_dtype));
        let rank = lhs.meta.num_dims();
        let result = if rank >= 2 && lhs.meta.shape()[rank - 2] == 1
            && [&lhs, &rhs, &output].iter().all(|tensor| {
                let rank = tensor.meta.num_dims();
                rank >= 2 && tensor.meta.shape()[..rank - 2].iter().all(|&size| size == 1)
            })
        {
            let matrix = |mut tensor: RudaTensor<R>| {
                let rank = tensor.meta.num_dims();
                let shape = [tensor.meta.shape()[rank - 2], tensor.meta.shape()[rank - 1]];
                let strides = [tensor.meta.strides()[rank - 2], tensor.meta.strides()[rank - 1]];
                *tensor.meta = ruda_core::tensor::Metadata::new(shape, strides);
                tensor
            };
            super::launch_gemv_low_precision(matrix(lhs.clone()), matrix(rhs.clone()), matrix(output.clone()))
        } else if !matches!(strategy, MatmulStrategy::Ruda) && rank >= 2 && lhs.meta.shape()[rank - 2] > 1
            && lhs.client.features().matmul.mma.iter().chain(lhs.client.features().matmul.cmma.iter())
                .any(|config| config.a_type == lhs.dtype.into() && config.b_type == rhs.dtype.into()
                    && config.cd_type == DType::F32.into())
        {
            launch_matmul_pipelined(lhs.clone(), rhs.clone(), output.clone(), f32_math)
        } else {
            Err(MatmulSetupError::InvalidConfig(Box::new("native matmul fast path is unavailable")))
        };
        match result {
            Ok(()) => return Ok(output),
            Err(MatmulSetupError::Unavailable(_) | MatmulSetupError::InvalidConfig(_)
                | MatmulSetupError::Vectorization(_)) => Some(output),
            Err(error) => return Err(error),
        }
    } else { out };
    match strategy {
        MatmulStrategy::CmmaResidueFirst => {
            if !matches!(lhs.dtype, DType::BF16 | DType::F16) || lhs.dtype != rhs.dtype
                || lhs.qparams.is_some() || rhs.qparams.is_some()
            {
                return Err(MatmulSetupError::InvalidConfig(Box::new("residue-first matmul requires matching unquantized BF16 or F16 inputs".to_string())));
            }
            let out = out.unwrap_or_else(|| init_matmul_output(&lhs, &rhs, out_dtype));
            launch_matmul(&Strategy::SimpleCyclicCmmaResidueFirst, lhs, rhs, out.clone())?;
            Ok(out)
        }
        MatmulStrategy::Ruda => {
            let out = out.unwrap_or_else(|| init_matmul_output(&lhs, &rhs, out_dtype));
            if super::skinny::try_launch(&lhs, &rhs, &out) {
                return Ok(out);
            }
            if super::f32_tiled::try_launch(&lhs, &rhs, &out, f32_math)? {
                return Ok(out);
            }
            if super::accurate::try_launch(&lhs, &rhs, &out, f32_math)? {
                return Ok(out);
            }
            if super::training_gemm::matmul_low_precision_tiled(&lhs, &rhs, &out) {
                return Ok(out);
            }
            if matches!(lhs.dtype, DType::BF16 | DType::F16) && lhs.dtype == rhs.dtype
                && lhs.qparams.is_none() && rhs.qparams.is_none()
                && lhs.meta.num_dims() == 2 && rhs.meta.num_dims() == 2
                && lhs.meta.shape()[0] >= 32 && lhs.meta.shape()[1] >= 256 && rhs.meta.shape()[1] >= 128
            {
                if lhs.dtype == DType::BF16 && rhs.meta.shape()[0] == lhs.meta.shape()[1]
                    && out.meta.num_dims() == 2
                    && out.meta.shape()[..] == [lhs.meta.shape()[0], rhs.meta.shape()[1]]
                    && matches!(out.dtype, DType::BF16 | DType::F32) && out.qparams.is_none()
                    && out.meta.strides()[1] == 1 && out.meta.strides()[0] == rhs.meta.shape()[1]
                    && lhs.client.same_execution_queue(&rhs.client) && lhs.client.same_execution_queue(&out.client)
                {
                    return Ok(super::training_tune::matmul(lhs, rhs, out, f32_math));
                }
            }
            let rank = lhs.meta.num_dims();
            if matches!(lhs.dtype, DType::F16 | DType::BF16) && rank >= 2
                && lhs.meta.shape()[rank - 2] > 1
                && lhs.meta.num_elements() > 0 && rhs.meta.num_elements() > 0
                && lhs.client.features().matmul.mma.iter().chain(lhs.client.features().matmul.cmma.iter())
                    .any(|config| config.a_type == lhs.dtype.into() && config.b_type == rhs.dtype.into()
                        && config.cd_type == DType::F32.into())
            {
                match launch_matmul_pipelined(lhs.clone(), rhs.clone(), out.clone(), f32_math) {
                    Ok(()) => return Ok(out),
                    Err(MatmulSetupError::Unavailable(_) | MatmulSetupError::InvalidConfig(_)
                        | MatmulSetupError::Vectorization(_)) => {},
                    Err(error) => return Err(error),
                }
            }
            if matches!(lhs.dtype, DType::BF16 | DType::F16) && lhs.dtype == rhs.dtype
                && lhs.qparams.is_none() && rhs.qparams.is_none()
                && lhs.meta.num_dims() == 2 && rhs.meta.num_dims() == 2
                && lhs.meta.shape()[0] >= 32 && lhs.meta.shape()[1] >= 256 && rhs.meta.shape()[1] >= 128
            {
                match launch_matmul_with_precision(&Strategy::DoubleAsyncCyclicCmma(Default::default()),
                    lhs.clone(), rhs.clone(), out.clone(), f32_math)
                {
                    Ok(()) => return Ok(out),
                    Err(MatmulSetupError::Unavailable(_) | MatmulSetupError::InvalidConfig(_)
                        | MatmulSetupError::Vectorization(_)) => {},
                    Err(error) => return Err(error),
                }
            }
            if lhs.dtype == DType::F64 {
                launch_matmul_naive(&Strategy::Naive, lhs, rhs, out.clone())?;
            } else {
                launch_matmul_with_precision(&Default::default(), lhs, rhs, out.clone(), f32_math)?;
            }
            Ok(out)
        }
        MatmulStrategy::Naive => {
            let out = out.unwrap_or_else(|| init_matmul_output(&lhs, &rhs, out_dtype));
            launch_matmul_naive(&Strategy::Naive, lhs, rhs, out.clone())?;
            Ok(out)
        }
        #[cfg(feature = "tensor-matmul-autotune")]
        MatmulStrategy::Autotune => Ok(matmul_autotune_with_precision(lhs, rhs, out, out_dtype, f32_math)),
    }
}

pub(crate) fn launch_matmul_naive<R: Runtime>(
    strategy: &Strategy,
    mut lhs: RudaTensor<R>,
    mut rhs: RudaTensor<R>,
    out: RudaTensor<R>,
) -> Result<(), MatmulSetupError> {
    // Naive has very specific layout requirements for block scaled tensors, so we need to manually
    // dequantize if it fails to launch normally. This is because naive is assumed to always work.
    if lhs.qparams.is_some() || rhs.qparams.is_some() {
        match launch_matmul(strategy, lhs.clone(), rhs.clone(), out.clone()) {
            Err(_) => {
                if lhs.qparams.is_some() {
                    lhs = dequantize(lhs, out.dtype);
                }
                if rhs.qparams.is_some() {
                    rhs = dequantize(rhs, out.dtype);
                }
                launch_matmul(strategy, lhs, rhs, out)
            }
            Ok(_) => Ok(()),
        }
    } else {
        launch_matmul(strategy, lhs, rhs, out)
    }
}

pub(crate) fn launch_matmul<R: Runtime>(
    strategy: &Strategy,
    lhs: RudaTensor<R>,
    rhs: RudaTensor<R>,
    out: RudaTensor<R>,
) -> Result<(), MatmulSetupError> {
    launch_matmul_with_precision(strategy, lhs, rhs, out, F32MathMode::Strict)
}

pub(crate) fn launch_matmul_with_precision<R: Runtime>(
    strategy: &Strategy,
    lhs: RudaTensor<R>,
    mut rhs: RudaTensor<R>,
    out: RudaTensor<R>,
    f32_math: F32MathMode,
) -> Result<(), MatmulSetupError> {
    let client = &out.client;

    let lhs_quant_handles = lhs.quantized_handles();
    let out_dtype: DType = out.dtype;

    let (lhs_dtype, lhs_handle) = match lhs_quant_handles {
        None => {
            let lhs_dtype = lhs.dtype;
            (
                lhs_dtype,
                InputBinding::new(lhs.binding(), lhs_dtype.into()),
            )
        }
        Some((data, scale)) => {
            let scheme = *lhs.scheme();
            let data_dtype = data.dtype;
            let scale_dtype = scale.dtype;
            (
                out_dtype,
                InputBinding::quantized(
                    data.binding(),
                    scale.binding(),
                    lhs.meta.shape().clone(),
                    scheme,
                    data_dtype.into(),
                    scale_dtype.into(),
                ),
            )
        }
    };

    let rhs_quant_handles = rhs.quantized_handles();

    let (rhs_dtype, rhs_handle) = match rhs_quant_handles {
        None => (
            lhs_dtype,
            InputBinding::new(rhs.binding(), lhs_dtype.into()),
        ),
        Some((data, scale)) => {
            // Extremely hacky fix to ensure naive can run in every case
            if matches!(strategy, Strategy::Naive)
                && matches!(rhs.scheme().level, QuantLevel::Block(_))
            {
                rhs = dequantize(rhs.clone(), lhs_dtype);
                let rhs_dtype = rhs.dtype;
                (
                    lhs_dtype,
                    InputBinding::new(rhs.binding(), rhs_dtype.into()),
                )
            } else {
                let scheme = *rhs.scheme();
                let data_dtype = data.dtype;
                let scale_dtype = scale.dtype;
                (
                    out_dtype,
                    InputBinding::quantized(
                        data.binding(),
                        scale.binding(),
                        rhs.meta.shape().clone(),
                        scheme,
                        data_dtype.into(),
                        scale_dtype.into(),
                    ),
                )
            }
        }
    };

    let mut dtypes = MatmulElems::from_globals(&MatmulGlobalElems {
        f32_math,
        lhs: lhs_dtype.into(),
        rhs: rhs_dtype.into(),
        out: out_dtype.into(),
    });

    crate::kernel_ir::launch::launch_ref(
        strategy,
        client,
        lhs_handle,
        rhs_handle,
        out.clone().binding(),
        &mut dtypes,
    )?;

    Ok(())
}
