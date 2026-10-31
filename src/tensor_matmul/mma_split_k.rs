use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::{dsl::prelude::*,tensor::RudaTensor};
use ruda_core::tensor::DType;

#[ruda(launch)]
fn reduce_vector<O:Float>(partials:&Array<Vector<f32,Const<4>>>,out:&mut Array<Vector<O,Const<4>>>,bias:&Array<O>,
    matrix_elements:usize,n:usize,alpha:f32,beta:f32,bias_row:usize,bias_column:usize,
    #[comptime] partitions:usize,#[comptime] epilogue:bool,#[comptime] use_bias:bool,#[define(O)] _output:StorageType) {
    let p=ABSOLUTE_POS as usize;
    if p<out.len() {
        let batch=p/(matrix_elements/4);let within=p%(matrix_elements/4);
        let mut sum=partials[batch*partitions*(matrix_elements/4)+within];
        #[unroll]
        for partition in 1usize..partitions {sum+=partials[(batch*partitions+partition)*(matrix_elements/4)+within];}
        if comptime!(epilogue) {
            #[unroll]
            for lane in 0usize..4 {
                let mut value=alpha*sum[lane];
                if comptime!(use_bias) {
                    let position=within*4+lane;
                    value+=beta*f32::cast_from(bias[(position/n)*bias_row+(position%n)*bias_column]);
                }
                sum[lane]=value;
            }
        }
        out[p]=Vector::cast_from(sum);
    }
}

#[ruda(launch)]
fn reduce_scalar<O:Float>(partials:&Array<f32>,out:&mut Array<O>,bias:&Array<O>,
    matrix_elements:usize,n:usize,alpha:f32,beta:f32,bias_row:usize,bias_column:usize,
    #[comptime] partitions:usize,#[comptime] epilogue:bool,#[comptime] use_bias:bool,#[define(O)] _output:StorageType) {
    let p=ABSOLUTE_POS as usize;
    if p<out.len() {
        let batch=p/matrix_elements;let within=p%matrix_elements;
        let mut sum=partials[batch*partitions*matrix_elements+within];
        #[unroll]
        for partition in 1usize..partitions {sum+=partials[(batch*partitions+partition)*matrix_elements+within];}
        if comptime!(epilogue) {
            sum=alpha*sum;
            if comptime!(use_bias) {sum+=beta*f32::cast_from(bias[(within/n)*bias_row+(within%n)*bias_column]);}
        }
        out[p]=O::cast_from(sum);
    }
}

pub(super) fn prepare_configured<R:Runtime>(lhs:RudaTensor<R>,rhs:RudaTensor<R>,out:RudaTensor<R>,
    epilogue:Option<(RudaTensor<R>,f32,f32)>,extent_m:usize,extent_n:usize,warp_rows:usize,warp_cols:usize,
    stages:usize,requested_partitions:usize)->Vec<PreparedKernel<R>> {
    let plan=super::gemm_policy::GemmPlan {blackwell:false,bm:extent_m,bn:extent_n,warp_rows,warp_cols,stages,partitions:requested_partitions};
    prepare_plan(lhs,rhs,out,epilogue,plan)
}

pub(super) fn prepare_plan<R:Runtime>(lhs:RudaTensor<R>,rhs:RudaTensor<R>,out:RudaTensor<R>,
    epilogue:Option<(RudaTensor<R>,f32,f32)>,mut plan:super::gemm_policy::GemmPlan)->Vec<PreparedKernel<R>> {
    let requested_partitions=plan.partitions;
    let rank=lhs.meta.num_dims();let m=lhs.meta.shape()[rank-2];let k=lhs.meta.shape()[rank-1];
    let n=*out.meta.shape().last().unwrap();let batches=out.meta.num_elements()/(m*n);
    assert!(requested_partitions>1 && k%64==0 && requested_partitions<=k/64);
    let partition_k=(k/64).div_ceil(requested_partitions)*64;let partitions=k.div_ceil(partition_k);
    assert!(partitions>1);
    let elements=out.meta.num_elements().checked_mul(partitions).expect("split-K workspace overflow");
    assert!(elements<=u32::MAX as usize && batches*partitions<=out.client.properties().hardware.max_ruda_count.2 as usize);
    let bytes=elements.checked_mul(4).expect("split-K workspace byte overflow");
    let client=out.client.clone();
    let partials=RudaTensor::new_contiguous(client.clone(),out.device.clone(),[batches*partitions,m,n].into(),client.empty(bytes),DType::F32);
    plan.partitions=partitions;
    let product=if plan.blackwell {
        super::blackwell::prepare(lhs,rhs,partials.clone(),None,plan)
    } else {
        super::mma_pipeline::prepare_split_partials(lhs,rhs,partials.clone(),plan.bm,plan.bn,plan.warp_rows,plan.warp_cols,plan.stages,partitions,partition_k)
    };
    let has_epilogue=epilogue.is_some();let (bias,alpha,beta)=epilogue.unwrap_or_else(||(out.clone(),1.0,0.0));
    let (bias_row,bias_column)=(bias.meta.strides()[0],bias.meta.strides()[1]);
    let vector=m*n%4==0;let width=if vector {4}else {1};
    let grid=RudaCount::Static(out.meta.num_elements().div_ceil(width*256) as u32,1,1);let output=out.dtype.into();
    let reduction=if vector {
        reduce_vector::prepare::<R>(&client,grid,RudaDim::new_1d(256),partials.into_array_arg(),out.into_array_arg(),bias.into_array_arg(),
            m*n,n,alpha,beta,bias_row,bias_column,partitions,has_epilogue,has_epilogue && beta!=0.0,output)
    }else {
        reduce_scalar::prepare::<R>(&client,grid,RudaDim::new_1d(256),partials.into_array_arg(),out.into_array_arg(),bias.into_array_arg(),
            m*n,n,alpha,beta,bias_row,bias_column,partitions,has_epilogue,has_epilogue && beta!=0.0,output)
    };
    vec![product,reduction]
}

pub(super) fn prepare<R:Runtime>(lhs:RudaTensor<R>,rhs:RudaTensor<R>,out:RudaTensor<R>,
    epilogue:Option<(RudaTensor<R>,f32,f32)>)->Option<Vec<PreparedKernel<R>>> {
    let rank=lhs.meta.num_dims();let m=lhs.meta.shape()[rank-2];let k=lhs.meta.shape()[rank-1];let n=*out.meta.shape().last()?;
    let h=&out.client.properties().hardware;let sms=h.num_streaming_multiprocessors? as usize;
    if k<64 || k%64!=0 || m<2 || n<64 || sms==0 {return None;}
    let batches=out.meta.num_elements()/(m*n);
    let plan=super::gemm_policy::plan(m,n,k,batches,sms,false,true);
    if plan.shared_bytes()>h.max_shared_memory_size as usize
        || plan.warp_rows*plan.warp_cols*32>h.max_units_per_ruda as usize
        || plan.warp_rows*plan.warp_cols>h.max_ruda_dim.1 as usize {return None;}
    let partitions=plan.partitions;
    let elements=out.meta.num_elements().checked_mul(partitions)?;
    if partitions<2 || elements>u32::MAX as usize || batches*partitions>h.max_ruda_count.2 as usize
        || out.meta.num_elements().div_ceil(256)>h.max_ruda_count.0 as usize {return None;}
    Some(prepare_plan(lhs,rhs,out,epilogue,plan))
}
