use crate::kernel_ir::definition::MatmulSetupError;
use ruda_core::{device::Device,tensor::{DType,Shape}};
use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::{dsl::{Runtime,calculate_ruda_count_elemwise,prelude::*},tensor::{RudaTensor,allocation::empty_device_contiguous_dtype}};

#[derive(Clone,Copy,Debug)]
pub struct TriangularSolveOptions {
    pub left:bool,pub upper:bool,pub transpose:bool,pub unit_diagonal:bool,pub alpha:f64,
}
impl Default for TriangularSolveOptions {
    fn default()->Self {Self {left:true,upper:true,transpose:false,unit_diagonal:false,alpha:1.}}
}
fn invalid(s:&str)->MatmulSetupError {MatmulSetupError::InvalidConfig(Box::new(s.to_string()))}

/// Solve op(A) X = alpha B (left) or X op(A) = alpha B (right).
/// A is square triangular, B is a matrix; leading batches broadcast and both
/// operands can be strided. Each GPU unit owns one RHS vector and performs
/// substitution without host numerical work. A's opposite triangle and a unit
/// diagonal are never read. A zero nonunit pivot retains IEEE division behavior.
pub fn triangular_solve<R:Runtime>(a:RudaTensor<R>,b:RudaTensor<R>,options:TriangularSolveOptions)
    ->Result<RudaTensor<R>,MatmulSetupError> {
    let (ra,rb)=(a.meta.num_dims(),b.meta.num_dims());
    if ra<2 || rb<2 || !matches!(a.dtype,DType::F32|DType::F64) || b.dtype!=a.dtype {
        return Err(invalid("triangular solve requires matching F32/F64 matrices"));
    }
    let order=a.meta.shape()[ra-1];let rows=b.meta.shape()[rb-2];let cols=b.meta.shape()[rb-1];
    if a.meta.shape()[ra-2]!=order || (if options.left {rows} else {cols})!=order {
        return Err(invalid("triangular solve matrix dimensions are incompatible"));
    }
    let rank=ra.max(rb);let mut dims=vec![1usize;rank];
    for axis in 0..rank-2 {
        let da=if axis+ra>=rank {a.meta.shape()[axis+ra-rank]}else {1};
        let db=if axis+rb>=rank {b.meta.shape()[axis+rb-rank]}else {1};
        if da!=db && da!=1 && db!=1 {return Err(invalid("triangular solve batch dimensions do not broadcast"));}
        dims[axis]=if da==1 {db} else {da};
    }
    dims[rank-2]=rows;dims[rank-1]=cols;
    let count=dims.iter().try_fold(1usize,|x,&d|x.checked_mul(d)).filter(|&x|x<=u32::MAX as usize)
        .ok_or_else(||invalid("triangular solve output exceeds U32 indexing"))?;
    for t in [&a,&b] {
        if t.qparams.is_some() || t.device.to_id()!=a.device.to_id() || !t.client.same_execution_queue(&a.client)
            || t.meta.shape().iter().try_fold(1usize,|x,&d|x.checked_mul(d)).is_none_or(|x|x>u32::MAX as usize)
            || (t.meta.num_elements()!=0 && t.meta.shape().iter().zip(t.meta.strides().iter())
                .try_fold(1usize,|span,(&d,&s)|(d-1).checked_mul(s).and_then(|n|span.checked_add(n)))
                .is_none_or(|span|span>u32::MAX as usize))
        {return Err(invalid("triangular solve requires same-device/queue unquantized tensors"));}
    }
    let out=empty_device_contiguous_dtype(a.client.clone(),a.device.clone(),Shape::from(dims),a.dtype);
    if count==0 {return Ok(out);}
    let vectors=count/order;let dim=RudaDim::new(a.client.properties(),vectors);
    let grid=calculate_ruda_count_elemwise(&a.client,vectors,dim);
    match a.dtype {
        DType::F32=>solve_kernel::launch::<f32,R>(&a.client,grid,dim,a.clone().into_tensor_arg(),b.into_tensor_arg(),out.clone().into_tensor_arg(),
            options.alpha as f32,options.left,options.upper,options.transpose,options.unit_diagonal),
        DType::F64=>solve_kernel::launch::<f64,R>(&a.client,grid,dim,a.clone().into_tensor_arg(),b.into_tensor_arg(),out.clone().into_tensor_arg(),
            options.alpha,options.left,options.upper,options.transpose,options.unit_diagonal),_=>unreachable!(),
    }
    Ok(out)
}

#[ruda]
fn base<F:Float>(input:&Tensor<F>,out:&Tensor<F>,batch:usize)->usize {
    let mut q=batch;let mut offset=0usize;let mut axis=out.rank()-2;
    while axis>0 {
        axis-=1;let coordinate=q%out.shape(axis);q/=out.shape(axis);
        if axis+input.rank()>=out.rank() {let ia=axis+input.rank()-out.rank();if input.shape(ia)!=1 {offset+=coordinate*input.stride(ia);}}
    }
    offset
}

#[ruda(launch)]
fn solve_kernel<F:Float+RudaElement>(a:&Tensor<F>,b:&Tensor<F>,out:&mut Tensor<F>,alpha:F,
    #[comptime] left:bool,#[comptime] upper:bool,#[comptime] transpose:bool,#[comptime] unit:bool) {
    let vector=ABSOLUTE_POS;let rows=out.shape(out.rank()-2);let cols=out.shape(out.rank()-1);
    let mut order=cols;let mut per_batch=rows;
    if comptime!(left) {order=rows;per_batch=cols;}
    if vector>=out.len()/order {terminate!();}
    let batch=vector/per_batch;let rhs=vector%per_batch;
    let ao=base(a,out,batch);let bo=base(b,out,batch);let oo=batch*rows*cols;
    let mut step=0usize;
    while step<order {
        let mut i=step;
        if comptime!((left && upper != transpose)||(!left && upper == transpose)) {i=order-1-step;}
        let mut row=rhs;let mut col=i;if comptime!(left) {row=i;col=rhs;}
        let dest=oo+row*cols+col;
        let mut value=F::new(0.0f32);
        if alpha!=F::new(0.0f32) {
            value=alpha*b[bo+row*b.stride(b.rank()-2)+col*b.stride(b.rank()-1)];
            let mut before=0usize;
            while before<step {
                let mut j=before;
                if comptime!((left && upper != transpose)||(!left && upper == transpose)) {j=order-1-before;}
                let mut ar=i;let mut ac=j;
                if comptime!(!left) {ar=j;ac=i;}
                if comptime!(transpose) {let temp=ar;ar=ac;ac=temp;}
                let coefficient=a[ao+ar*a.stride(a.rank()-2)+ac*a.stride(a.rank()-1)];
                let mut previous=oo+rhs*cols+j;
                if comptime!(left) {previous=oo+j*cols+rhs;}
                value-=coefficient*out[previous];before+=1;
            }
            if comptime!(!unit) {value/=a[ao+i*a.stride(a.rank()-2)+i*a.stride(a.rank()-1)];}
        }
        out[dest]=value;step+=1;
    }
}
