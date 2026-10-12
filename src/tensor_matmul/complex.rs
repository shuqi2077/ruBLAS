use crate::kernel_ir::definition::MatmulSetupError;
use ruda_core::{device::Device, tensor::{DType, calculate_matmul_output}};
use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::{dsl::{Runtime, calculate_ruda_count_elemwise, prelude::*}, tensor::{RudaTensor, allocation::empty_device_contiguous_dtype}};

#[derive(Clone, Copy, Debug)]
pub struct ComplexGemmOptions {
    pub alpha: [f64; 2],
    pub beta: [f64; 2],
    pub transpose_lhs: bool,
    pub transpose_rhs: bool,
    pub conjugate_lhs: bool,
    pub conjugate_rhs: bool,
}
impl Default for ComplexGemmOptions {
    fn default() -> Self { Self { alpha:[1.,0.], beta:[0.,0.], transpose_lhs:false,
        transpose_rhs:false, conjugate_lhs:false, conjugate_rhs:false } }
}
fn invalid(reason:&str)->MatmulSetupError {MatmulSetupError::InvalidConfig(Box::new(reason.to_string()))}

/// Split-storage complex GEMM. Matrix transposition and conjugation are
/// independent; leading batches broadcast. Arbitrary nonnegative strides are
/// read directly, including different real/imaginary strides. No CPU payload
/// transfer or F64 downcast is performed. C, when supplied, has output shape.
pub fn complex_gemm<R:Runtime>(
    ar:RudaTensor<R>,ai:RudaTensor<R>,br:RudaTensor<R>,bi:RudaTensor<R>,
    c:Option<(RudaTensor<R>,RudaTensor<R>)>, mut options:ComplexGemmOptions,
)->Result<(RudaTensor<R>,RudaTensor<R>),MatmulSetupError> {
    if ar.meta.num_dims()<2 || br.meta.num_dims()<2 || ar.meta.shape()!=ai.meta.shape()
        || br.meta.shape()!=bi.meta.shape() || !matches!(ar.dtype,DType::F32|DType::F64)
    {return Err(invalid("complex GEMM requires split F32/F64 matrix tensors with matching component shapes"));}
    if ar.dtype==DType::F32 {
        options.alpha=options.alpha.map(|x|x as f32 as f64);options.beta=options.beta.map(|x|x as f32 as f64);
    }
    let mut ash=ar.meta.shape().clone();let mut bsh=br.meta.shape().clone();
    let ra=ash.len();let rb=bsh.len();
    if options.transpose_lhs {ash.swap(ra-2,ra-1);}
    if options.transpose_rhs {bsh.swap(rb-2,rb-1);}
    let shape=calculate_matmul_output(&ash,&bsh).map_err(|_|invalid("complex GEMM matrix/batch dimensions are incompatible"))?;
    let count=shape.iter().try_fold(1usize,|a,&b|a.checked_mul(b)).filter(|&n|n<=u32::MAX as usize)
        .ok_or_else(||invalid("complex GEMM output exceeds U32 indexing"))?;
    let mut operands=vec![&ar,&ai,&br,&bi];
    if let Some((cr,ci))=&c {
        if cr.meta.shape()!=&shape || ci.meta.shape()!=&shape {return Err(invalid("complex GEMM C must have exact output shape"));}
        operands.push(cr);operands.push(ci);
    } else if options.beta!=[0.,0.] {return Err(invalid("complex GEMM nonzero beta requires C"));}
    for t in operands {
        if t.dtype!=ar.dtype || t.qparams.is_some() || t.device.to_id()!=ar.device.to_id()
            || !t.client.same_execution_queue(&ar.client)
            || t.meta.shape().iter().try_fold(1usize,|a,&b|a.checked_mul(b)).is_none_or(|n|n>u32::MAX as usize)
            || (t.meta.num_elements()!=0 && t.meta.shape().iter().zip(t.meta.strides().iter())
                .try_fold(1usize,|span,(&d,&s)|(d-1).checked_mul(s).and_then(|n|span.checked_add(n)))
                .is_none_or(|span|span>u32::MAX as usize))
        {return Err(invalid("complex GEMM components must share real precision, device and execution queue"));}
    }
    let real=empty_device_contiguous_dtype(ar.client.clone(),ar.device.clone(),shape.clone(),ar.dtype);
    let imag=empty_device_contiguous_dtype(ar.client.clone(),ar.device.clone(),shape,ar.dtype);
    if count==0 {return Ok((real,imag));}
    let read_c=options.beta!=[0.,0.];let (cr,ci)=c.unwrap_or_else(||(ar.clone(),ai.clone()));
    let dim=RudaDim::new(ar.client.properties(),count);let grid=calculate_ruda_count_elemwise(&ar.client,count,dim);
    macro_rules! launch {($f:ty)=> {complex_kernel::launch::<$f,R>(&ar.client,grid,dim,
        ar.clone().into_tensor_arg(),ai.into_tensor_arg(),br.into_tensor_arg(),bi.into_tensor_arg(),
        cr.into_tensor_arg(),ci.into_tensor_arg(),real.clone().into_tensor_arg(),imag.clone().into_tensor_arg(),
        options.alpha[0] as $f,options.alpha[1] as $f,options.beta[0] as $f,options.beta[1] as $f,
        ash[ra-1] as u32,options.transpose_lhs,options.transpose_rhs,options.conjugate_lhs,options.conjugate_rhs,
        options.alpha!=[0.,0.] && ash[ra-1]!=0,read_c)}}
    match ar.dtype {DType::F32=>launch!(f32),DType::F64=>launch!(f64),_=>unreachable!()}
    Ok((real,imag))
}

#[ruda]
fn batch_offset<F:Float>(input:&Tensor<F>,output:&Tensor<F>,batch:usize)->usize {
    let mut remaining=batch;let mut offset=0usize;let mut axis=output.rank()-2;
    while axis>0 {
        axis-=1;let coordinate=remaining%output.shape(axis);remaining/=output.shape(axis);
        if axis+input.rank()>=output.rank() {
            let ia=axis+input.rank()-output.rank();
            if input.shape(ia)!=1 {offset+=coordinate*input.stride(ia);}
        }
    }
    offset
}

#[ruda(launch)]
fn complex_kernel<F:Float+RudaElement>(
    ar:&Tensor<F>,ai:&Tensor<F>,br:&Tensor<F>,bi:&Tensor<F>,cr:&Tensor<F>,ci:&Tensor<F>,
    real:&mut Tensor<F>,imag:&mut Tensor<F>,alpha_r:F,alpha_i:F,beta_r:F,beta_i:F,inner:u32,
    #[comptime] ta:bool,#[comptime] tb:bool,#[comptime] ca:bool,#[comptime] cb:bool,
    #[comptime] read_product:bool,#[comptime] read_c:bool,
) {
    let p=ABSOLUTE_POS;if p>=real.len() {terminate!();}
    let m=real.shape(real.rank()-2);let n=real.shape(real.rank()-1);
    let row=p/n%m;let col=p%n;let batch=p/(m*n);
    let mut sr=F::new(0.0f32);let mut si=F::new(0.0f32);
    if comptime!(read_product) {
        let aro=batch_offset(ar,real,batch);let aio=batch_offset(ai,real,batch);
        let bro=batch_offset(br,real,batch);let bio=batch_offset(bi,real,batch);
        let mut j=0usize;
        while j<inner as usize {
            let mut a0=row;let mut a1=j;let mut b0=j;let mut b1=col;
            if comptime!(ta) {a0=j;a1=row;}if comptime!(tb) {b0=col;b1=j;}
            let av=ar[aro+a0*ar.stride(ar.rank()-2)+a1*ar.stride(ar.rank()-1)];
            let mut ax=ai[aio+a0*ai.stride(ai.rank()-2)+a1*ai.stride(ai.rank()-1)];
            let bv=br[bro+b0*br.stride(br.rank()-2)+b1*br.stride(br.rank()-1)];
            let mut bx=bi[bio+b0*bi.stride(bi.rank()-2)+b1*bi.stride(bi.rank()-1)];
            if comptime!(ca) {ax=-ax;}if comptime!(cb) {bx=-bx;}
            sr+=av*bv-ax*bx;si+=av*bx+ax*bv;j+=1;
        }
    }
    let mut rr=F::new(0.0f32);let mut ri=F::new(0.0f32);
    if comptime!(read_product) {rr=alpha_r*sr-alpha_i*si;ri=alpha_r*si+alpha_i*sr;}
    if comptime!(read_c) {
        let co=batch_offset(cr,real,batch)+row*cr.stride(cr.rank()-2)+col*cr.stride(cr.rank()-1);
        let io=batch_offset(ci,real,batch)+row*ci.stride(ci.rank()-2)+col*ci.stride(ci.rank()-1);
        let cv=cr[co];let cx=ci[io];rr+=beta_r*cv-beta_i*cx;ri+=beta_r*cx+beta_i*cv;
    }
    real[p]=rr;imag[p]=ri;
}
