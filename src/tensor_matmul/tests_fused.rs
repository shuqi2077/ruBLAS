use super::*;
use half::{bf16, f16};
use ruda_core::tensor::{DType, Shape, data::TensorData};
use ruda_kernel::tensor::{RudaTensor, transfer::from_data, readback::into_data_sync, permutation::swap_dims};
use ruda_test_runtime::TestRuntime;
use crate::tensor_grouped::{GroupedStrategy, grouped_matmul_nt_with_strategy};

type Tensor = RudaTensor<TestRuntime>;
fn required() {
    assert!(std::any::type_name::<TestRuntime>().contains("CudaRuntime"), "these tests require actual CUDA");
    assert_eq!(std::env::var("RUDA_REQUIRE_GPU").as_deref(),Ok("1"));
}
fn round(x:f32,d:DType)->f32 {match d {DType::F16=>f16::from_f32(x).to_f32(),DType::BF16=>bf16::from_f32(x).to_f32(),_=>x}}
fn values(n:usize,seed:usize,d:DType)->Vec<f32> {(0..n).map(|i|round((((i*17+seed)%67) as f32-33.)/64.,d)).collect()}
fn tensor(v:&[f32],shape:impl Into<Shape>,d:DType)->Tensor {
    let shape=shape.into();let data=match d {
        DType::F16=>TensorData::new(v.iter().map(|&v|f16::from_f32(v)).collect::<Vec<_>>(),shape),
        DType::BF16=>TensorData::new(v.iter().map(|&v|bf16::from_f32(v)).collect::<Vec<_>>(),shape),
        _=>TensorData::new(v.to_vec(),shape)};
    from_data(data,&Default::default())
}
fn read(t:Tensor)->Vec<f32> {let d=t.dtype;let v=into_data_sync(t);match d {
    DType::F16=>v.to_vec::<f16>().unwrap().into_iter().map(f16::to_f32).collect(),
    DType::BF16=>v.to_vec::<bf16>().unwrap().into_iter().map(bf16::to_f32).collect(),
    _=>v.to_vec::<f32>().unwrap()}}
fn close(actual:&[f32],expected:&[f32],tol:f32) {
    assert_eq!(actual.len(),expected.len());
    for (i,(&a,&b)) in actual.iter().zip(expected).enumerate() {assert!(a.is_finite() && (a-b).abs()<=tol*b.abs().max(1.),"{i}: {a} != {b}");}
}
fn activation(x:f64,a:GemmActivation)->f64 {match a {GemmActivation::None=>x,GemmActivation::Relu=>x.max(0.),
    GemmActivation::Gelu=>0.5*x*(1.+(std::f64::consts::FRAC_2_PI.sqrt()*(x+0.044715*x*x*x)).tanh())}}
fn fused_case(m:usize,n:usize,k:usize,d:DType,a:GemmActivation,bd:DType,od:DType,s:FusedGemmStrategy,strided:bool) {
    let x=values(m*k,3,d);let w=values(k*n,5,d);let b=values(n,11,bd);
    let lhs=if strided {let mut t=vec![0.;m*k];for r in 0..m {for j in 0..k {t[j*m+r]=x[r*k+j];}}swap_dims(tensor(&t,[k,m],d),0,1)} else {tensor(&x,[m,k],d)};
    let actual=read(matmul_bias_activation(lhs,tensor(&w,[k,n],d),tensor(&b,[n],bd),a,s,od).unwrap());
    let expected=(0..m*n).map(|p| {let row=p/n;let col=p%n;
        round(activation((0..k).map(|j|x[row*k+j] as f64*w[j*n+col] as f64).sum::<f64>()+b[col] as f64,a) as f32,od)}).collect::<Vec<_>>();
    close(&actual,&expected,if od==DType::BF16 {0.012} else if od==DType::F16 {0.002} else {4e-5});
}
#[test]
fn software_five_blas_fp16_fused_epilogues() {
    required();
    for s in [FusedGemmStrategy::Scalar,FusedGemmStrategy::TensorCore,FusedGemmStrategy::Auto] {
        for a in [GemmActivation::None,GemmActivation::Relu,GemmActivation::Gelu] {
            fused_case(19,33,35,DType::F16,a,DType::F32,DType::F32,s,true);
            fused_case(16,16,32,DType::F16,a,DType::F16,DType::F16,s,false);
            fused_case(3,7,0,DType::F16,a,DType::F32,DType::F16,s,false);
        }
    }
    fused_case(0,7,3,DType::F16,GemmActivation::Relu,DType::F32,DType::F16,FusedGemmStrategy::TensorCore,false);
    println!("RUDA_FIVE_BLAS_FUSED_FP16=28");
}
#[test]
fn software_five_blas_bf16_fused_epilogues() {
    required();
    for a in [GemmActivation::None,GemmActivation::Relu,GemmActivation::Gelu] {
        fused_case(17,19,33,DType::BF16,a,DType::BF16,DType::BF16,FusedGemmStrategy::TensorCore,false);
        fused_case(1,3,17,DType::BF16,a,DType::F32,DType::F32,FusedGemmStrategy::TensorCore,true);
    }
    println!("RUDA_FIVE_BLAS_FUSED_BF16=6");
}
fn grouped_case(d:DType,s:GroupedStrategy,all_padding:bool) {
    let (m,n,k,e)=(37,33,35,5);let x=values(m*k,13,d);let w=values(e*n*k,19,d);
    let ids=(0..m).map(|r|if all_padding || r%7==0 {u32::MAX} else {((r*3)%4) as u32}).collect::<Vec<_>>();
    let actual=read(grouped_matmul_nt_with_strategy(tensor(&x,[m,k],d),tensor(&w,[e,n,k],d),
        from_data(TensorData::new(ids.clone(),[m]),&Default::default()),s).unwrap());
    let expected=(0..m*n).map(|p| {let row=p/n;let col=p%n;let expert=ids[row] as usize;
        if expert>=e {0.} else {round((0..k).map(|j|x[row*k+j] as f64*w[(expert*n+col)*k+j] as f64).sum::<f64>() as f32,d)}}).collect::<Vec<_>>();
    close(&actual,&expected,if d==DType::BF16 {0.012} else if d==DType::F16 {0.003} else {4e-5});
}
#[test]
fn software_five_blas_fp16_arbitrary_grouped() {
    required();
    for s in [GroupedStrategy::Scalar,GroupedStrategy::TensorCore,GroupedStrategy::Auto] {
        grouped_case(DType::F16,s,false);grouped_case(DType::F16,s,true);
    }
    grouped_case(DType::F32,GroupedStrategy::Auto,false);
    println!("RUDA_FIVE_BLAS_GROUPED_FP16=7");
}
#[test]
fn software_five_blas_bf16_arbitrary_grouped() {
    required();grouped_case(DType::BF16,GroupedStrategy::TensorCore,false);grouped_case(DType::BF16,GroupedStrategy::TensorCore,true);
    println!("RUDA_FIVE_BLAS_GROUPED_BF16=2");
}
