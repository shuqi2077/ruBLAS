//! Device-only regrouping for arbitrary per-row expert IDs. There is no tensor
//! payload or routing metadata readback, and the matrix multiply uses CMMA.
use super::tensorcore;
use ruda_core::{ir::{ElemType, FloatKind, features::MmaConfig}, tensor::{DType, Shape}};
use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::{dsl::{Runtime, calculate_ruda_count_elemwise, prelude::*}, tensor::{RudaTensor, allocation::empty_device_contiguous_dtype, contiguous::into_contiguous}};

pub(super) fn supported<R: Runtime>(input: &RudaTensor<R>, experts: usize, columns: usize) -> bool {
    let cfg = MmaConfig { a_type: input.dtype.into(), b_type: input.dtype.into(),
        cd_type: ElemType::Float(FloatKind::F32).into(), m:16, n:16, k:16 };
    let hw = &input.client.properties().hardware;
    matches!(input.dtype, DType::F16 | DType::BF16)
        && hw.plane_size_min == 32 && hw.plane_size_max == 32
        && hw.max_shared_memory_size >= 2048
        && input.client.features().matmul.cmma.contains(&cfg)
        && columns.div_ceil(16) <= hw.max_ruda_count.0 as usize
        && experts <= hw.max_ruda_count.1 as usize
}

pub(super) fn launch<R: Runtime>(input: RudaTensor<R>, weights: RudaTensor<R>, ids: RudaTensor<R>) -> RudaTensor<R> {
    let (m,k,e,n) = (input.meta.shape()[0],input.meta.shape()[1],weights.meta.shape()[0],weights.meta.shape()[1]);
    let output = empty_device_contiguous_dtype(input.client.clone(),input.device.clone(),Shape::from([m,n]),input.dtype);
    if m == 0 { return output; }
    let input=into_contiguous(input); let weights=into_contiguous(weights); let ids=into_contiguous(ids);
    let allocate = |shape: Shape, dtype: DType| empty_device_contiguous_dtype(input.client.clone(),input.device.clone(),shape,dtype);
    let counts=allocate(Shape::from([e]),DType::U32);
    let ranks=allocate(Shape::from([m]),DType::U32);
    let offsets=allocate(Shape::from([e+1]),DType::U32);
    let packed=allocate(Shape::from([m,k]),input.dtype);
    let result=allocate(Shape::from([m,n]),input.dtype);
    let dim=RudaDim::new(input.client.properties(),e);
    clear::launch::<R>(&input.client,calculate_ruda_count_elemwise(&input.client,e,dim),dim,counts.clone().into_array_arg());
    let dim=RudaDim::new(input.client.properties(),m);
    count::launch::<R>(&input.client,calculate_ruda_count_elemwise(&input.client,m,dim),dim,
        ids.clone().into_array_arg(),counts.clone().into_array_arg(),ranks.clone().into_array_arg(),e as u32);
    prefix::launch::<R>(&input.client,RudaCount::Static(1,1,1),RudaDim::new_1d(1),
        counts.into_array_arg(),offsets.clone().into_array_arg());
    let dim=RudaDim::new(input.client.properties(),m*k);
    pack::launch::<R>(&input.client,calculate_ruda_count_elemwise(&input.client,m*k,dim),dim,
        input.clone().into_array_arg(),ids.clone().into_array_arg(),ranks.clone().into_array_arg(),offsets.clone().into_array_arg(),
        packed.clone().into_array_arg(),e as u32,k as u32,input.dtype.into());
    // offsets end at the number of valid rows; invalid rows have no packed
    // payload and are never consumed by the segmented kernel.
    tensorcore::segmented::launch::<R>(&input.client,
        RudaCount::Static(n.div_ceil(16) as u32,e as u32,1),RudaDim::new_1d(32),
        packed.into_array_arg(),weights.into_array_arg(),offsets.clone().into_array_arg(),result.clone().into_array_arg(),
        n as u32,k as u32,input.dtype.into());
    let dim=RudaDim::new(input.client.properties(),m*n);
    unpack::launch::<R>(&input.client,calculate_ruda_count_elemwise(&input.client,m*n,dim),dim,
        result.into_array_arg(),ids.into_array_arg(),ranks.into_array_arg(),offsets.into_array_arg(),output.clone().into_array_arg(),
        e as u32,n as u32,input.dtype.into());
    output
}

pub(super) fn prepare<R:Runtime>(input:&RudaTensor<R>,weights:&RudaTensor<R>,ids:&RudaTensor<R>,output:&RudaTensor<R>)
    ->super::GroupedKernelPlan<R>
{
    let (m,k,e,n)=(input.meta.shape()[0],input.meta.shape()[1],weights.meta.shape()[0],weights.meta.shape()[1]);
    let mut kernels=Vec::new();if m==0 {return super::GroupedKernelPlan {kernels,workspace:Vec::new()};}
    let allocate=|shape:Shape,dtype:DType|empty_device_contiguous_dtype(input.client.clone(),input.device.clone(),shape,dtype);
    let counts=allocate([e].into(),DType::U32);let ranks=allocate([m].into(),DType::U32);
    let offsets=allocate([e+1].into(),DType::U32);let packed=allocate([m,k].into(),input.dtype);
    let result=allocate([m,n].into(),input.dtype);let client=&input.client;
    let dim=RudaDim::new(client.properties(),e);
    kernels.push(clear::prepare::<R>(client,calculate_ruda_count_elemwise(client,e,dim),dim,counts.clone().into_array_arg()));
    let dim=RudaDim::new(client.properties(),m);
    kernels.push(count::prepare::<R>(client,calculate_ruda_count_elemwise(client,m,dim),dim,
        ids.clone().into_array_arg(),counts.clone().into_array_arg(),ranks.clone().into_array_arg(),e as u32));
    kernels.push(prefix::prepare::<R>(client,RudaCount::Static(1,1,1),RudaDim::new_1d(1),
        counts.clone().into_array_arg(),offsets.clone().into_array_arg()));
    let dim=RudaDim::new(client.properties(),m*k);
    kernels.push(pack::prepare::<R>(client,calculate_ruda_count_elemwise(client,m*k,dim),dim,
        input.clone().into_array_arg(),ids.clone().into_array_arg(),ranks.clone().into_array_arg(),offsets.clone().into_array_arg(),
        packed.clone().into_array_arg(),e as u32,k as u32,input.dtype.into()));
    kernels.push(tensorcore::segmented::prepare::<R>(client,RudaCount::Static(n.div_ceil(16) as u32,e as u32,1),RudaDim::new_1d(32),
        packed.clone().into_array_arg(),weights.clone().into_array_arg(),offsets.clone().into_array_arg(),result.clone().into_array_arg(),
        n as u32,k as u32,input.dtype.into()));
    let dim=RudaDim::new(client.properties(),m*n);
    kernels.push(unpack::prepare::<R>(client,calculate_ruda_count_elemwise(client,m*n,dim),dim,
        result.clone().into_array_arg(),ids.clone().into_array_arg(),ranks.clone().into_array_arg(),offsets.clone().into_array_arg(),
        output.clone().into_array_arg(),e as u32,n as u32,input.dtype.into()));
    super::GroupedKernelPlan {kernels,workspace:vec![counts,ranks,offsets,packed,result]}
}

#[ruda(launch)]
fn clear(counts: &mut Array<u32>) {
    if ABSOLUTE_POS < counts.len() { counts[ABSOLUTE_POS]=0; }
}

#[ruda(launch)]
fn count(ids: &Array<u32>, counts: &mut Array<Atomic<u32>>, ranks: &mut Array<u32>, experts: u32) {
    let row=ABSOLUTE_POS;
    if row >= ids.len() { terminate!(); }
    let expert=ids[row];
    if expert < experts { ranks[row]=counts[expert as usize].fetch_add(1u32); }
}

#[ruda(launch)]
fn prefix(counts: &Array<u32>, offsets: &mut Array<u32>) {
    let mut expert=0usize; let mut total=0u32; offsets[0]=0;
    while expert < counts.len() {
        total+=counts[expert]; offsets[expert+1]=total; expert+=1;
    }
}

#[ruda(launch)]
fn pack<F: Float>(input: &Array<F>, ids: &Array<u32>, ranks: &Array<u32>, offsets: &Array<u32>, packed: &mut Array<F>,
    experts: u32, inner: u32, #[define(F)] _dtype: StorageType) {
    let p=ABSOLUTE_POS;
    if p >= input.len() { terminate!(); }
    let row=p/inner as usize; let expert=ids[row];
    if expert < experts {
        let packed_row=(offsets[expert as usize]+ranks[row]) as usize;
        packed[packed_row*inner as usize+p%inner as usize]=input[p];
    }
}

#[ruda(launch)]
fn unpack<F: Float>(result: &Array<F>, ids: &Array<u32>, ranks: &Array<u32>, offsets: &Array<u32>, output: &mut Array<F>,
    experts: u32, columns: u32, #[define(F)] _dtype: StorageType) {
    let p=ABSOLUTE_POS;
    if p >= output.len() { terminate!(); }
    let row=p/columns as usize; let expert=ids[row];
    let mut value=F::cast_from(0.0f32);
    if expert < experts {
        let packed_row=(offsets[expert as usize]+ranks[row]) as usize;
        value=result[packed_row*columns as usize+p%columns as usize];
    }
    output[p]=value;
}
