use ruda_core::tensor::DType;
use ruda_kernel::{dsl as kernel_dsl, dsl::prelude::*, tensor::RudaTensor};
use ruda_kernel::dsl::prelude::barrier::{copy_async, commit_copy_group, wait_copy_group};
use crate::kernel_ir::{definition::MatmulSetupError, launch::Strategy};
use super::{F32MathMode, launch_matmul_with_precision};

pub(super) fn try_launch<R: Runtime>(a: &RudaTensor<R>, b: &RudaTensor<R>, out: &RudaTensor<R>,
    math: F32MathMode) -> Result<bool, MatmulSetupError> {
    if launch(a, b, out, math, None)? { return Ok(true); }
    if math == F32MathMode::AllowTf32 && R::name(&a.client) == "cuda"
        && a.dtype == DType::F32 && b.dtype == DType::F32 && out.dtype == DType::F32
        && [a, b, out].iter().all(|x| x.qparams.is_none() && x.meta.num_dims() == a.meta.num_dims()
            && x.client.same_execution_queue(&a.client) && x.meta.shape().iter().all(|&v| v != 0))
        && matches!(a.meta.num_dims(), 2 | 3) {
        for strategy in [Strategy::DoubleAsyncCyclicCmma(Default::default()),
            Strategy::DoubleCyclicCmma(Default::default()), Strategy::SimpleCyclicCmma(Default::default())] {
            match launch_matmul_with_precision(&strategy, a.clone(), b.clone(), out.clone(), math) {
                Ok(()) => return Ok(true),
                Err(MatmulSetupError::Unavailable(_) | MatmulSetupError::InvalidConfig(_)
                    | MatmulSetupError::Vectorization(_)) => {},
                Err(error) => return Err(error),
            }
        }
    }
    if math == F32MathMode::AllowTf32 { return launch(a, b, out, F32MathMode::Strict, None); }
    Ok(false)
}

pub fn try_addmm_f32_tiled<R: Runtime>(bias: &RudaTensor<R>, a: &RudaTensor<R>, b: &RudaTensor<R>,
    out: &RudaTensor<R>, alpha: f32, beta: f32, math: F32MathMode) -> Result<bool, MatmulSetupError> {
    if math != F32MathMode::Strict || alpha == 0.0 || out.meta.num_dims() != 2 || bias.meta.shape() != out.meta.shape()
        || bias.dtype != DType::F32 || bias.qparams.is_some() || !bias.client.same_execution_queue(&out.client)
        || bias.meta.shape().iter().zip(bias.meta.strides().iter()).try_fold(1usize, |span, (&size, &stride)|
            size.checked_sub(1).and_then(|v| v.checked_mul(stride)).and_then(|v| span.checked_add(v)))
            .is_none_or(|span| span > u32::MAX as usize)
    { return Ok(false); }
    launch(a, b, out, math, Some((bias, alpha, beta)))
}

fn launch<R: Runtime>(a: &RudaTensor<R>, b: &RudaTensor<R>, out: &RudaTensor<R>,
    math: F32MathMode, epilogue: Option<(&RudaTensor<R>, f32, f32)>) -> Result<bool, MatmulSetupError> {
    if R::name(&a.client) != "cuda" || a.dtype != DType::F32 || b.dtype != a.dtype || out.dtype != a.dtype
        || a.qparams.is_some() || b.qparams.is_some() || out.qparams.is_some()
        || !matches!(a.meta.num_dims(), 2 | 3) || b.meta.num_dims() != a.meta.num_dims()
        || out.meta.num_dims() != a.meta.num_dims()
        || !a.client.same_execution_queue(&b.client) || !a.client.same_execution_queue(&out.client)
    { return Ok(false); }
    let d = a.meta.num_dims() - 2;
    let (m, k, n) = (a.meta.shape()[d], a.meta.shape()[d+1], b.meta.shape()[d+1]);
    if m == 0 || n == 0 || k == 0 || b.meta.shape()[d] != k { return Ok(false); }
    let ta = a.meta.strides()[d] == 1;
    let nt = b.meta.strides()[d] == 1;
    let lda = a.meta.strides()[d + usize::from(ta)];
    let ldb = b.meta.strides()[d + usize::from(nt)];
    let batch = if d == 0 { 1 } else { a.meta.shape()[0] };
    let sa = if d == 0 { 0 } else { a.meta.strides()[0] };
    let sb = if d == 0 { 0 } else { b.meta.strides()[0] };
    if m % 32 != 0 || n % 32 != 0 || k % 8 != 0 || batch == 0
        || (!ta && a.meta.strides()[d+1] != 1) || (!nt && b.meta.strides()[d+1] != 1)
        || lda < (if ta { m } else { k }) || ldb < (if nt { k } else { n })
        || lda % 4 != 0 || ldb % 4 != 0 || sa % 4 != 0 || sb % 4 != 0
        || out.meta.shape()[d..] != [m, n] || out.meta.strides()[d..] != [n, 1]
        || (d == 1 && (b.meta.shape()[0] != batch || out.meta.shape()[0] != batch
            || m.checked_mul(n) != Some(out.meta.strides()[0])))
        || [a, b].iter().any(|x| x.handle.offset_start.unwrap_or(0) % 16 != 0)
        || [a, b, out].iter().any(|x| x.meta.shape().iter().zip(x.meta.strides().iter()).try_fold(1usize,
            |span, (&size, &stride)| size.checked_sub(1).and_then(|v|v.checked_mul(stride)).and_then(|v|span.checked_add(v)))
            .is_none_or(|span| span > u32::MAX as usize))
    { return Ok(false); }
    if math == F32MathMode::AllowTf32 {
        return Ok(super::tf32_tiled::launch(a, b, out, m, n, k, lda, ldb, sa, sb, ta, nt, batch));
    }
    let bm = if m % 128 == 0 { 128 } else if m >= 64 { 64 } else { 32 };
    let bn = if n % 128 == 0 { 128 } else if n >= 64 { 64 } else { 32 };
    let bk = if k % 32 == 0 { 32 } else if k % 16 == 0 { 16 } else { 8 };
    let a_stage = if ta { bk * (bm / 4 + 1) } else { bm * (bk / 4 + 1) };
    let b_stage = if nt { bn * (bk / 4 + 1) } else { bk * (bn / 4 + 1) };
    let hw = &a.client.properties().hardware;
    if hw.max_units_per_ruda < 256 || hw.max_shared_memory_size < (a_stage + b_stage) * 32
        || m.div_ceil(bm) * n.div_ceil(bn) > hw.max_ruda_count.0 as usize || batch > hw.max_ruda_count.1 as usize
    { return Ok(false); }
    let output_offset = out.handle.offset_start.unwrap_or(0);
    let vector = if bn >= 64 && output_offset % 16 == 0 { 4 } else if output_offset % 8 == 0 { 2 } else { 1 };
    let (bias, alpha, beta) = epilogue.unwrap_or((out, 1.0, 0.0));
    let bias_row = if epilogue.is_some() { bias.meta.strides()[0] } else { 0 };
    let bias_col = if epilogue.is_some() { bias.meta.strides()[1] } else { 0 };
    unsafe { tiled::launch_unchecked::<R>(&a.client,
        RudaCount::Static((m.div_ceil(bm) * n.div_ceil(bn)) as u32, batch as u32, 1), RudaDim::new_1d(256),
        a.clone().into_array_arg(), b.clone().into_array_arg(), bias.clone().into_array_arg(), out.clone().into_array_arg(),
        alpha, beta, m, n, k, lda, ldb, sa, sb, ta, nt, bm, bn, bk, a.client.features().copy_async,
        epilogue.is_some(), beta != 0.0, bias_row, bias_col, vector); }
    Ok(true)
}

#[ruda]
pub(super) fn load(a: &Array<Vector<f32, Const<4>>>, b: &Array<Vector<f32, Const<4>>>,
    sa: &mut SharedMemory<Vector<f32, Const<4>>>, sb: &mut SharedMemory<Vector<f32, Const<4>>>,
    row: usize, col: usize, block: usize, bank: usize,
    #[comptime] lda: usize, #[comptime] ldb: usize, #[comptime] a_batch: usize, #[comptime] b_batch: usize,
    #[comptime] ta: bool, #[comptime] nt: bool, #[comptime] bm: usize, #[comptime] bn: usize,
    #[comptime] bk: usize, #[comptime] asynchronous: bool,
    #[comptime] m: usize, #[comptime] n: usize, #[comptime] threads: usize,
) {
    let ac = comptime!(if ta { bm / 4 } else { bk / 4 });
    let bc = comptime!(if nt { bk / 4 } else { bn / 4 });
    let astage = comptime!(if ta { bk * (bm / 4 + 1) } else { bm * (bk / 4 + 1) });
    let bstage = comptime!(if nt { bn * (bk / 4 + 1) } else { bk * (bn / 4 + 1) });
    #[unroll]
    for step in 0usize..comptime!((bm * bk / 4).div_ceil(threads)) {
        let p = UNIT_POS as usize + step * threads;
        if p < bm * bk / 4 {
            let mut source = RUDA_POS_Y as usize * (a_batch / 4);
            if comptime!(ta) { source += (block * bk + p / ac) * (lda / 4) + row / 4 + p % ac; }
            else { source += (row + p / ac) * (lda / 4) + block * (bk / 4) + p % ac; }
            let target = bank * astage + p / ac * (ac + 1) + p % ac;
            let mut valid = true;
            if comptime!(m % bm != 0) {
                if comptime!(ta) { valid = row + (p % ac) * 4 < m; }
                else { valid = row + p / ac < m; }
            }
            if valid {
                if comptime!(asynchronous) { copy_async(&a.slice(source, source+1), &mut sa.slice_mut(target, target+1), 4u32); }
                else { sa[target] = a[source]; }
            } else { sa[target] = Vector::new(0.0f32); }
        }
    }
    #[unroll]
    for step in 0usize..comptime!((bn * bk / 4).div_ceil(threads)) {
        let p = UNIT_POS as usize + step * threads;
        if p < bn * bk / 4 {
            let mut source = RUDA_POS_Y as usize * (b_batch / 4);
            if comptime!(nt) { source += (col + p / bc) * (ldb / 4) + block * (bk / 4) + p % bc; }
            else { source += (block * bk + p / bc) * (ldb / 4) + col / 4 + p % bc; }
            let target = bank * bstage + p / bc * (bc + 1) + p % bc;
            let mut valid = true;
            if comptime!(n % bn != 0) {
                if comptime!(nt) { valid = col + p / bc < n; }
                else { valid = col + (p % bc) * 4 < n; }
            }
            if valid {
                if comptime!(asynchronous) { copy_async(&b.slice(source, source+1), &mut sb.slice_mut(target, target+1), 4u32); }
                else { sb[target] = b[source]; }
            } else { sb[target] = Vector::new(0.0f32); }
        }
    }
}

#[ruda(launch_unchecked)]
fn tiled<OS: Size>(a: &Array<Vector<f32, Const<4>>>, b: &Array<Vector<f32, Const<4>>>,
    bias: &Array<f32>, out: &mut Array<Vector<f32, OS>>, alpha: f32, beta: f32,
    #[comptime] m: usize, #[comptime] n: usize, #[comptime] k: usize,
    #[comptime] lda: usize, #[comptime] ldb: usize, #[comptime] a_batch: usize, #[comptime] b_batch: usize,
    #[comptime] ta: bool, #[comptime] nt: bool, #[comptime] bm: usize, #[comptime] bn: usize,
    #[comptime] bk: usize, #[comptime] asynchronous: bool,
    #[comptime] epilogue: bool, #[comptime] use_bias: bool,
    #[comptime] bias_row: usize, #[comptime] bias_col: usize, #[define(OS)] _vector: usize,
) {
    let row = RUDA_POS_X as usize % comptime!(m.div_ceil(bm)) * bm;
    let col = RUDA_POS_X as usize / comptime!(m.div_ceil(bm)) * bn;
    let ty = UNIT_POS as usize / 16;
    let tx = UNIT_POS as usize % 16;
    let tm = comptime!(bm / 16);
    let tn = comptime!(bn / 16);
    let astage = comptime!(if ta { bk * (bm / 4 + 1) } else { bm * (bk / 4 + 1) });
    let bstage = comptime!(if nt { bn * (bk / 4 + 1) } else { bk * (bn / 4 + 1) });
    let mut sa = SharedMemory::<Vector<f32, Const<4>>>::new_aligned(astage * 2, 32usize);
    let mut sb = SharedMemory::<Vector<f32, Const<4>>>::new_aligned(bstage * 2, 32usize);
    let mut acc = Array::<f32>::new(tm * tn);
    #[unroll] for i in 0usize..(tm * tn) { acc[i] = 0.0f32; }
    load(a, b, &mut sa, &mut sb, row, col, 0usize, 0usize, lda, ldb, a_batch, b_batch, ta, nt, bm, bn, bk, asynchronous, m, n, 256usize);
    if comptime!(asynchronous) { commit_copy_group(); }
    for block in 0usize..(k / bk) {
        let bank = block % 2;
        if comptime!(asynchronous) { wait_copy_group(0u32); }
        sync_ruda();
        if block + 1 < k / bk {
            load(a, b, &mut sa, &mut sb, row, col, block+1, 1-bank, lda, ldb, a_batch, b_batch, ta, nt, bm, bn, bk, asynchronous, m, n, 256usize);
            if comptime!(asynchronous) { commit_copy_group(); }
        }
        #[unroll]
        for inner in 0usize..bk {
            let mut ar = Array::<f32>::new(tm);
            let mut br = Array::<f32>::new(tn);
            #[unroll]
            for i in 0usize..tm {
                let r = ty * comptime!(tm.min(4)) + comptime!(i % 4 + i / 4 * 64);
                if comptime!(ta) { ar[i] = sa[bank * astage + inner * (bm / 4 + 1) + r / 4][r % 4]; }
                else { ar[i] = sa[bank * astage + r * (bk / 4 + 1) + inner / 4][inner % 4]; }
            }
            #[unroll]
            for j in 0usize..tn {
                let c = tx * comptime!(tn.min(4)) + comptime!(j % 4 + j / 4 * 64);
                if comptime!(nt) { br[j] = sb[bank * bstage + c * (bk / 4 + 1) + inner / 4][inner % 4]; }
                else { br[j] = sb[bank * bstage + inner * (bn / 4 + 1) + c / 4][c % 4]; }
            }
            #[unroll] for i in 0usize..tm {
                #[unroll] for j in 0usize..tn { acc[i * tn + j] = fma(ar[i], br[j], acc[i * tn + j]); }
            }
        }
    }
    let vector = out.vector_size().comptime();
    #[unroll] for i in 0usize..tm {
        let r = row + ty * comptime!(tm.min(4)) + comptime!(i % 4 + i / 4 * 64);
        #[unroll] for j in 0usize..comptime!(tn / vector) {
            let c = col + tx * comptime!(tn.min(4)) + comptime!((j * vector) % 4 + (j * vector) / 4 * 64);
            if r < m && c < n {
            let mut result = Vector::<f32, OS>::empty();
            #[unroll] for lane in 0usize..vector {
                let mut value = acc[i * tn + j * vector + lane];
                if comptime!(epilogue) {
                    value *= alpha;
                    if comptime!(use_bias) { value += beta * bias[r * bias_row + (c + lane) * bias_col]; }
                }
                result[lane] = value;
            }
            out[(RUDA_POS_Y as usize * m * n + r * n + c) / vector] = result;
            }
        }
    }
}
