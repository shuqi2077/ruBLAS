use ruda_core::{ir::features::MmaConfig, tensor::DType};
use ruda_kernel::{dsl as kernel_dsl, dsl::{cmma, prelude::{*, barrier::{Barrier, copy_async}}}, tensor::RudaTensor};

/// Fixed-layout BF16 training GEMM. Accumulation stays FP32, in increasing K order.
#[allow(unsafe_code)]
pub fn matmul_bf16_training<R: Runtime>(a: &RudaTensor<R>, b: &RudaTensor<R>, out: &RudaTensor<R>) -> bool {
    a.dtype == DType::BF16 && matmul_low_precision_tiled(a, b, out)
}

pub(super) fn matmul_low_precision_tiled<R: Runtime>(a: &RudaTensor<R>, b: &RudaTensor<R>, out: &RudaTensor<R>) -> bool {
    if R::name(&a.client) != "cuda"
        || !matches!(a.dtype, DType::BF16 | DType::F16) || b.dtype != a.dtype
        || (out.dtype != a.dtype && out.dtype != DType::F32)
        || a.qparams.is_some() || b.qparams.is_some() || out.qparams.is_some()
        || !matches!(a.meta.num_dims(), 2 | 3) || b.meta.num_dims() != a.meta.num_dims() || out.meta.num_dims() != a.meta.num_dims()
        || !a.client.same_execution_queue(&b.client) || !a.client.same_execution_queue(&out.client)
    { return false; }
    let d = a.meta.num_dims() - 2;
    let (m, k, n) = (a.meta.shape()[d], a.meta.shape()[d+1], b.meta.shape()[d+1]);
    let ta = a.meta.strides()[d] == 1;
    let nt = b.meta.strides()[d] == 1;
    let lda = if ta { a.meta.strides()[d+1] } else { a.meta.strides()[d] };
    let ldb = if nt { b.meta.strides()[d+1] } else { b.meta.strides()[d] };
    if m == 0 || m % 32 != 0 || k == 0 || k % 32 != 0 || n == 0 || n % 32 != 0
        || b.meta.shape()[d] != k || (!ta && a.meta.strides()[d+1] != 1)
        || lda < (if ta { m } else { k }) || lda % 8 != 0
        || (!nt && b.meta.strides()[d+1] != 1) || ldb < (if nt { k } else { n }) || ldb % 8 != 0
        || out.meta.shape()[d..] != [m, n] || out.meta.strides()[d..] != [n, 1]
        || a.handle.offset_start.unwrap_or(0) % 16 != 0 || b.handle.offset_start.unwrap_or(0) % 16 != 0
    { return false; }
    let Some(a_span) = (if ta { k - 1 } else { m - 1 }).checked_mul(lda)
        .and_then(|v| v.checked_add(if ta { m } else { k })) else { return false; };
    let Some(b_span) = (if nt { n - 1 } else { k - 1 }).checked_mul(ldb)
        .and_then(|v| v.checked_add(if nt { k } else { n })) else { return false; };
    let batch = if d == 0 { 1 } else { a.meta.shape()[0] };
    let a_batch = if d == 0 { 0 } else { a.meta.strides()[0] };
    let b_batch = if d == 0 { 0 } else { b.meta.strides()[0] };
    if batch == 0 || batch > a.client.properties().hardware.max_ruda_count.1 as usize
        || (d == 1 && (b.meta.shape()[0] != batch || out.meta.shape()[0] != batch
            || a_batch % 8 != 0 || b_batch % 8 != 0
            || m.checked_mul(n) != Some(out.meta.strides()[0])))
        || (batch - 1).checked_mul(a_batch).and_then(|v| v.checked_add(a_span)).is_none_or(|v| v > u32::MAX as usize)
        || (batch - 1).checked_mul(b_batch).and_then(|v| v.checked_add(b_span)).is_none_or(|v| v > u32::MAX as usize)
        || m.checked_mul(n).and_then(|v| v.checked_mul(batch)).is_none_or(|v| v > u32::MAX as usize)
        || !a.client.features().copy_async
        || !a.client.features().matmul.cmma.contains(&MmaConfig {
            a_type: a.dtype.into(), b_type: b.dtype.into(), cd_type: DType::F32.into(),
            m: 16, n: 16, k: 16,
        })
    { return false; }
    super::training_tile_tune::matmul(a.clone(), b.clone(), out.clone(), k);
    true
}

pub(super) fn split_k<R: Runtime>(a: &RudaTensor<R>, b: &RudaTensor<R>, partial: &RudaTensor<R>,
    parts: usize, block: usize) -> bool {
    let (m, k, n) = (a.meta.shape()[0], a.meta.shape()[1], b.meta.shape()[1]);
    let nt = b.meta.strides()[0] == 1;
    let lda = a.meta.strides()[0];
    let ldb = if nt { b.meta.strides()[1] } else { b.meta.strides()[0] };
    if R::name(&a.client) != "cuda" || m == 0 || m % 32 != 0 || n == 0 || n % 32 != 0
        || a.meta.strides()[1] != 1 || lda < k || lda % 8 != 0
        || (!nt && b.meta.strides()[1] != 1) || ldb < (if nt { k } else { n }) || ldb % 8 != 0
        || a.handle.offset_start.unwrap_or(0) % 16 != 0 || b.handle.offset_start.unwrap_or(0) % 16 != 0
        || (m - 1).checked_mul(lda).and_then(|v| v.checked_add(k)).is_none_or(|v| v > u32::MAX as usize)
        || (if nt { n - 1 } else { k - 1 }).checked_mul(ldb)
            .and_then(|v| v.checked_add(if nt { k } else { n })).is_none_or(|v| v > u32::MAX as usize)
        || !a.client.features().copy_async || !a.client.features().matmul.cmma.contains(&MmaConfig {
            a_type: DType::BF16.into(), b_type: DType::BF16.into(), cd_type: DType::F32.into(), m: 16, n: 16, k: 16,
        })
    { return false; }
    let mut left = a.clone(); left.meta.shape_mut()[1] = block; left.meta.insert(0, parts, block);
    let mut right = b.clone(); right.meta.shape_mut()[0] = block;
    right.meta.insert(0, parts, block * b.meta.strides()[0]);
    super::training_tile_tune::matmul(left, right, partial.clone(), k - (parts-1) * block);
    true
}

#[allow(unsafe_code)]
pub(super) fn launch_tile<R: Runtime>(a: RudaTensor<R>, b: RudaTensor<R>, out: RudaTensor<R>,
    last_k: usize, bm: usize, bn: usize, bk: usize, asynchronous: bool, warp_m: usize, manual: bool, stages: usize, prefetch: bool) -> Result<RudaTensor<R>, String> {
    launch_tile_warp_grid(a, b, out, last_k, bm, bn, bk, asynchronous, warp_m, 2, manual, stages, prefetch)
}

#[allow(unsafe_code)]
pub(super) fn launch_tile_warp_grid<R: Runtime>(a: RudaTensor<R>, b: RudaTensor<R>, out: RudaTensor<R>,
    last_k: usize, bm: usize, bn: usize, bk: usize, asynchronous: bool, warp_m: usize, warp_cols: usize,
    manual: bool, stages: usize, prefetch: bool) -> Result<RudaTensor<R>, String> {
    if !matches!(warp_cols, 2 | 4) || (!manual && warp_cols != 2) || bn % (warp_cols * 16) != 0 {
        return Err("training warp columns do not divide the block columns".into());
    }
    if manual && asynchronous && !matches!(stages, 2 | 3) {
        return Err("manual training pipeline requires two or three stages".into());
    }
    if manual && !a.client.features().matmul.mma.contains(&MmaConfig {
        a_type: a.dtype.into(), b_type: b.dtype.into(), cd_type: DType::F32.into(),
        m: 16, n: 8, k: 16,
    }) { return Err("manual low-precision MMA is unavailable on this device".into()); }
    let d = a.meta.num_dims() - 2;
    let (m, k, n) = (a.meta.shape()[d], a.meta.shape()[d+1], b.meta.shape()[d+1]);
    if m % bm != 0 || n % bn != 0 { return Err("training tile does not divide the output geometry".into()); }
    if k % bk != 0 || last_k == 0 || last_k > k || last_k % bk != 0 {
        return Err("training K tile does not divide the reduction".into());
    }
    if !matches!(warp_m, 32 | 64) || bm % warp_m != 0 {
        return Err("training warp tile does not divide the block rows".into());
    }
    let threads = bm / warp_m * warp_cols * 32;
    if bm * bk % (threads * 8) != 0 || bn * bk % (threads * 8) != 0 {
        return Err("training tile does not divide the cooperative load".into());
    }
    let ta = a.meta.strides()[d] == 1;
    let nt = b.meta.strides()[d] == 1;
    let lda = if ta { a.meta.strides()[d+1] } else { a.meta.strides()[d] };
    let ldb = if nt { b.meta.strides()[d+1] } else { b.meta.strides()[d] };
    let batch = if d == 0 { 1 } else { a.meta.shape()[0] };
    let a_batch = if d == 0 { 0 } else { a.meta.strides()[0] };
    let b_batch = if d == 0 { 0 } else { b.meta.strides()[0] };
    let a_stage = if ta { bk * (bm / 8 + 1) } else { bm * (bk / 8 + 1) };
    let b_stage = if nt { bn * (bk / 8 + 1) } else { bk * (bn / 8 + 1) };
    let shared = if manual {
        (bm + bn) * bk * 2 * (if asynchronous { stages } else { 1 }) + threads / 32 * 256 * 4
    } else {
        (a_stage + b_stage) * (if asynchronous { 2 } else { 1 }) * 16 + threads / 32 * 256 * 4 + 32
    };
    if shared > a.client.properties().hardware.max_shared_memory_size
        || threads > a.client.properties().hardware.max_units_per_ruda as usize {
        return Err("training tile exceeds the device block resources".into());
    }
    if n / bn * (m / bm) > a.client.properties().hardware.max_ruda_count.0 as usize {
        return Err("training output grid exceeds the device limit".into());
    }
    let a_span = (batch - 1) * a_batch + if ta { (last_k - 1) * lda + m } else { (m - 1) * lda + last_k };
    let b_span = (batch - 1) * b_batch + if nt { (n - 1) * ldb + last_k } else { (last_k - 1) * ldb + n };
    let mut left = a.clone();
    let mut right = b.clone();
    if d == 1 { left.meta.remove(0); right.meta.remove(0); }
    left.meta.shape_mut()[0] = a_span; left.meta.shape_mut()[1] = 1;
    right.meta.shape_mut()[0] = b_span; right.meta.shape_mut()[1] = 1;
    if manual {
        unsafe { super::training_mma::tiled_warp_grid::launch_unchecked::<R>(&a.client, RudaCount::Static((n / bn * (m / bm)) as u32, batch as u32, 1), RudaDim::new_1d(threads as u32),
            left.into_array_arg(), right.into_array_arg(), out.clone().into_array_arg(), k, n, lda, ldb, ta, nt,
            m, batch, last_k, a_batch, b_batch, bm, bn, bk, asynchronous, warp_m,
            stages, prefetch, warp_cols, a.dtype.into(), out.dtype.into(), output_vector_size(&out)); }
    } else {
        unsafe { tiled::launch_unchecked::<R>(&a.client, RudaCount::Static((n / bn * (m / bm)) as u32, batch as u32, 1), RudaDim::new_1d(threads as u32),
            left.into_array_arg(), right.into_array_arg(), out.clone().into_array_arg(), k, n, lda, ldb, ta, nt,
            m, batch, last_k, a_batch, b_batch, bm, bn, bk, asynchronous, warp_m,
            a.dtype.into(), out.dtype.into(), output_vector_size(&out)); }
    }
    Ok(out)
}

pub(super) fn output_vector_size<R: Runtime>(out: &RudaTensor<R>) -> usize {
    if out.handle.offset_start.unwrap_or(0) % 16 == 0 {
        if matches!(out.dtype, DType::BF16 | DType::F16) { 8 } else { 4 }
    } else { 1 }
}

#[ruda(launch_unchecked)]
fn tiled<F: Float, O: Float, OS: Size>(a: &Array<Vector<F, Const<8>>>, b: &Array<Vector<F, Const<8>>>,
    out: &mut Array<Vector<O, OS>>, #[comptime] k: usize, #[comptime] n: usize,
    #[comptime] lda: usize, #[comptime] ldb: usize, #[comptime] ta: bool, #[comptime] nt: bool,
    #[comptime] m: usize, #[comptime] batch: usize, #[comptime] last_k: usize,
    #[comptime] a_batch: usize, #[comptime] b_batch: usize,
    #[comptime] bm: usize, #[comptime] bn: usize, #[comptime] bk: usize, #[comptime] asynchronous: bool,
    #[comptime] warp_m: usize,
    #[define(F)] _input: StorageType, #[define(O)] _output: StorageType,
    #[define(OS)] _output_vector_size: usize) {
    let thread = UNIT_POS as usize;
    let lane = thread % 32;
    let warp = thread / 32;
    let output_width = out.vector_size().comptime();
    // Adjacent CTAs reuse the weight tile; smaller tiles keep serial-K views occupied.
    let row = RUDA_POS_X as usize % (m / bm) * bm;
    let col = RUDA_POS_X as usize / (m / bm) * bn;
    let wm = warp / 2 * warp_m;
    let wn = warp % 2 * (bn / 2);
    let banks = comptime!(if asynchronous { 2usize } else { 1usize });
    let a_stage = comptime!(if ta { bk * (bm / 8 + 1) } else { bm * (bk / 8 + 1) });
    let b_stage = comptime!(if nt { bn * (bk / 8 + 1) } else { bk * (bn / 8 + 1) });
    let mut sa = SharedMemory::<Vector<F, Const<8>>>::new_aligned(a_stage * banks, 32usize);
    let mut sb = SharedMemory::<Vector<F, Const<8>>>::new_aligned(b_stage * banks, 32usize);
    let mut epilogue = SharedMemory::<f32>::new_aligned(bm / warp_m * 512usize, 32usize);
    let loaded = Barrier::shared(RUDA_DIM, UNIT_POS == 0);
    let mut acc = Sequence::<cmma::Matrix<f32>>::new();
    #[unroll]
    for _ in 0usize..(warp_m / 16 * (bn / 32)) {
        acc.push(cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator,
            16usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0f32));
    }
    if comptime!(asynchronous) {
        load_stage(a, b, &mut sa, &mut sb, row, col, 0usize, 0usize, lda, ldb, ta, nt, a_batch, b_batch, bm, bn, bk, asynchronous, warp_m);
        loaded.commit_copy_async();
    }
    let blocks = k / bk - usize::cast_from(RUDA_POS_Y as usize == batch - 1) * ((k - last_k) / bk);
    for block in 0usize..blocks {
        let bank = block % banks;
        if comptime!(asynchronous) {
            loaded.arrive_and_wait();
            if block + 1 < blocks {
                load_stage(a, b, &mut sa, &mut sb, row, col, block + 1, 1 - bank, lda, ldb, ta, nt, a_batch, b_batch, bm, bn, bk, asynchronous, warp_m);
                loaded.commit_copy_async();
            }
        } else {
            load_stage(a, b, &mut sa, &mut sb, row, col, block, bank, lda, ldb, ta, nt, a_batch, b_batch, bm, bn, bk, asynchronous, warp_m);
            sync_ruda();
        }
        #[unroll]
        for half in 0usize..(bk / 16) {
            let mut ar = Sequence::<cmma::Matrix<Vector<F, Const<8>>>>::new();
            let mut br = Sequence::<cmma::Matrix<Vector<F, Const<8>>>>::new();
            #[unroll]
            for i in 0usize..(warp_m / 16) {
                if comptime!(ta) {
                    let offset = bank * a_stage + half * 16 * (bm / 8 + 1) + (wm + i * 16) / 8;
                    ar.push(cmma::Matrix::<Vector<F, Const<8>>>::from_slice(cmma::MatrixIdent::A,
                        16usize, 16usize, 16usize, cmma::MatrixLayout::ColMajor,
                        &sa.slice(offset, a_stage * banks), (bm + 8) as u32));
                } else {
                    let offset = bank * a_stage + (wm + i * 16) * (bk / 8 + 1) + half * 2;
                    ar.push(cmma::Matrix::<Vector<F, Const<8>>>::from_slice(cmma::MatrixIdent::A,
                        16usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor,
                        &sa.slice(offset, a_stage * banks), (bk + 8) as u32));
                }
            }
            #[unroll]
            for i in 0usize..(bn / 32) {
                if comptime!(nt) {
                    let offset = bank * b_stage + (wn + i * 16) * (bk / 8 + 1) + half * 2;
                    br.push(cmma::Matrix::<Vector<F, Const<8>>>::from_slice(cmma::MatrixIdent::B,
                        16usize, 16usize, 16usize, cmma::MatrixLayout::ColMajor,
                        &sb.slice(offset, b_stage * banks), (bk + 8) as u32));
                } else {
                    let offset = bank * b_stage + half * 16 * (bn / 8 + 1) + wn / 8 + i * 2;
                    br.push(cmma::Matrix::<Vector<F, Const<8>>>::from_slice(cmma::MatrixIdent::B,
                        16usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor,
                        &sb.slice(offset, b_stage * banks), (bn + 8) as u32));
                }
            }
            #[unroll]
            for i in 0usize..(warp_m / 16) {
                #[unroll]
                for j in 0usize..(bn / 32) {
                    cmma::execute(&ar[i], &br[j], &acc[i * (bn / 32) + j], &acc[i * (bn / 32) + j]);
                }
            }
        }
        if comptime!(!asynchronous) { sync_ruda(); }
    }
    #[unroll]
    for i in 0usize..(warp_m / 16) {
        #[unroll]
        for j in 0usize..(bn / 32) {
            cmma::store(&mut epilogue.slice_mut(warp * 256, warp * 256 + 256),
                &acc[i * (bn / 32) + j], 16u32, cmma::MatrixLayout::RowMajor);
            sync_plane();
            #[unroll]
            for t in 0usize..(8 / output_width) {
                let pos = (lane + t * 32) * output_width;
                let mut values = Vector::<O, OS>::empty();
                #[unroll]
                for v in 0usize..output_width {
                    values[v] = O::cast_from(epilogue[warp * 256 + pos + v]);
                }
                out[((RUDA_POS_Y as usize * m + row + wm + i * 16 + pos / 16) * n + col + wn + j * 16 + pos % 16) / output_width]
                    = values;
            }
            sync_plane();
        }
    }
}

#[ruda]
pub(super) fn load_stage<F: Float>(a: &Array<Vector<F, Const<8>>>, b: &Array<Vector<F, Const<8>>>,
    sa: &mut SharedMemory<Vector<F, Const<8>>>, sb: &mut SharedMemory<Vector<F, Const<8>>>,
    row: usize, col: usize, block: usize, bank: usize,
    #[comptime] lda: usize, #[comptime] ldb: usize, #[comptime] ta: bool, #[comptime] nt: bool,
    #[comptime] a_batch: usize, #[comptime] b_batch: usize,
    #[comptime] bm: usize, #[comptime] bn: usize, #[comptime] bk: usize, #[comptime] asynchronous: bool,
    #[comptime] warp_m: usize) {
    let threads = comptime!(bm / warp_m * 64);
    #[unroll]
    for i in 0usize..(bm * bk / (threads * 8)) {
        let p = UNIT_POS as usize + i * threads;
        if comptime!(ta) {
            let source = (RUDA_POS_Y as usize * a_batch + (block * bk + p / (bm / 8)) * lda + row) / 8 + p % (bm / 8);
            let target = bank * bk * (bm / 8 + 1) + (p / (bm / 8)) * (bm / 8 + 1) + p % (bm / 8);
            if comptime!(asynchronous) {
                copy_async(&a.slice(source, source + 1), &mut sa.slice_mut(target, target + 1), 8u32);
            } else { sa[target] = a[source]; }
        } else {
            let source = (RUDA_POS_Y as usize * a_batch + (row + p / (bk / 8)) * lda + block * bk) / 8 + p % (bk / 8);
            let target = bank * bm * (bk / 8 + 1) + (p / (bk / 8)) * (bk / 8 + 1) + p % (bk / 8);
            if comptime!(asynchronous) {
                copy_async(&a.slice(source, source + 1), &mut sa.slice_mut(target, target + 1), 8u32);
            } else { sa[target] = a[source]; }
        }
    }
    #[unroll]
    for i in 0usize..(bn * bk / (threads * 8)) {
        let p = UNIT_POS as usize + i * threads;
        if comptime!(nt) {
            let source = (RUDA_POS_Y as usize * b_batch + (col + p / (bk / 8)) * ldb + block * bk) / 8 + p % (bk / 8);
            let target = bank * bn * (bk / 8 + 1) + (p / (bk / 8)) * (bk / 8 + 1) + p % (bk / 8);
            if comptime!(asynchronous) {
                copy_async(&b.slice(source, source + 1), &mut sb.slice_mut(target, target + 1), 8u32);
            } else { sb[target] = b[source]; }
        } else {
            let source = (RUDA_POS_Y as usize * b_batch + (block * bk + p / (bn / 8)) * ldb + col) / 8 + p % (bn / 8);
            let target = bank * bk * (bn / 8 + 1) + (p / (bn / 8)) * (bn / 8 + 1) + p % (bn / 8);
            if comptime!(asynchronous) {
                copy_async(&b.slice(source, source + 1), &mut sb.slice_mut(target, target + 1), 8u32);
            } else { sb[target] = b[source]; }
        }
    }
}
