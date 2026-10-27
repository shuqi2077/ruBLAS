use ruda_core::ir::{ElemType, FloatKind, features::MmaConfig};
use ruda_kernel::{dsl as kernel_dsl, dsl::{cmma, prelude::*}, tensor::RudaTensor};
use ruda_kernel::dsl::prelude::barrier::{commit_copy_group, wait_copy_group};
use super::f32_tiled::load;

// Inputs have already passed the F32 tiled layout, span and alignment checks.
pub(super) fn launch<R: Runtime>(a: &RudaTensor<R>, b: &RudaTensor<R>, out: &RudaTensor<R>,
    m: usize, n: usize, k: usize, lda: usize, ldb: usize, a_batch: usize, b_batch: usize,
    ta: bool, nt: bool, batch: usize) -> bool {
    let client = &a.client;
    let hw = &client.properties().hardware;
    let storage = ElemType::Float(FloatKind::TF32).into();
    if !client.features().matmul.mma.contains(&MmaConfig { a_type: storage, b_type: storage,
        cd_type: ElemType::Float(FloatKind::F32).into(), m: 16, n: 8, k: 8 })
        || hw.plane_size_min != 32 || hw.plane_size_max != 32 || hw.max_units_per_ruda < 128
    { return false; }
    let bm = if m >= 128 && n >= 128 { 128 } else { 64 };
    let bn = 64;
    let bk = if k % 32 == 0 { 32 } else if k % 16 == 0 { 16 } else { 8 };
    let astage = if ta { bk * (bm / 4 + 1) } else { bm * (bk / 4 + 1) };
    let bstage = if nt { bn * (bk / 4 + 1) } else { bk * (bn / 4 + 1) };
    if hw.max_shared_memory_size < (astage + bstage) * 32
        || m.div_ceil(bm) * n.div_ceil(bn) > hw.max_ruda_count.0 as usize
        || batch > hw.max_ruda_count.1 as usize { return false; }
    unsafe { tiled::launch_unchecked::<R>(client,
        RudaCount::Static((m.div_ceil(bm) * n.div_ceil(bn)) as u32, batch as u32, 1),
        RudaDim::new_1d(128), a.clone().into_array_arg(), b.clone().into_array_arg(), out.clone().into_array_arg(),
        m, n, k, lda, ldb, a_batch, b_batch, ta, nt, bm, bn, bk, client.features().copy_async); }
    true
}

#[ruda(launch_unchecked)]
fn tiled(a: &Array<Vector<f32, Const<4>>>, b: &Array<Vector<f32, Const<4>>>, out: &mut Array<f32>,
    #[comptime] m: usize, #[comptime] n: usize, #[comptime] k: usize,
    #[comptime] lda: usize, #[comptime] ldb: usize, #[comptime] a_batch: usize, #[comptime] b_batch: usize,
    #[comptime] ta: bool, #[comptime] nt: bool, #[comptime] bm: usize, #[comptime] bn: usize,
    #[comptime] bk: usize, #[comptime] asynchronous: bool) {
    let lane = UNIT_POS as usize % 32;
    let warp = UNIT_POS as usize / 32;
    let row = RUDA_POS_X as usize % comptime!(m.div_ceil(bm)) * bm;
    let col = RUDA_POS_X as usize / comptime!(m.div_ceil(bm)) * bn;
    let wm = warp / 2 * (bm / 2);
    let wn = warp % 2 * (bn / 2);
    let mt = comptime!(bm / 32);
    let ntile = comptime!(bn / 16);
    let astage = comptime!(if ta { bk * (bm / 4 + 1) } else { bm * (bk / 4 + 1) });
    let bstage = comptime!(if nt { bn * (bk / 4 + 1) } else { bk * (bn / 4 + 1) });
    let mut sa = SharedMemory::<Vector<f32, Const<4>>>::new_aligned(astage * 2, 32usize);
    let mut sb = SharedMemory::<Vector<f32, Const<4>>>::new_aligned(bstage * 2, 32usize);
    let scalar_a = sa.to_slice().with_vector_size::<Const<1>>();
    let scalar_b = sb.to_slice().with_vector_size::<Const<1>>();
    let mut acc = Sequence::<Vector<f32, Const<4>>>::new();
    #[unroll] for _ in 0usize..(mt * ntile) { acc.push(Vector::new(0.0f32)); }
    let def = cmma::MmaDefinition::<tf32, tf32, f32>::new(16usize, 8usize, 8usize);
    load(a, b, &mut sa, &mut sb, row, col, 0usize, 0usize, lda, ldb, a_batch, b_batch,
        ta, nt, bm, bn, bk, asynchronous, m, n, 128usize);
    if comptime!(asynchronous) { commit_copy_group(); }
    for block in 0usize..(k / bk) {
        let bank = block % 2;
        if comptime!(asynchronous) { wait_copy_group(0u32); }
        sync_ruda();
        if block + 1 < k / bk {
            load(a, b, &mut sa, &mut sb, row, col, block + 1, 1 - bank, lda, ldb, a_batch, b_batch,
                ta, nt, bm, bn, bk, asynchronous, m, n, 128usize);
            if comptime!(asynchronous) { commit_copy_group(); }
        }
        #[unroll] for step in 0usize..(bk / 8) {
            let mut ar = Array::<Vector<tf32, Const<4>>>::new(mt);
            #[unroll] for i in 0usize..mt {
                let mut fragment = Vector::<tf32, Const<4>>::empty();
                #[unroll] for v in 0usize..4 {
                    let r = wm + i * 16 + lane / 4 + (v % 2) * 8;
                    let c = step * 8 + lane % 4 + (v / 2) * 4;
                    let mut value = 0.0f32;
                    if comptime!(ta) { value = scalar_a[bank * astage * 4 + c * (bm + 4) + r][0usize]; }
                    else { value = scalar_a[bank * astage * 4 + r * (bk + 4) + c][0usize]; }
                    fragment[v] = tf32::cast_from(value);
                }
                ar[i] = fragment;
            }
            #[unroll] for j in 0usize..ntile {
                let mut br = Vector::<tf32, Const<2>>::empty();
                #[unroll] for v in 0usize..2 {
                    let r = step * 8 + lane % 4 + v * 4;
                    let c = wn + j * 8 + lane / 4;
                    let mut value = 0.0f32;
                    if comptime!(nt) { value = scalar_b[bank * bstage * 4 + c * (bk + 4) + r][0usize]; }
                    else { value = scalar_b[bank * bstage * 4 + r * (bn + 4) + c][0usize]; }
                    br[v] = tf32::cast_from(value);
                }
                #[unroll] for i in 0usize..mt {
                    def.execute_vectors_inplace(&ar[i], &br, acc.index_mut(i * ntile + j));
                }
            }
        }
    }
    #[unroll] for i in 0usize..mt {
        #[unroll] for j in 0usize..ntile {
            let values = acc[i * ntile + j];
            #[unroll] for v in 0usize..4 {
                let r = row + wm + i * 16 + lane / 4 + (v / 2) * 8;
                let c = col + wn + j * 8 + (lane % 4) * 2 + v % 2;
                if r < m && c < n { out[RUDA_POS_Y as usize * m * n + r * n + c] = values[v]; }
            }
        }
    }
}
