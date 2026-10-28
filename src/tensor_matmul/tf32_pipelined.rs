use ruda_core::ir::{ElemType, FloatKind, features::MmaConfig};
use ruda_kernel::{dsl as kernel_dsl, dsl::{cmma, prelude::*}, tensor::RudaTensor};
use ruda_kernel::dsl::prelude::barrier::{copy_async, commit_copy_group, wait_copy_group};

// Multi-stage TF32 tensor-core GEMM for row-major A [M,K] and row-major B [K,N].
//
// * cp.async global->shared copies with `stages` buffers in flight.
// * XOR-swizzled shared tiles (no padding) so every fragment read is free of
//   bank conflicts: A fragments use 64-bit loads, B fragments 128-bit loads.
// * The K index inside each k8 step is permuted consistently for A and B
//   (logical k=t -> physical 2t, logical k=t+4 -> physical 2t+1), which is
//   exact because the MMA sums over K. The N index of every warp tile is
//   permuted (logical tile j, column g -> physical q*32 + g*4 + j%4) so that one
//   128-bit shared load feeds four n8 tiles and the accumulators of one thread
//   cover eight contiguous output columns, stored with 128-bit writes.
#[derive(Clone, Copy)]
struct Config { bm: usize, bn: usize, warp_m: usize, warp_n: usize, stages: usize, resident: usize, efficiency: f64 }

const CONFIGS: [Config; 3] = [
    Config { bm: 128, bn: 128, warp_m: 32, warp_n: 64, stages: 2, resident: 2, efficiency: 1.0 },
    Config { bm: 128, bn: 64, warp_m: 32, warp_n: 32, stages: 3, resident: 2, efficiency: 0.85 },
    Config { bm: 64, bn: 64, warp_m: 32, warp_n: 32, stages: 3, resident: 3, efficiency: 0.55 },
];

#[cfg(test)]
mod tests {
    use super::CONFIGS;

    #[test]
    fn affine_copy_offsets_preserve_every_swizzled_element() {
        for config in CONFIGS {
            let threads = config.bm / config.warp_m * (config.bn / config.warp_n) * 32;
            let quads = config.bn / 4;
            assert_eq!(config.bm * 8 % threads, 0);
            assert_eq!(config.bn * 8 % threads, 0);
            assert_eq!(threads % 32, 0);
            assert_eq!(threads % (quads * 8), 0);
            for pitch in [config.bm + 4, config.bn + 12, 1024] {
                for thread in 0..threads {
                    let a_base = thread / 8 * pitch + thread % 8;
                    let b_base = thread / quads * pitch + thread % quads;
                    let a_shared = thread / 8 * 8 + (thread % 8 ^ (thread / 8 % 4) * 2);
                    let b_shared = thread / quads * quads + (thread % quads ^ (thread / quads / 2 % 4) * 2);
                    for i in 0..config.bm * 8 / threads {
                        let p = thread + i * threads;
                        assert_eq!(a_base + i * (threads / 8) * pitch, p / 8 * pitch + p % 8);
                        assert_eq!(a_shared + i * threads, p / 8 * 8 + (p % 8 ^ (p / 8 % 4) * 2));
                    }
                    for i in 0..config.bn * 8 / threads {
                        let p = thread + i * threads;
                        assert_eq!(b_base + i * (threads / quads) * pitch, p / quads * pitch + p % quads);
                        assert_eq!(b_shared + i * threads, p / quads * quads + (p % quads ^ (p / quads / 2 % 4) * 2));
                    }
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn launch<R: Runtime>(a: &RudaTensor<R>, b: &RudaTensor<R>, out: &RudaTensor<R>,
    m: usize, n: usize, k: usize, lda: usize, ldb: usize, a_batch: usize, b_batch: usize,
    batch: usize) -> bool {
    if std::env::var("RUDA_TF32_GEMM").is_ok_and(|value| value == "legacy") { return false; }
    let client = &a.client;
    let hw = &client.properties().hardware;
    let storage = ElemType::Float(FloatKind::TF32).into();
    if !client.features().copy_async || k % 32 != 0 || m % 32 != 0 || n % 32 != 0
        || !client.features().matmul.mma.contains(&MmaConfig { a_type: storage, b_type: storage,
            cd_type: ElemType::Float(FloatKind::F32).into(), m: 16, n: 8, k: 8 })
        || hw.plane_size_min != 32 || hw.plane_size_max != 32
        || out.handle.offset_start.unwrap_or(0) % 16 != 0
        || batch > hw.max_ruda_count.1 as usize
    { return false; }
    let sms = hw.num_streaming_multiprocessors.unwrap_or(108).max(1) as usize;
    let mut best: Option<(f64, Config)> = None;
    for config in CONFIGS {
        let threads = config.bm / config.warp_m * (config.bn / config.warp_n) * 32;
        let shared = config.stages * (config.bm * 32 + 32 * config.bn) * 4;
        let tiles = m.div_ceil(config.bm) * n.div_ceil(config.bn);
        if threads as u32 > hw.max_units_per_ruda || shared > hw.max_shared_memory_size
            || tiles > hw.max_ruda_count.0 as usize { continue; }
        let resident = config.resident.min(hw.max_shared_memory_size / shared).max(1);
        let waves = (tiles * batch).div_ceil(sms * resident);
        let cost = (waves * resident * config.bm * config.bn) as f64 / config.efficiency;
        if best.is_none_or(|(value, _)| cost < value) { best = Some((cost, config)); }
    }
    let Some((_, config)) = best else { return false; };
    let tiles_m = m.div_ceil(config.bm);
    let group = if tiles_m % 8 == 0 { 8 } else { tiles_m };
    let threads = config.bm / config.warp_m * (config.bn / config.warp_n) * 32;
    unsafe { pipelined::launch_unchecked::<R>(client,
        RudaCount::Static((tiles_m * n.div_ceil(config.bn)) as u32, batch as u32, 1),
        RudaDim::new_1d(threads as u32), a.clone().into_array_arg(), b.clone().into_array_arg(),
        out.clone().into_array_arg(), m, n, k, lda, ldb, a_batch, b_batch,
        config.bm, config.bn, config.warp_m, config.warp_n, config.stages, group); }
    true
}

#[ruda]
fn load_stage(a: &Array<Vector<f32, Const<4>>>, b: &Array<Vector<f32, Const<4>>>,
    sa: &mut SharedMemory<Vector<f32, Const<4>>>, sb: &mut SharedMemory<Vector<f32, Const<4>>>,
    a_source: usize, b_source: usize, a_target: usize, b_target: usize,
    a_row: usize, b_col: usize, block: usize, bank: usize,
    #[comptime] m: usize, #[comptime] n: usize, #[comptime] lda: usize, #[comptime] ldb: usize,
    #[comptime] bm: usize, #[comptime] bn: usize, #[comptime] threads: usize) {
    let a_global = a_source + block * 8;
    let b_global = b_source + block * (32 * (ldb / 4));
    let a_shared = bank * (bm * 8) + a_target;
    let b_shared = bank * (bn * 8) + b_target;
    #[unroll]
    for i in 0usize..comptime!(bm * 8 / threads) {
        let target = a_shared + i * threads;
        let source = a_global + i * ((threads / 8) * (lda / 4));
        let mut valid = true;
        if comptime!(m % bm != 0) { valid = a_row + i * (threads / 8) < m; }
        if valid {
            copy_async(&a.slice(source, source + 1), &mut sa.slice_mut(target, target + 1), 4u32);
        } else {
            sa[target] = Vector::new(0.0f32);
        }
    }
    let mut b_valid = true;
    if comptime!(n % bn != 0) { b_valid = b_col < n; }
    #[unroll]
    for i in 0usize..comptime!(bn * 8 / threads) {
        let target = b_shared + i * threads;
        let source = b_global + i * ((threads / (bn / 4)) * (ldb / 4));
        if b_valid {
            copy_async(&b.slice(source, source + 1), &mut sb.slice_mut(target, target + 1), 4u32);
        } else {
            sb[target] = Vector::new(0.0f32);
        }
    }
}

#[ruda(launch_unchecked)]
fn pipelined(a: &Array<Vector<f32, Const<4>>>, b: &Array<Vector<f32, Const<4>>>,
    out: &mut Array<Vector<f32, Const<4>>>,
    #[comptime] m: usize, #[comptime] n: usize, #[comptime] k: usize,
    #[comptime] lda: usize, #[comptime] ldb: usize, #[comptime] a_batch: usize, #[comptime] b_batch: usize,
    #[comptime] bm: usize, #[comptime] bn: usize, #[comptime] warp_m: usize, #[comptime] warp_n: usize,
    #[comptime] stages: usize, #[comptime] group: usize) {
    let thread = UNIT_POS as usize;
    let lane = thread % 32;
    let warp = thread / 32;
    let g = lane / 4;
    let t = lane % 4;
    let warp_cols = comptime!(bn / warp_n);
    let threads = comptime!(bm / warp_m * (bn / warp_n) * 32);
    let per_group = comptime!(group * n.div_ceil(bn));
    let pid = RUDA_POS_X as usize;
    let local = pid % per_group;
    let row = (pid / per_group * group + local % group) * bm;
    let col = local / group * bn;
    let wm = warp / warp_cols * warp_m;
    let wn = warp % warp_cols * warp_n;
    let mt = comptime!(warp_m / 16);
    let ntile = comptime!(warp_n / 8);
    let quads = comptime!(warp_n / 32);
    let a_stage = comptime!(bm * 8);
    let b_stage = comptime!(bn * 8);
    let b_row = comptime!(bn / 4);
    let blocks = comptime!(k / 32);
    let a_copy_row = row + thread / 8;
    let b_copy_col = col + (thread % b_row) * 4;
    let a_source = RUDA_POS_Y as usize * (a_batch / 4) + a_copy_row * (lda / 4) + thread % 8;
    let b_source = RUDA_POS_Y as usize * (b_batch / 4) + (thread / b_row) * (ldb / 4) + b_copy_col / 4;
    let a_target = (thread / 8) * 8 + ((thread % 8) ^ ((thread / 8 % 4) * 2));
    let b_target = (thread / b_row) * b_row + ((thread % b_row) ^ ((thread / b_row / 2 % 4) * 2));
    let mut sa = SharedMemory::<Vector<f32, Const<4>>>::new_aligned(a_stage * stages, 32usize);
    let mut sb = SharedMemory::<Vector<f32, Const<4>>>::new_aligned(b_stage * stages, 32usize);
    let pairs = sa.to_slice().with_vector_size::<Const<2>>();
    let mut acc = Sequence::<Vector<f32, Const<4>>>::new();
    #[unroll] for _ in 0usize..(mt * ntile) { acc.push(Vector::new(0.0f32)); }
    let def = cmma::MmaDefinition::<tf32, tf32, f32>::new(16usize, 8usize, 8usize);
    #[unroll]
    for stage in 0usize..(stages - 1) {
        if comptime!(stage < k / 32) {
            load_stage(a, b, &mut sa, &mut sb, a_source, b_source, a_target, b_target,
                a_copy_row, b_copy_col, stage, stage, m, n, lda, ldb, bm, bn, threads);
        }
        commit_copy_group();
    }
    // Per-thread swizzle terms: A rows have row % 4 == g % 4, B rows of one
    // k8 step have (row / 2) % 4 == t.
    let xa = (g % 4) * 2;
    let a_base = (wm + g) * 16 + t % 2;
    let b_col = wn / 4 + (g ^ (t * 2));
    let b_base = t * 2 * b_row + b_col;
    let mut bank = 0usize;
    let mut write_bank = usize::cast_from(stages - 1);
    for block in 0usize..blocks {
        wait_copy_group(comptime!((stages - 2) as u32));
        sync_ruda();
        let next = block + stages - 1;
        if next < blocks {
            load_stage(a, b, &mut sa, &mut sb, a_source, b_source, a_target, b_target,
                a_copy_row, b_copy_col, next, write_bank, m, n, lda, ldb, bm, bn, threads);
        }
        commit_copy_group();
        let a_read = bank * (a_stage * 2) + a_base;
        let b_read = bank * b_stage + b_base;
        #[unroll]
        for s in 0usize..4 {
            let mut ar = Sequence::<Vector<tf32, Const<4>>>::new();
            let chunk = ((s * 2) ^ xa) + t / 2;
            #[unroll]
            for i in 0usize..mt {
                let index = a_read + i * 256 + chunk * 2;
                let low = pairs[index];
                let high = pairs[index + 128];
                let mut fragment = Vector::<tf32, Const<4>>::empty();
                fragment[0usize] = tf32::cast_from(low[0usize]);
                fragment[1usize] = tf32::cast_from(high[0usize]);
                fragment[2usize] = tf32::cast_from(low[1usize]);
                fragment[3usize] = tf32::cast_from(high[1usize]);
                ar.push(fragment);
            }
            #[unroll]
            for q in 0usize..quads {
                let index = b_read + s * (8 * b_row) + q * 8;
                let first = sb[index];
                let second = sb[index + b_row];
                #[unroll]
                for jj in 0usize..4 {
                    let mut br = Vector::<tf32, Const<2>>::empty();
                    br[0usize] = tf32::cast_from(first[jj]);
                    br[1usize] = tf32::cast_from(second[jj]);
                    #[unroll]
                    for i in 0usize..mt {
                        def.execute_vectors_inplace(&ar[i], &br, acc.index_mut(i * ntile + q * 4 + jj));
                    }
                }
            }
        }
        bank += 1;
        if bank == stages { bank = 0; }
        write_bank += 1;
        if write_bank == stages { write_bank = 0; }
    }
    wait_copy_group(0u32);
    let base = RUDA_POS_Y as usize * m * n;
    #[unroll]
    for i in 0usize..mt {
        #[unroll]
        for h in 0usize..2 {
            let r = row + wm + i * 16 + g + h * 8;
            #[unroll]
            for q in 0usize..quads {
                let c = col + wn + q * 32 + t * 8;
                let mut low = Vector::<f32, Const<4>>::empty();
                let mut high = Vector::<f32, Const<4>>::empty();
                #[unroll]
                for jj in 0usize..4 {
                    let value = acc[i * ntile + q * 4 + jj];
                    low[jj] = value[h * 2];
                    high[jj] = value[h * 2 + 1];
                }
                let mut valid = true;
                if comptime!(m % bm != 0) { valid = valid && r < m; }
                if comptime!(n % bn != 0) { valid = valid && c < n; }
                if valid {
                    out[(base + r * n + c) / 4] = low;
                    out[(base + r * n + c + 4) / 4] = high;
                }
            }
        }
    }
}
