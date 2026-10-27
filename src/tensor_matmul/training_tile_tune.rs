use ruda::runtime::tune::{LocalTuner, Tunable, TunableSet};
use ruda_kernel::{dsl as kernel_dsl, dsl::{AutotuneKey, Runtime, RudaTuneId, ir::StorageType}, tensor::RudaTensor};
use serde::{Deserialize, Serialize};

#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize, AutotuneKey)]
struct TileKey {
    batch: usize,
    a_batch: usize,
    b_batch: usize,
    m: usize,
    k: usize,
    last_k: usize,
    n: usize,
    lda: usize,
    ldb: usize,
    ta: bool,
    nt: bool,
    input: StorageType,
    output: StorageType,
    output_vector_size: usize,
}

fn key<R: Runtime>((a, b, out, last_k): &(RudaTensor<R>, RudaTensor<R>, RudaTensor<R>, usize)) -> TileKey {
    let d = a.meta.num_dims() - 2;
    let ta = a.meta.strides()[d] == 1;
    let nt = b.meta.strides()[d] == 1;
    TileKey { batch: if d == 0 { 1 } else { a.meta.shape()[0] },
        a_batch: if d == 0 { 0 } else { a.meta.strides()[0] }, b_batch: if d == 0 { 0 } else { b.meta.strides()[0] },
        m: a.meta.shape()[d], k: a.meta.shape()[d+1], last_k: *last_k, n: b.meta.shape()[d+1],
        lda: if ta { a.meta.strides()[d+1] } else { a.meta.strides()[d] },
        ldb: if nt { b.meta.strides()[d+1] } else { b.meta.strides()[d] }, ta, nt, input: a.dtype.into(), output: out.dtype.into(),
        output_vector_size: super::training_gemm::output_vector_size(out) }
}

fn inputs<R: Runtime>(_: &TileKey,
    (a, b, out, last_k): &(RudaTensor<R>, RudaTensor<R>, RudaTensor<R>, usize),
) -> (RudaTensor<R>, RudaTensor<R>, RudaTensor<R>, usize) {
    let mut scratch = out.clone();
    let mut handle = out.client.empty(usize::try_from(out.handle.size()).expect("training matmul span exceeds usize"));
    handle.offset_start = out.handle.offset_start;
    handle.offset_end = out.handle.offset_end;
    scratch.handle = handle;
    (a.clone(), b.clone(), scratch, *last_k)
}

pub(super) fn matmul<R: Runtime>(a: RudaTensor<R>, b: RudaTensor<R>, out: RudaTensor<R>, last_k: usize) {
    static TUNER: LocalTuner<TileKey, RudaTuneId> = LocalTuner::new("rublas::low_precision::ordered_tiles::v26");
    let client = a.client.clone();
    let id = RudaTuneId::new(&client, &a.device);
    let candidates = TUNER.init_for_device(&id, || {
        let mut set = TunableSet::new(key::<R>, inputs::<R>);
        for (name, m, n, k, asynchronous) in [
            ("sync_128x128x32", 128, 128, 32, false),
            ("async_64x64x32", 64, 64, 32, true),
            ("async_128x64x32", 128, 64, 32, true),
            ("async_128x128x32", 128, 128, 32, true),
            ("async_128x64x64", 128, 64, 64, true),
            ("async_128x128x64", 128, 128, 64, true),
            ("sync_32x128x32", 32, 128, 32, false),
            ("async_32x128x32", 32, 128, 32, true),
            ("async_32x128x64", 32, 128, 64, true),
            ("sync_32x64x32", 32, 64, 32, false),
            ("async_32x64x32", 32, 64, 32, true),
            ("async_32x64x64", 32, 64, 64, true),
            ("sync_32x32x32", 32, 32, 32, false),
            ("async_32x32x32", 32, 32, 32, true),
            ("async_64x32x32", 64, 32, 32, true),
            ("async_128x32x64", 128, 32, 64, true),
            ("sync_64x128x32", 64, 128, 32, false),
            ("async_64x128x32", 64, 128, 32, true),
            ("async_64x128x64", 64, 128, 64, true),
        ] {
            for (manual, stages, prefetch) in [(false, 2, false), (true, 2, false), (true, 3, false), (true, 2, true), (true, 3, true)] {
                if !asynchronous && stages == 3 { continue; }
                set = set.with(Tunable::new(&if manual { format!("mma_s{stages}_prefetch{prefetch}_{name}") } else { name.into() },
                    move |(a, b, out, last_k): (RudaTensor<R>, RudaTensor<R>, RudaTensor<R>, usize)| {
                        super::training_gemm::launch_tile(a, b, out, last_k, m, n, k, asynchronous, 32, manual, stages, prefetch)
                    }));
            }
        }
        for (name, m, n, k) in [
            ("async_wm64_128x128x32", 128, 128, 32),
            ("async_wm64_128x128x64", 128, 128, 64),
            ("async_wm64_256x128x32", 256, 128, 32),
            ("async_wm64_256x128x64", 256, 128, 64),
            ("async_wm64_256x64x32", 256, 64, 32),
            ("async_wm64_256x64x64", 256, 64, 64),
        ] {
            for (manual, stages, prefetch) in [(false, 2, false), (true, 2, false), (true, 3, false), (true, 2, true), (true, 3, true)] {
                set = set.with(Tunable::new(&if manual { format!("mma_s{stages}_prefetch{prefetch}_{name}") } else { name.into() },
                    move |(a, b, out, last_k): (RudaTensor<R>, RudaTensor<R>, RudaTensor<R>, usize)| {
                        super::training_gemm::launch_tile(a, b, out, last_k, m, n, k, true, 64, manual, stages, prefetch)
                    }));
            }
        }
        for (name, m, n, k) in [
            ("mma_async_64x256x32", 64, 256, 32),
            ("mma_async_128x256x32", 128, 256, 32),
            ("mma_async_128x256x64", 128, 256, 64),
        ] {
            for (stages, prefetch) in [(2, false), (3, false), (2, true), (3, true)] {
                set = set.with(Tunable::new(&format!("{name}_s{stages}_prefetch{prefetch}"),
                    move |(a, b, out, last_k): (RudaTensor<R>, RudaTensor<R>, RudaTensor<R>, usize)| {
                        super::training_gemm::launch_tile(a, b, out, last_k, m, n, k, true, 32, true, stages, prefetch)
                    }));
            }
        }
        for (m, n, k, warp_m) in [(64, 256, 32, 32), (64, 256, 64, 32),
                                  (128, 256, 32, 64), (128, 256, 64, 64)] {
            for (stages, prefetch) in [(2, false), (3, false), (2, true), (3, true)] {
                set = set.with(Tunable::new(&format!("mma_wc4_wm{warp_m}_{m}x{n}x{k}_s{stages}_prefetch{prefetch}"),
                    move |(a, b, out, last_k): (RudaTensor<R>, RudaTensor<R>, RudaTensor<R>, usize)| {
                        super::training_gemm::launch_tile_warp_grid(a, b, out, last_k,
                            m, n, k, true, warp_m, 4, true, stages, prefetch)
                    }));
            }
        }
        set
    });
    TUNER.execute(&id, &client, candidates, (a, b, out, last_k));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ordered_tile_keys_do_not_bucket_k_or_strides() {
        let a = TileKey { batch: 1, a_batch: 0, b_batch: 0, m: 512, k: 96, last_k: 96, n: 256, lda: 128, ldb: 256, ta: false, nt: false,
            input: ruda_core::tensor::DType::BF16.into(), output: ruda_core::tensor::DType::F32.into(), output_vector_size: 4 };
        assert_ne!(a, TileKey { m: 32, ..a.clone() });
        assert_ne!(a, TileKey { batch: 2, ..a.clone() });
        assert_ne!(a, TileKey { a_batch: 65536, ..a.clone() });
        assert_ne!(a, TileKey { b_batch: 32768, ..a.clone() });
        assert_ne!(a, TileKey { k: 128, ..a.clone() });
        assert_ne!(a, TileKey { last_k: 64, ..a.clone() });
        assert_ne!(a, TileKey { lda: 256, ..a.clone() });
        assert_ne!(a, TileKey { ldb: 512, ..a.clone() });
        assert_ne!(a, TileKey { ta: true, ..a.clone() });
        assert_ne!(a, TileKey { nt: true, ..a.clone() });
        assert_ne!(a, TileKey { input: ruda_core::tensor::DType::F16.into(), ..a.clone() });
        assert_ne!(a, TileKey { output: ruda_core::tensor::DType::BF16.into(), ..a.clone() });
        assert_ne!(a, TileKey { output_vector_size: 1, ..a.clone() });
    }
}
