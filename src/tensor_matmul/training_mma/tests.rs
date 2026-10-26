use crate::tensor_matmul::training_gemm::{launch_tile, launch_tile_warp_grid};
use half::bf16;
use ruda_core::tensor::{DType, Metadata, Shape, data::TensorData};
use ruda_kernel::tensor::{RudaTensor, transfer::from_data, readback::into_data_sync, permutation::swap_dims};
use ruda_test_runtime::TestRuntime;

type Tensor = RudaTensor<TestRuntime>;

#[test]
fn manual_training_warp_grid_covers_each_output_once() {
    for bm in [32usize, 64, 128, 256] {
        for bn in [64usize, 128, 256] {
            for warp_m in [32usize, 64] {
                if bm % warp_m != 0 { continue; }
                for warp_cols in [2usize, 4] {
                    let warp_n = bn / warp_cols;
                    let warps = bm / warp_m * warp_cols;
                    let mut counts = vec![0usize; bm * bn];
                    for warp in 0..warps {
                        let wm = warp / warp_cols * warp_m;
                        let wn = warp % warp_cols * warp_n;
                        for i in 0..warp_m / 16 {
                            for j in 0..warp_n / 8 {
                                for lane in 0..32 {
                                    for r in 0..2 {
                                        for c in 0..2 {
                                            let row = wm + i * 16 + lane / 4 + r * 8;
                                            let col = wn + j * 8 + lane % 4 * 2 + c;
                                            assert!(row < bm && col < bn);
                                            counts[row * bn + col] += 1;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    assert!(counts.iter().all(|&count| count == 1));
                    for bk in [32usize, 64] {
                        let threads = warps * 32;
                        if bm * bk % (threads * 8) != 0 || bn * bk % (threads * 8) != 0 { continue; }
                        for elements in [bm * bk / 8, bn * bk / 8] {
                            let mut copied = vec![0usize; elements];
                            let copies = elements / threads;
                            let per_group = copies.div_ceil(bk / 16);
                            for group in 0..bk / 16 {
                                for i in (group * per_group).min(copies)..((group + 1) * per_group).min(copies) {
                                    for thread in 0..threads { copied[thread + i * threads] += 1; }
                                }
                            }
                            assert!(copied.iter().all(|&count| count == 1));
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn manual_training_epilogue_swizzle_preserves_coordinates_and_bank_transactions() {
    let index = |row: usize, col: usize| row * 16 + (col ^ ((row / 2 % 2) * 12));
    let mut shared = [usize::MAX; 256];
    for part in 0..2 {
        for r in 0..2 {
            for half_warp in 0..2 {
                let mut banks = [0usize; 32];
                for lane in half_warp * 16..(half_warp + 1) * 16 {
                    let (row, col) = (lane / 4 + r * 8, part * 8 + lane % 4 * 2);
                    let start = index(row, col);
                    assert_eq!(start % 2, 0);
                    for c in 0..2 {
                        assert_eq!(index(row, col + c), start + c);
                        assert_eq!(shared[start + c], usize::MAX);
                        shared[start + c] = row * 16 + col + c;
                        banks[(start + c) % 32] += 1;
                    }
                }
                assert_eq!(banks, [1; 32]);
            }
        }
    }
    for width in [1, 4, 8] {
        let vector = width.min(4);
        let lanes = 32 / vector;
        for t in 0..8 / width {
            for group in 0..width / vector {
                for transaction in 0..32 / lanes {
                    let mut banks = [0usize; 32];
                    for lane in transaction * lanes..(transaction + 1) * lanes {
                        let logical = (lane + t * 32) * width + group * vector;
                        let start = index(logical / 16, logical % 16);
                        assert_eq!(start % vector, 0);
                        for v in 0..vector {
                            assert_eq!(shared[start + v], logical + v);
                            banks[(start + v) % 32] += 1;
                        }
                    }
                    assert_eq!(banks, [1; 32]);
                }
            }
        }
    }
}

#[test]
fn manual_training_compact_offsets_match_full_fragment_addresses() {
    let index = |row: usize, chunk: usize, chunks: usize| {
        row * chunks + (chunk ^ ((row / 8usize.div_ceil(chunks)) % chunks.min(8)))
    };
    for bm in [32usize, 64, 96, 128, 256] {
        for bn in [32usize, 64, 96, 128, 256] {
            for bk in [32usize, 64, 96, 128] {
                for warp_m in [32usize, 64] {
                    if bm % warp_m != 0 { continue; }
                    for warp in 0..bm / warp_m * 2 {
                        let wm = warp / 2 * warp_m;
                        let wn = warp % 2 * (bn / 2);
                        for lane in 0..32 {
                            for bank in 0..3 {
                                for half in 0..bk / 16 {
                                    for transposed in [false, true] {
                                        for i in 0..warp_m / 16 {
                                            let base = bank * bm * bk / 8 + if transposed {
                                                index(lane % 8 + lane / 16 * 8, (wm + i * 16) / 8 + lane % 16 / 8, bm / 8)
                                            } else { index(wm + i * 16 + lane % 16, lane / 16, bk / 8) };
                                            let expected = bank * bm * bk / 8 + if transposed {
                                                index(half * 16 + lane % 8 + lane / 16 * 8, (wm + i * 16) / 8 + lane % 16 / 8, bm / 8)
                                            } else { index(wm + i * 16 + lane % 16, half * 2 + lane / 16, bk / 8) };
                                            let actual = if transposed { base + half * 16 * (bm / 8) }
                                                else if bk.is_power_of_two() { base ^ (half * 2) }
                                                else { expected };
                                            assert_eq!(actual, expected, "A: {bm}x{bn}x{bk}, warp={warp}, lane={lane}, bank={bank}, half={half}");
                                        }
                                        for j in 0..bn / 16 {
                                            let base = bank * bn * bk / 8 + if transposed {
                                                index(wn + j * 8 + lane % 8, lane % 16 / 8, bk / 8)
                                            } else { index(lane % 16, wn / 8 + j, bn / 8) };
                                            let expected = bank * bn * bk / 8 + if transposed {
                                                index(wn + j * 8 + lane % 8, half * 2 + lane % 16 / 8, bk / 8)
                                            } else { index(half * 16 + lane % 16, wn / 8 + j, bn / 8) };
                                            let actual = if !transposed { base + half * 16 * (bn / 8) }
                                                else if bk.is_power_of_two() { base ^ (half * 2) }
                                                else { expected };
                                            assert_eq!(actual, expected, "B: {bm}x{bn}x{bk}, warp={warp}, lane={lane}, bank={bank}, half={half}");
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn manual_training_vector_load_offsets_preserve_coordinates() {
    for tile in [32usize, 64, 128, 256] {
        for bk in [32usize, 64, 128] {
            for batch in [0usize, 1, 2] {
                for batch_stride in [0usize, 1048576] {
                    for stride in [1024usize, 1032, 2048] {
                        for start in [0usize, 32, 512] {
                            for block in [0usize, 1, 7] {
                                for p in 0..tile * bk / 8 {
                                    let old_row = (batch * batch_stride + (start + p / (bk / 8)) * stride + block * bk) / 8 + p % (bk / 8);
                                    let new_row = batch * (batch_stride / 8) + (start + p / (bk / 8)) * (stride / 8) + block * (bk / 8) + p % (bk / 8);
                                    let old_col = (batch * batch_stride + (block * bk + p / (tile / 8)) * stride + start) / 8 + p % (tile / 8);
                                    let new_col = batch * (batch_stride / 8) + (block * bk + p / (tile / 8)) * (stride / 8) + start / 8 + p % (tile / 8);
                                    assert_eq!(new_row, old_row);
                                    assert_eq!(new_col, old_col);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn manual_training_fragment_prefetch_preserves_k_order() {
    for bk in [32usize, 64, 96, 128] {
        for warp_m in [32usize, 64] {
            for bn in [32usize, 64, 128, 256] {
                let (na, nb) = (warp_m / 16, bn / 16);
                let mut left = (0..na).map(|i| (0, i)).collect::<Vec<_>>();
                let mut right = (0..nb).map(|j| (0, j)).collect::<Vec<_>>();
                let mut order = vec![Vec::new(); na * nb];
                for half in 0..bk / 16 {
                    if half + 1 < bk / 16 {
                        left.extend((0..na).map(|i| (half + 1, i)));
                        right.extend((0..nb).map(|j| (half + 1, j)));
                    }
                    assert!((1..=2).contains(&(left.len() / na - half)));
                    assert!((1..=2).contains(&(right.len() / nb - half)));
                    for j in 0..nb {
                        for i in 0..na {
                            assert_eq!(left[half * na + i], (half, i));
                            assert_eq!(right[half * nb + j], (half, j));
                            order[i * nb + j].push(half);
                        }
                    }
                }
                assert_eq!(left.len(), bk / 16 * na);
                assert_eq!(right.len(), bk / 16 * nb);
                for sequence in order { assert_eq!(sequence, (0..bk / 16).collect::<Vec<_>>()); }
            }
        }
    }
}

fn values(count: usize, seed: usize) -> Vec<bf16> {
    (0..count).map(|i| {
        let bits = ((i * 17 + seed) % 128) | (((i * 7 + seed) % 12 + 119) << 7)
            | (((i * 13 + seed) % 2) << 15);
        bf16::from_bits(bits as u16)
    }).collect()
}

fn output(batch: usize, m: usize, n: usize, dtype: DType, offset: usize) -> (Tensor, Tensor) {
    let count = batch * m * n;
    let shape = Shape::from([count + offset + 8]);
    let data = if dtype == DType::BF16 {
        TensorData::new(vec![bf16::from_f32(-7.0); count + offset + 8], shape)
    } else {
        TensorData::new(vec![-7.0f32; count + offset + 8], shape)
    };
    let storage: Tensor = from_data(data, &Default::default());
    let mut view = storage.clone();
    *view.meta = Metadata::new([batch, m, n], vec![m * n, n, 1]);
    view.handle.offset_start = Some((offset * dtype.size()) as u64);
    (view, storage)
}

fn read(tensor: Tensor) -> Vec<f32> {
    let dtype = tensor.dtype;
    let data = into_data_sync(tensor);
    if dtype == DType::BF16 {
        data.to_vec::<bf16>().unwrap().into_iter().map(bf16::to_f32).collect()
    } else { data.to_vec::<f32>().unwrap() }
}

#[test]
#[ignore = "requires an actual CUDA device and RUDA_REQUIRE_GPU=1"]
fn manual_training_tiles_match_wmma_for_layouts_pipeline_tails_and_offsets() {
    let cases = check_tiles(&[
        (32, 32, 32, false, 32, 1, 2),
        (128, 128, 32, true, 32, 2, 2), (128, 128, 32, true, 32, 3, 2),
        (128, 32, 64, true, 32, 2, 2), (128, 32, 64, true, 32, 3, 2),
        (128, 128, 64, true, 64, 2, 2), (128, 128, 64, true, 64, 3, 2),
        (128, 256, 64, true, 32, 2, 2), (128, 256, 64, true, 32, 3, 2),
    ], &[1, 2, 4, 7]);
    assert_eq!(cases, 576);
}

#[test]
#[ignore = "requires an actual CUDA device and RUDA_REQUIRE_GPU=1"]
fn manual_training_warp_grids_match_wmma_for_layouts_pipeline_tails_and_offsets() {
    let cases = check_tiles(&[
        (64, 256, 32, true, 32, 2, 4), (64, 256, 64, true, 32, 3, 4),
        (128, 256, 32, true, 64, 3, 4), (128, 256, 64, true, 64, 2, 4),
        (128, 128, 64, true, 32, 2, 2), (128, 128, 64, true, 64, 2, 2),
    ], &[1, 4, 7]);
    assert_eq!(cases, 288);
}

fn check_tiles(tiles: &[(usize, usize, usize, bool, usize, usize, usize)], block_counts: &[usize]) -> usize {
    assert!(std::any::type_name::<TestRuntime>().contains("CudaRuntime"));
    assert_eq!(std::env::var("RUDA_REQUIRE_GPU").as_deref(), Ok("1"));
    let mut cases = 0;
    for &(bm, bn, bk, asynchronous, warp_m, stages, warp_cols) in tiles {
        for ta in [false, true] {
            for nt in [false, true] {
                for &blocks in block_counts {
                    let (batch, m, k, n) = (2usize, bm, bk * blocks, bn * 2);
                    let last_k = if blocks == 1 { k } else { k - bk };
                    let pad = if blocks % 2 == 0 { 8 } else { 0 };
                    let (ar, ac) = if ta { (k, m) } else { (m, k) };
                    let (br, bc) = if nt { (n, k) } else { (k, n) };
                    let left = values(batch * ar * (ac + pad), 3);
                    let right = values(br * (bc + pad), 19);
                    let mut a: Tensor = from_data(TensorData::new(left.clone(), [batch, ar, ac + pad]), &Default::default());
                    let mut b: Tensor = from_data(TensorData::new(right.clone(), [1, br, bc + pad]), &Default::default());
                    a.meta.shape_mut()[2] = ac;
                    b.meta.shape_mut()[2] = bc;
                    b.meta.shape_mut()[0] = batch;
                    b.meta.strides_mut()[0] = 0;
                    if ta { a = swap_dims(a, 1, 2); }
                    if nt { b = swap_dims(b, 1, 2); }
                    for (dtype, prefetch) in [(DType::BF16, false), (DType::F32, false), (DType::BF16, true), (DType::F32, true)] {
                        let offset = if blocks == 4 { 3 } else { 0 };
                        let (out, storage) = output(batch, m, n, dtype, offset);
                        let (reference, _) = output(batch, m, n, dtype, 0);
                        let actual = read(launch_tile_warp_grid(a.clone(), b.clone(), out, last_k,
                            bm, bn, bk, asynchronous, warp_m, warp_cols, true, stages, prefetch).unwrap());
                        let expected = read(launch_tile(a.clone(), b.clone(), reference, last_k,
                            32, 32, 32, false, 32, false, 1, false).unwrap());
                        assert_eq!(actual.len(), expected.len());
                        let mismatch = actual.iter().zip(&expected).position(|(a, b)| a.to_bits() != b.to_bits());
                        assert_eq!(mismatch, None,
                            "tile={bm}x{bn}x{bk}, warp_cols={warp_cols}, stages={stages}, prefetch={prefetch}, ta={ta}, nt={nt}, blocks={blocks}, dtype={dtype:?}");
                        let whole = read(storage);
                        assert!(whole[..offset].iter().all(|v| *v == -7.0));
                        assert!(whole[offset + actual.len()..].iter().all(|v| *v == -7.0));
                        for z in 0..batch {
                            for (row, col) in [(0, 0), (m / 2, n / 2), (m - 1, n - 1)] {
                                let limit = if z + 1 == batch { last_k } else { k };
                                let sum: f64 = (0..limit).map(|t| {
                                    let ai = z * ar * (ac + pad) + if ta { t * (ac + pad) + row } else { row * (ac + pad) + t };
                                    let bi = if nt { col * (bc + pad) + t } else { t * (bc + pad) + col };
                                    f64::from(left[ai].to_f32()) * f64::from(right[bi].to_f32())
                                }).sum();
                                let tolerance = if dtype == DType::BF16 { 0.008 } else { 2e-5 };
                                let value = f64::from(actual[(z * m + row) * n + col]);
                                assert!(value.is_finite() && (value - sum).abs() <= tolerance * sum.abs().max(1.0));
                            }
                        }
                        cases += 1;
                    }
                }
            }
        }
    }
    cases
}
