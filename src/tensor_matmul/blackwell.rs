use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::{dsl::prelude::*, tensor::RudaTensor};
use ruda_core::tensor::{DType, Metadata};

#[ruda]
fn load_tile<I: Float>(barrier: &barrier::Barrier, source: &TensorMap<I, Tiled>, target: &mut SliceMut<I>,
    batch: i32, row: i32, column: i32, #[comptime] batched: bool) {
    if comptime!(batched) { barrier.tma_load_3d(source, target, batch, row, column); }
    else { barrier.tma_load_2d(source, target, row, column); }
}

#[ruda(launch)]
pub fn blackwell_gemm<I: Float, O: Float>(
    lhs: &TensorMap<I, Tiled>, rhs: &TensorMap<I, Tiled>, out: &mut Array<O>, bias: &Array<O>,
    m: u32, n: u32, k: u32, partition_k: u32, workers: u32,
    alpha: f32, beta: f32, bias_row: usize, bias_column: usize,
    #[define(I)] _input: StorageType, #[define(O)] _output: StorageType,
    #[comptime] bn: usize, #[comptime] stages: usize, #[comptime] bf16: bool,
    #[comptime] partitions: usize, #[comptime] epilogue: bool, #[comptime] use_bias: bool,
    #[comptime] batch_shape: Vec<usize>, #[comptime] batch_a: Vec<usize>, #[comptime] batch_b: Vec<usize>,
) {
    let mut a = SharedMemory::<I>::new_aligned(128 * 64 * stages, 1024usize);
    let mut b = SharedMemory::<I>::new_aligned(bn * 64 * stages, 1024usize);
    let mut transpose = SharedMemory::<f32>::new_aligned(128usize * 33, 128usize);
    let mut allocation = SharedMemory::<u32>::new(1usize);
    let mut ready = Sequence::<Shared<barrier::Barrier>>::new();
    let mut complete = Sequence::<Shared<barrier::Barrier>>::new();
    #[unroll]
    for _ in 0usize..stages {
        ready.push(barrier::Barrier::shared(1u32, UNIT_POS == 0));
        complete.push(barrier::Barrier::shared(1u32, UNIT_POS == 0));
    }
    sync_async_proxy_shared();
    if UNIT_POS < 32 {
        tcgen05::allocate(&mut allocation.slice_mut(0usize, 1usize), bn as u32);
        tcgen05::relinquish();
    }
    sync_ruda();
    let address = allocation[0];
    let count_m = (m as usize + 127) / 128;
    let count_n = (n as usize + bn - 1) / bn;
    let partition = RUDA_POS_Z as usize;
    let batch = RUDA_POS_Y as usize;
    let mut remaining = batch;
    let mut left_batch = 0usize;
    let mut right_batch = 0usize;
    #[unroll]
    for step in 0usize..batch_shape.len() {
        let axis = comptime!(batch_shape.len() - 1 - step);
        let coordinate = remaining % comptime!(batch_shape[axis]);
        remaining /= comptime!(batch_shape[axis]);
        left_batch += coordinate * comptime!(batch_a[axis]);
        right_batch += coordinate * comptime!(batch_b[axis]);
    }
    let start_k = partition * partition_k as usize;
    let mut span_k = k as usize;
    if comptime!(partitions > 1) { span_k = partition_k as usize; }
    let tiles_k = span_k / 64;
    let mut tile = RUDA_POS_X as usize;
    let mut tile_iteration = 0usize;
    while tile < count_m * count_n {
        let group = tile / (count_n * 8) * 8;
        let mut height = 8usize;
        if count_m - group < height { height = count_m - group; }
        let local = tile % (count_n * 8);
        let row = (group + local % height) * 128;
        let column = (local / height) * bn;
        if UNIT_POS == 0 {
            for cycle in 0usize..((tiles_k + stages - 1) / stages) {
                #[unroll]
                for stage in 0usize..stages {
                    let step = cycle * stages + stage;
                    if step < tiles_k {
                        if cycle > 0 {
                            let prior = tile_iteration * ((tiles_k + stages - 1 - stage) / stages);
                            complete[stage].wait_parity(((prior + cycle - 1) & 1) as u32);
                        }
                        ready[stage].arrive_and_expect_tx(1u32, ((128 + bn) * 64 * 2) as u32);
                        load_tile(&ready[stage], lhs, &mut a.slice_mut(stage*128*64, (stage+1)*128*64), left_batch as i32, row as i32, (start_k + step*64) as i32, comptime!(!batch_shape.is_empty()));
                        load_tile(&ready[stage], rhs, &mut b.slice_mut(stage*bn*64, (stage+1)*bn*64), right_batch as i32, column as i32, (start_k + step*64) as i32, comptime!(!batch_shape.is_empty()));
                    }
                }
            }
        } else if UNIT_POS == 32 {
            for cycle in 0usize..((tiles_k + stages - 1) / stages) {
                #[unroll]
                for stage in 0usize..stages {
                    let step = cycle * stages + stage;
                    if step < tiles_k {
                        let prior = tile_iteration * ((tiles_k + stages - 1 - stage) / stages);
                        let phase = ((prior + cycle) & 1) as u32;
                        ready[stage].wait_parity(phase);
                        tcgen05::fence(false);
                        #[unroll]
                        for inner in 0usize..4 {
                            let ad = tcgen05::descriptor(&a.slice(stage*128*64 + inner*16, stage*128*64 + inner*16 + 8));
                            let bd = tcgen05::descriptor(&b.slice(stage*bn*64 + inner*16, stage*bn*64 + inner*16 + 8));
                            tcgen05::mma(address, ad, bd, step != 0 || inner != 0, bn as u32, bf16);
                        }
                        tcgen05::commit(&complete[stage]);
                    }
                }
            }
            #[unroll]
            for stage in 0usize..stages {
                if stage < tiles_k {
                    let completed = (tile_iteration + 1) * ((tiles_k + stages - 1 - stage) / stages) - 1;
                    complete[stage].wait_parity((completed & 1) as u32);
                }
            }
            tcgen05::fence(true);
        }
        sync_ruda();
        tcgen05::fence(false);
        #[unroll]
        for chunk in 0usize..(bn / 32) {
            let registers = tcgen05::load(address + ((UNIT_POS / 32 * 32) << 16) + (chunk * 32) as u32);
            #[unroll]
            for component in 0usize..32 {
                transpose[UNIT_POS as usize * 33 + component] = registers[component];
            }
            sync_ruda();
            #[unroll]
            for copy in 0usize..32 {
                let position = UNIT_POS as usize + copy * 128;
                let r = row + position / 32;
                let c = column + chunk * 32 + position % 32;
                if r < m as usize && c < n as usize {
                    let mut value = transpose[(position / 32) * 33 + position % 32];
                    if comptime!(epilogue) {
                        value = alpha * value;
                        if comptime!(use_bias) { value += beta * f32::cast_from(bias[r * bias_row + c * bias_column]); }
                    }
                    out[((batch * partitions + partition) * m as usize + r) * n as usize + c] = O::cast_from(value);
                }
            }
            if comptime!(chunk + 1 == bn / 32) { tcgen05::fence(true); }
            sync_ruda();
        }
        // Each stage completes ceil((tiles_k-stage)/stages) barrier phases per tile.
        tile_iteration += 1;
        tile += workers as usize;
    }
    tcgen05::fence(true);
    sync_ruda();
    if UNIT_POS < 32 { tcgen05::deallocate(address, bn as u32); }
}

fn batch_layout<R: Runtime>(tensor: &RudaTensor<R>, output_shape: &[usize], transposed: bool)
    -> Option<(usize, usize, Vec<usize>)> {
    let shape = tensor.meta.shape();
    let strides = tensor.meta.strides();
    let rank = shape.len();
    if rank < 2 || rank - 2 > output_shape.len() { return None; }
    let row_axis = rank - 2 + usize::from(transposed);
    let k_axis = rank - 1 - usize::from(transposed);
    let (rows, columns) = (shape[row_axis], shape[k_axis]);
    let row_stride = strides[row_axis];
    if rows == 0 || columns == 0 || strides[k_axis] != 1 || row_stride < columns || row_stride % 8 != 0
        || rows > i32::MAX as usize || columns > i32::MAX as usize
        || (row_stride as u64).checked_mul(2)? >= (1u64 << 40)
        || tensor.handle.offset_start.unwrap_or(0) % 16 != 0 { return None; }
    let mut batch_strides = vec![0; output_shape.len()];
    let padding = output_shape.len() - (rank - 2);
    for axis in 0..rank - 2 {
        if shape[axis] != 1 && shape[axis] != output_shape[padding + axis] { return None; }
        if shape[axis] != 1 { batch_strides[padding + axis] = strides[axis]; }
    }
    let mut unit = 0usize;
    for &stride in &batch_strides {
        let mut remainder = stride;
        while remainder != 0 { let next = unit % remainder; unit = remainder; remainder = next; }
    }
    let matrix_span = (rows - 1).checked_mul(row_stride)?.checked_add(columns)?;
    let matrix_pitch = rows.checked_mul(row_stride)?;
    if unit == 0 { unit = matrix_pitch; }
    if unit < matrix_pitch || unit % 8 != 0 || (unit as u64).checked_mul(2)? >= (1u64 << 40) { return None; }
    let span = output_shape.iter().zip(&batch_strides).try_fold(0usize, |span, (&size, &stride)|
        span.checked_add(size.checked_sub(1)?.checked_mul(stride)?))?;
    if span.checked_add(matrix_span)? > u32::MAX as usize { return None; }
    let planes = span / unit + 1;
    if planes > i32::MAX as usize { return None; }
    for stride in &mut batch_strides { *stride /= unit; }
    Some((planes, unit, batch_strides))
}

pub(super) fn prepare_batched<R: Runtime>(lhs: &RudaTensor<R>, rhs: &RudaTensor<R>, out: &RudaTensor<R>,
    epilogue: Option<(RudaTensor<R>, f32, f32)>, allow_split: bool) -> Option<Vec<PreparedKernel<R>>> {
    let client = &out.client;
    let h = &client.properties().hardware;
    let output_rank = out.meta.num_dims();
    if !client.features().matmul.tcgen05 || h.plane_size_min != 32 || h.plane_size_max != 32
        || h.max_units_per_ruda < 128 || h.max_ruda_dim.0 < 128 || output_rank < 2 { return None; }
    let batch_shape = &out.meta.shape()[..output_rank - 2];
    let (m, n) = (out.meta.shape()[output_rank - 2], out.meta.shape()[output_rank - 1]);
    let k = *lhs.meta.shape().last()?;
    if m < super::gemm_policy::BLACKWELL_MIN_M || n < 64 || k == 0 || k % 64 != 0
        || epilogue.is_some() && !batch_shape.is_empty() { return None; }
    batch_layout(lhs, batch_shape, false)?;
    batch_layout(rhs, batch_shape, true)?;
    let batches = batch_shape.iter().try_fold(1usize, |value, &size| value.checked_mul(size))?;
    if batches == 0 || batches > h.max_ruda_count.1 as usize { return None; }
    let mut plan = super::gemm_policy::plan(m, n, k, batches,
        h.num_streaming_multiprocessors.unwrap_or(1) as usize, true, allow_split);
    if !plan.blackwell || plan.shared_bytes() > h.max_shared_memory_size as usize { return None; }
    if out.meta.num_elements().checked_mul(plan.partitions).is_none_or(|size| size > u32::MAX as usize)
        || batches.checked_mul(plan.partitions)? > h.max_ruda_count.2 as usize
        || out.meta.num_elements().div_ceil(256) > h.max_ruda_count.0 as usize { plan.partitions = 1; }
    if out.meta.num_elements() > u32::MAX as usize { return None; }
    let expand = |source: &RudaTensor<R>| {
        let mut tensor = source.clone();
        let rank = source.meta.num_dims();
        let mut shape = batch_shape.to_vec();
        let mut strides = vec![0usize; batch_shape.len()];
        let padding = batch_shape.len() - (rank - 2);
        for axis in 0..rank - 2 {
            if source.meta.shape()[axis] != 1 { strides[padding + axis] = source.meta.strides()[axis]; }
        }
        shape.extend_from_slice(&source.meta.shape()[rank - 2..]);
        strides.extend_from_slice(&source.meta.strides()[rank - 2..]);
        *tensor.meta = Metadata::new(shape, strides);
        tensor
    };
    let lhs = expand(lhs);
    let rhs = expand(rhs);
    Some(if plan.partitions > 1 {
        super::mma_split_k::prepare_plan(lhs, rhs, out.clone(), epilogue, plan)
    } else { vec![prepare(lhs, rhs, out.clone(), epilogue, plan)] })
}

pub(super) fn prepare<R: Runtime>(mut lhs: RudaTensor<R>, mut rhs: RudaTensor<R>, out: RudaTensor<R>,
    epilogue: Option<(RudaTensor<R>, f32, f32)>, plan: super::gemm_policy::GemmPlan) -> PreparedKernel<R> {
    let left_rank = lhs.meta.num_dims();
    let right_rank = rhs.meta.num_dims();
    let m = lhs.meta.shape()[left_rank - 2];
    let k = lhs.meta.shape()[left_rank - 1];
    let n = rhs.meta.shape()[right_rank - 1];
    let batch_rank = left_rank.max(right_rank) - 2;
    let mut batch_shape = vec![1usize; batch_rank];
    for tensor in [&lhs, &rhs] {
        let rank = tensor.meta.num_dims() - 2;
        for axis in 0..rank { batch_shape[batch_rank - rank + axis] = batch_shape[batch_rank - rank + axis].max(tensor.meta.shape()[axis]); }
    }
    let (planes_a, stride_a, batch_a) = batch_layout(&lhs, &batch_shape, false).expect("validated SM100 lhs layout");
    let (planes_b, stride_b, batch_b) = batch_layout(&rhs, &batch_shape, true).expect("validated SM100 rhs layout");
    let batches = batch_shape.iter().product::<usize>();
    let left_stride = lhs.meta.strides()[left_rank - 2];
    let right_stride = rhs.meta.strides()[right_rank - 1];
    let mut tile_a = vec![128, 64];
    let mut tile_b = vec![plan.bn, 64];
    if batch_rank == 0 {
        *lhs.meta = Metadata::new([m, k], [left_stride, 1]);
        *rhs.meta = Metadata::new([n, k], [right_stride, 1]);
    } else {
        *lhs.meta = Metadata::new([planes_a, m, k], [stride_a, left_stride, 1]);
        *rhs.meta = Metadata::new([planes_b, n, k], [stride_b, right_stride, 1]);
        tile_a.insert(0, 1); tile_b.insert(0, 1);
    }
    let client = out.client.clone();
    let sms = client.properties().hardware.num_streaming_multiprocessors.unwrap_or(1) as usize;
    let workers = (m.div_ceil(128) * n.div_ceil(plan.bn)).min(sms.max(1));
    let input: StorageType = lhs.dtype.into();
    let output: StorageType = out.dtype.into();
    let bf16 = lhs.dtype == DType::BF16;
    let left = TensorMapArg::<R, Tiled>::new(TiledArgs {tile_size: tile_a.into()}, lhs.into_tensor_arg(), input)
        .with_swizzle(TensorMapSwizzle::B128);
    let right = TensorMapArg::<R, Tiled>::new(TiledArgs {tile_size: tile_b.into()}, rhs.into_tensor_arg(), input)
        .with_swizzle(TensorMapSwizzle::B128);
    let has_epilogue = epilogue.is_some();
    let (bias, alpha, beta) = epilogue.unwrap_or_else(|| (out.clone(), 1.0, 0.0));
    let (bias_row, bias_column) = (bias.meta.strides()[0], bias.meta.strides()[1]);
    blackwell_gemm::prepare::<R>(&client, RudaCount::Static(workers as u32, batches as u32, plan.partitions as u32),
        RudaDim::new_1d(128), left, right, out.into_array_arg(), bias.into_array_arg(),
        m as u32, n as u32, k as u32, (k / plan.partitions) as u32, workers as u32,
        alpha, beta, bias_row, bias_column, input, output, plan.bn, plan.stages, bf16,
        plan.partitions, has_epilogue, has_epilogue && beta != 0.0, batch_shape, batch_a, batch_b)
}
