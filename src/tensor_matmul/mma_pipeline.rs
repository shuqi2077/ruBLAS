use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::{dsl::prelude::*, tensor::RudaTensor};

#[ruda]
fn copy_tile<I: Float>(source: &Slice<I>, destination: &mut SliceMut<I>, base: usize,
    first_row: usize, stride: usize, shared_base: u32, limit: u32, tile: usize, #[comptime] rows: usize, #[comptime] units: usize,
    #[comptime] step_k: usize, #[comptime] tails: bool, #[comptime] partition: usize, #[comptime] partitions: usize) {
    #[unroll]
    for copy in 0usize..(rows * step_k / (8 * units)) {
        if comptime!(copy % partitions == partition) {
        let shared = (shared_base + (copy * units * 8) as u32) as usize;
        let offset = base + copy * stride + tile * step_k;
        if comptime!(tails) {
            let mut start=0usize;let mut length=0usize;
            if first_row+copy*(units*8/step_k)<limit as usize {start=offset;length=8;}
            barrier::copy_async_checked(&source.slice(start,start+length),&mut destination.slice_mut(shared,shared+8),8u32);
        } else {
            barrier::copy_async(&source.slice(offset,offset+8),&mut destination.slice_mut(shared,shared+8),8u32);
        }
        }
    }
}

#[ruda]
fn load_warp_tiles<I: Float>(a: &Slice<I>, b: &Slice<I>,
    left: &mut Sequence<Array<Vector<I, Const<2>>>>, right: &mut Sequence<Array<Vector<I, Const<2>>>>,
    a_base: u32, b_base: u32, #[comptime] extent_m: usize, #[comptime] extent_n: usize,
    #[comptime] warp_rows: usize, #[comptime] warp_cols: usize, #[comptime] step_k: usize,
    stage: usize, #[comptime] inner: usize, #[comptime] slot: usize,
) {
    let def = cmma::MmaDefinition::<I, I, f32>::new(16usize, 8usize, 16usize);
    #[unroll]
    for mi in 0usize..(extent_m / (16 * warp_rows)) {
        let at = ((stage * extent_m * step_k) as u32 + (a_base ^ (mi * 16 * step_k + inner * 16) as u32)) as usize;
        let loaded = def.load_matrix::<I, Const<2>>(&a.slice(at, at + 8), cmma::MatrixIdent::A, 4usize, false);
        let target = left.index_mut(slot * (extent_m / (16 * warp_rows)) + mi);
        #[unroll]
        for register in 0usize..4 { target[register] = loaded[register]; }
    }
    #[unroll]
    for pair in 0usize..(extent_n / (16 * warp_cols)) {
        let at = ((stage * extent_n * step_k) as u32 + (b_base ^ (pair * 16 * step_k + inner * 16) as u32)) as usize;
        let loaded = def.load_matrix::<I, Const<2>>(&b.slice(at, at + 8), cmma::MatrixIdent::B, 4usize, false);
        let first = right.index_mut(slot * (extent_n / (8 * warp_cols)) + pair * 2);
        first[0] = loaded[0];first[1] = loaded[1];
        let second = right.index_mut(slot * (extent_n / (8 * warp_cols)) + pair * 2 + 1);
        second[0] = loaded[2];second[1] = loaded[3];
    }
}

#[ruda(launch)]
fn mma_pipeline<I: Float, O: Float>(
    lhs: &Array<I>, rhs: &Array<I>, out: &mut Array<O>, bias: &Array<O>,
    m: u32, n: u32, full_k: u32, partition_k: u32, alpha: f32, beta: f32, bias_row_stride: usize, bias_column_stride: usize,
    #[define(I)] _input: StorageType, #[define(O)] _output: StorageType,
    #[comptime] warp_rows: usize, #[comptime] warp_cols: usize, #[comptime] extent_m: usize, #[comptime] extent_n: usize,
    #[comptime] tails: bool, #[comptime] stages: usize, #[comptime] preload: usize, #[comptime] step_k: usize,
    #[comptime] copy_groups: bool,
    #[comptime] grouped_tiles: bool,
    #[comptime] interleaved: bool,
    #[comptime] epilogue: bool, #[comptime] use_bias: bool, #[comptime] partitions: usize,
) {
    let lane = UNIT_POS_X as usize;
    let warp = UNIT_POS_Y as usize;
    let output_batch = RUDA_POS_Z as usize;
    let batch = output_batch / partitions;
    let mut reduction_start = 0usize;
    let mut k = full_k;
    if comptime!(partitions > 1) {
        reduction_start = (output_batch % partitions) * partition_k as usize;
        k = partition_k;
        if reduction_start + k as usize > full_k as usize { k = full_k - reduction_start as u32; }
    }
    let mut tile_m=RUDA_POS_X as usize;let mut tile_n=RUDA_POS_Y as usize;
    if comptime!(grouped_tiles) {
        let count_m=(m as usize+extent_m-1)/extent_m;
        let count_n=(n as usize+extent_n-1)/extent_n;
        let linear=tile_m+tile_n*count_m;
        let group_n=linear/(count_m*8)*8;
        let mut width=8usize;if count_n-group_n<width {width=count_n-group_n;}
        let within=linear%(count_m*8);
        let group_m=within/(4*width)*4;
        let mut height=4usize;if count_m-group_m<height {height=count_m-group_m;}
        let local=within%(4*width);
        tile_m=group_m+local%height;tile_n=group_n+local/height;
    }
    let row = tile_m * extent_m;
    let column = tile_n * extent_n;
    let left = lhs.slice(batch * m as usize * full_k as usize + reduction_start, (batch + 1) * m as usize * full_k as usize);
    let right = rhs.slice(batch * n as usize * full_k as usize + reduction_start, (batch + 1) * n as usize * full_k as usize);
    let copy_row = UNIT_POS as usize / (step_k / 8);
    let copy_column = UNIT_POS as usize % (step_k / 8) * 8;
    let copy_shared = (copy_row * step_k + (copy_column ^ ((copy_row / (64 / step_k) % (step_k / 8)) * 8))) as u32;
    let left_base = (row + copy_row) * full_k as usize + copy_column;
    let right_base = (column + copy_row) * full_k as usize + copy_column;
    let copy_stride = (32 * warp_rows * warp_cols * 8 / step_k) * full_k as usize;
    let a_row = (warp / warp_cols * (extent_m / warp_rows) + lane % 16) as u32;
    let a_column = (lane / 16 * 8) as u32;
    let a_base = a_row * step_k as u32 + (a_column ^ ((a_row / (64 / step_k) as u32 % (step_k / 8) as u32) * 8));
    let mut b_row = (warp % warp_cols * (extent_n / warp_cols) + lane % comptime!(if preload >= 2 { 16usize } else { 8usize })) as u32;
    let mut b_column = (lane / comptime!(if preload >= 2 { 16usize } else { 8usize }) % 2 * 8) as u32;
    if comptime!(copy_groups && step_k == 64 && preload >= 2 && stages >= 3) {
        b_row = (warp % warp_cols * (extent_n / warp_cols) + lane % 8 + lane / 16 * 8) as u32;
        b_column = (lane / 8 % 2 * 8) as u32;
    }
    let b_base = b_row * step_k as u32 + (b_column ^ ((b_row / (64 / step_k) as u32 % (step_k / 8) as u32) * 8));
    let mut a = SharedMemory::<I>::new_aligned(extent_m * step_k * stages, 128usize);
    let mut b = SharedMemory::<I>::new_aligned(extent_n * step_k * stages, 128usize);
    let mut ready = Sequence::<Shared<barrier::Barrier>>::new();
    #[unroll]
    for _ in 0usize..comptime!(if copy_groups { 0 } else { stages }) {
        ready.push(barrier::Barrier::shared((32 * warp_rows * warp_cols) as u32, UNIT_POS == 0));
    }
    let def = cmma::MmaDefinition::<I, I, f32>::new(16usize, 8usize, 16usize);
    let mut accumulators = Sequence::<Array<Vector<f32, Const<2>>>>::new();
    #[unroll]
    for _ in 0usize..(extent_m * extent_n / (128 * warp_rows * warp_cols)) {
        let mut acc = Array::<Vector<f32, Const<2>>>::new(2usize);
        acc[0] = Vector::new(0.0f32);
        acc[1] = Vector::new(0.0f32);
        accumulators.push(acc);
    }
    #[unroll]
    for stage in 0usize..comptime!(if copy_groups { stages - 1 } else { stages }) {
        if stage * step_k < k as usize {
            copy_tile(&left,&mut a.slice_mut(stage*extent_m*step_k,(stage+1)*extent_m*step_k),left_base,row+copy_row,copy_stride,copy_shared,m,stage,extent_m,32*warp_rows*warp_cols,step_k,tails,0usize,1usize);
            copy_tile(&right,&mut b.slice_mut(stage*extent_n*step_k,(stage+1)*extent_n*step_k),right_base,column+copy_row,copy_stride,copy_shared,n,stage,extent_n,32*warp_rows*warp_cols,step_k,tails,0usize,1usize);
            if !comptime!(copy_groups) { ready[stage].commit_copy_async(); }
        }
        if comptime!(copy_groups) { barrier::commit_copy_group(); }
    }
    if comptime!(copy_groups && step_k == 64 && preload >= 2 && stages >= 3) {
        let mut left_registers = Sequence::<Array<Vector<I, Const<2>>>>::new();
        let mut right_registers = Sequence::<Array<Vector<I, Const<2>>>>::new();
        #[unroll]
        for _ in 0usize..(2 * extent_m / (16 * warp_rows)) {
            left_registers.push(Array::<Vector<I, Const<2>>>::new(4usize));
        }
        #[unroll]
        for _ in 0usize..(2 * extent_n / (8 * warp_cols)) {
            right_registers.push(Array::<Vector<I, Const<2>>>::new(2usize));
        }
        barrier::wait_copy_group((stages - 2) as u32);
        sync_ruda();
        if k > 0 {
            load_warp_tiles(&a.slice(0usize,a.len()),&b.slice(0usize,b.len()),&mut left_registers,&mut right_registers,
                a_base,b_base,extent_m,extent_n,warp_rows,warp_cols,step_k,0usize,0usize,0usize);
        }
        let mut pipeline_stage = 0usize;
        let mut pipeline_write_stage = 0usize;
        pipeline_write_stage = stages - 1;
        for tile in 0usize..(k as usize / step_k) {
                    let mut pipeline_next_stage = pipeline_stage + 1;
                    if pipeline_next_stage == stages { pipeline_next_stage = 0; }
                    #[unroll]
                    for inner in 0usize..(step_k / 16) {
                        if comptime!(inner + 1 < step_k / 16) {
                            load_warp_tiles(&a.slice(0usize,a.len()),&b.slice(0usize,b.len()),&mut left_registers,&mut right_registers,
                                a_base,b_base,extent_m,extent_n,warp_rows,warp_cols,step_k,pipeline_stage,inner+1,(inner+1)%2);
                        } else if (tile + 1) * step_k < k as usize {
                            load_warp_tiles(&a.slice(0usize,a.len()),&b.slice(0usize,b.len()),&mut left_registers,&mut right_registers,
                                a_base,b_base,extent_m,extent_n,warp_rows,warp_cols,step_k,pipeline_next_stage,0usize,0usize);
                        }
                        #[unroll]
                        for ni in 0usize..(extent_n / (8 * warp_cols)) {
                            #[unroll]
                            for mi in 0usize..(extent_m / (16 * warp_rows)) {
                                def.execute_inplace(
                                    &left_registers[(inner%2)*(extent_m/(16*warp_rows))+mi],
                                    &right_registers[(inner%2)*(extent_n/(8*warp_cols))+ni],
                                    accumulators.index_mut(mi*(extent_n/(8*warp_cols))+ni));
                            }
                        }
                        if comptime!(inner + 1 < step_k / 16) {
                            if (tile + stages - 1) * step_k < k as usize {
                                #[unroll]
                                for group in 0usize..comptime!(if inner+2==step_k/16 {2usize} else {1usize}) {
                                    copy_tile(&left,&mut a.slice_mut(pipeline_write_stage*extent_m*step_k,(pipeline_write_stage+1)*extent_m*step_k),left_base,row+copy_row,copy_stride,copy_shared,m,tile+stages-1,extent_m,32*warp_rows*warp_cols,step_k,tails,inner+group,step_k/16);
                                    copy_tile(&right,&mut b.slice_mut(pipeline_write_stage*extent_n*step_k,(pipeline_write_stage+1)*extent_n*step_k),right_base,column+copy_row,copy_stride,copy_shared,n,tile+stages-1,extent_n,32*warp_rows*warp_cols,step_k,tails,inner+group,step_k/16);
                                }
                            }
                        }
                        if comptime!(inner + 2 == step_k / 16) {
                            barrier::commit_copy_group();
                            if (tile + 1) * step_k < k as usize {
                                barrier::wait_copy_group((stages - 2) as u32);
                                sync_ruda();
                            }
                        }
                    }
                    pipeline_stage = pipeline_next_stage;
                    pipeline_write_stage += 1;
                    if pipeline_write_stage == stages { pipeline_write_stage = 0; }
        }
    } else {
    for cycle in 0usize..((k as usize / step_k + stages - 1) / stages) {
        #[unroll]
        for stage in 0usize..stages {
            let tile = cycle * stages + stage;
            if tile * step_k < k as usize {
                if comptime!(copy_groups) {
                    barrier::wait_copy_group((stages - 2) as u32);
                    sync_ruda();
                    if !comptime!(interleaved) {
                        let next=comptime!((stage+stages-1)%stages);
                        if (tile+stages-1)*step_k<k as usize {
                            copy_tile(&left,&mut a.slice_mut(next*extent_m*step_k,(next+1)*extent_m*step_k),left_base,row+copy_row,copy_stride,copy_shared,m,tile+stages-1,extent_m,32*warp_rows*warp_cols,step_k,tails,0usize,1usize);
                            copy_tile(&right,&mut b.slice_mut(next*extent_n*step_k,(next+1)*extent_n*step_k),right_base,column+copy_row,copy_stride,copy_shared,n,tile+stages-1,extent_n,32*warp_rows*warp_cols,step_k,tails,0usize,1usize);
                        }
                        barrier::commit_copy_group();
                    }
                } else { ready[stage].arrive_and_wait(); }
                let current_m = stage * extent_m * step_k;
                let current_n = stage * extent_n * step_k;
                #[unroll]
                for inner in 0usize..(step_k / 16) {
                    if comptime!(copy_groups && interleaved) {
                        let next=comptime!((stage+stages-1)%stages);
                        if (tile+stages-1)*step_k<k as usize {
                            copy_tile(&left,&mut a.slice_mut(next*extent_m*step_k,(next+1)*extent_m*step_k),left_base,row+copy_row,copy_stride,copy_shared,m,tile+stages-1,extent_m,32*warp_rows*warp_cols,step_k,tails,inner,step_k/16);
                            copy_tile(&right,&mut b.slice_mut(next*extent_n*step_k,(next+1)*extent_n*step_k),right_base,column+copy_row,copy_stride,copy_shared,n,tile+stages-1,extent_n,32*warp_rows*warp_cols,step_k,tails,inner,step_k/16);
                        }
                    }
                    let mut fragments = Sequence::<Array<Vector<I, Const<2>>>>::new();
                    #[unroll]
                    for mi in 0usize..(extent_m / (16 * warp_rows)) {
                        let at = ((stage * extent_m * step_k) as u32 + (a_base ^ (mi * 16 * step_k + inner * 16) as u32)) as usize;
                        fragments.push(def.load_matrix::<I, Const<2>>(&a.slice(at, at + 8), cmma::MatrixIdent::A, 4usize, false));
                    }
                    #[unroll]
                    for pair in 0usize..(extent_n / (8 * warp_cols * preload)) {
                        let mut right_fragments = Sequence::<Array<Vector<I, Const<2>>>>::new();
                        if comptime!(preload >= 2) {
                            #[unroll]
                            for half in 0usize..(preload / 2) {
                                let at = ((stage * extent_n * step_k) as u32 + (b_base ^ ((pair * preload + half * 2) * 8 * step_k + inner * 16) as u32)) as usize;
                                let loaded = def.load_matrix::<I, Const<2>>(&b.slice(at, at + 8), cmma::MatrixIdent::B, 4usize, false);
                                let mut first = Array::<Vector<I, Const<2>>>::new(2usize);
                                let mut second = Array::<Vector<I, Const<2>>>::new(2usize);
                                first[0] = loaded[0];first[1] = loaded[2];
                                second[0] = loaded[1];second[1] = loaded[3];
                                right_fragments.push(first);right_fragments.push(second);
                            }
                        } else {
                            let at = ((stage * extent_n * step_k) as u32 + (b_base ^ (pair * 8 * step_k + inner * 16) as u32)) as usize;
                            right_fragments.push(def.load_matrix::<I, Const<2>>(&b.slice(at, at + 8), cmma::MatrixIdent::B, 2usize, false));
                        }
                        #[unroll]
                        for half in 0usize..preload {
                            #[unroll]
                            for mi in 0usize..(extent_m / (16 * warp_rows)) {
                                def.execute_inplace(&fragments[mi], &right_fragments[half], accumulators.index_mut(mi * (extent_n / (8 * warp_cols)) + pair * preload + half));
                            }
                        }
                    }
                }
                if comptime!(copy_groups) {
                    if comptime!(interleaved) { barrier::commit_copy_group(); }
                }
                else {
                    sync_ruda();
                    if (tile + stages) * step_k < k as usize {
                        copy_tile(&left,&mut a.slice_mut(current_m,current_m+extent_m*step_k),left_base,row+copy_row,copy_stride,copy_shared,m,tile+stages,extent_m,32*warp_rows*warp_cols,step_k,tails,0usize,1usize);
                        copy_tile(&right,&mut b.slice_mut(current_n,current_n+extent_n*step_k),right_base,column+copy_row,copy_stride,copy_shared,n,tile+stages,extent_n,32*warp_rows*warp_cols,step_k,tails,0usize,1usize);
                        ready[stage].commit_copy_async();
                    }
                }
            }
        }
    }
    }
    if comptime!(copy_groups) { barrier::wait_copy_group(0u32); }
    #[unroll]
    for mi in 0usize..(extent_m / (16 * warp_rows)) {
        #[unroll]
        for ni in 0usize..(extent_n / (8 * warp_cols)) {
            #[unroll]
            for half in 0usize..2 {
                let r = row + warp / warp_cols * (extent_m / warp_rows) + mi * 16 + lane / 4 + half * 8;
                let c = column + warp % warp_cols * (extent_n / warp_cols) + ni * 8 + lane % 4 * 2;
                let mut pair = accumulators[mi * (extent_n / (8 * warp_cols)) + ni][half];
                if comptime!(epilogue) {
                    #[unroll]
                    for element in 0usize..2 {
                        let mut value=alpha*pair[element];
                        if comptime!(use_bias) {
                            if !comptime!(tails) || (r<m as usize && c+element<n as usize) {
                                value+=beta*f32::cast_from(bias[r*bias_row_stride+(c+element)*bias_column_stride]);
                            }
                        }
                        pair[element]=value;
                    }
                }
                if !comptime!(tails) {
                    let mut packed=out.slice_mut(0usize,out.len()).into_vectorized().with_vector_size::<Const<2>>();
                    packed[((output_batch*m as usize+r)*n as usize+c)/2]=Vector::cast_from(pair);
                } else {
                    if r < m as usize && c < n as usize {
                        out[(output_batch * m as usize + r) * n as usize + c] = O::cast_from(pair[0]);
                    }
                    if r < m as usize && c + 1 < n as usize {
                        out[(output_batch * m as usize + r) * n as usize + c + 1] = O::cast_from(pair[1]);
                    }
                }
            }
        }
    }
}

pub(super) fn prepare<R: Runtime>(lhs: RudaTensor<R>, rhs: RudaTensor<R>, out: RudaTensor<R>) -> PreparedKernel<R> {
    prepare_with_epilogue(lhs,rhs,out,None)
}

pub(super) fn prepare_addmm<R: Runtime>(lhs:RudaTensor<R>,rhs:RudaTensor<R>,out:RudaTensor<R>,
    bias:RudaTensor<R>,alpha:f32,beta:f32)->PreparedKernel<R> {
    prepare_with_epilogue(lhs,rhs,out,Some((bias,alpha,beta)))
}

fn prepare_with_epilogue<R: Runtime>(lhs:RudaTensor<R>,rhs:RudaTensor<R>,out:RudaTensor<R>,
    epilogue:Option<(RudaTensor<R>,f32,f32)>)->PreparedKernel<R> {
    let rank = lhs.meta.num_dims();
    let m = lhs.meta.shape()[rank - 2];
    let n = out.meta.shape()[rank - 1];
    let k = lhs.meta.shape()[rank - 1];
    let hardware = &lhs.client.properties().hardware;
    let grouped_tiles = m >= 512 && n >= 512 && k >= 1024;
    if k % 64 == 0 && hardware.max_shared_memory_size >= 32784 {
        if m >= 128 && n >= 2 * k && hardware.max_shared_memory_size >= 65552
            && hardware.max_units_per_ruda >= 256 && hardware.max_ruda_dim.1 >= 8 {
            let stages = if m >= 512 && m % 128 == 0 && n % 128 == 0 && k >= 2048 && hardware.max_shared_memory_size >= 98328 { 3 } else { 2 };
            return prepare_configured_epilogue(lhs, rhs, out, 128, 128, 2, 4, stages, 4, 64, true, grouped_tiles,epilogue,1,0);
        }
        let rectangular = m >= 128 && hardware.max_shared_memory_size >= 73752;
        let (extent_m, stages) = if rectangular { (128, 3) } else { (64, 2) };
        return prepare_configured_epilogue(lhs, rhs, out, extent_m, 64, 2, 2, stages, 4, 64, true, grouped_tiles,epilogue,1,0);
    }
    let extent = if m % 128 == 0 && n % 128 == 0 { 128 } else { 64 };
    let warp_rows = if extent == 128 && hardware.max_units_per_ruda >= 256 && hardware.max_ruda_dim.1 >= 8 { 4 } else { 2 };
    let stages = if hardware.max_shared_memory_size as usize >= extent * 128 * 3 + 24 { 3 } else { 2 };
    prepare_configured_epilogue(lhs, rhs, out, extent, extent, warp_rows, 2, stages, 2, 32, true, grouped_tiles,epilogue,1,0)
}

pub(super) fn prepare_configured<R: Runtime>(lhs: RudaTensor<R>, rhs: RudaTensor<R>, out: RudaTensor<R>,
    extent_m: usize, extent_n: usize, warp_rows: usize, warp_cols: usize, stages: usize, preload: usize, step_k: usize, copy_groups: bool, grouped_tiles: bool) -> PreparedKernel<R> {
    prepare_configured_epilogue(lhs,rhs,out,extent_m,extent_n,warp_rows,warp_cols,stages,preload,step_k,copy_groups,grouped_tiles,None,1,0)
}

pub(super) fn prepare_split_partials<R: Runtime>(lhs:RudaTensor<R>,rhs:RudaTensor<R>,out:RudaTensor<R>,
    extent_m:usize,extent_n:usize,warp_rows:usize,warp_cols:usize,stages:usize,partitions:usize,partition_k:usize)->PreparedKernel<R> {
    assert!(out.dtype==ruda_core::tensor::DType::F32 && partitions>1 && partition_k%64==0);
    prepare_configured_epilogue(lhs,rhs,out,extent_m,extent_n,warp_rows,warp_cols,stages,4,64,true,true,None,partitions,partition_k)
}

fn prepare_configured_epilogue<R: Runtime>(lhs: RudaTensor<R>, rhs: RudaTensor<R>, out: RudaTensor<R>,
    extent_m: usize, extent_n: usize, warp_rows: usize, warp_cols: usize, stages: usize, preload: usize, step_k: usize, copy_groups: bool, grouped_tiles: bool,
    epilogue:Option<(RudaTensor<R>,f32,f32)>,partitions:usize,partition_k:usize) -> PreparedKernel<R> {
    let rank = lhs.meta.num_dims();
    let m = lhs.meta.shape()[rank - 2];
    let k = lhs.meta.shape()[rank - 1];
    let n = *out.meta.shape().last().unwrap();
    let batches = out.meta.num_elements() / (m * n);
    let client = lhs.client.clone();
    let input = lhs.dtype.into();
    let output = out.dtype.into();
    let hardware = &client.properties().hardware;
    assert!(matches!(extent_m,64|128|256) && matches!(extent_n,64|128|256) && matches!(warp_rows,2|4) && matches!(warp_cols,2|4) && warp_rows*warp_cols<=8 && matches!(stages,2..=4));
    assert!(matches!(preload,1|2|4|8|16) && extent_n / (8 * warp_cols) % preload == 0);
    assert!(matches!(step_k,32|64) && k % step_k == 0);
    assert!(hardware.max_shared_memory_size as usize >= (extent_m+extent_n) * step_k * 2 * stages + stages * 8);
    assert!(hardware.max_units_per_ruda as usize >= warp_rows * warp_cols * 32 && hardware.max_ruda_dim.1 as usize >= warp_rows * warp_cols);
    let has_epilogue=epilogue.is_some();
    let (bias,alpha,beta)=epilogue.unwrap_or_else(||(out.clone(),1.0,0.0));
    let (bias_row_stride,bias_column_stride)=(bias.meta.strides()[0],bias.meta.strides()[1]);
    mma_pipeline::prepare::<R>(&client, RudaCount::Static(m.div_ceil(extent_m) as u32, n.div_ceil(extent_n) as u32, batches as u32),
        RudaDim::new_2d(32, (warp_cols * warp_rows) as u32), lhs.into_array_arg(), rhs.into_array_arg(), out.into_array_arg(),bias.into_array_arg(),
        m as u32, n as u32, k as u32,partition_k as u32,alpha,beta,bias_row_stride,bias_column_stride, input, output, warp_rows, warp_cols, extent_m, extent_n, m % extent_m != 0 || n % extent_n != 0, stages, preload, step_k, copy_groups, grouped_tiles, step_k == 32 && warp_rows * warp_cols == 8,has_epilogue,has_epilogue && beta!=0.0,partitions)
}
