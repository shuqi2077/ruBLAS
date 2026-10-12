use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::*;

#[ruda]
fn tile<G: Float, W: Float>(
    input: &Array<G>, qweight: &Array<i32>, qzeros: &Array<i32>,
    scales: &Array<W>, bias: &Array<W>, output: &mut Array<G>,
    k: usize, n: usize, group_size: usize, has_bias: u32,
    row: usize, column: usize, expert: usize,
    #[comptime] columns: u32,
) {
    let packed_columns = n / 8;
    let groups = k / group_size;
    let packed_column = column / 8;
    let weight_base = expert * k * packed_columns;
    let zero_base = expert * groups * packed_columns;
    let scale_base = expert * groups * n;
    let mut sums = Array::<f32>::new(columns as usize);
    let mut group_scales = Array::<f32>::new(columns as usize);
    let mut group_zeros = Array::<f32>::new(columns as usize);
    #[unroll]
    for offset in 0u32..columns {
        sums[offset as usize] = 0.0f32;
    }
    let mut group = 0usize;
    let mut inner = 0usize;
    while group < groups {
        let zero_word = u32::cast_from(qzeros[zero_base + group * packed_columns + packed_column]);
        #[unroll]
        for offset in 0u32..columns {
            let channel = column + offset as usize;
            let lane = channel % 8;
            let shift = ((lane % 2) * 4 + lane / 2) as u32 * 4;
            group_zeros[offset as usize] = f32::cast_from((zero_word >> shift) & 15u32);
            group_scales[offset as usize] = f32::cast_from(scales[scale_base + group * n + channel]);
        }
        let end = inner + group_size;
        while inner < end {
            let word = u32::cast_from(qweight[weight_base + inner * packed_columns + packed_column]);
            let activation = f32::cast_from(input[row * k + inner]);
            #[unroll]
            for offset in 0u32..columns {
                let lane = (column + offset as usize) % 8;
                let shift = ((lane % 2) * 4 + lane / 2) as u32 * 4;
                let quant = f32::cast_from((word >> shift) & 15u32);
                let weight = W::cast_from((quant - group_zeros[offset as usize]) * group_scales[offset as usize]);
                sums[offset as usize] += activation * f32::cast_from(weight);
            }
            inner += 1;
        }
        group += 1;
    }
    #[unroll]
    for offset in 0u32..columns {
        let channel = column + offset as usize;
        let mut value = G::cast_from(sums[offset as usize]);
        if has_bias != 0 {
            value += G::cast_from(bias[expert * n + channel]);
        }
        output[row * n + channel] = value;
    }
}

#[ruda(launch)]
pub(super) fn forward<G: Float, W: Float>(
    input: &Array<G>, qweight: &Array<i32>, qzeros: &Array<i32>,
    scales: &Array<W>, bias: &Array<W>, output: &mut Array<G>,
    input_features: u32, output_features: u32, group_size: u32, has_bias: u32,
    #[comptime] columns: u32,
    #[define(G)] _activation: StorageType, #[define(W)] _weight: StorageType,
) {
    let position = ABSOLUTE_POS as usize;
    if position >= output.len() / columns as usize { terminate!(); }
    let n = output_features as usize;
    let row = position / (n / columns as usize);
    let column = (position % (n / columns as usize)) * columns as usize;
    tile::<G, W>(input, qweight, qzeros, scales, bias, output,
        input_features as usize, n, group_size as usize, has_bias, row, column, 0, columns);
}

#[ruda(launch)]
pub(super) fn grouped_forward<G: Float, W: Float>(
    input: &Array<G>, qweight: &Array<i32>, qzeros: &Array<i32>,
    scales: &Array<W>, bias: &Array<W>, ids: &Array<u32>, output: &mut Array<G>,
    experts: u32, input_features: u32, output_features: u32, group_size: u32, has_bias: u32,
    #[comptime] columns: u32,
    #[define(G)] _activation: StorageType, #[define(W)] _weight: StorageType,
) {
    let position = ABSOLUTE_POS as usize;
    if position >= output.len() / columns as usize { terminate!(); }
    let n = output_features as usize;
    let row = position / (n / columns as usize);
    let column = (position % (n / columns as usize)) * columns as usize;
    let expert = ids[row] as usize;
    if expert >= experts as usize {
        #[unroll]
        for offset in 0u32..columns { output[row * n + column + offset as usize] = G::cast_from(0.0f32); }
        terminate!();
    }
    tile::<G, W>(input, qweight, qzeros, scales, bias, output,
        input_features as usize, n, group_size as usize, has_bias, row, column, expert, columns);
}
