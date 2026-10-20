// GGML block definitions and codebooks: da8e2a54b919557b816b1c2beb432ccb766f7ce0.
// Copyright (c) 2023-2026 The ggml authors. MIT license: LICENSE-GGML.
use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::*;

#[ruda]
fn byte(p: &Tensor<u8>, i: usize) -> u32 { u32::cast_from(p[i]) }

#[ruda]
fn word(p: &Tensor<u8>, i: usize, #[comptime] big: bool) -> u32 {
    let mut value = byte(p, i) | (byte(p, i + 1) << 8);
    if comptime!(big) { value = (byte(p, i) << 8) | byte(p, i + 1); }
    value
}

#[ruda]
fn dword(p: &Tensor<u8>, i: usize, #[comptime] big: bool) -> u32 {
    let mut value = word(p, i, false) | (word(p, i + 2, false) << 16);
    if comptime!(big) { value = (word(p, i, true) << 16) | word(p, i + 2, true); }
    value
}

#[ruda]
fn signed(v: u32) -> f32 {
    let mut n = i32::cast_from(v);
    if v >= 128 { n -= 256; }
    f32::cast_from(n)
}

#[ruda]
fn half_value(bits: u32) -> f32 {
    let sign = (bits & 0x8000) << 16;
    let exponent = (bits >> 10) & 31;
    let mantissa = bits & 1023;
    let mut value = f32::from_bits(sign | ((exponent + 112) << 23) | (mantissa << 13));
    if exponent == 0 {
        value = f32::cast_from(mantissa) * 0.000000059604644775390625f32;
        if sign != 0 { value = -value; }
    }
    if exponent == 31 { value = f32::from_bits(sign | 0x7f800000 | (mantissa << 13)); }
    value
}

#[ruda]
fn h(p: &Tensor<u8>, i: usize, #[comptime] big: bool) -> f32 { half_value(word(p, i, big)) }

#[ruda]
fn parity_sign(code: u32, lane: usize) -> f32 {
    let mut parity = code;
    parity ^= parity >> 4;
    parity ^= parity >> 2;
    parity ^= parity >> 1;
    let mask = code | ((parity & 1) << 7);
    let mut result = 1.0f32;
    if ((mask >> u32::cast_from(lane)) & 1) != 0 { result = -1.0; }
    result
}

#[ruda]
fn sign_bit(mask: u32, lane: usize) -> f32 {
    let mut result = 1.0f32;
    if ((mask >> u32::cast_from(lane)) & 1) != 0 { result = -1.0; }
    result
}

#[ruda]
fn scale_k(p: &Tensor<u8>, base: usize, group: usize, #[comptime] minimum: bool) -> u32 {
    let mut value = 0u32;
    if group < 4 {
        if comptime!(minimum) { value = byte(p, base + group + 4) & 63; }
        else { value = byte(p, base + group) & 63; }
    } else {
        if comptime!(minimum) {
            value = (byte(p, base + group + 4) >> 4) | ((byte(p, base + group) >> 6) << 4);
        } else {
            value = (byte(p, base + group + 4) & 15) | ((byte(p, base + group - 4) >> 6) << 4);
        }
    }
    value
}

/// Decodes one original weight into a register, never a dense weight allocation.
#[ruda]
pub(super) fn weight(
    p: &Tensor<u8>, table: &Tensor<u8>, base: usize, j: usize,
    #[comptime] format: u32, #[comptime] big: bool,
) -> f32 {
    let mut value = 0.0f32;
    if comptime!(format == 0) { value = f32::from_bits(dword(p, base, big)); }
    if comptime!(format == 1) { value = h(p, base, big); }
    if comptime!(format == 30) { value = f32::from_bits(word(p, base, big) << 16); }
    if comptime!(format == 2 || format == 3 || format == 6 || format == 7) {
        let mut qs = base + 2;
        if comptime!(format == 3 || format == 7) { qs += 2; }
        if comptime!(format == 6 || format == 7) { qs += 4; }
        let mut q = (byte(p, qs + j % 16) >> u32::cast_from(4 * (j / 16))) & 15;
        if comptime!(format == 6 || format == 7) {
            q |= ((byte(p, qs - 4 + j / 8) >> u32::cast_from(j % 8)) & 1) << 4;
        }
        let mut qf = f32::cast_from(q);
        if comptime!(format == 2) { qf -= 8.0; }
        if comptime!(format == 6) { qf -= 16.0; }
        value = h(p, base, big) * qf;
        if comptime!(format == 3 || format == 7) { value += h(p, base + 2, big); }
    }
    if comptime!(format == 8 || format == 9) {
        let mut qs = base + 2;
        if comptime!(format == 9) { qs += 2; }
        value = h(p, base, big) * signed(byte(p, qs + j));
    }
    if comptime!(format == 10) {
        let scale = byte(p, base + j / 16);
        let q = (byte(p, base + 16 + j / 128 * 32 + j % 32) >> u32::cast_from(2 * (j % 128 / 32))) & 3;
        value = h(p, base + 80, big) * f32::cast_from(scale & 15) * f32::cast_from(q)
            - h(p, base + 82, big) * f32::cast_from(scale >> 4);
    }
    if comptime!(format == 11) {
        let group = j / 16;
        let low = (byte(p, base + 96 + group % 8) >> u32::cast_from(4 * (group / 8))) & 15;
        let high = (byte(p, base + 104 + group % 4) >> u32::cast_from(2 * (group / 4))) & 3;
        let scale = f32::cast_from(low | (high << 4)) - 32.0;
        let q = (byte(p, base + 32 + j / 128 * 32 + j % 32) >> u32::cast_from(2 * (j % 128 / 32))) & 3;
        let mut qf = f32::cast_from(q);
        if ((byte(p, base + j % 32) >> u32::cast_from(j / 32)) & 1) == 0 { qf -= 4.0; }
        value = h(p, base + 108, big) * scale * qf;
    }
    if comptime!(format == 12 || format == 13) {
        let group = j / 32;
        let mut qs = base + 16;
        if comptime!(format == 13) { qs += 32; }
        let mut q = (byte(p, qs + j / 64 * 32 + j % 32) >> u32::cast_from(4 * (group % 2))) & 15;
        if comptime!(format == 13) {
            q |= ((byte(p, base + 16 + j % 32) >> u32::cast_from(group)) & 1) << 4;
        }
        value = h(p, base, big) * f32::cast_from(scale_k(p, base + 4, group, false)) * f32::cast_from(q)
            - h(p, base + 2, big) * f32::cast_from(scale_k(p, base + 4, group, true));
    }
    if comptime!(format == 14) {
        let low = (byte(p, base + j / 128 * 64 + j % 64) >> u32::cast_from(4 * (j % 128 / 64))) & 15;
        let high = (byte(p, base + 128 + j / 128 * 32 + j % 32) >> u32::cast_from(2 * (j % 128 / 32))) & 3;
        value = h(p, base + 208, big) * signed(byte(p, base + 192 + j / 16))
            * (f32::cast_from(low | (high << 4)) - 32.0);
    }
    if comptime!(format == 15) {
        value = f32::from_bits(dword(p, base, big)) * signed(byte(p, base + 4 + j));
    }
    if comptime!(format == 16) {
        let group = j / 32;
        let lane = j % 32 / 8;
        let grid = (word(p, base + 2 + group * 8 + lane / 2 * 2, big) >> u32::cast_from(8 * (lane % 2))) & 255;
        let flags = word(p, base + 6 + group * 8, big) | (word(p, base + 8 + group * 8, big) << 16);
        value = h(p, base, big) * (0.5 + f32::cast_from(flags >> 28)) * 0.25
            * f32::cast_from(table[grid as usize * 8 + j % 8])
            * parity_sign((flags >> u32::cast_from(7 * lane)) & 127, j % 8);
    }
    if comptime!(format == 17) {
        let code = word(p, base + 2 + j / 8 * 2, big);
        let scale = (byte(p, base + 66 + j / 32) >> u32::cast_from(4 * (j % 32 / 16))) & 15;
        value = h(p, base, big) * (0.5 + f32::cast_from(scale)) * 0.25
            * f32::cast_from(table[(code & 511) as usize * 8 + j % 8])
            * parity_sign(code >> 9, j % 8);
    }
    if comptime!(format == 22) {
        let group = j / 32;
        let lane = j % 32 / 8;
        let grid = byte(p, base + 2 + j / 8)
            | (((byte(p, base + 66 + group) >> u32::cast_from(2 * lane)) & 3) << 8);
        let scale = (byte(p, base + 74 + group) >> u32::cast_from(4 * (j % 32 / 16))) & 15;
        value = h(p, base, big) * (0.5 + f32::cast_from(scale)) * 0.25
            * f32::cast_from(table[grid as usize * 8 + j % 8])
            * sign_bit(byte(p, base + 34 + j / 8), j % 8);
    }
    if comptime!(format == 18) {
        let flags = dword(p, base + 66 + j / 32 * 4, false);
        let grid = byte(p, base + 2 + j / 4);
        value = h(p, base, big) * (0.5 + f32::cast_from(flags >> 28)) * 0.5
            * f32::cast_from(table[grid as usize * 4 + j % 4])
            * parity_sign((flags >> u32::cast_from(7 * (j % 32 / 8))) & 127, j % 8);
    }
    if comptime!(format == 21) {
        let group = j / 32;
        let grid = byte(p, base + 2 + j / 4)
            | (((byte(p, base + 66 + group) >> u32::cast_from(j % 32 / 4)) & 1) << 8);
        let scale = (byte(p, base + 106 + group / 2) >> u32::cast_from(4 * (group % 2))) & 15;
        value = h(p, base, big) * f32::cast_from(1 + 2 * scale)
            * f32::cast_from(table[grid as usize * 4 + j % 4])
            * sign_bit(byte(p, base + 74 + j / 8), j % 8);
    }
    if comptime!(format == 19) {
        let high = word(p, base + 34 + j / 32 * 2, big);
        let grid = byte(p, base + 2 + j / 8) | (((high >> u32::cast_from(3 * (j % 32 / 8))) & 7) << 8);
        let delta = 0.125 * sign_bit(high, 15);
        value = h(p, base, big) * f32::cast_from(2 * ((high >> 12) & 7) + 1)
            * (signed(u32::cast_from(table[grid as usize * 8 + j % 8])) + delta);
    }
    if comptime!(format == 29) {
        let scale_bits = (word(p, base + 48, false) >> 12)
            | ((word(p, base + 50, false) >> 8) & 0x00f0)
            | ((word(p, base + 52, false) >> 4) & 0x0f00)
            | (word(p, base + 54, false) & 0xf000);
        let scale = (word(p, base + 48 + j / 64 * 2, false) >> u32::cast_from(3 * (j % 64 / 16))) & 7;
        let flags = byte(p, base + 32 + j / 16) >> u32::cast_from(4 * (j % 16 / 8));
        let grid = byte(p, base + j / 8) | ((flags & 7) << 8);
        value = half_value(scale_bits) * f32::cast_from(2 * scale + 1)
            * (signed(u32::cast_from(table[grid as usize * 8 + j % 8])) + 0.125 * sign_bit(flags, 3));
    }
    if comptime!(format == 20 || format == 23) {
        let mut qs = base + 2;
        let mut scale = h(p, base, big);
        if comptime!(format == 23) {
            let group = j / 32;
            qs = base + 8 + group * 16;
            let low = (byte(p, base + 4 + group / 2) >> u32::cast_from(4 * (group % 2))) & 15;
            let high = (word(p, base + 2, big) >> u32::cast_from(2 * group)) & 3;
            scale *= f32::cast_from(low | (high << 4)) - 32.0;
        }
        let code = (byte(p, qs + j % 16) >> u32::cast_from(4 * (j % 32 / 16))) & 15;
        value = scale * signed(u32::cast_from(table[code as usize]));
    }
    if comptime!(format == 34) {
        let mut offset = j % 32;
        let mut power = j / 32;
        if j >= 160 { offset = 32 + (j - 160) % 16; power = (j - 160) / 16; }
        if j >= 240 { offset = 48 + (j - 240) % 4; power = (j - 240) / 4; }
        let mut factor = 1u32;
        for _i in 0..power { factor *= 3; }
        let q = ((byte(p, base + offset) * factor) & 255) * 3 >> 8;
        value = h(p, base + 52, big) * (f32::cast_from(q) - 1.0);
    }
    if comptime!(format == 35) {
        let q = (byte(p, base + j / 128 * 32 + j % 32) >> u32::cast_from(2 * (j % 128 / 32))) & 3;
        value = h(p, base + 64, big) * (f32::cast_from(q) - 1.0);
    }
    if comptime!(format == 39) {
        let exponent = byte(p, base);
        let mut bits = 0u32;
        if exponent < 2 { bits = 0x00200000u32 << exponent; }
        else { bits = (exponent - 1) << 23; }
        let code = (byte(p, base + 1 + j % 16) >> u32::cast_from(4 * (j / 16))) & 15;
        value = f32::from_bits(bits) * signed(u32::cast_from(table[code as usize]));
    }
    if comptime!(format == 40) {
        let bits = byte(p, base + j / 16);
        let exponent = (bits >> 3) & 15;
        let mantissa = bits & 7;
        let mut scale = f32::cast_from(mantissa) * 0.0009765625;
        if exponent != 0 {
            scale = f32::from_bits((exponent + 119) << 23) * (1.0 + f32::cast_from(mantissa) * 0.125);
        }
        if bits == 0 || bits == 127 { scale = 0.0; }
        let code = (byte(p, base + 4 + j / 16 * 8 + j % 8) >> u32::cast_from(4 * (j % 16 / 8))) & 15;
        value = scale * signed(u32::cast_from(table[code as usize]));
    }
    if comptime!(format == 41) {
        let q = (byte(p, base + 2 + j / 8) >> u32::cast_from(j % 8)) & 1;
        value = h(p, base, big) * (2.0 * f32::cast_from(q) - 1.0);
    }
    if comptime!(format == 42) {
        let q = (byte(p, base + 2 + j / 4) >> u32::cast_from(2 * (j % 4))) & 3;
        value = h(p, base, big) * (f32::cast_from(q) - 1.0);
    }
    value
}

#[ruda(launch)]
pub(super) fn vector<F: Float + RudaElement>(
    input: &Tensor<F>, packed: &Tensor<u8>, table: &Tensor<u8>, output: &mut Tensor<F>,
    width: usize, #[comptime] multiply: bool,
    #[comptime] block: usize, #[comptime] bytes: usize, #[comptime] format: u32, #[comptime] big: bool,
) {
    let position = ABSOLUTE_POS;
    if position < output.len() {
        let k = position % width;
        let w = weight(packed, table, k / block * bytes, k % block, format, big);
        let x = f32::cast_from(input[position]);
        let mut y = x + w;
        if comptime!(multiply) { y = x * w; }
        output[position] = F::cast_from(y);
    }
}

#[ruda(launch)]
pub(super) fn matmul<F: Float + RudaElement>(
    input: &Tensor<F>, packed: &Tensor<u8>, table: &Tensor<u8>, output: &mut Tensor<F>,
    width: usize, columns: usize, row_bytes: usize,
    #[comptime] block: usize, #[comptime] bytes: usize, #[comptime] format: u32, #[comptime] big: bool,
) {
    let position = ABSOLUTE_POS;
    if position < output.len() {
        let row = position / columns;
        let column = position % columns;
        let mut sum = 0.0f32;
        for k in 0..width {
            let w = weight(packed, table, column * row_bytes + k / block * bytes, k % block, format, big);
            sum += f32::cast_from(input[row * width + k]) * w;
        }
        output[position] = F::cast_from(sum);
    }
}

#[ruda(launch)]
pub(super) fn matmul_selected<F: Float + RudaElement, I: Int + RudaElement>(
    input: &Tensor<F>, indices: &Tensor<I>, packed: &Tensor<u8>, table: &Tensor<u8>,
    output: &mut Tensor<F>, status: &mut Array<Atomic<u32>>, width: usize, columns: usize, experts: usize,
    selected: usize, input_slots: usize, row_bytes: usize,
    #[comptime] block: usize, #[comptime] bytes: usize, #[comptime] format: u32, #[comptime] big: bool,
) {
    let position = ABSOLUTE_POS;
    if position < output.len() {
        let slot = position / columns;
        let index = indices[slot];
        let expert = usize::cast_from(index);
        if index < I::cast_from(0) || I::cast_from(expert) != index || expert >= experts {
            status[0].store(1u32);
            terminate!();
        }
        let token = slot / selected;
        let input_row = token * input_slots + slot % selected % input_slots;
        let weight_row = expert * columns + position % columns;
        let mut sum = 0.0f32;
        for k in 0..width {
            let w = weight(packed, table, weight_row * row_bytes + k / block * bytes, k % block, format, big);
            sum += f32::cast_from(input[input_row * width + k]) * w;
        }
        output[position] = F::cast_from(sum);
    }
}

#[ruda(launch)]
pub(super) fn gather<F: Float + RudaElement, I: Int + RudaElement>(
    indices: &Tensor<I>, packed: &Tensor<u8>, table: &Tensor<u8>, output: &mut Tensor<F>,
    status: &mut Array<Atomic<u32>>,
    width: usize, rows: usize, row_bytes: usize,
    #[comptime] block: usize, #[comptime] bytes: usize, #[comptime] format: u32, #[comptime] big: bool,
) {
    let position = ABSOLUTE_POS;
    if position < output.len() {
        let index = indices[position / width];
        let row = usize::cast_from(index);
        if index < I::cast_from(0) || I::cast_from(row) != index || row >= rows {
            status[0].store(1u32);
            terminate!();
        }
        let k = position % width;
        output[position] = F::cast_from(weight(packed, table, row * row_bytes + k / block * bytes, k % block, format, big));
    }
}
