mod codebooks;
mod kernels;
#[cfg(test)]
mod tests;

use half::{bf16, f16};
use ruda_core::{
    device::Device,
    tensor::{DType, Shape, data::TensorData, quantization::GgmlType},
};
use ruda_kernel::{
    dsl::{Runtime, calculate_ruda_count_elemwise, prelude::RudaDim},
    tensor::{RudaTensor, allocation::empty_device_contiguous_dtype, contiguous::into_contiguous, transfer::from_data},
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GgmlError(pub String);
impl core::fmt::Display for GgmlError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result { f.write_str(&self.0) }
}
impl std::error::Error for GgmlError {}

/// Original GGML block storage in logical `[rows, width]` order.
/// Only the packed file bytes and the format's small shared codebook are resident.
#[derive(Clone, Debug)]
pub struct GgmlGemm<R: Runtime> {
    packed: RudaTensor<R>,
    table: RudaTensor<R>,
    format: GgmlType,
    big_endian: bool,
    rows: usize,
    width: usize,
    row_bytes: usize,
}

impl<R: Runtime> GgmlGemm<R> {
    pub fn from_bytes(
        data: Vec<u8>, format: GgmlType, rows: usize, width: usize,
        big_endian: bool, device: &R::Device,
    ) -> Result<Self, GgmlError> {
        let row_bytes = format.row_bytes(width).ok_or_else(|| GgmlError("GGML row is not block aligned".into()))?;
        if rows == 0 || width == 0 || rows.checked_mul(row_bytes) != Some(data.len())
            || data.len() > u32::MAX as usize || width > u32::MAX as usize || rows > u32::MAX as usize
        { return Err(GgmlError("GGML packed geometry or byte length is invalid".into())); }
        if !format.is_quantized() && !matches!(format, GgmlType::F32 | GgmlType::F16 | GgmlType::BF16) {
            return Err(GgmlError(format!("{format:?} is not a supported floating GGML projection format")));
        }
        let table: Vec<u8> = match format {
            GgmlType::IQ2_XXS => codebooks::IQ2XXS_GRID.iter().flat_map(|n| n.to_le_bytes()).collect(),
            GgmlType::IQ2_XS => codebooks::IQ2XS_GRID.iter().flat_map(|n| n.to_le_bytes()).collect(),
            GgmlType::IQ2_S => codebooks::IQ2S_GRID.iter().flat_map(|n| n.to_le_bytes()).collect(),
            GgmlType::IQ3_XXS => codebooks::IQ3XXS_GRID.iter().flat_map(|n| n.to_le_bytes()).collect(),
            GgmlType::IQ3_S => codebooks::IQ3S_GRID.iter().flat_map(|n| n.to_le_bytes()).collect(),
            GgmlType::IQ1_S | GgmlType::IQ1_M => codebooks::IQ1S_GRID.iter().flat_map(|n| n.to_le_bytes()).collect(),
            GgmlType::IQ4_NL | GgmlType::IQ4_XS => codebooks::KVALUES_IQ4NL.iter().map(|n| *n as u8).collect(),
            GgmlType::MXFP4 | GgmlType::NVFP4 => codebooks::KVALUES_FP4.iter().map(|n| *n as u8).collect(),
            _ => vec![0u8],
        };
        let size = data.len();
        let packed = from_data(TensorData::new(data, [size]), device);
        let table_len = table.len();
        let table = from_data(TensorData::new(table, [table_len]), device);
        Ok(Self { packed, table, format, big_endian, rows, width, row_bytes })
    }

    pub fn format(&self) -> GgmlType { self.format }
    pub fn rows(&self) -> usize { self.rows }
    pub fn width(&self) -> usize { self.width }
    pub fn packed_bytes(&self) -> usize { self.rows * self.row_bytes }
    pub fn original_storage(&self) -> &RudaTensor<R> { &self.packed }

    pub fn apply_vector(&self, input: RudaTensor<R>, multiply: bool) -> Result<RudaTensor<R>, GgmlError> {
        self.same_device(&input)?;
        let shape = input.meta.shape().clone();
        let rank = shape.num_dims();
        if self.rows != 1 || rank == 0 || shape[rank - 1] != self.width
            || !matches!(input.dtype, DType::F32 | DType::F16 | DType::BF16)
            || input.meta.num_elements() > u32::MAX as usize
        { return Err(GgmlError("GGML vector operation requires matching floating input".into())); }
        let input = into_contiguous(input);
        let size = input.meta.num_elements();
        let output = empty_device_contiguous_dtype(input.client.clone(), input.device.clone(), shape, input.dtype);
        if size != 0 {
            let dim = RudaDim::new(input.client.properties(), size);
            let grid = calculate_ruda_count_elemwise(&input.client, size, dim);
            let (block, bytes) = self.format.block_layout();
            macro_rules! run {
                ($f:ty) => { unsafe { kernels::vector::launch::<$f, R>(
                    &input.client, grid, dim, input.clone().into_tensor_arg(),
                    self.packed.clone().into_tensor_arg(), self.table.clone().into_tensor_arg(),
                    output.clone().into_tensor_arg(), self.width, multiply,
                    block, bytes, self.format as u32, self.big_endian,
                ) } };
            }
            match input.dtype { DType::F32 => run!(f32), DType::F16 => run!(f16), DType::BF16 => run!(bf16), _ => unreachable!() }
        }
        Ok(output)
    }

    fn same_device(&self, tensor: &RudaTensor<R>) -> Result<(), GgmlError> {
        if tensor.device.to_id() != self.packed.device.to_id()
            || !tensor.client.same_execution_queue(&self.packed.client) || tensor.qparams.is_some()
        { return Err(GgmlError("GGML input and packed weight must share device and execution queue".into())); }
        Ok(())
    }

    pub fn forward(&self, input: RudaTensor<R>) -> Result<RudaTensor<R>, GgmlError> {
        self.same_device(&input)?;
        let mut shape = input.meta.shape().clone();
        let rank = shape.num_dims();
        if rank == 0 || shape[rank - 1] != self.width
            || !matches!(input.dtype, DType::F32 | DType::F16 | DType::BF16)
        { return Err(GgmlError("GGML projection requires matching floating input width".into())); }
        shape[rank - 1] = self.rows;
        let size = shape.iter().try_fold(1usize, |n, &d| n.checked_mul(d))
            .ok_or_else(|| GgmlError("GGML output size overflows".into()))?;
        if size > u32::MAX as usize || input.meta.num_elements() > u32::MAX as usize {
            return Err(GgmlError("GGML activation exceeds native indexing range".into()));
        }
        let input = into_contiguous(input);
        let output = empty_device_contiguous_dtype(input.client.clone(), input.device.clone(), shape, input.dtype);
        if size != 0 {
            let dim = RudaDim::new(input.client.properties(), size);
            let grid = calculate_ruda_count_elemwise(&input.client, size, dim);
            let (block, bytes) = self.format.block_layout();
            macro_rules! run {
                ($f:ty) => { unsafe { kernels::matmul::launch::<$f, R>(
                    &input.client, grid, dim, input.clone().into_tensor_arg(),
                    self.packed.clone().into_tensor_arg(), self.table.clone().into_tensor_arg(),
                    output.clone().into_tensor_arg(), self.width, self.rows, self.row_bytes,
                    block, bytes, self.format as u32, self.big_endian,
                ) } };
            }
            match input.dtype { DType::F32 => run!(f32), DType::F16 => run!(f16), DType::BF16 => run!(bf16), _ => unreachable!() }
        }
        Ok(output)
    }

    /// Applies selected expert matrices to `[tokens, 1 | selected, width]`.
    /// Packed rows are `[experts, columns]`; indices are `[tokens, selected]`.
    pub fn forward_selected(
        &self, input: RudaTensor<R>, indices: RudaTensor<R>, experts: usize,
    ) -> Result<RudaTensor<R>, GgmlError> {
        self.same_device(&input)?;
        self.same_device(&indices)?;
        let shape = input.meta.shape();
        let index_shape = indices.meta.shape();
        if experts == 0 || self.rows % experts != 0 || shape.num_dims() != 3
            || index_shape.num_dims() != 2 || shape[0] != index_shape[0]
            || index_shape[1] == 0 || (shape[1] != 1 && shape[1] != index_shape[1])
            || shape[2] != self.width
            || !matches!(input.dtype, DType::F32 | DType::F16 | DType::BF16)
            || !matches!(indices.dtype, DType::I32 | DType::I64)
        { return Err(GgmlError("GGML selected expert geometry or dtype is invalid".into())); }
        let columns = self.rows / experts;
        let selected = index_shape[1];
        let input_slots = shape[1];
        let output_shape = Shape::from([shape[0], selected, columns]);
        let size = shape[0].checked_mul(selected).and_then(|n| n.checked_mul(columns))
            .filter(|n| *n <= u32::MAX as usize)
            .ok_or_else(|| GgmlError("GGML selected expert output exceeds native indexing range".into()))?;
        if input.meta.num_elements() > u32::MAX as usize || indices.meta.num_elements() > u32::MAX as usize {
            return Err(GgmlError("GGML selected expert input exceeds native indexing range".into()));
        }
        let input = into_contiguous(input);
        let indices = into_contiguous(indices);
        let output = empty_device_contiguous_dtype(
            input.client.clone(), input.device.clone(), output_shape, input.dtype);
        if size != 0 {
            let dim = RudaDim::new(input.client.properties(), size);
            let grid = calculate_ruda_count_elemwise(&input.client, size, dim);
            let (block, bytes) = self.format.block_layout();
            macro_rules! run {
                ($f:ty, $i:ty) => { unsafe { kernels::matmul_selected::launch::<$f, $i, R>(
                    &input.client, grid, dim, input.clone().into_tensor_arg(), indices.clone().into_tensor_arg(),
                    self.packed.clone().into_tensor_arg(), self.table.clone().into_tensor_arg(),
                    output.clone().into_tensor_arg(), self.width, columns, experts,
                    selected, input_slots, self.row_bytes, block, bytes, self.format as u32, self.big_endian,
                ) } };
            }
            match (input.dtype, indices.dtype) {
                (DType::F32, DType::I32) => run!(f32, i32), (DType::F32, DType::I64) => run!(f32, i64),
                (DType::F16, DType::I32) => run!(f16, i32), (DType::F16, DType::I64) => run!(f16, i64),
                (DType::BF16, DType::I32) => run!(bf16, i32), (DType::BF16, DType::I64) => run!(bf16, i64),
                _ => unreachable!(),
            }
        }
        Ok(output)
    }

    /// Gathers only requested embeddings; the vocabulary table stays packed.
    pub fn gather(&self, indices: RudaTensor<R>, dtype: DType) -> Result<RudaTensor<R>, GgmlError> {
        self.same_device(&indices)?;
        if !matches!(indices.dtype, DType::I32 | DType::I64)
            || !matches!(dtype, DType::F32 | DType::F16 | DType::BF16)
        { return Err(GgmlError("GGML gather requires integer indices and floating output".into())); }
        let mut shape = indices.meta.shape().to_vec();
        shape.push(self.width);
        let size = indices.meta.num_elements().checked_mul(self.width)
            .filter(|n| *n <= u32::MAX as usize).ok_or_else(|| GgmlError("GGML gather size overflows".into()))?;
        let indices = into_contiguous(indices);
        let output = empty_device_contiguous_dtype(indices.client.clone(), indices.device.clone(), Shape::from(shape), dtype);
        if size != 0 {
            let dim = RudaDim::new(indices.client.properties(), size);
            let grid = calculate_ruda_count_elemwise(&indices.client, size, dim);
            let (block, bytes) = self.format.block_layout();
            macro_rules! run {
                ($f:ty, $i:ty) => { unsafe { kernels::gather::launch::<$f, $i, R>(
                    &indices.client, grid, dim, indices.clone().into_tensor_arg(),
                    self.packed.clone().into_tensor_arg(), self.table.clone().into_tensor_arg(),
                    output.clone().into_tensor_arg(), self.width, self.rows, self.row_bytes,
                    block, bytes, self.format as u32, self.big_endian,
                ) } };
            }
            match (dtype, indices.dtype) {
                (DType::F32, DType::I32) => run!(f32, i32), (DType::F32, DType::I64) => run!(f32, i64),
                (DType::F16, DType::I32) => run!(f16, i32), (DType::F16, DType::I64) => run!(f16, i64),
                (DType::BF16, DType::I32) => run!(bf16, i32), (DType::BF16, DType::I64) => run!(bf16, i64),
                _ => unreachable!(),
            }
        }
        Ok(output)
    }
}
