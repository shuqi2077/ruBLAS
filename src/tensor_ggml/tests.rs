use super::*;
use ruda_kernel::tensor::readback::into_data_sync;
use ruda_test_runtime::TestRuntime;

#[test]
fn packed_q4_projection_and_embedding_do_not_create_dense_weight_storage() {
    for big in [false, true] {
        let device = Default::default();
        let mut bytes = Vec::new();
        for scale in [1.0f32, 2.0] {
            let bits = f16::from_f32(scale).to_bits();
            bytes.extend_from_slice(&if big { bits.to_be_bytes() } else { bits.to_le_bytes() });
            bytes.extend_from_slice(&[0x98; 16]);
        }
        let weights = GgmlGemm::<TestRuntime>::from_bytes(bytes.clone(), GgmlType::Q4_0, 2, 32, big, &device).unwrap();
        assert_eq!(weights.packed_bytes(), 36);
        assert_eq!(weights.original_storage().dtype, DType::U8);
        let input = from_data(TensorData::new((1..=32).map(|i| i as f32).collect::<Vec<_>>(), [1, 32]), &device);
        let actual = into_data_sync(weights.forward(input).unwrap()).to_vec::<f32>().unwrap();
        assert_eq!(actual, [392.0, 784.0]);
        let indices = from_data(TensorData::new(vec![1i32, 0], [2]), &device);
        let actual = into_data_sync(weights.gather(indices, DType::F32).unwrap()).to_vec::<f32>().unwrap();
        let expected = [vec![0.0; 16], vec![2.0; 16], vec![0.0; 16], vec![1.0; 16]].concat();
        assert_eq!(actual, expected);
        assert_eq!(into_data_sync(weights.original_storage().clone()).to_vec::<u8>().unwrap(), bytes);
    }
}

#[test]
fn packed_ggml_indices_reject_negative_out_of_range_and_truncated_values() {
    let device = Default::default();
    let mut bytes = Vec::new();
    for scale in [1.0f32, 2.0] {
        bytes.extend_from_slice(&f16::from_f32(scale).to_bits().to_le_bytes());
        bytes.extend_from_slice(&[0x99; 16]);
    }
    let weights = GgmlGemm::<TestRuntime>::from_bytes(bytes.clone(), GgmlType::Q4_0, 2, 32, false, &device).unwrap();
    let input = from_data(TensorData::new(vec![1.0f32; 32], [1, 1, 32]), &device);
    let invalid = [-1i32, 2, i32::MAX].into_iter()
        .map(|index| TensorData::new(vec![index], [1, 1]))
        .chain([-1i64, 2, 1i64 << 32, (1i64 << 32) + 1, i64::MAX].into_iter()
            .map(|index| TensorData::new(vec![index], [1, 1])));
    for data in invalid {
        let indices = from_data(data, &device);
        assert_eq!(weights.gather(indices.clone(), DType::F32).unwrap_err(),
            GgmlError("GGUF embedding index out of bounds".into()));
        assert_eq!(weights.forward_selected(input.clone(), indices, 2).unwrap_err(),
            GgmlError("GGUF expert index out of bounds".into()));
    }
    for data in [TensorData::new(vec![1i32], [1, 1]), TensorData::new(vec![1i64], [1, 1])] {
        let indices = from_data(data, &device);
        assert_eq!(into_data_sync(weights.gather(indices.clone(), DType::F32).unwrap()).to_vec::<f32>().unwrap(), vec![2.0; 32]);
        assert_eq!(into_data_sync(weights.forward_selected(input.clone(), indices, 2).unwrap()).to_vec::<f32>().unwrap(), vec![64.0]);
    }
    assert_eq!(into_data_sync(weights.original_storage().clone()).to_vec::<u8>().unwrap(), bytes);
}

#[test]
#[ignore = "requires pinned GGML reference fixture in RUDA_GGML_ORACLE"]
fn all_ggml_formats_match_independent_ggml_projection_gather_and_vector_reference() {
    use std::io::Read;
    fn word(input: &mut &[u8]) -> u32 {
        let mut bytes = [0; 4];
        input.read_exact(&mut bytes).unwrap();
        u32::from_le_bytes(bytes)
    }
    fn near(actual: f32, expected: f32, tolerance: f32, format: GgmlType, index: usize) {
        assert!(actual.is_finite() && (actual - expected).abs() <= tolerance * expected.abs().max(1.0),
            "{format:?} at {index}: {actual} != {expected}");
    }
    let bytes = std::fs::read(std::env::var_os("RUDA_GGML_ORACLE")
        .expect("RUDA_GGML_ORACLE must identify the independently generated fixture")).unwrap();
    let mut data = bytes.as_slice();
    assert_eq!(&data[..4], b"GGOR");
    data = &data[4..];
    let count = word(&mut data);
    let mut formats = Vec::new();
    let device = Default::default();
    for _ in 0..count {
        let id = word(&mut data);
        formats.push(id);
        let format = GgmlType::from_id(id).unwrap();
        let width = word(&mut data) as usize;
        let rows = word(&mut data) as usize;
        let length = word(&mut data) as usize;
        let packed = data[..length].to_vec();
        data = &data[length..];
        let reference = (0..rows * width).map(|_| f32::from_bits(word(&mut data))).collect::<Vec<_>>();
        let weights = GgmlGemm::<TestRuntime>::from_bytes(packed.clone(), format, rows, width, false, &device).unwrap();
        assert_eq!(weights.packed_bytes(), packed.len());
        let ids = vec![2i32, 0, 1, 2];
        let indices = from_data(TensorData::new(ids.clone(), [ids.len()]), &device);
        let actual = into_data_sync(weights.gather(indices, DType::F32).unwrap()).to_vec::<f32>().unwrap();
        for (i, value) in actual.into_iter().enumerate() {
            near(value, reference[ids[i / width] as usize * width + i % width], 2e-6, format, i);
        }
        let input = (0..5 * width).map(|i| ((i * 17 % 23) as f32 - 11.0) / 7.0).collect::<Vec<_>>();
        let tensor = from_data(TensorData::new(input.clone(), [5, width]), &device);
        let actual = into_data_sync(weights.forward(tensor).unwrap()).to_vec::<f32>().unwrap();
        for (i, value) in actual.into_iter().enumerate() {
            let expected = (0..width).map(|k| input[i / rows * width + k] * reference[i % rows * width + k]).sum::<f32>();
            near(value, expected, 2e-4, format, i);
        }
        let vector = GgmlGemm::<TestRuntime>::from_bytes(packed[..length / rows].to_vec(),
            format, 1, width, false, &device).unwrap();
        for multiply in [false, true] {
            let tensor = from_data(TensorData::new(input.clone(), [5, width]), &device);
            let actual = into_data_sync(vector.apply_vector(tensor, multiply).unwrap()).to_vec::<f32>().unwrap();
            for (i, value) in actual.into_iter().enumerate() {
                let expected = if multiply { input[i] * reference[i % width] } else { input[i] + reference[i % width] };
                near(value, expected, 2e-6, format, i);
            }
        }
        assert_eq!(into_data_sync(weights.original_storage().clone()).to_vec::<u8>().unwrap(), packed);
    }
    assert_eq!(formats, [0,1,2,3,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,29,30,34,35,39,40,41,42]);
    assert!(data.is_empty());
}
