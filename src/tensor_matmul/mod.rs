mod base;
mod skinny;
mod accurate;
mod sliced;
mod training_tune;
mod training_gemm;
mod training_mma;
mod training_tile_tune;
#[allow(unsafe_code)]
mod gemm;
mod fused;
mod complex;
mod triangular;
#[cfg(all(test, feature = "tensor-grouped"))]
mod tests_fused;
mod gemv_search;
mod gemv;
mod mma_pipeline;
mod mma_split_k;
mod tune;

/// Contains utilities for matmul operation
pub mod utils;

pub use base::*;
pub use training_gemm::matmul_bf16_training;
pub use sliced::{matmul_bf16_sliced_k2_bias, matmul_bf16_serial_k3, matmul_bf16_head_parallel_k6};
pub use gemm::{GemmOptions, gemm};
pub use fused::{GemmActivation, FusedGemmStrategy, matmul_bias_activation};
pub use complex::{ComplexGemmOptions, complex_gemm};
pub use triangular::{TriangularSolveOptions, triangular_solve};
pub use gemv_search::*;
pub use gemv::{launch_gemv_low_precision, prepare_gemv_low_precision};
#[cfg(feature = "tensor-matmul-autotune")]
pub use tune::*;
pub use utils::*;
