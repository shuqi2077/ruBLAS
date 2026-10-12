mod base;
#[allow(unsafe_code)]
mod gemm;
mod fused;
#[cfg(all(test, feature = "tensor-grouped"))]
mod tests_fused;
mod gemv_search;
mod tune;

/// Contains utilities for matmul operation
pub mod utils;

pub use base::*;
pub use gemm::{GemmOptions, gemm};
pub use fused::{GemmActivation, FusedGemmStrategy, matmul_bias_activation};
pub use gemv_search::*;
#[cfg(feature = "tensor-matmul-autotune")]
pub use tune::*;
pub use utils::*;
