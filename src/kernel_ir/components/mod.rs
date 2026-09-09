pub mod batch;
pub mod global;
pub mod stage;
pub mod tile;

// Internal-only — external crates import this directly from cubek-std.
pub(crate) use ruda_kernel::tiling::CubeDimResource;
