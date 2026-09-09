use ruda_kernel::dsl as cubecl;
use ruda_kernel::dsl::prelude::*;

use crate::kernel_ir::{
    components::{
        global::{SharedGlobalMatmulConfig, read::SyncStrategy},
        stage::StageConfig,
    },
    definition::MatmulTypes,
};

/// Simple synchronous barrier, using `cube_sync()`
pub struct Synchronous {}

#[cube]
impl SyncStrategy for Synchronous {
    type Barrier = ();

    fn create_barrier() -> Self::Barrier {}

    fn sync<MP: MatmulTypes, S: StageConfig>(
        _barrier: &mut Self::Barrier,
        #[comptime] _config: SharedGlobalMatmulConfig<S>,
    ) {
        sync_cube();
    }
}
