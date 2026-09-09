pub mod row_fp {
    use super::*;
    use rublas::kernel_ir::definition::{MatmulProblem, TilingScheme};
    use ruda_kernel::tiling::cube_count::{CubeCountStrategy, GlobalOrder, HypercubeBlueprint, SmAllocation};

    fn hypercube_blueprint(
        tiling_scheme: &TilingScheme,
        problem: &MatmulProblem,
    ) -> HypercubeBlueprint {
        HypercubeBlueprint::builder()
            .global_order(GlobalOrder::RowMajor)
            .cube_count_strategy(CubeCountStrategy::FromProblem)
            .build()
    }

    include!("partition_buffering.rs");
}

mod swizzlecol_fp {
    use super::*;
    use rublas::kernel_ir::definition::TilingScheme;
    use ruda_kernel::tiling::cube_count::{CubeCountStrategy, GlobalOrder, HypercubeBlueprint, SmAllocation};

    fn hypercube_blueprint(
        tiling_scheme: &TilingScheme,
        problem: &MatmulProblem,
    ) -> HypercubeBlueprint {
        HypercubeBlueprint::builder()
            .global_order(GlobalOrder::SwizzleCol(2))
            .cube_count_strategy(CubeCountStrategy::FromProblem)
            .build()
    }

    include!("partition_buffering.rs");
}

mod col_fl {
    use super::*;
    use rublas::kernel_ir::definition::TilingScheme;
    use ruda_kernel::tiling::cube_count::{CubeCountStrategy, GlobalOrder, HypercubeBlueprint, SmAllocation};

    fn hypercube_blueprint(
        tiling_scheme: &TilingScheme,
        problem: &MatmulProblem,
    ) -> HypercubeBlueprint {
        HypercubeBlueprint::builder()
            .global_order(GlobalOrder::ColMajor)
            .cube_count_strategy(CubeCountStrategy::Flattened)
            .build()
    }

    include!("partition_buffering.rs");
}

mod swizzlerow_fl {
    use super::*;
    use rublas::kernel_ir::definition::TilingScheme;
    use ruda_kernel::tiling::cube_count::{CubeCountStrategy, GlobalOrder, HypercubeBlueprint, SmAllocation};

    fn hypercube_blueprint(
        tiling_scheme: &TilingScheme,
        problem: &MatmulProblem,
    ) -> HypercubeBlueprint {
        HypercubeBlueprint::builder()
            .global_order(GlobalOrder::SwizzleRow(2))
            .cube_count_strategy(CubeCountStrategy::Flattened)
            .build()
    }

    include!("partition_buffering.rs");
}

mod row_sm_exact {
    use super::*;
    use rublas::kernel_ir::definition::TilingScheme;
    use ruda_kernel::tiling::cube_count::{CubeCountStrategy, GlobalOrder, HypercubeBlueprint, SmAllocation};

    fn hypercube_blueprint(
        tiling_scheme: &TilingScheme,
        problem: &MatmulProblem,
    ) -> HypercubeBlueprint {
        HypercubeBlueprint::builder()
            .global_order(GlobalOrder::RowMajor)
            .cube_count_strategy(CubeCountStrategy::Sm {
                num_sms: 4,
                sm_usage: SmAllocation::Exact,
                cubes_first: false,
            })
            .build()
    }

    include!("partition_buffering.rs");
}

mod row_sm_full {
    use super::*;
    use rublas::kernel_ir::definition::TilingScheme;
    use ruda_kernel::tiling::cube_count::{CubeCountStrategy, GlobalOrder, HypercubeBlueprint, SmAllocation};

    fn hypercube_blueprint(
        tiling_scheme: &TilingScheme,
        problem: &MatmulProblem,
    ) -> HypercubeBlueprint {
        HypercubeBlueprint::builder()
            .global_order(GlobalOrder::RowMajor)
            .cube_count_strategy(CubeCountStrategy::Sm {
                num_sms: 4,
                sm_usage: SmAllocation::Full,
                cubes_first: false,
            })
            .build()
    }

    include!("partition_buffering.rs");
}

mod swizzlerow_cube_full {
    use super::*;
    use rublas::kernel_ir::definition::TilingScheme;
    use ruda_kernel::tiling::cube_count::{CubeCountStrategy, GlobalOrder, HypercubeBlueprint, SmAllocation};

    fn hypercube_blueprint(
        tiling_scheme: &TilingScheme,
        problem: &MatmulProblem,
    ) -> HypercubeBlueprint {
        HypercubeBlueprint::builder()
            .global_order(GlobalOrder::SwizzleRow(2))
            .cube_count_strategy(CubeCountStrategy::Sm {
                num_sms: 4,
                sm_usage: SmAllocation::Full,
                cubes_first: true,
            })
            .build()
    }

    include!("partition_buffering.rs");
}

mod swizzlerow_spread {
    use super::*;
    use rublas::kernel_ir::definition::TilingScheme;
    use ruda_kernel::tiling::cube_count::{CubeCountStrategy, GlobalOrder, HypercubeBlueprint, SmAllocation};

    fn hypercube_blueprint(
        tiling_scheme: &TilingScheme,
        problem: &MatmulProblem,
    ) -> HypercubeBlueprint {
        HypercubeBlueprint::builder()
            .global_order(GlobalOrder::SwizzleRow(2))
            .cube_count_strategy(CubeCountStrategy::Spread)
            .build()
    }

    include!("partition_buffering.rs");
}
