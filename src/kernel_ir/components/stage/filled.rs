use ruda_kernel::dsl as cubecl;
use ruda_kernel::dsl::prelude::*;
use ruda_kernel::library::tensor::layout::Coords2d;
use ruda_kernel::tiling::tile::{Tile, TileScope, Value};

use crate::kernel_ir::components::stage::{Stage, StageFamily, TilingLayout};

pub struct FilledStageFamily;

impl StageFamily for FilledStageFamily {
    type Stage<ES: Numeric, NS: Size, T: TilingLayout> = FilledStage<ES>;
}

#[derive(CubeType, Clone)]
pub struct FilledStage<ES: Numeric> {
    value: ES,
}

#[cube]
impl<ES: Numeric> FilledStage<ES> {
    pub fn new(value: ES) -> Self {
        FilledStage::<ES> { value }
    }
}

#[cube]
impl<ES: Numeric> Stage<ES, ReadOnly> for FilledStage<ES> {
    fn tile<Sc: TileScope>(this: &Self, _tile: Coords2d) -> Tile<ES, Sc, ReadOnly> {
        Tile::new_Broadcasted(Value::<ES> { val: this.value })
    }
}
