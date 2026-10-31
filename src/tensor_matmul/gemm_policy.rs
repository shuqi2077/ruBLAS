#[derive(Clone, Copy, Debug)]
pub struct GemmPlan {
    pub blackwell: bool,
    pub bm: usize,
    pub bn: usize,
    pub warp_rows: usize,
    pub warp_cols: usize,
    pub stages: usize,
    pub partitions: usize,
}

pub const SMALL_TILES: [[usize; 6]; 3] = [
    [16, 16, 64, 1, 2, 3],
    [32, 32, 64, 1, 2, 3],
    [127, 64, 64, 2, 2, 2],
];
pub const BLACKWELL_MIN_M: usize = 65;
pub const BLACKWELL_WIDE_MIN_M: usize = 512;
pub const SPLIT_WAVES: usize = 2;
pub const BLACKWELL_SPLIT_WAVES: usize = 1;
pub const MAX_SPLITS: usize = 16;
pub const MIN_PARTITION_K: usize = 256;
pub const SPARSE_PARTITION_K: usize = 128;

pub fn plan(m: usize, n: usize, k: usize, batches: usize, sms: usize,
    blackwell: bool, allow_split: bool) -> GemmPlan {
    let blackwell = blackwell && m >= BLACKWELL_MIN_M;
    let (bm, bn, warp_rows, warp_cols, stages) = if blackwell {
        let wide = m >= BLACKWELL_WIDE_MIN_M && n >= 256
            && m.div_ceil(128) * n.div_ceil(256) * batches >= sms.max(1);
        (128, if wide {256} else {128}, 1, 4, 3)
    } else if let Some(t) = SMALL_TILES.iter().find(|t| m <= t[0]) {
        (t[1], t[2], t[3], t[4], t[5])
    } else if n >= 2 * k {
        (128, 128, 2, 4, 3)
    } else {
        (128, 64, 2, 2, 3)
    };
    let tiles = m.div_ceil(bm) * n.div_ceil(bn) * batches;
    let minimum_k = if tiles.saturating_mul(4) < sms {SPARSE_PARTITION_K} else {MIN_PARTITION_K};
    let mut partitions = 1;
    let waves = if blackwell {BLACKWELL_SPLIT_WAVES} else {SPLIT_WAVES};
    if allow_split && tiles > 0 && sms > 0 && k % 64 == 0 {
        while partitions < MAX_SPLITS && tiles * partitions < sms * waves
            && k % (64 * partitions * 2) == 0 && k / (partitions * 2) >= minimum_k {
            partitions *= 2;
        }
    }
    GemmPlan {blackwell, bm, bn, warp_rows, warp_cols, stages, partitions}
}

impl GemmPlan {
    pub fn shared_bytes(self) -> usize {
        (self.bm + self.bn) * 64 * 2 * self.stages
            + if self.blackwell {128 * 33 * 4 + 2048} else {self.stages * 8}
    }
}

#[cfg(feature = "ptx-export")]
pub fn export_policy() -> String {
    format!("{{\"small_tiles\":{:?},\"blackwell_min_m\":{},\"wide_min_m\":{},\"split_waves\":{},\"blackwell_split_waves\":{},\"max_splits\":{},\"minimum_k\":{},\"sparse_k\":{}}}",
        SMALL_TILES, BLACKWELL_MIN_M, BLACKWELL_WIDE_MIN_M, SPLIT_WAVES, BLACKWELL_SPLIT_WAVES, MAX_SPLITS,
        MIN_PARTITION_K, SPARSE_PARTITION_K)
}
