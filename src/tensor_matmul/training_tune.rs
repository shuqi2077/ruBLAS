use ruda::runtime::tune::{LocalTuner, Tunable, TunableSet};
use ruda_kernel::{dsl::{Runtime, RudaTuneId}, tensor::RudaTensor};
use crate::kernel_ir::launch::{MatmulAutotuneKey, Strategy};
use super::{F32MathMode, launch_matmul_with_precision};

fn key<R: Runtime>((lhs, rhs, out): &(RudaTensor<R>, RudaTensor<R>, RudaTensor<R>)) -> MatmulAutotuneKey {
    MatmulAutotuneKey::generate(&lhs.client, lhs.meta.shape(), rhs.meta.shape(),
        lhs.meta.strides(), rhs.meta.strides(), lhs.dtype.into(), rhs.dtype.into(),
        out.dtype.into(), lhs.try_scheme(), rhs.try_scheme())
}

fn inputs<R: Runtime>(_: &MatmulAutotuneKey,
    (lhs, rhs, out): &(RudaTensor<R>, RudaTensor<R>, RudaTensor<R>),
) -> (RudaTensor<R>, RudaTensor<R>, RudaTensor<R>) {
    // Preserve output layout/offset without reading uninitialized output data.
    let mut scratch = out.clone();
    let mut handle = out.client.empty(usize::try_from(out.handle.size()).expect("matmul scratch span exceeds usize"));
    handle.offset_start = out.handle.offset_start;
    handle.offset_end = out.handle.offset_end;
    scratch.handle = handle;
    (lhs.clone(), rhs.clone(), scratch)
}

pub(super) fn matmul<R: Runtime>(lhs: RudaTensor<R>, rhs: RudaTensor<R>,
    out: RudaTensor<R>, math: F32MathMode,
) -> RudaTensor<R> {
    static STRICT: LocalTuner<MatmulAutotuneKey, RudaTuneId> = LocalTuner::new("rublas::training_bf16::strict_f32::v3");
    static TF32: LocalTuner<MatmulAutotuneKey, RudaTuneId> = LocalTuner::new("rublas::training_bf16::allow_tf32::v3");
    let tuner = match math { F32MathMode::Strict => &STRICT, F32MathMode::AllowTf32 => &TF32 };
    let client = lhs.client.clone();
    let id = RudaTuneId::new(&client, &lhs.device);
    let candidates = tuner.init_for_device(&id, move || {
        let mut set = TunableSet::new(key::<R>, inputs::<R>);
        // Keep the known-compatible native selector first. No TMA/Hopper-only
        // candidates or reduced-precision conversions are introduced here.
        for strategy in [Strategy::Auto,
            Strategy::DoubleAsyncCyclicCmma(Default::default()),
            Strategy::DoubleCyclicCmma(Default::default()),
            Strategy::SimpleAsyncCyclicCmma(Default::default()),
            Strategy::OrderedDoubleCmma(Default::default())]
        {
            set = set.with(Tunable::new(&strategy.to_string(),
                move |(a, b, output): (RudaTensor<R>, RudaTensor<R>, RudaTensor<R>)| {
                    launch_matmul_with_precision(&strategy, a, b, output.clone(), math)
                        .map(|()| output).map_err(|error| format!("{error:?}"))
                }));
        }
        set
    });
    tuner.execute(&id, &client, candidates, (lhs, rhs, out))
}
