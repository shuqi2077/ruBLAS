use super::{AwqGemmLayout, grouped_kernel, kernel, tiled_kernel};
#[cfg(feature = "tensor-int4-autotune")]
use ruda::runtime::tune::{LocalTuner, Tunable, TunableSet};
#[cfg(feature = "tensor-int4-autotune")]
use ruda_kernel::dsl::RudaTuneId;
use ruda_kernel::dsl::{Runtime, calculate_ruda_count_elemwise, prelude::RudaDim};
use ruda_kernel::tensor::RudaTensor;

#[derive(Clone)]
pub(super) struct ForwardArgs<R: Runtime> {
    pub input: RudaTensor<R>,
    pub qweight: RudaTensor<R>,
    pub qzeros: RudaTensor<R>,
    pub scales: RudaTensor<R>,
    pub bias: Option<RudaTensor<R>>,
    pub output: RudaTensor<R>,
    pub layout: AwqGemmLayout,
    pub experts: usize,
    pub ids: Option<RudaTensor<R>>,
}

#[cfg(feature = "tensor-int4-autotune")]
fn key<R: Runtime>(args: &ForwardArgs<R>) -> String {
    format!("{:?};experts={};input={};weight={};zeros={};scales={};bias={:?};ids={:?};output={}",
        args.layout, args.experts, args.input.autotune_signature(), args.qweight.autotune_signature(),
        args.qzeros.autotune_signature(), args.scales.autotune_signature(),
        args.bias.as_ref().map(RudaTensor::autotune_signature),
        args.ids.as_ref().map(RudaTensor::autotune_signature), args.output.autotune_signature())
}

#[cfg(feature = "tensor-int4-autotune")]
fn scratch<R: Runtime>(_key: &String, args: &ForwardArgs<R>) -> ForwardArgs<R> {
    let mut args = args.clone();
    args.output.handle = args.output.client.empty(args.output.meta.num_elements() * args.output.dtype.size());
    args
}

#[cfg(not(feature = "tensor-int4-autotune"))]
pub(super) fn forward<R: Runtime>(args: ForwardArgs<R>) -> RudaTensor<R> {
    launch(args, 1, 0)
}

#[cfg(feature = "tensor-int4-autotune")]
pub(super) fn forward<R: Runtime>(args: ForwardArgs<R>) -> RudaTensor<R> {
    static TUNER: LocalTuner<String, RudaTuneId> = LocalTuner::new("rublas::awq::packed_ordered::v2");
    let client = args.input.client.clone();
    let id = RudaTuneId::new(&client, &args.input.device);
    let max_units = client.properties().hardware.max_units_per_ruda.min(client.properties().hardware.max_ruda_dim.0);
    let tunables = TUNER.init_for_device(&id, move || {
        let mut set = TunableSet::new(key::<R>, scratch::<R>)
            .with_stack_tuning(0, "packed-ordered-v2", key::<R>);
        for (columns, units) in [(0u32,0u32), (1,0), (2,0), (4,0), (8,0),
            (1,32), (2,32), (4,32), (8,32), (1,128), (2,128), (4,128), (8,128)] {
            if units != 0 && units > max_units { continue; }
            let name = if columns == 0 { "awq_original".into() } else { format!("awq_grouped_columns_{columns}_units_{units}") };
            set = set.with(Tunable::new(&name, move |args: ForwardArgs<R>| Ok::<_, String>(launch(args, columns, units))));
        }
        set
    });
    TUNER.execute(&id, &client, tunables, args)
}

fn launch<R: Runtime>(args: ForwardArgs<R>, columns: u32, units: u32) -> RudaTensor<R> {
    let elements = args.output.meta.num_elements();
    let work = elements / columns.max(1) as usize;
    let dim = if units == 0 { RudaDim::new(args.input.client.properties(), work) } else { RudaDim::new_1d(units) };
    let count = calculate_ruda_count_elemwise(&args.input.client, work, dim);
    let bias = args.bias.as_ref().unwrap_or(&args.scales).clone().into_array_arg();
    let k = args.layout.input_features as u32;
    let n = args.layout.output_features as u32;
    let group = args.layout.group_size as u32;
    let has_bias = u32::from(args.bias.is_some());
    if let Some(ids) = args.ids {
        if columns == 0 {
            grouped_kernel::forward::launch::<R>(&args.input.client, count, dim,
                args.input.clone().into_array_arg(), args.qweight.into_array_arg(), args.qzeros.into_array_arg(),
                args.scales.clone().into_array_arg(), bias, ids.into_array_arg(), args.output.clone().into_array_arg(),
                args.experts as u32, k, n, group, has_bias, args.input.dtype.into(), args.scales.dtype.into());
        } else {
            tiled_kernel::grouped_forward::launch::<R>(&args.input.client, count, dim,
                args.input.clone().into_array_arg(), args.qweight.into_array_arg(), args.qzeros.into_array_arg(),
                args.scales.clone().into_array_arg(), bias, ids.into_array_arg(), args.output.clone().into_array_arg(),
                args.experts as u32, k, n, group, has_bias, columns, args.input.dtype.into(), args.scales.dtype.into());
        }
    } else if columns != 0 {
        tiled_kernel::forward::launch::<R>(&args.input.client, count, dim,
            args.input.clone().into_array_arg(), args.qweight.into_array_arg(), args.qzeros.into_array_arg(),
            args.scales.clone().into_array_arg(), bias, args.output.clone().into_array_arg(),
            k, n, group, has_bias, columns, args.input.dtype.into(), args.scales.dtype.into());
    } else if args.input.dtype == args.scales.dtype {
        kernel::awq_gemm::launch::<R>(&args.input.client, count, dim,
            args.input.clone().into_array_arg(), args.qweight.into_array_arg(), args.qzeros.into_array_arg(),
            args.scales.into_array_arg(), bias, args.output.clone().into_array_arg(),
            k, n, group, has_bias, args.input.dtype.into());
    } else {
        kernel::awq_gemm_mixed::launch::<R>(&args.input.client, count, dim,
            args.input.clone().into_array_arg(), args.qweight.into_array_arg(), args.qzeros.into_array_arg(),
            args.scales.clone().into_array_arg(), bias, args.output.clone().into_array_arg(),
            k, n, group, has_bias, args.input.dtype.into(), args.scales.dtype.into());
    }
    args.output
}
