//! Naive matmul kernel implementation
//!
//! Each local unit will compute a single element of the output matrix.
use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::*;
use ruda_kernel::library::tensor::MatrixBatchLayout;
use ruda_kernel::library::tensor::matrix_batch_layout;
use ruda_kernel::dsl::tensor_vector_size_parallel;
use ruda_kernel::tiling::InputBinding;

use crate::kernel_ir::{
    definition::{MatmulElems, MatmulProblem, MatmulSetupError},
    definition::{MatmulVectorSizes, ruda_mapping_launch},
};

use crate::kernel_ir::{
    launch::InputArg,
    launch::{ConcreteInputsFactory, ConcreteOutputFactory, OutputArg, TensorArgs},
    routines::naive::NaiveRoutine,
    routines::{BlueprintStrategy, Routine as _},
};

/// Prepare the existing naive kernel without allocating or launching layout copies.
/// LHS must be row-major, RHS column-major, and output row-major.
pub fn prepare_canonical_ref<R: Runtime>(
    client: &ComputeClient<R>, lhs: InputBinding<R>, rhs: InputBinding<R>,
    out: TensorBinding<R>, dtypes: &MatmulElems,
) -> Result<PreparedKernel<R>, MatmulSetupError> {
    use crate::kernel_ir::{components::batch::naive::matmul_entry, launch::*};
    let rank = lhs.shape().len();
    if rank < 2 || rhs.shape().len() != rank || lhs.scheme().is_some() || rhs.scheme().is_some()
        || !matches!(matrix_batch_layout(&lhs.data().strides, None), MatrixBatchLayout::Contiguous)
        || rhs.data().strides[rank - 2] != 1
        || rhs.data().strides[rank - 1] != rhs.shape()[rank - 2]
        || !matches!(matrix_batch_layout(&out.strides, None), MatrixBatchLayout::Contiguous)
    {
        return Err(MatmulSetupError::InvalidConfig(Box::new("prepared naive matmul requires canonical unquantized layouts")));
    }
    let vector_sizes = MatmulVectorSizes {
        lhs: tensor_vector_size_parallel(client.io_optimized_vector_sizes(dtypes.lhs_global.size()),
            &lhs.data().shape, &lhs.data().strides, rank - 1),
        rhs: tensor_vector_size_parallel(client.io_optimized_vector_sizes(dtypes.rhs_global.size()),
            &rhs.data().shape, &rhs.data().strides, rank - 2),
        out: 1,
    };
    let address_type = lhs.required_address_type().max(rhs.required_address_type())
        .max(out.required_address_type(dtypes.acc_global.size()));
    let problem = MatmulProblem::from_shapes_and_strides(lhs.shape().into(), rhs.shape().into(),
        (&out.shape).into(), lhs.data().strides.clone(), rhs.data().strides.clone(), out.strides.clone(),
        dtypes.as_global_elems(), address_type, None, None)?;
    let settings = NaiveRoutine::device_settings(client, vector_sizes);
    let expanded = NaiveRoutine::expand_blueprint(&problem, &settings, &BlueprintStrategy::Inferred(().into()))?;
    let info = NaiveRoutine::prepare(&problem, &settings, expanded)?;
    let input = <InputArg<TensorArgs> as ConcreteInputsFactory<NaiveRoutine>>::create(
        lhs, rhs, &info.blueprint, &problem, &info.vector_sizes, dtypes);
    let output = <OutputArg<TensorArgs> as ConcreteOutputFactory<NaiveRoutine>>::create(
        out, &info.blueprint, &problem, &info.vector_sizes, dtypes);
    Ok(matmul_entry::prepare::<TensorArgs, Lhs, LhsSize, Rhs, RhsSize, Acc, AccSize, R>(
        client, info.ruda_count_plan.resolve(), info.ruda_dim, info.address_type, input, output, (),
        ruda_mapping_launch(&info.ruda_count_plan), info.blueprint,
        include_str!("../components/batch/naive/matmul.rs").to_owned(),
        [dtypes.lhs_global, dtypes.rhs_global, dtypes.acc_global],
        [info.vector_sizes.lhs, info.vector_sizes.rhs, info.vector_sizes.out]))
}

#[allow(clippy::result_large_err)]
pub fn launch_ref<R: Runtime>(
    client: &ComputeClient<R>,
    lhs: InputBinding<R>,
    rhs: InputBinding<R>,
    out: TensorBinding<R>,
    dtypes: &MatmulElems,
) -> Result<(), MatmulSetupError> {
    let rank = lhs.shape().len();
    let dim1 = rank - 1;
    let dim2 = rank - 2;

    let lhs_layout = matrix_batch_layout(&lhs.data().strides, lhs.scheme());
    let rhs_layout = matrix_batch_layout(&rhs.data().strides, rhs.scheme());

    let lhs = if !matches!(lhs_layout, MatrixBatchLayout::Contiguous) {
        lhs.into_contiguous(client)?
    } else {
        lhs
    };

    // we swap the dimensions to achieve memory-coalescing:
    // consecutive elements of a column in the original rhs tensor will now be stored
    // consecutively in memory, which allows to fetch them with fewer memory instructions
    let correct_rhs_layout = |mut rhs: InputBinding<R>| {
        rhs.swap_dims(dim1, dim2);
        let mut rhs = rhs.into_contiguous(client)?;

        rhs.swap_dims(dim1, dim2);

        let returned: Result<InputBinding<R>, LaunchError> = Ok(rhs);
        returned
    };

    let rhs = match rhs_layout {
        MatrixBatchLayout::Contiguous => correct_rhs_layout(rhs)?,
        MatrixBatchLayout::MildlyPermuted {
            transposed,
            batch_swap,
        } => {
            if transposed && !batch_swap {
                rhs
            } else {
                correct_rhs_layout(rhs)?
            }
        }
        MatrixBatchLayout::HighlyPermuted => correct_rhs_layout(rhs)?,
    };

    let lhs_shape = lhs.shape();
    let rhs_shape = rhs.shape();
    let out_shape = &out.shape;

    let lhs_vector_size = tensor_vector_size_parallel(
        client.io_optimized_vector_sizes(dtypes.lhs_global.size()),
        &lhs.data().shape,
        &lhs.data().strides,
        rank - 1,
    );
    let rhs_vector_size = tensor_vector_size_parallel(
        client.io_optimized_vector_sizes(dtypes.rhs_global.size()),
        &rhs.data().shape,
        &rhs.data().strides,
        rank - 2,
    );
    let vector_sizes = MatmulVectorSizes {
        lhs: lhs_vector_size,
        rhs: rhs_vector_size,
        out: 1,
    };

    let mut view_vector_sizes = vector_sizes;

    if let InputBinding::Quantized { scheme, .. } = lhs {
        view_vector_sizes.lhs *= scheme.num_quants();
    }
    if let InputBinding::Quantized { scheme, .. } = rhs {
        view_vector_sizes.rhs *= scheme.num_quants();
    }

    let address_type = lhs
        .required_address_type()
        .max(rhs.required_address_type())
        .max(out.required_address_type(dtypes.acc_global.size()));

    let problem = MatmulProblem::from_shapes_and_strides(
        lhs_shape.into(),
        rhs_shape.into(),
        out_shape.into(),
        lhs.data().strides.clone(),
        rhs.data().strides.clone(),
        out.strides.clone(),
        dtypes.as_global_elems(),
        address_type,
        lhs.scheme(),
        rhs.scheme(),
    )?;

    let device_settings = NaiveRoutine::device_settings(client, view_vector_sizes);
    let expand_info = NaiveRoutine::expand_blueprint(
        &problem,
        &device_settings,
        &BlueprintStrategy::Inferred(().into()),
    )?;
    let launch_info = NaiveRoutine::prepare(&problem, &device_settings, expand_info)?;

    let input = <InputArg<TensorArgs> as ConcreteInputsFactory<NaiveRoutine>>::create(
        lhs,
        rhs,
        &launch_info.blueprint,
        &problem,
        &launch_info.vector_sizes,
        dtypes,
    );
    let output = <OutputArg<TensorArgs> as ConcreteOutputFactory<NaiveRoutine>>::create(
        out,
        &launch_info.blueprint,
        &problem,
        &launch_info.vector_sizes,
        dtypes,
    );

    NaiveRoutine::launch::<TensorArgs, R>(
        client,
        launch_info.ruda_dim,
        launch_info.ruda_count_plan.resolve(),
        launch_info.address_type,
        input,
        output,
        (),
        ruda_mapping_launch(&launch_info.ruda_count_plan),
        launch_info.blueprint,
        dtypes,
        &launch_info.vector_sizes,
    )
}
