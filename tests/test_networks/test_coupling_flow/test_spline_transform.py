import keras
import numpy as np
import pytest

from bayesflow._backend import value_and_grad
from bayesflow.networks.inference.coupling.transforms import SplineTransform


def _unconstrained_parameters(transform, *, batch_size, dimensions):
    return keras.ops.zeros((batch_size, dimensions * transform.params_per_dim))


def _constrained_parameters(transform, unconstrained):
    return transform.constrain_parameters(transform.split_parameters(unconstrained))


@pytest.mark.parametrize(
    ("transform_kwargs", "expected_domain"),
    [
        ({}, (-3.0, 3.0, -3.0, 3.0)),
        ({"default_domain": (-5.0, 5.0, -5.0, 5.0)}, (-5.0, 5.0, -5.0, 5.0)),
    ],
    ids=["default-domain", "wide-domain"],
)
def test_zero_parameters_realize_configured_domain_after_serialization(transform_kwargs, expected_domain):
    transform = SplineTransform(**transform_kwargs)
    serialized = keras.saving.serialize_keras_object(transform)
    transforms = (transform, keras.saving.deserialize_keras_object(serialized))

    for candidate in transforms:
        unconstrained = _unconstrained_parameters(candidate, batch_size=2, dimensions=3)
        parameters = _constrained_parameters(candidate, unconstrained)
        horizontal_edges = keras.ops.convert_to_numpy(parameters["horizontal_edges"])
        vertical_edges = keras.ops.convert_to_numpy(parameters["vertical_edges"])

        np.testing.assert_array_equal(horizontal_edges[..., 0], expected_domain[0])
        np.testing.assert_array_equal(horizontal_edges[..., -1], expected_domain[1])
        np.testing.assert_array_equal(vertical_edges[..., 0], expected_domain[2])
        np.testing.assert_array_equal(vertical_edges[..., -1], expected_domain[3])


def test_bin_indices_expose_forward_and_inverse_tail_occupancy():
    transform = SplineTransform(bins=4)
    horizontal_edges = np.broadcast_to(np.asarray([-3.0, -1.0, 0.0, 1.0, 3.0], dtype=np.float32), (1, 7, 5)).copy()
    vertical_edges = np.broadcast_to(np.asarray([-5.0, -2.0, 0.0, 2.0, 5.0], dtype=np.float32), (1, 7, 5)).copy()
    parameters = {
        "horizontal_edges": keras.ops.convert_to_tensor(horizontal_edges),
        "vertical_edges": keras.ops.convert_to_tensor(vertical_edges),
    }
    forward_values = keras.ops.convert_to_tensor([[-4.0, -3.0, -2.0, 0.5, 2.0, 3.0, 4.0]])
    inverse_values = keras.ops.convert_to_tensor([[-6.0, -5.0, -3.0, 1.0, 3.0, 5.0, 6.0]])

    forward_bins = transform.bin_indices(forward_values, parameters)
    inverse_bins = transform.bin_indices(inverse_values, parameters, inverse=True)

    expected = np.asarray([[-1, -1, 0, 2, 3, 3, 4]], dtype=np.int32)
    np.testing.assert_array_equal(keras.ops.convert_to_numpy(forward_bins), expected)
    np.testing.assert_array_equal(keras.ops.convert_to_numpy(inverse_bins), expected)


@pytest.mark.parametrize("inverse", [False, True], ids=["forward", "inverse"])
def test_extreme_affine_tails_have_finite_outputs_and_gradients(inverse):
    transform = SplineTransform(bins=8)
    values = keras.ops.convert_to_tensor([[-1.0e20, 1.0e20]])
    weights = keras.ops.convert_to_tensor([[1.0e-20, 1.0e-20]])
    unconstrained = _unconstrained_parameters(transform, batch_size=1, dimensions=2)

    def objective_from_parameters(raw_parameters):
        parameters = _constrained_parameters(transform, raw_parameters)
        outputs, log_det = transform(values, parameters=parameters, inverse=inverse)
        return keras.ops.sum(outputs * weights) + keras.ops.sum(log_det)

    fixed_parameters = _constrained_parameters(transform, unconstrained)

    def objective_from_values(raw_values):
        outputs, log_det = transform(raw_values, parameters=fixed_parameters, inverse=inverse)
        return keras.ops.sum(outputs * weights) + keras.ops.sum(log_det)

    parameter_value, parameter_gradients = value_and_grad(objective_from_parameters)(unconstrained)
    input_value, input_gradients = value_and_grad(objective_from_values)(values)
    outputs, log_det = transform(values, parameters=fixed_parameters, inverse=inverse)

    np.testing.assert_allclose(keras.ops.convert_to_numpy(outputs), keras.ops.convert_to_numpy(values))
    np.testing.assert_allclose(keras.ops.convert_to_numpy(log_det), np.zeros((1,)))
    for tensor in (parameter_value, parameter_gradients, input_value, input_gradients):
        assert np.all(np.isfinite(keras.ops.convert_to_numpy(tensor)))


@pytest.mark.parametrize("inverse", [False, True], ids=["forward", "inverse"])
def test_spline_branch_receives_safe_values(inverse):
    transform = SplineTransform(bins=8)
    values = keras.ops.convert_to_tensor([[-1.0e20, -2.5, 0.5, 2.5, 1.0e20]])
    unconstrained = _unconstrained_parameters(transform, batch_size=1, dimensions=5)
    parameters = _constrained_parameters(transform, unconstrained)
    method = transform.method_fn
    seen = []

    def recording_method(spline_input, *args, **kwargs):
        seen.append(keras.ops.convert_to_numpy(spline_input))
        return method(spline_input, *args, **kwargs)

    transform.method_fn = recording_method
    transform(values, parameters=parameters, inverse=inverse)

    bins = transform.bin_indices(values, parameters, inverse=inverse)
    inside = (bins >= 0) & (bins < transform.bins)
    edge_name = "vertical_edges" if inverse else "horizontal_edges"
    safe_boundary = parameters[edge_name][..., 0]
    expected = keras.ops.where(inside, values, safe_boundary)

    np.testing.assert_array_equal(seen[0], keras.ops.convert_to_numpy(expected))
