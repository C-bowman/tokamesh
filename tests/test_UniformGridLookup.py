import numpy as np
import pytest
from hypothesis import given, strategies as st
from tokamesh.utilities import BinaryTree, UniformGridLookup


@pytest.mark.parametrize("layers", [0, 1, 5, 10])
@pytest.mark.parametrize(
    "limits",
    [
        (0.0, 1.0),
        (-4.0, 8.0),
        (0.1, 1.3),
        tuple(np.array([0.1, 1.3], dtype=np.float32)),
        (1e8, 1e8 + 1.3),
        (-1e8, -1e8 + 1.3),
        (0.0, 1e-300),
        (-1e300, 1e300),
    ],
)
def test_uniform_grid_boundaries(layers, limits):
    grid = UniformGridLookup(layers, limits)
    tree = BinaryTree(layers, limits)
    values = np.concatenate(
        [
            tree.edges,
            np.nextafter(tree.edges, -np.inf),
            np.nextafter(tree.edges, np.inf),
            [-np.inf, np.inf, np.nan, -np.finfo(float).max, np.finfo(float).max],
        ]
    )

    with np.errstate(divide="raise", invalid="raise", over="raise"):
        actual = grid.lookup_index(values)

    np.testing.assert_array_equal(actual, tree.lookup_index(values))
    np.testing.assert_array_equal(grid.edges, tree.edges)
    np.testing.assert_allclose(grid.mids, tree.mids)
    assert grid.nodes == tree.nodes
    assert grid.layers == tree.layers


@pytest.mark.parametrize(
    "limits", [(0.1, 1.3), tuple(np.array([0.1, 1.3], dtype=np.float32))]
)
@given(value=st.floats(), layers=st.integers(min_value=0, max_value=10))
def test_uniform_grid_scalar(limits, value, layers):
    grid = UniformGridLookup(layers, limits)
    actual = grid.lookup_index(value)

    assert isinstance(actual, np.int64)
    assert actual == BinaryTree(layers, limits).lookup_index(value)


@pytest.mark.parametrize("layers", [0, 1, 3, 5])
@pytest.mark.parametrize("limits", [(np.float32(0), np.float32(1e-43)), (0.0, 1e-320)])
def test_uniform_grid_subnormal_steps(layers, limits):
    grid = UniformGridLookup(layers, limits)
    tree = BinaryTree(layers, limits)
    values = np.concatenate(
        [
            tree.edges,
            np.nextafter(tree.edges, -np.inf),
            np.nextafter(tree.edges, np.inf),
        ]
    )

    np.testing.assert_array_equal(grid.lookup_index(values), tree.lookup_index(values))


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64, np.uint64])
def test_uniform_grid_array_shapes(dtype):
    grid = UniformGridLookup(3, (0.0, 4.0))
    tree = BinaryTree(3, (0.0, 4.0))
    values = np.array([[0, 1, 2], [2, 3, 4]], dtype=dtype).T

    actual = grid.lookup_index(values)

    assert actual.shape == values.shape
    assert actual.dtype == np.int64
    np.testing.assert_array_equal(actual, tree.lookup_index(values))
    np.testing.assert_array_equal(grid.lookup_index(values.tolist()), actual)


@pytest.mark.parametrize("shape", [(0,), (2, 0)])
def test_uniform_grid_empty(shape):
    actual = UniformGridLookup(3, (0.0, 1.0)).lookup_index(np.empty(shape))

    assert actual.shape == shape
    assert actual.dtype == np.int64


@pytest.mark.parametrize(
    "layers, limits",
    [
        (-1, (0.0, 1.0)),
        (2, (0.0,)),
        (2, (0.0, 0.0)),
        (2, (1.0, 0.0)),
        (2, (np.nan, 1.0)),
        (2, (0.0, np.inf)),
        (2, (-1e308, 1e308)),
        (2, (1e16, 1e16 + 2.0)),
    ],
)
def test_uniform_grid_invalid_parameters(layers, limits):
    with pytest.raises(ValueError):
        UniformGridLookup(layers, limits)


def test_uniform_grid_noninteger_layers():
    with pytest.raises(TypeError):
        UniformGridLookup(1.5, (0.0, 1.0))


def test_uniform_grid_does_not_search(monkeypatch):
    def unexpected_search(*args, **kwargs):
        pytest.fail("UniformGridLookup must not use searchsorted")

    monkeypatch.setattr("tokamesh.utilities.searchsorted", unexpected_search)
    grid = UniformGridLookup(2, (0.0, 1.0))

    np.testing.assert_array_equal(
        grid.lookup_index([0.0, 0.25, 0.5, 0.75, 1.0]), [-1, 0, 1, 2, 3]
    )
