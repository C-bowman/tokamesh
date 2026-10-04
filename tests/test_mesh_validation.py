import numpy as np
import pytest
from tokamesh.geometry import GeometryCalculator
from tokamesh.mesh import TriangularMesh, validate_mesh_data


@pytest.fixture
def mesh_data():
    return {
        "R": np.array([0.0, 2.0, 1.0, 3.0]),
        "z": np.array([0.0, 0.0, 2.0, 1.0]),
        "triangles": np.array([[0, 1, 2]]),
    }


@pytest.mark.parametrize("name", ["R", "z"])
@pytest.mark.parametrize("shape", [(), (4, 1), (1, 4)])
def test_coordinate_dimensions(mesh_data, name, shape):
    mesh_data[name] = np.ones(shape)

    with pytest.raises(ValueError, match="1D array"):
        validate_mesh_data(**mesh_data)


@pytest.mark.parametrize("name", ["R", "z"])
def test_empty_coordinates(mesh_data, name):
    mesh_data[name] = np.array([])

    with pytest.raises(ValueError, match="empty"):
        validate_mesh_data(**mesh_data)


@pytest.mark.parametrize("name", ["R", "z"])
@pytest.mark.parametrize("dtype", [bool, complex, object, str, "datetime64[D]"])
def test_coordinate_dtypes(mesh_data, name, dtype):
    mesh_data[name] = mesh_data[name].astype(dtype)

    with pytest.raises(TypeError, match="real numeric"):
        validate_mesh_data(**mesh_data)


@pytest.mark.parametrize("name", ["R", "z"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_coordinates(mesh_data, name, value):
    mesh_data[name][-1] = value  # Also reject invalid unused vertices.

    with pytest.raises(ValueError, match="finite"):
        validate_mesh_data(**mesh_data)


@pytest.mark.parametrize("dtype", [float, complex, bool, object, str])
def test_connectivity_dtype(mesh_data, dtype):
    mesh_data["triangles"] = mesh_data["triangles"].astype(dtype)

    with pytest.raises(TypeError, match="integer dtype"):
        validate_mesh_data(**mesh_data)


def test_empty_connectivity(mesh_data):
    mesh_data["triangles"] = np.empty((0, 3), dtype=int)

    with pytest.raises(ValueError, match="at least one triangle"):
        validate_mesh_data(**mesh_data)


@pytest.mark.parametrize("shape", [(), (3,), (1, 2), (1, 3, 1)])
def test_connectivity_shape(mesh_data, shape):
    mesh_data["triangles"] = np.zeros(shape, dtype=int)

    with pytest.raises(ValueError, match="shape"):
        validate_mesh_data(**mesh_data)


@pytest.mark.parametrize(
    "triangles",
    [
        np.array([[-1, 1, 2]]),
        np.array([[0, 1, 4]]),
        np.array([[0, 1, np.iinfo(np.uint64).max]], dtype=np.uint64),
    ],
)
def test_connectivity_range(mesh_data, triangles):
    mesh_data["triangles"] = triangles

    with pytest.raises(ValueError, match="range"):
        validate_mesh_data(**mesh_data)


@pytest.mark.parametrize("vertices", [(0, 0, 1), (0, 1, 1), (0, 1, 0)])
def test_duplicate_vertex_indices(mesh_data, vertices):
    mesh_data["triangles"] = np.array([vertices])

    with pytest.raises(ValueError, match="duplicate vertices"):
        validate_mesh_data(**mesh_data)


@pytest.mark.parametrize(
    "R, z",
    [
        ([0.0, 1.0, 2.0], [0.0, 1.0, 2.0]),
        ([0.0, 1.0, 2.0], [1.0, 1.0, 1.0]),
        ([1.0, 1.0, 1.0], [0.0, 1.0, 2.0]),
        ([0.0, 0.0, 1.0], [0.0, 0.0, 1.0]),
    ],
)
def test_degenerate_triangles(R, z):
    with np.errstate(all="raise"), pytest.raises(ValueError, match="non-zero area"):
        validate_mesh_data(np.array(R), np.array(z), np.array([[0, 1, 2]]))


def test_degenerate_triangle_after_valid_triangle(mesh_data):
    mesh_data["R"][3] = 1.0
    mesh_data["z"][3] = 0.0
    mesh_data["triangles"] = np.array([[0, 1, 2], [0, 3, 1]])

    with pytest.raises(ValueError, match="Triangle 1"):
        validate_mesh_data(**mesh_data)


@pytest.mark.parametrize("scale", [1e200, 1e-200])
def test_unrepresentable_areas(mesh_data, scale):
    mesh_data["R"] *= scale
    mesh_data["z"] *= scale

    with np.errstate(all="raise"), pytest.raises(ValueError, match="non-zero area"):
        validate_mesh_data(**mesh_data)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64, np.uint64])
@pytest.mark.parametrize("reverse", [False, True])
def test_valid_numeric_coordinates(mesh_data, dtype, reverse):
    mesh_data["R"] = mesh_data["R"].astype(dtype)
    mesh_data["z"] = mesh_data["z"].astype(dtype)
    if reverse:
        mesh_data["triangles"] = mesh_data["triangles"][:, ::-1]
    originals = {name: value.copy() for name, value in mesh_data.items()}

    assert validate_mesh_data(**mesh_data) is None
    for name, value in mesh_data.items():
        np.testing.assert_array_equal(value, originals[name])
        assert value.dtype == originals[name].dtype


@pytest.mark.parametrize("dtype", [np.int8, np.int32, np.int64, np.uint64])
def test_valid_connectivity_dtypes(mesh_data, dtype):
    mesh_data["triangles"] = mesh_data["triangles"].astype(dtype)

    assert validate_mesh_data(**mesh_data) is None


@pytest.mark.parametrize("scale", [1e-20, 1e-100])
def test_small_nondegenerate_triangles(mesh_data, scale):
    mesh_data["R"] *= scale
    mesh_data["z"] *= scale

    assert validate_mesh_data(**mesh_data) is None


def test_thin_nondegenerate_triangle(mesh_data):
    mesh_data["z"] *= 1e-20

    assert validate_mesh_data(**mesh_data) is None


@pytest.mark.parametrize("invalid", ["dtype", "nonfinite", "degenerate"])
@pytest.mark.parametrize("constructor", [TriangularMesh, GeometryCalculator])
def test_constructors_validate_mesh_data(mesh_data, invalid, constructor):
    error = ValueError
    if invalid == "dtype":
        mesh_data["triangles"] = mesh_data["triangles"].astype(float)
        error = TypeError
    elif invalid == "nonfinite":
        mesh_data["R"][0] = np.nan
    else:
        mesh_data["z"][:] = 0.0
    if constructor is GeometryCalculator:
        mesh_data["ray_origins"] = np.array([[1.0, 0.0, 0.0]])
        mesh_data["ray_ends"] = np.array([[2.0, 1.0, 1.0]])

    with pytest.raises(error, match=constructor.__name__):
        constructor(**mesh_data)


def test_validation_error_source(mesh_data):
    mesh_data["R"][0] = np.nan

    with pytest.raises(ValueError, match="CustomCaller"):
        validate_mesh_data(**mesh_data, error_source="CustomCaller")
