import pytest
from numpy import arange, array, sin, cos, pi, isclose, ones, sqrt, sinc, zeros
from numpy.random import uniform, seed, multivariate_normal
from tokamesh import TriangularMesh
from tokamesh.utilities import BinaryTree, UniformGridLookup
from tokamesh.construction import equilateral_mesh
import matplotlib.pyplot as plt
from hypothesis import given, strategies as st


@given(st.floats(-4, 8))
def test_binary_tree(value):
    limit_left, limit_right = -4, 8

    tree = BinaryTree(4, [limit_left, limit_right])
    index = tree.lookup_index(value)
    # tree only returns the index if the value is inside the
    # range, exclusive of the lower limit, else it returns -1
    if limit_left < value <= limit_right:
        left = tree.edges[index]
        right = tree.edges[index + 1]
        assert left <= value <= right
    else:
        assert index == -1


@pytest.fixture
def mesh():
    # create a test mesh
    scale = 0.1
    R, z, triangles = equilateral_mesh(
        R_range=(0.0, 1.0), z_range=(0.0, 1.0), resolution=scale
    )
    # perturb the mesh to make it non-uniform
    seed(1)
    L = uniform(0.0, 0.3 * scale, size=R.size)
    theta = uniform(0.0, 2 * pi, size=R.size)
    R += L * cos(theta)
    z += L * sin(theta)
    # build the mesh
    return TriangularMesh(R=R, z=z, triangles=triangles)


def test_TriangularMesh_input_types():
    with pytest.raises(TypeError):
        TriangularMesh(ones(5), ones(5), [1, 2, 3])


def test_TriangularMesh_vertices_shape():
    with pytest.raises(ValueError):
        TriangularMesh(ones([5, 2]), ones(5), ones([4, 3]))


def test_TriangularMesh_inconsistent_sizes():
    with pytest.raises(ValueError):
        TriangularMesh(ones(6), ones(5), ones([4, 3]))


def test_TriangularMesh_triangles_shape():
    with pytest.raises(ValueError):
        TriangularMesh(ones([5, 2]), ones(5), ones([4, 3, 2]))


def test_interpolate(mesh):
    # As barycentric interpolation is linear, if we use a plane as the test
    # function, it should agree nearly exactly with interpolation result.
    def plane(x, y):
        return 0.37 - 5.31 * x + 2.09 * y

    vertex_values = plane(mesh.R, mesh.z)
    # create a series of random test-points
    R_test = uniform(0.2, 0.8, size=50)
    z_test = uniform(0.2, 0.8, size=50)
    # check the exact and interpolated values are equal
    interpolated = mesh.interpolate(R_test, z_test, vertex_values)
    assert isclose(interpolated, plane(R_test, z_test)).all()

    # now test multidimensional inputs
    R_test = uniform(0.2, 0.8, size=[12, 3, 8])
    z_test = uniform(0.2, 0.8, size=[12, 3, 8])
    # check the exact and interpolated values are equal
    interpolated = mesh.interpolate(R_test, z_test, vertex_values)
    assert isclose(interpolated, plane(R_test, z_test)).all()

    # now test giving just floats
    interpolated = mesh.interpolate(0.31, 0.54, vertex_values)
    assert isclose(interpolated, plane(0.31, 0.54)).all()


def test_build_interpolator_matrix(mesh):
    # As barycentric interpolation is linear, if we use a plane as the test
    # function, it should agree nearly exactly with interpolation result.
    def plane(x, y):
        return 0.21 - 4.91 * x + 21.9 * y

    vertex_values = plane(mesh.R, mesh.z)
    # create a series of random test-points
    R_test = uniform(0.2, 0.8, size=5000)
    z_test = uniform(0.2, 0.8, size=5000)
    G = mesh.build_interpolator_matrix(R_test, z_test)
    interpolated = G.dot(vertex_values)
    # check the exact and interpolated values are equal
    assert isclose(interpolated, plane(R_test, z_test)).all()

    # now test giving just floats
    r, z = 0.61, 0.74
    G = mesh.build_interpolator_matrix(r, z)
    interpolated = G.dot(vertex_values)
    assert isclose(interpolated, plane(r, z)).all()


def test_interpolate_inconsistent_shapes(mesh):
    with pytest.raises(ValueError):
        mesh.interpolate(ones([2, 1]), ones([2, 3]), ones(mesh.n_vertices))


def test_interpolate_vertices_size(mesh):
    with pytest.raises(ValueError):
        mesh.interpolate(ones(5), ones(5), ones(mesh.n_vertices - 2))


def test_interpolate_vertices_type(mesh):
    with pytest.raises(TypeError):
        mesh.interpolate(ones(5), ones(5), [1.0] * mesh.n_vertices)


def test_find_triangle(mesh):
    # first check all the barycentres
    R_centre = mesh.R[mesh.triangle_vertices].mean(axis=1)
    z_centre = mesh.z[mesh.triangle_vertices].mean(axis=1)
    tri_inds = mesh.find_triangle(R_centre, z_centre)
    assert (tri_inds == arange(mesh.triangle_vertices.shape[0])).all()

    # now check some points outside the mesh
    R_outside = array([mesh.R[mesh.z.argmax()] - 1e-4, mesh.R_limits[1] + 1.0])
    z_outside = array([mesh.z.max(), mesh.z_limits[1] + 1.0])
    tri_inds = mesh.find_triangle(R_outside, z_outside)
    assert (tri_inds == -1).all()
    # ideally we should find a way to generate lots of random
    # points just outside the boundary of the mesh to check.


@pytest.mark.parametrize("vertices", [(0, 1, 2), (0, 2, 1)])
def test_queries_in_cells_contained_by_triangle(vertices):
    mesh = TriangularMesh(
        R=array([0.0, 4.0, 1.0]),
        z=array([0.0, 1.0, 4.0]),
        triangles=array([vertices]),
    )
    # The cell [1, 2] x [1, 2] is strictly inside the triangle.
    R_test = array([0.25, 1.25, 1.5, 1.75, 2.0, 3.5, 5.0])
    z_test = array([0.25, 1.75, 1.5, 1.25, 2.0, 3.5, 1.0])
    vertex_values = 3.0 + mesh.R + 2.0 * mesh.z
    expected = 3.0 + R_test + 2.0 * z_test
    expected[-2:] = 0.0

    assert (mesh.find_triangle(R_test, z_test) == [0, 0, 0, 0, 0, -1, -1]).all()
    assert isclose(mesh.interpolate(R_test, z_test, vertex_values), expected).all()
    matrix = mesh.build_interpolator_matrix(R_test, z_test)
    assert matrix.shape == (R_test.size, mesh.n_vertices)
    assert isclose(matrix @ vertex_values, expected).all()


def test_triangle_candidates_on_cell_boundaries():
    mesh = TriangularMesh(
        R=array([0.0, 4.0, 0.0, 4.0]),
        z=array([0.0, 0.0, 4.0, 4.0]),
        triangles=array([[0, 1, 2], [1, 3, 2]]),
    )
    R_test = array([1.0, 2.0, 3.0, 4.0, 1.0])
    z_test = array([1.0, 2.0, 3.0, 1.0, 4.0])

    assert (mesh.find_triangle(R_test, z_test) == [0, 1, 1, 1, 1]).all()


@pytest.mark.parametrize("missing_value", [float("nan"), float("inf")])
def test_interpolate_outside_triangle_bounding_box_candidates(missing_value):
    mesh = TriangularMesh(
        R=array([0.0, 4.0, 1.0]),
        z=array([0.0, 1.0, 4.0]),
        triangles=array([[0, 1, 2]]),
    )
    vertex_values = array([missing_value, 1.0, 1.0])

    assert mesh.interpolate(3.5, 3.5, vertex_values)[0] == 0.0


def test_find_triangle_inconsistent_shapes(mesh):
    with pytest.raises(ValueError):
        mesh.find_triangle(ones([2, 1]), ones([2, 3]))


def test_uniform_lookup_matches_binary_tree_queries(mesh):
    assert isinstance(mesh.R_tree, UniformGridLookup)
    assert isinstance(mesh.z_tree, UniformGridLookup)
    R_test = uniform(mesh.R_limits[0], mesh.R_limits[1], size=1000)
    z_test = uniform(mesh.z_limits[0], mesh.z_limits[1], size=1000)
    values = 3.0 + mesh.R + 2.0 * mesh.z
    indices = mesh.find_triangle(R_test, z_test)
    interpolated = mesh.interpolate(R_test, z_test, values)
    matrix = mesh.build_interpolator_matrix(R_test, z_test)

    mesh.R_tree = BinaryTree(mesh.R_tree.layers, mesh.R_limits)
    mesh.z_tree = BinaryTree(mesh.z_tree.layers, mesh.z_limits)

    assert (mesh.find_triangle(R_test, z_test) == indices).all()
    assert isclose(mesh.interpolate(R_test, z_test, values), interpolated).all()
    assert isclose(mesh.build_interpolator_matrix(R_test, z_test), matrix).all()


def test_plot_field(mesh):
    # generate a random test field using a gaussian process
    distance = sqrt(
        (mesh.R[:, None] - mesh.R[None, :]) ** 2
        + (mesh.z[:, None] - mesh.z[None, :]) ** 2
    )
    scale = 0.04
    covariance = sinc(distance / scale) ** 2
    field = multivariate_normal(zeros(mesh.R.size), covariance)
    field -= field.min()

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(1, 1, 1)
    mesh.plot_field(
        ax=ax,
        vertex_values=field,
        colormap="Blues",
        mesh_color="black",
        mesh_thickness=0.4,
    )
    plt.close()
