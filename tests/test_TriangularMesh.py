import pytest
from numpy import arange, array, sin, cos, pi, isclose, ones
from numpy.random import uniform, seed
from tokamesh import TriangularMesh
from tokamesh.construction import equilateral_mesh

@pytest.fixture
def mesh():
    # create a test mesh
    scale = 0.1
    R, z, triangles = equilateral_mesh(
        R_range=(0., 1.),
        z_range=(0., 1.),
        resolution=scale
    )
    # perturb the mesh to make it non-uniform
    seed(1)
    L = uniform(0., 0.3 * scale, size=R.size)
    theta = uniform(0., 2 * pi, size=R.size)
    R += L * cos(theta)
    z += L * sin(theta)
    # build the mesh
    return TriangularMesh(R=R, z=z, triangles=triangles)


def test_TriangularMesh_input_types():
    with pytest.raises(TypeError):
        TriangularMesh(ones(5), ones(5), [1, 2, 3])


def test_TriangularMesh_vertices_shape():
    with pytest.raises(ValueError):
        TriangularMesh(ones([5,2]), ones(5), ones([4,3]))


def test_TriangularMesh_inconsistent_sizes():
    with pytest.raises(ValueError):
        TriangularMesh(ones(6), ones(5), ones([4,3]))


def test_TriangularMesh_triangles_shape():
    with pytest.raises(ValueError):
        TriangularMesh(ones([5,2]), ones(5), ones([4,3,2]))


def test_interpolate(mesh):
    # As barycentric interpolation is linear, if we use a plane as the test
    # function, it should agree nearly exactly with interpolation result.
    def plane(x, y):
        return 0.37 - 5.31*x + 2.09*y

    vertex_values = plane(mesh.R, mesh.z)
    # create a series of random test-points
    R_test = uniform(0.2, 0.8, size=50)
    z_test = uniform(0.2, 0.8, size=50)
    # check the exact and interpolated values are equal
    interpolated = mesh.interpolate(R_test, z_test, vertex_values)
    assert isclose(interpolated, plane(R_test, z_test)).all()

    # now test multi-dimensional inputs
    R_test = uniform(0.2, 0.8, size=[12, 3, 8])
    z_test = uniform(0.2, 0.8, size=[12, 3, 8])
    # check the exact and interpolated values are equal
    interpolated = mesh.interpolate(R_test, z_test, vertex_values)
    assert isclose(interpolated, plane(R_test, z_test)).all()

    # now test giving just floats
    interpolated = mesh.interpolate(0.31, 0.54, vertex_values)
    assert isclose(interpolated, plane(0.31, 0.54)).all()


def test_interpolate_inconsistent_shapes(mesh):
    with pytest.raises(ValueError):
        mesh.interpolate(ones([2, 1]), ones([2, 3]), ones(mesh.n_vertices))


def test_interpolate_vertices_size(mesh):
    with pytest.raises(ValueError):
        mesh.interpolate(ones(5), ones(5), ones(mesh.n_vertices-2))


def test_interpolate_vertices_type(mesh):
    with pytest.raises(TypeError):
        mesh.interpolate(ones(5), ones(5), [1.]*mesh.n_vertices)


def test_find_triangle(mesh):
    # first check all the barycentres
    R_centre = mesh.R[mesh.triangle_vertices].mean(axis=1)
    z_centre = mesh.z[mesh.triangle_vertices].mean(axis=1)
    tri_inds = mesh.find_triangle(R_centre, z_centre)
    assert (tri_inds == arange(mesh.triangle_vertices.shape[0])).all()

    # now check some points outside the mesh
    R_outside = array([mesh.R[mesh.z.argmax()]-1e-4, mesh.R_limits[1] + 1.])
    z_outside = array([mesh.z.max(), mesh.z_limits[1] + 1.])
    tri_inds = mesh.find_triangle(R_outside, z_outside)
    assert (tri_inds == -1).all()
    # ideally we should find a way to generate lots of random
    # points just outside the boundary of the mesh to check.


def test_find_triangle_inconsistent_shapes(mesh):
    with pytest.raises(ValueError):
        mesh.find_triangle(ones([2, 1]), ones([2, 3]))
