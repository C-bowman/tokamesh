
from numpy import sin, cos, pi, isclose
from numpy.random import uniform
from tokamesh import TriangularMesh
from tokamesh.construction import equilateral_mesh


def test_interpolate():
    # create a test mesh
    scale = 0.1
    R, z, triangles = equilateral_mesh(
        R_range=(0., 1.),
        z_range=(0., 1.),
        resolution=scale
    )
    # perturb the mesh to make it non-uniform
    L = uniform(0., 0.3 * scale, size=R.size)
    theta = uniform(0., 2 * pi, size=R.size)
    R += L * cos(theta)
    z += L * sin(theta)
    # build the mesh
    mesh = TriangularMesh(R=R, z=z, triangles=triangles)

    # As barycentric interpolation is linear, if we use a plane as the test
    # function, it should agree nearly exactly with interpolation result.
    def plane(x, y):
        return 0.37 - 5.31*x + 2.09*y

    vertex_values = plane(R, z)
    # create a series of random test-points
    R_test = uniform(0.2, 0.8, size=50)
    z_test = uniform(0.2, 0.8, size=50)
    # check the exact and interpolated values are equal
    interpolated = mesh.interpolate(R_test, z_test, vertex_values)
    exact = plane(R_test, z_test)
    assert isclose(interpolated, exact).all()
