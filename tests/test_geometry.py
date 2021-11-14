
from numpy import linspace, array, zeros, sqrt, sinc
from scipy.sparse import csc_matrix
from numpy.random import multivariate_normal
from scipy.integrate import quad, simps

from tokamesh.construction import equilateral_mesh
from tokamesh import TriangularMesh
from tokamesh.geometry import Camera, BarycentricGeometryMatrix
from tokamesh.geometry import radius_hyperbolic_integral


def test_BarycentricGeometryMatrix():
    # build a basic mesh out of equilateral triangles
    R, z, triangles = equilateral_mesh(resolution=0.01, R_range=[0, 0.2], z_range=[0, 0.2])
    mesh = TriangularMesh(R, z, triangles)

    # generate a random test field using a gaussian process
    distance = sqrt((R[:, None] - R[None, :]) ** 2 + (z[:, None] - z[None, :]) ** 2)
    scale = 0.04
    covariance = sinc(distance / scale) ** 2
    field = multivariate_normal(zeros(R.size), covariance)
    field -= field.min()

    # generate a synthetic camera to image the field
    cam_position = array([0.17, 0.19, 0.18])
    cam_direction = array([-0.1, -0.1, -0.06])
    cam = Camera(position=cam_position, direction=cam_direction, max_distance=1., num_x=5, num_y=15)

    # calculate the geometry matrix data
    BGM = BarycentricGeometryMatrix(
        R=R,
        z=z,
        triangles=triangles,
        ray_origins=cam.ray_starts,
        ray_ends=cam.ray_ends)

    matrix_data = BGM.calculate()

    # extract the data and build a sparse matrix
    entry_values = matrix_data['entry_values']
    row_values = matrix_data['row_indices']
    col_values = matrix_data['col_indices']
    shape = matrix_data['shape']
    G = csc_matrix((entry_values, (row_values, col_values)), shape=shape)

    # get the geometry matrix prediction of the line-integrals
    matrix_integrals = G.dot(field)

    # manually calculate the line integrals for comparison
    L = linspace(0, 0.5, 3000)  # build a distance axis for the integrals
    R_projection, z_projection = cam.project_rays(L)  # get the position of each ray at each distance
    # directly integrate along each ray
    direct_integrals = zeros(R_projection.shape[1])
    for i in range(R_projection.shape[1]):
        samples = mesh.interpolate(R_projection[:, i], z_projection[:, i], vertex_values=field)
        direct_integrals[i] = simps(samples, x=L)

    assert (abs(direct_integrals - matrix_integrals) < 1e-3).all()


def test_radius_hyperbolic_integral():
    # testing analytic integral solution of following function:
    def R(l, l_tan, R_tan_sq, sqrt_q2):
        return sqrt((sqrt_q2*(l - l_tan))**2 + R_tan_sq)

    l1, l2 = (0.1, 1.7)  # start / end points
    l_tan = 1.2  # line distance of tangency point
    R_tan_sq = 0.2  # squared major radius of tangency point
    sqrt_q2 = 0.35  # sqrt of quadratic coefficient of the ray
    # numerically evaluate the integral
    numeric, err = quad(func=R, a=l1, b=l2, args=(l_tan, R_tan_sq, sqrt_q2))
    # get the analytic solution
    exact = radius_hyperbolic_integral(l1, l2, l_tan, R_tan_sq, sqrt_q2)
    # confirm that they agree
    assert abs(exact - numeric) < 2*abs(err)
