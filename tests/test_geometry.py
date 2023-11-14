from numpy import sqrt, sinc, sin, cos, pi, exp
from numpy import array, linspace, zeros, array_equal, piecewise
from scipy.sparse import csc_matrix
from numpy.random import multivariate_normal
from scipy.integrate import quad, simps

from tokamesh.construction import equilateral_mesh
from tokamesh import TriangularMesh
from tokamesh.geometry import BarycentricGeometryMatrix, linear_geometry_matrix
from tokamesh.geometry import radius_hyperbolic_integral
from tokamesh.utilities import build_edge_map, Camera


def test_BarycentricGeometryMatrix():
    # build a basic mesh out of equilateral triangles
    R, z, triangles = equilateral_mesh(
        resolution=0.01, R_range=[0, 0.2], z_range=[0, 0.2]
    )
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
    cam = Camera(
        position=cam_position,
        direction=cam_direction,
        max_distance=1.0,
        num_x=5,
        num_y=15,
    )

    # calculate the geometry matrix data
    BGM = BarycentricGeometryMatrix(
        R=R, z=z, triangles=triangles, ray_origins=cam.ray_starts, ray_ends=cam.ray_ends
    )

    matrix_data = BGM.calculate()

    # extract the data and build a sparse matrix
    entry_values = matrix_data["entry_values"]
    row_values = matrix_data["row_indices"]
    col_values = matrix_data["col_indices"]
    shape = matrix_data["shape"]
    G = csc_matrix((entry_values, (row_values, col_values)), shape=shape)

    # get the geometry matrix prediction of the line-integrals
    matrix_integrals = G.dot(field)

    # manually calculate the line integrals for comparison
    L = linspace(0, 0.5, 3000)  # build a distance axis for the integrals
    # get the position of each ray at each distance
    R_projection, z_projection = cam.project_rays(L)
    # directly integrate along each ray
    direct_integrals = zeros(R_projection.shape[1])
    for i in range(R_projection.shape[1]):
        samples = mesh.interpolate(
            R_projection[:, i], z_projection[:, i], vertex_values=field
        )
        direct_integrals[i] = simps(samples, x=L)

    assert (abs(direct_integrals - matrix_integrals) < 1e-3).all()


def test_radius_hyperbolic_integral():
    # testing analytic integral solution of following function:
    def R(l, l_tan, R_tan_sq, sqrt_q2):
        return sqrt((sqrt_q2 * (l - l_tan)) ** 2 + R_tan_sq)

    l1, l2 = (0.1, 1.7)  # start / end points
    l_tan = 1.2  # line distance of tangency point
    R_tan_sq = 0.2  # squared major radius of tangency point
    sqrt_q2 = 0.35  # sqrt of quadratic coefficient of the ray
    # numerically evaluate the integral
    numeric, err = quad(func=R, a=l1, b=l2, args=(l_tan, R_tan_sq, sqrt_q2))
    # get the analytic solution
    exact = radius_hyperbolic_integral(l1, l2, l_tan, R_tan_sq, sqrt_q2)
    # confirm that they agree
    assert abs(exact - numeric) < 2 * abs(err)


def test_build_edge_map():
    triangles = array(((0, 1, 2), (3, 1, 2), (4, 3, 2)))

    triangle_edges, edge_vertices, edge_map = build_edge_map(triangles)

    expected_triangle_edges = array(((0, 1, 2), (3, 1, 4), (5, 4, 6)))
    expected_edge_vertices = array(
        ((0, 1), (1, 2), (0, 2), (1, 3), (2, 3), (3, 4), (2, 4))
    )
    expected_edge_map = {0: [0], 1: [0, 1], 2: [0], 3: [1], 4: [1, 2], 5: [2], 6: [2]}

    assert array_equal(triangle_edges, expected_triangle_edges)
    assert array_equal(edge_vertices, expected_edge_vertices)
    assert edge_map == expected_edge_map


def test_linear_geometry_matrix():
    class LinearBF:
        def __init__(self, a, b, c):
            self.a, self.b, self.c = a, b, c
            self.m1 = 1 / (b - a)
            self.c1 = -a * self.m1
            self.m2 = -1 / (c - b)
            self.c2 = -c * self.m2

        def __call__(self, x):
            conds = [(self.a <= x) & (x < self.b), (self.b <= x) & (x < self.c)]
            funcs = [self.lhs, self.rhs, lambda z: z * 0.0]
            return piecewise(x, conds, funcs)

        def lhs(self, x):
            return self.m1 * x + self.c1

        def rhs(self, x):
            return self.m2 * x + self.c2

    def solve_quadratic(a, b, c):
        descrim = b**2 - 4 * a * c
        if descrim >= 0:
            t1, t2 = -0.5 * b / a, 0.5 * sqrt(descrim) / a
            return t1 - t2, t1 + t2
        else:
            return []

    # define the rays
    n_rays = 100
    ray_angles = linspace(0, 0.275 * pi, n_rays)
    origins = zeros([n_rays, 3])
    origins[:, 0] = -1.95
    origins[:, 1] = 0.05
    origins[:, 2] = 0.01

    rays = zeros([n_rays, 3])
    rays[:, 0] = cos(ray_angles)
    rays[:, 1] = sin(ray_angles)

    # calculate ray lengths from centre-column / wall intersections
    lengths = zeros(n_rays)
    for i in range(n_rays):
        q2 = rays[i, 0] ** 2 + rays[i, 1] ** 2
        q1 = 2 * (origins[i, 0] * rays[i, 0] + origins[i, 1] * rays[i, 1])
        q0 = origins[i, 0] ** 2 + origins[i, 1] ** 2
        col_roots = solve_quadratic(a=q2, b=q1, c=q0 - 0.2**2)
        wall_roots = solve_quadratic(a=q2, b=q1, c=q0 - 2**2)
        roots = [*col_roots, *wall_roots]
        lengths[i] = min([r for r in roots if r > 0.0])
    ends = origins + rays * lengths[:, None]

    # define the emission as a function of the radius
    n_points = 50
    R = linspace(0.25, 1.6, n_points)
    z1 = (R - 1.4) / 0.05
    z2 = (R - 0.35) / 0.02
    emission = exp(-0.5 * z1**2) + 0.5 * exp(-0.5 * z2**2) + 0.005

    # Use geometry matrix to calculate the analytic integral result
    G = linear_geometry_matrix(R=R, ray_origins=origins, ray_ends=ends)
    analytic_integral = G.dot(emission)

    # Directly integrate over the basis functions to get the numerical integral
    basis = [LinearBF(R[0] - 1e-5, R[0], R[1])]
    basis.extend([LinearBF(R[i - 1], R[i], R[i + 1]) for i in range(1, n_points - 1)])
    basis.append(LinearBF(R[-2], R[-1], R[-1] + 1e-5))
    numerical_integral = zeros(n_rays)
    for i in range(n_rays):
        L = linspace(0, lengths[i], 5000)
        x_ray = rays[i, 0] * L + origins[i, 0]
        y_ray = rays[i, 1] * L + origins[i, 1]
        R = sqrt(x_ray**2 + y_ray**2)
        F = sum(w * lbf(R) for lbf, w in zip(basis, emission))
        numerical_integral[i] = simps(F, x=L)

    # check that analytic integral agrees with numerical one
    assert abs(analytic_integral - numerical_integral).max() < 1e-5
