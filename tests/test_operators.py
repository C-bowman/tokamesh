from numpy import arange, zeros, sin, exp, sqrt, array, isclose
from numpy.random import default_rng
from tokamesh.operators import parallel_derivative, perpendicular_derivative
from tokamesh.operators import edge_difference_matrix, umbrella_matrix


def single_hexagon_mesh(scale=1.0):
    a = 0.5 * sqrt(3)
    unit_hexagon = [
        (0.0, 0.0),
        (0.0, 1.0),
        (a, 0.5),
        (a, -0.5),
        (0.0, -1),
        (-a, -0.5),
        (-a, 0.5),
    ]
    triangles = array([[0, i, i % 6 + 1] for i in range(1, 7)])
    R, z = [array([p[i] for p in unit_hexagon]) for i in [0, 1]]
    R *= scale
    z *= scale
    return R, z, triangles


def test_edge_difference_matrix():
    R, z, triangles = single_hexagon_mesh(scale=2.0)
    vertex_vals = zeros(7)
    vertex_vals[0] = 2.0
    operator = edge_difference_matrix(R=R, z=z, triangles=triangles)
    diffs = operator @ vertex_vals
    assert ((diffs == 0.0) | (diffs == 2.0)).all()

    operator = edge_difference_matrix(R=R, z=z, triangles=triangles, normalised=True)
    diffs = operator @ vertex_vals
    assert (isclose(diffs, 0.0) | isclose(diffs, 1.0)).all()


def test_umbrella_matrix():
    R, z, triangles = single_hexagon_mesh(scale=2.0)
    # making vertex values lie in a plane should make umbrella matrix return zeros
    vertex_vals = R * 1.7 - z * 0.4
    operator = umbrella_matrix(R=R, z=z, triangles=triangles)

    diffs = operator @ vertex_vals
    assert (diffs == 0.0).all()

    # perturb the vertex positions and check the inverse-distance-weighted
    # operator returns a result closer to zero than the un-weighted operator
    rng = default_rng(123)
    R = rng.normal(loc=R, scale=0.2)
    z = rng.normal(loc=z, scale=0.2)
    vertex_vals = R * 1.7 - z * 0.4

    weighted_operator = umbrella_matrix(
        R=R, z=z, triangles=triangles, inverse_distance_weighting=True
    )
    diffs = operator @ vertex_vals
    weighted_diffs = weighted_operator @ vertex_vals
    assert abs(diffs[0]) > abs(weighted_diffs[0])


def test_field_aligned_operators():
    # first create a rectangular grid with irregular spacing
    n, m = 31, 29
    R_axis = (2 + sin(arange(n))).cumsum()
    R_axis = (R_axis - R_axis[0]) / (R_axis - R_axis[0]).max()
    z_axis = (2 + sin(arange(m))).cumsum()
    z_axis = (z_axis - z_axis[0]) / (z_axis - z_axis[0]).max()

    # generate R and z at each vertex, and the index grid
    R = zeros(n * m)
    z = zeros(n * m)
    inds = arange(n * m).reshape((n, m))
    for i in range(n):
        for j in range(m):
            k = inds[i, j]
            R[k] = R_axis[i]
            z[k] = z_axis[j]

    # define a test field at each vertex and find the derivatives
    field = exp(-R) + exp(-z)
    d1_para_analytic = -exp(-R)
    d1_perp_analytic = -exp(-z)
    d2_para_analytic = exp(-R)
    d2_perp_analytic = exp(-z)

    # estimate the derivatives via finite difference
    d1_para = parallel_derivative(R=R, z=z, index_grid=inds, order=1) @ field
    d1_perp = perpendicular_derivative(R=R, z=z, index_grid=inds, order=1) @ field
    d2_para = parallel_derivative(R=R, z=z, index_grid=inds, order=2) @ field
    d2_perp = perpendicular_derivative(R=R, z=z, index_grid=inds, order=2) @ field

    # check that the findiff estimates agree with the exact values
    assert abs(d1_para / d1_para_analytic - 1).mean() < 1e-3
    assert abs(d1_perp / d1_perp_analytic - 1).mean() < 1e-3
    assert abs(d2_para / d2_para_analytic - 1).mean() < 1e-2
    assert abs(d2_perp / d2_perp_analytic - 1).mean() < 1e-2
