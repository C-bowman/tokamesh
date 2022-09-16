from numpy import arange, zeros, sin, exp
from tokamesh.operators import parallel_derivative, perpendicular_derivative


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
