from numpy import array, concatenate, sqrt, exp
from scipy.integrate import quad
from tokamesh.tokamaks import mastu_boundary
from tokamesh.utilities import Camera
from tokamesh.construction import Polygon
from tokamesh.diagnostics import find_ray_boundary_intersections
from tokamesh.diagnostics import line_integration_matrix
from tokamesh.diagnostics import radial_fan, tangential_fan


def test_ray_boundary_intersections():
    # get some rays using a synthetic camera
    cam_position = array([1.3, 1.1, -0.5])
    cam_direction = array([-0.1, -0.15, 0.04])

    cam = Camera(
        position=cam_position,
        direction=cam_direction,
        max_distance=10.0,
        num_x=5,
        num_y=5,
        fov=60.0,
    )

    # generate both an upper and lower fan of lines on the outboard side of MAST-U
    upper_fan_origins, upper_fan_directions = radial_fan(
        n_lines=8, poloidal_position=(1.75, 0.85), phi=0.3, angle_range=(180, 235)
    )

    lower_fan_origins, lower_fan_directions = radial_fan(
        n_lines=8, poloidal_position=(1.75, -0.85), phi=0.3, angle_range=(125, 180)
    )

    tan_origins, tan_directions = tangential_fan(
        n_lines=8, poloidal_position=(1.6, 0.05), phi=0.3, angle_range=(0, 60)
    )

    # concatenate the upper and lower lines into a single group
    fan_origins = concatenate([upper_fan_origins, lower_fan_origins], axis=0)
    fan_directions = concatenate([upper_fan_directions, lower_fan_directions], axis=0)

    ray_sets = [
        (cam.ray_starts, cam.ray_directions),
        (fan_origins, fan_directions),
        (tan_origins, tan_directions),
    ]

    # Create a boundary polygon to check distances to the wall
    R_bnd, z_bnd = mastu_boundary()
    poly = Polygon(x=R_bnd, y=z_bnd)

    for ray_origins, ray_directions in ray_sets:
        # find the endpoints of each ray
        endpoints = find_ray_boundary_intersections(
            R_boundary=R_bnd,
            z_boundary=z_bnd,
            ray_origins=ray_origins,
            ray_directions=ray_directions,
            min_distance=0.0,
        )

        # get the (R, z) locations of the ray endpoints
        R_ends = sqrt(endpoints[:, 0] ** 2 + endpoints[:, 1] ** 2)
        z_ends = endpoints[:, 2]

        # verify that the distances between the endpoints and the boundary are almost zero
        boundary_distances = poly.distance(x=R_ends, y=z_ends)
        assert (abs(boundary_distances) < 1e-12).all()


def test_line_integration_matrix():

    # define a gaussian field to test the integration through
    def gauss_test_field(R, z):
        rho_sqr = R**2 + z**2
        return exp(-0.5 * rho_sqr)

    def integration_test_func(l, v0, dv):
        v = v0 + l * dv
        return gauss_test_field(R=sqrt(v[0] ** 2 + v[1] ** 2), z=v[2])

    # get some rays using a synthetic camera
    cam_position = array([1.3, 1.1, -0.5])
    cam_direction = array([-0.1, -0.15, 0.04])

    cam = Camera(
        position=cam_position,
        direction=cam_direction,
        max_distance=10.0,
        num_x=3,
        num_y=3,
        fov=60.0,
    )

    directions = cam.ray_directions
    origins = cam.ray_starts

    # get endpoints of the rays for the MAST-U boundary
    R_bnd, z_bnd = mastu_boundary()
    endpoints = find_ray_boundary_intersections(
        R_boundary=R_bnd,
        z_boundary=z_bnd,
        ray_origins=cam.ray_starts,
        ray_directions=cam.ray_directions,
        min_distance=0.0,
    )

    # build the line-integrals
    R_integral, z_integral, integration_matrix = line_integration_matrix(
        origins=origins, endpoints=endpoints, target_resolution=0.01
    )

    # evaluate the line integrals using the matrix
    matrix_results = integration_matrix @ gauss_test_field(R_integral, z_integral)

    # evaluate the line integrals directly with quadrature
    distances = sqrt(((endpoints - origins) ** 2).sum(axis=1))
    quad_results = [
        quad(
            func=integration_test_func,
            a=0.0,
            b=distances[i],
            args=(origins[i, :], directions[i, :]),
        )[0]
        for i in range(origins.shape[0])
    ]

    quad_results = array(quad_results)

    # compare the values to confirm they agree
    max_abs_frac_error = abs(quad_results / matrix_results - 1).max()
    assert max_abs_frac_error < 1e-4
