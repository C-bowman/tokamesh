from numpy import array, concatenate, sqrt
from tokamesh.tokamaks import mastu_boundary
from tokamesh.utilities import Camera
from tokamesh.construction import Polygon
from tokamesh.diagnostics import find_ray_boundary_intersections, radial_fan


def test_ray_boundary_intersections():
    # get some rays using a synthetic camera
    cam_position = array([1.3, 1.1, -0.5])
    cam_direction = array([-0.1, -0.15, 0.04])

    cam = Camera(
        position=cam_position,
        direction=cam_direction,
        max_distance=10.,
        num_x=5,
        num_y=5,
        fov=60.,
    )

    # generate both an upper and lower fan of lines on the outboard side of MAST-U
    upper_fan_origins, upper_fan_directions = radial_fan(
        n_lines=8, poloidal_position=(1.75, 0.85), phi=0.3, angle_range=(180, 235)
    )

    lower_fan_origins, lower_fan_directions = radial_fan(
        n_lines=8, poloidal_position=(1.75, -0.85), phi=0.3, angle_range=(125, 180)
    )

    # concatenate the upper and lower lines into a single group
    fan_origins = concatenate([upper_fan_origins, lower_fan_origins], axis=0)
    fan_directions = concatenate([upper_fan_directions, lower_fan_directions], axis=0)

    ray_sets = [
        (cam.ray_starts, cam.ray_directions),
        (fan_origins, fan_directions),
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
            min_distance=0.
        )

        # get the (R, z) locations of the ray endpoints
        R_ends = sqrt(endpoints[:, 0] ** 2 + endpoints[:, 1] ** 2)
        z_ends = endpoints[:, 2]

        # verify that the distances between the endpoints and the boundary are almost zero
        boundary_distances = poly.distance(x=R_ends, y=z_ends)
        assert (abs(boundary_distances) < 1e-12).all()
