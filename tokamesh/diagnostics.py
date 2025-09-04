from numpy import sqrt, sin, cos, tan, ceil, dot, pi, cross
from numpy import array, linspace, meshgrid, ndarray, zeros


def radial_fan(
    n_lines: int,
    poloidal_position: tuple[float, float],
    phi: float,
    angle_range: tuple[float, float],
) -> tuple[ndarray, ndarray]:
    R, z = poloidal_position
    origins = zeros([n_lines, 3])
    origins[:, 0] = R * cos(phi)
    origins[:, 1] = R * sin(phi)
    origins[:, 2] = z

    theta_axis = linspace(*angle_range, n_lines) * (pi / 180)
    R_unit = cos(theta_axis)
    directions = zeros([n_lines, 3])
    directions[:, 0] = cos(phi) * R_unit
    directions[:, 1] = sin(phi) * R_unit
    directions[:, 2] = sin(theta_axis)
    return origins, directions


def conical_ray_bundle(
    origin: ndarray, direction: ndarray, angular_radius=10.0
) -> tuple[ndarray, ndarray]:
    u0 = origin
    du = direction
    phi = angular_radius * pi / 360.0

    x_angles, y_angles = unit_circle_hexgrid(order=3)
    x_angles *= phi
    y_angles *= phi

    # make sure the direction is normalised
    du /= sqrt(dot(du, du))

    # find the first perpendicular
    K = du[1] / du[0]
    b = 1.0 / sqrt(K**2 + 1.0)
    a = -K * b
    p1 = array([a, b, 0.0])

    # use cross-product to find second perpendicular
    p2 = cross(du, p1)

    # calculate the ray directions
    tan_x = tan(x_angles)
    tan_y = tan(y_angles)
    norm = sqrt(1 + tan_x**2 + tan_y**2)
    v = du[None, :] + tan_x[:, None] * p1[None, :] + tan_y[:, None] * p2[None, :]
    ray_directions = v / norm[:, None]
    ray_origins = zeros(ray_directions.shape) + u0[None, :]
    return ray_origins, ray_directions


def unit_circle_hexgrid(order: int) -> tuple[ndarray, ndarray]:
    eps = 1e-12
    # create a square grid and offset every other row to make hexagons
    grid_size = int(ceil(2 * order / sqrt(3)))
    grid_size = grid_size + 1 if grid_size % 2 == 0 else grid_size
    k = linspace(-grid_size, grid_size, 2 * grid_size + 1)
    x, y = meshgrid(k, k)
    y[:, ::2] += 0.5
    # re-scale x-axis to make the hexagons regular
    x *= sqrt(3) * 0.5
    # unravel the grid and shrink it to fit the unit circle
    x = x.flatten() / order
    y = y.flatten() / order
    # remove any points outside the unit circle
    r_sqr = x**2 + y**2
    inside = r_sqr <= (1 + eps)
    return x[inside], y[inside]
