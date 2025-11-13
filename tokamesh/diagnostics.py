from numpy import sqrt, sin, cos, tan, ceil, dot, pi, cross, inf
from numpy import array, arange, full, linspace, meshgrid, ndarray, zeros
from numpy import isfinite, minimum, stack, concatenate, atleast_1d
from scipy.sparse import csc_matrix
from tokamesh.geometry import RayCollection


def radial_fan(
    poloidal_position: tuple[float, float],
    phi: float,
    n_lines: int = 8,
    angle_range: tuple[float, float] = (130.0, 230.0),
    angles: ndarray[float] = None,
) -> tuple[ndarray, ndarray]:
    """
    Generate a fan of lines-of-sight which are purely radial (i.e. have a constant
    toroidal angle).

    :param poloidal_position: \
        The ``(R, z)`` position of the origin point of the fan as a tuple of two floats.

    :param phi: \
        The toroidal angle of the fan origin point.

    :param n_lines: \
        The number of lines-of-sight in the fan.

    :param angle_range: \
        The range of angles in the poloidal plane over which the fan is spread as
        a tuple of two floats.

    :param angles: \
        An array of angles (in degrees) for each line-of-sight. If provided, this
        argument overrides the ``n_lines`` and ``angle_range`` arguments.

    :return: \
        The ray origins and the ray directions as a pair of numpy arrays.
    """
    R, z = poloidal_position
    origins = zeros([n_lines, 3])
    origins[:, 0] = R * cos(phi)
    origins[:, 1] = R * sin(phi)
    origins[:, 2] = z

    if angles is None:
        angles = linspace(*angle_range, n_lines)

    theta_axis = angles * (pi / 180)
    R_unit = cos(theta_axis)
    directions = zeros([n_lines, 3])
    directions[:, 0] = cos(phi) * R_unit
    directions[:, 1] = sin(phi) * R_unit
    directions[:, 2] = sin(theta_axis)
    return origins, directions


def tangential_fan(
    poloidal_position: tuple[float, float],
    phi: float,
    n_lines: int = 8,
    angle_range: tuple[float, float] = (0.0, 60.0),
    angles: ndarray[float] = None,
) -> tuple[ndarray, ndarray]:
    """
    Generate a fan of lines-of-sight with different tangency radii but a constant
    z-height value.

    :param poloidal_position: \
        The ``(R, z)`` position of the origin point of the fan as a tuple of two floats.

    :param phi: \
        The toroidal angle of the fan origin point.

    :param n_lines: \
        The number of lines-of-sight in the fan.

    :param angle_range: \
        The range of angles (in degrees) between the inward major radius vector and the
        lines of sight. Angles of 0 and 90 degrees therefore correspond to purely radial
        and purely tangential lines of sight respectively.

    :param angles: \
        An array of angles (in degrees) for each line-of-sight. If provided, this
        argument overrides the ``n_lines`` and ``angle_range`` arguments.

    :return: \
        The ray origins and the ray directions as a pair of numpy arrays.
    """
    if angles is None:
        angles = linspace(*angle_range, n_lines)

    R, z = poloidal_position
    inward = [-cos(phi), -sin(phi)]

    directions = zeros([angles.size, 3])
    origins = zeros([angles.size, 3])
    origins[:, 0] = R * cos(phi)
    origins[:, 1] = R * sin(phi)
    origins[:, 2] = z

    rads = angles * pi / 180
    cos_t = cos(rads)
    sin_t = sin(rads)
    directions[:, 0] = inward[0] * cos_t - inward[1] * sin_t
    directions[:, 1] = inward[0] * sin_t + inward[1] * cos_t

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


def find_ray_boundary_intersections(
    R_boundary: ndarray,
    z_boundary: ndarray,
    ray_origins: ndarray,
    ray_directions: ndarray,
    min_distance: float = 0.0,
) -> ndarray:
    """
    Calculate the end-points of a series of lines-of-sight based on the first
    position along each line which intersects with a polygon representing
    the machine boundary.

    :param R_boundary: \
        The major radius values of the boundary polygon as a 1D numpy array.

    :param z_boundary: \
        The z-height values of the boundary polygon as a 1D numpy array.

    :param ray_origins: \
        The ``(x,y,z)`` position vectors of the origin of each ray (i.e. line-of-sight)
        as a 2D numpy array. The array must have shape ``(M,3)`` where ``M`` is the
        total number of rays.

    :param ray_directions: \
        The ``(x,y,z)`` direction unit-vectors of each ray (i.e. line-of-sight)
        as a 2D numpy array. The array must have shape ``(M,3)`` where ``M`` is the
        total number of rays.

    :param min_distance: \
        The minimum allowed distance of an intersection point from the ray origin.
        Any intersections with the boundary at a distance less than this value
        are ignored.
    """

    valid_rays = (
        ray_origins.ndim == ray_directions.ndim == 2
        and ray_origins.shape[0] == ray_directions.shape[0]
        and ray_origins.shape[1] == ray_directions.shape[1] == 3
    )
    if not valid_rays:
        raise ValueError(
            f"""\n
            \r[ find_ray_boundary_intersections error ]
            \r>> 'ray_origins' and 'ray_directions' must both have shape (N, 3)
            \r>> where 'N' is the number of rays. Instead, they have shapes
            \r>> {ray_origins.shape}, {ray_directions.shape}
            \r>> respectively.
            """
        )

    rays = RayCollection(origins=ray_origins, directions=ray_directions)

    # ensure that the given boundary forms a closed loop
    if (R_boundary[0] != R_boundary[-1]) or (z_boundary[0] != z_boundary[-1]):
        R_boundary = concatenate([R_boundary, atleast_1d(R_boundary[0])])
        z_boundary = concatenate([z_boundary, atleast_1d(z_boundary[0])])

    # pair-up adjacent boundary points to form the edges
    R_edges = stack([R_boundary[:-1], R_boundary[1:]], axis=1)
    z_edges = stack([z_boundary[:-1], z_boundary[1:]], axis=1)
    n_edges = R_edges.shape[0]

    # pre-calculate the properties of each edge
    R_edge_mid = R_edges.mean(axis=1)
    z_edge_mid = z_edges.mean(axis=1)
    edge_lengths = sqrt(
        (R_edges[:, 0] - R_edges[:, 1]) ** 2 + (z_edges[:, 0] - z_edges[:, 1]) ** 2
    )
    edge_drn = zeros([n_edges, 2])
    edge_drn[:, 0] = R_edges[:, 1] - R_edges[:, 0]
    edge_drn[:, 1] = z_edges[:, 1] - z_edges[:, 0]
    edge_drn /= edge_lengths[:, None]

    # for edges in the mesh which are very close to horizontal, the regular
    # intersection calculation becomes inaccurate. Here we set a lower-limit
    # on the allowed size of the z-component of the edge unit vector. Edges
    # with a z-component below this limit will instead use a different
    # intersection calculation for horizontal edges.
    min_z_component = 1e-12

    # we are looking for the intersection with the shortest distance from the origin
    # which is also above the set minimum distance. Using this function to update
    # the intersections as we loop through boundary edges ensures the correct result.
    def update_intersections(current_isec: ndarray, new_isec: ndarray):
        valid = isfinite(new_isec) & (new_isec > min_distance)
        current_isec[valid] = minimum(current_isec[valid], new_isec[valid])

    intersections = full(rays.size, fill_value=inf, dtype=float)
    for edge in range(n_edges):
        R0 = R_edge_mid[edge]
        z0 = z_edge_mid[edge]
        uR, uz = edge_drn[edge, :]
        w = edge_lengths[edge]

        # if the edge is horizontal, a simplified method can be used
        if abs(uz) < min_z_component:
            new_intersects = rays.horizontal_hyperbola_intersections(R0, z0, w)
            update_intersections(intersections, new_intersects)

        else:  # else we need the general intersection calculation
            new_intersects = rays.edge_hyperbola_intersections(R0, z0, uR, uz, w)
            update_intersections(intersections, new_intersects[:, 0])
            update_intersections(intersections, new_intersects[:, 1])

    # finally project the rays to the calculated intersection distances
    # to calculate the end-points
    ray_ends = rays.origins + rays.directions * intersections[:, None]
    return ray_ends


def line_integration_matrix(
    origins: ndarray, endpoints: ndarray, target_resolution: float
) -> tuple[ndarray, ndarray, csc_matrix]:
    """
    Generate a series of (R, z) points to be used in calculating line-integrals along
    a set of given lines-of-sight, and a corresponding sparse-matrix operator which
    computes the integrals when multiplied with a vector of field values at those
    (R, z) positions.

    :param origins: \
        The ``(x,y,z)`` position vectors of the origin of each ray (i.e. line-of-sight)
        as a 2D numpy array. The array must have shape ``(M,3)`` where ``M`` is the
        total number of rays.

    :param endpoints: \
        The ``(x,y,z)`` position vectors of the end-point of each ray (i.e. line-of-sight)
        as a 2D numpy array. The array must have shape ``(M,3)`` where ``M`` is the
        total number of rays.

    :param target_resolution: \
        The target distance between integration points for each line-of-sight.
    """
    entries = []
    rows = []
    R = []
    z = []
    distances = sqrt(((endpoints - origins) ** 2).sum(axis=1))
    n_lines = origins.shape[0]

    # loop over each line-of-sight
    for i in range(n_lines):
        # split the line into equal segments close to the target resolution
        n_points = int(ceil(distances[i] / target_resolution)) + 1
        dl = distances[i] / (n_points - 1)

        # get the (R, z) coordinates of the integration points
        fader = linspace(0, 1, n_points)
        line = (
            origins[i, :][:, None] * (1 - fader[None, :])
            + fader[None, :] * endpoints[i, :][:, None]
        )
        R.append(sqrt(line[0, :] ** 2 + line[1, :] ** 2))
        z.append(line[2, :])

        # build the trapezium integration weights
        weights = full(n_points, fill_value=dl)
        weights[0] = dl * 0.5
        weights[-1] = dl * 0.5
        entries.append(weights)

        # build the row indices for the current line
        rows.append(full(n_points, fill_value=i, dtype=int))

    # combine the data for all lines into single arrays
    R = concatenate(R)
    z = concatenate(z)
    entries = concatenate(entries)
    rows = concatenate(rows)

    total_points = R.size
    cols = arange(total_points, dtype=int)

    # build a sparse matrix which computes the line-integrals
    total_points = R.size
    matrix_shape = (n_lines, total_points)
    integration_matrix = csc_matrix((entries, (rows, cols)), shape=matrix_shape)
    return R, z, integration_matrix
