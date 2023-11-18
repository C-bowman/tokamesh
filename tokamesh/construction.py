from numpy import sqrt, ceil, sin, cos, arctan2, diff, minimum, maximum, cumsum
from numpy import array, ones, zeros, full, linspace, arange, concatenate, vstack
from numpy import in1d, unique, isclose, nan, atleast_1d, intersect1d
from numpy import int64, ndarray
from warnings import warn

from tokamesh.utilities import build_edge_map


MeshData = tuple[ndarray, ndarray, ndarray]
floatpair = tuple[float, float]


def equilateral_mesh(
    R_range: floatpair,
    z_range: floatpair,
    resolution: float,
    rotation: float = None,
    pivot: floatpair = (0.0, 0.0),
) -> MeshData:
    """
    Construct a mesh from equilateral triangles which fills a rectangular region.

    :param R_range: \
        A tuple in the form ``(R_min, R_max)`` specifying the range of major radius
        values to cover with triangles.

    :param z_range: \
        A tuple in the form ``(z_min, z_max)`` specifying the range of z-height values
        to cover with triangles.

    :param float resolution: \
        The side-length of the triangles.

    :param rotation: \
        Angle (in radians) by which the mesh will be rotated.

    :param pivot: \
        Pivot point around which the rotation is applied.

    :return: \
        A tuple containing ``R_vert``, ``z_vert`` and ``triangles``.
        ``R_vert`` is the major radius of the vertices as a 1D array. ``z_vert`` the is
        z-height of the vertices as a 1D array. ``triangles`` is a 2D array of integers
        of shape ``(N,3)`` specifying the indices of the vertices which form each
        triangle in the mesh, where ``N`` is the total number of triangles.
    """
    # determine how many rows / columns of triangles to create
    N = int(ceil((R_range[1] - R_range[0]) / resolution))
    M = int(ceil((z_range[1] - z_range[0]) / (resolution * 0.5 * sqrt(3))))

    # create the vertices by producing a rectangular grid
    # and shifting every other row
    x_ax = linspace(0, N - 1, N) * resolution
    y_ax = linspace(0, M - 1, M) * resolution * 0.5 * sqrt(3)

    x = zeros([N, M])
    y = zeros([N, M])
    y[:, :] = y_ax[None, :] + z_range[0]
    x[:, :] = x_ax[:, None] + R_range[0]
    x[:, 1::2] += 0.5 * resolution

    # rotate the vertices around a point if requested
    if rotation is not None:
        x, y = rotate(x, y, rotation, pivot)

    # divide up the grid into triangles
    triangle_inds = []
    for i in range(N - 1):
        for j in range(M - 1):
            v1 = M * i + j
            v2 = M * (i + 1) + j
            v3 = M * i + j + 1
            v4 = M * (i + 1) + j + 1

            if j % 2 == 0:
                triangle_inds.append([v1, v2, v3])
                triangle_inds.append([v2, v3, v4])
            else:
                triangle_inds.append([v1, v3, v4])
                triangle_inds.append([v1, v2, v4])

    return x.flatten(), y.flatten(), array(triangle_inds)


def rotate(R, z, angle, pivot):
    """Rotate the point `(R, z)` anti-clockwise by `angle` about the point `pivot`"""
    d = sqrt((R - pivot[0]) ** 2 + (z - pivot[1]) ** 2)
    theta = arctan2(z - pivot[1], R - pivot[0]) + angle
    return d * cos(theta) + pivot[0], d * sin(theta) + pivot[1]


def trim_vertices(
    R: ndarray, z: ndarray, triangles: ndarray, trim_bools: ndarray
) -> MeshData:
    """
    Removes chosen vertices (and any triangles containing those vertices) from a mesh.

    :param R: \
        The major radius of each mesh vertex as a 1D numpy array.

    :param z: \
        The z-height of each mesh vertex as a 1D numpy array.

    :param triangles: \
        A 2D numpy array of integers specifying the indices of the vertices which form
        each of the triangles in the mesh. The array must have shape ``(N,3)`` where ``N``
        is the total number of triangles.

    :param trim_bools: \
        A 1D array of boolean values corresponding to the vertices, which is ``True`` for
        any vertices which are to be removed from the mesh.

    :return R, z, triangles: \
        The ``R``, ``z`` and ``triangles`` arrays (defined as described above) with the
        specified vertices removed.
    """
    vert_inds = (~trim_bools).nonzero()[0]
    tri_bools = in1d(triangles[:, 0], vert_inds)
    tri_bools &= in1d(triangles[:, 1], vert_inds)
    tri_bools &= in1d(triangles[:, 2], vert_inds)
    tri_inds = tri_bools.nonzero()[0]
    index_converter = zeros(R.size, dtype=int64)
    index_converter[vert_inds] = arange(vert_inds.size)
    trim_triangles = index_converter[triangles[tri_inds, :]]
    trim_triangles.sort(axis=1)
    return R[vert_inds], z[vert_inds], trim_triangles


class Polygon:
    """
    Class for evaluating whether a given point is inside a polygon,
    or the distance between it and the nearest point on the polygon.

    :param x: \
        The x-values of the polygon vertices as a 1D numpy array.

    :param y: \
        The y-values of the polygon vertices as a 1D numpy array.
    """

    def __init__(self, x, y):
        self.x = array(x)
        self.y = array(y)
        if (self.x[0] != self.x[-1]) or (self.y[0] != self.y[-1]):
            self.x = concatenate([self.x, atleast_1d(self.x[0])])
            self.y = concatenate([self.y, atleast_1d(self.y[0])])

        self.n = len(x)

        self.dx = diff(self.x)
        self.dy = diff(self.y)
        self.im = full(self.dx.size, fill_value=nan)
        self.c = full(self.dx.size, fill_value=nan)
        im_inds = (self.dy != 0.0).nonzero()[0]
        c_inds = (self.dx != 0.0).nonzero()[0]
        self.im[im_inds] = self.dx[im_inds] / self.dy[im_inds]
        self.c[c_inds] = (
            self.y[:-1][c_inds]
            - self.x[:-1][c_inds] * self.dy[c_inds] / self.dx[c_inds]
        )

        # pre-calculate the bounding rectangle of each edge for intersection testing
        self.x_upr = maximum(self.x[1:], self.x[:-1])
        self.x_lwr = minimum(self.x[1:], self.x[:-1])
        self.y_upr = maximum(self.y[1:], self.y[:-1])
        self.y_lwr = minimum(self.y[1:], self.y[:-1])

        # normalise the unit vectors
        self.lengths = sqrt(self.dx**2 + self.dy**2)
        self.dx /= self.lengths
        self.dy /= self.lengths

        self.zero_im = self.im == 0.0

    def is_inside(self, v):
        x, y = v
        k = (y - self.c) * self.im

        limits_check = (self.y_lwr < y) & (y < self.y_upr) & (x < self.x_upr)
        isec_check = (x < k) | self.zero_im
        intersections = (limits_check & isec_check).sum()
        return True if intersections % 2 == 1 else False

    def distance(self, v):
        x, y = v
        dx = x - self.x[:-1]
        dy = y - self.y[:-1]

        L = (dx * self.dx + dy * self.dy) / self.lengths
        D = dx * self.dy - dy * self.dx
        booles = (0 <= L) & (L <= 1)

        points_min = sqrt(dx**2 + dy**2).min()

        if booles.any():
            perp_min = abs(D[booles]).min()
            return min(perp_min, points_min)
        else:
            return points_min

    def diagnostic_plot(self):
        xmin = self.x.min()
        xmax = self.x.max()
        ymin = self.y.min()
        ymax = self.y.max()
        xpad = (xmax - xmin) * 0.15
        ypad = (ymax - ymin) * 0.15

        N = 200
        x_ax = linspace(xmin - xpad, xmax + xpad, N)
        y_ax = linspace(ymin - ypad, ymax + ypad, N)

        inside = zeros([N, N])
        distance = zeros([N, N])
        for i in range(N):
            for j in range(N):
                v = [x_ax[i], y_ax[j]]
                inside[i, j] = self.is_inside(v)
                distance[i, j] = self.distance(v)

        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(12, 4))
        ax1 = fig.add_subplot(131)
        ax1.contourf(x_ax, y_ax, inside.T)
        ax1.plot(self.x, self.y, ".-", c="white", lw=2)
        ax1.set_title("point is inside polygon")

        ax2 = fig.add_subplot(132)
        ax2.contourf(x_ax, y_ax, distance.T, 100)
        ax2.plot(self.x, self.y, ".-", c="white", lw=2)
        ax2.set_title("distance from polygon")

        ax3 = fig.add_subplot(133)
        ax3.contourf(x_ax, y_ax, (distance * inside).T, 100)
        ax3.plot(self.x, self.y, ".-", c="white", lw=2)
        ax3.set_title("interior point distance from polygon")

        plt.tight_layout()
        plt.show()


def find_boundaries(triangles: ndarray) -> list[ndarray]:
    """
    Find all the boundaries of a given mesh.

    :param triangles: \
        A 2D numpy array of integers specifying the indices of the vertices which form
        each of the triangles in the mesh. The array must have shape ``(N, 3)`` where
        ``N`` is the total number of triangles.

    :return: \
        A list of 1D numpy arrays containing the indices of the vertices in each boundary.
    """
    # Construct a mapping from triangles to edges, and edges to vertices
    triangle_edges, edge_vertices, _ = build_edge_map(triangles)
    # identify edges on the boundary by finding edges which are only part of one triangle
    unique_vals, counts = unique(triangle_edges, return_counts=True)
    boundary_edges_indices = (counts == 1).nonzero()[0]
    boundary_edges = edge_vertices[boundary_edges_indices, :]

    # now create a map between an edge, and the other edges to which it's connected
    boundary_connections = {}
    for i in range(boundary_edges.shape[0]):
        edges = (
            (boundary_edges[i, 0] == boundary_edges)
            | (boundary_edges[i, 1] == boundary_edges)
        ).nonzero()[0]
        boundary_connections[i] = [e for e in edges if e != i]

    # we use a set to keep track of which edges have already been used as part of a boundary
    unused_edges = {i for i in range(boundary_edges.shape[0])}

    # now follow the connections map to build the boundaries
    boundaries = []
    while len(unused_edges) > 0:
        current_boundary = [unused_edges.pop()]  # start at an arbitrary unused edge
        while True:
            connected_edges = boundary_connections[current_boundary[-1]]
            for edge in connected_edges:
                if edge in unused_edges:
                    current_boundary.append(edge)
                    unused_edges.remove(edge)
                    break
            else:
                break
        boundaries.append(boundary_edges_indices[current_boundary])

    _, edges_per_vertex = unique(boundary_edges, return_counts=True)
    if edges_per_vertex.max() > 2:
        warn(
            """\n
            [ find_boundaries warning ]
            >> The given mesh contains at least two sub-meshes which
            >> are connected by only one vertex. Currently, it is not
            >> guaranteed that find_boundaries will draw separate
            >> boundaries for each sub-mesh.
            """
        )

    # Now we need to convert the boundaries from edge indices to vertex indices
    vertex_boundaries = []
    for boundary in boundaries:
        # the order of the first two vertex indices needs to match the direction
        # in which the boundary is being traced.
        v1, v2 = edge_vertices[boundary[0], :]
        if v1 in edge_vertices[boundary[1], :]:
            vertex_boundary = [v2, v1]
        else:
            vertex_boundary = [v1, v2]

        # now loop over all the other edges and add the new vertex that appears
        for edge in boundary[1:]:
            v1, v2 = edge_vertices[edge, :]
            next_vertex = v1 if (v1 not in vertex_boundary) else v2
            vertex_boundary.append(next_vertex)

        vertex_boundaries.append(array(vertex_boundary))

    return vertex_boundaries


def build_central_mesh(
    R_boundary: ndarray,
    z_boundary: ndarray,
    resolution: float,
    padding_factor: float = 1.0,
    rotation: float = None,
) -> MeshData:
    """
    Generate an equilateral mesh which fills the space inside a given boundary,
    up to a chosen distance to the boundary edge.

    :param R_boundary: \
        The major-radius values of the boundary as a 1D numpy array.

    :param z_boundary: \
        The z-height values of the boundary as a 1D numpy array.

    :param resolution: \
        The side-length of the equilateral triangles.

    :param padding_factor: \
        A multiplicative factor which defines the minimum distance to the boundary
        such that ``min_distance = padding_factor * resolution``. No vertices in the
        returned mesh will be closer to the boundary than ``min_distance``.

    :param rotation: \
        Angle (in radians) by which the orientations of mesh triangles are rotated,
        relative to their default orientation.

    :return: \
        A tuple containing ``R_vert``, ``z_vert`` and ``triangles``.
        ``R_vert`` is the major-radius of the vertices as a 1D array. ``z_vert`` the is
        z-height of the vertices as a 1D array. ``triangles`` is a 2D array of integers
        of shape ``(N,3)`` specifying the indices of the vertices which form each triangle
        in the mesh, where ``N`` is the total number of triangles.
    """
    poly = Polygon(R_boundary, z_boundary)
    pad = 2 * 0.5 * sqrt(3) * resolution
    if rotation is None:
        R_range = (R_boundary.min() - pad, R_boundary.max() + pad)
        z_range = (z_boundary.min() - pad, z_boundary.max() + pad)
        R, z, triangles = equilateral_mesh(
            R_range=R_range, z_range=z_range, resolution=resolution
        )
    else:
        rot_R, rot_z = rotate(R_boundary, z_boundary, -rotation, [0.0, 0.0])
        R_range = (rot_R.min() - pad, rot_R.max() + pad)
        z_range = (rot_z.min() - pad, rot_z.max() + pad)
        R, z, triangles = equilateral_mesh(
            R_range=R_range, z_range=z_range, resolution=resolution
        )
        R, z = rotate(R, z, rotation, [0.0, 0.0])

    # remove all triangles which are too close to or inside walls
    bools = array(
        [
            poly.is_inside(p) * poly.distance(p) < resolution * padding_factor
            for p in zip(R, z)
        ]
    )

    return trim_vertices(R, z, triangles, bools)


def refine_mesh(
    R: ndarray, z: ndarray, triangles: ndarray, refinement_bools: ndarray
) -> MeshData:
    """
    Refine a mesh by partitioning specified triangles into 4 sub-triangles.
    Triangles sharing one or more edges with those being refined will also
    be partitioned in such a way to ensure the resulting mesh is valid.

    :param R: \
        The major radius of each mesh vertex as a 1D numpy array.

    :param z: \
        The z-height of each mesh vertex as a 1D numpy array.

    :param triangles: \
        A 2D numpy array of integers specifying the indices of the vertices which form
        each of the triangles in the mesh. The array must have shape ``(N,3)`` where ``N``
        is the total number of triangles.

    :param refinement_bools: \
        A numpy array of bools specifying which triangles will be refined. Triangles with
        indices corresponding to ``True`` values in the bool array will be refined.

    :return R, z, triangles: \
        The ``R``, ``z`` and ``triangles`` arrays (defined as described above) for the
        refined mesh.
    """
    # convert the bools to indices
    refinement_inds = refinement_bools.nonzero()[0]
    # build a set as we'll be performing membership checks
    refine_set = {i for i in refinement_inds}

    # get the edge mapping data
    triangle_edges, edge_vertices, edge_to_triangles = build_edge_map(triangles)

    new_mesh_triangles = []
    for t in range(triangles.shape[0]):
        # for the current triangle, find which of its neighbours are being refined
        refined_neighbours = []
        for edge in triangle_edges[t, :]:
            refined_neighbours.extend(
                [i for i in edge_to_triangles[edge] if i != t and i in refine_set]
            )

        vertices = [(R[i], z[i]) for i in triangles[t, :]]

        # if either the triangle itself, or all of its neighbours, are being refined, it must be quadrisected
        if t in refine_set or len(refined_neighbours) == 3:
            v1, v2, v3 = vertices
            # get the mid-point of each side
            m12 = (0.5 * (v1[0] + v2[0]), 0.5 * (v1[1] + v2[1]))
            m23 = (0.5 * (v2[0] + v3[0]), 0.5 * (v2[1] + v3[1]))
            m31 = (0.5 * (v3[0] + v1[0]), 0.5 * (v3[1] + v1[1]))
            # add the new triangles
            new_mesh_triangles.extend(
                [[v1, m12, m31], [v2, m12, m23], [v3, m23, m31], [m12, m23, m31]]
            )

        # if no neighbours are being refined, the triangle remains unchanged
        elif len(refined_neighbours) == 0:
            new_mesh_triangles.append(vertices)

        # if the triangle has two refined neighbours, it must be trisected
        elif len(refined_neighbours) == 2:
            # first we need to find the two edges shared with the neighbours
            shared_edges = [
                intersect1d(triangle_edges[k, :], triangle_edges[t, :])
                for k in refined_neighbours
            ]

            # now find the point that these two edges share
            shared_vertex_index = intersect1d(
                *[edge_vertices[k, :] for k in shared_edges]
            )
            shared_vertex = (R[shared_vertex_index[0]], z[shared_vertex_index[0]])

            # get the two points which are not shared
            v1, v2 = [v for v in vertices if v != shared_vertex]

            # get the mid points of the shared sides
            midpoint_1 = (
                0.5 * (v1[0] + shared_vertex[0]),
                0.5 * (v1[1] + shared_vertex[1]),
            )
            midpoint_2 = (
                0.5 * (v2[0] + shared_vertex[0]),
                0.5 * (v2[1] + shared_vertex[1]),
            )

            # add the new triangles
            new_mesh_triangles.append([midpoint_1, midpoint_2, shared_vertex])
            new_mesh_triangles.append([midpoint_1, v1, v2])
            new_mesh_triangles.append([midpoint_1, midpoint_2, v2])

        # if the triangle has one refined neighbour, it must be bisected
        elif len(refined_neighbours) == 1:
            # find the shared edge
            shared_edge = intersect1d(
                triangle_edges[refined_neighbours[0], :], triangle_edges[t, :]
            )

            # get the vertices of the shared edge
            v1, v2 = [(R[i], z[i]) for i in edge_vertices[shared_edge, :].squeeze()]

            # get the remaining vertex
            v3 = [v for v in vertices if v not in [v1, v2]][0]

            # get the midpoint of the shared edge
            midpoint = (0.5 * (v1[0] + v2[0]), 0.5 * (v1[1] + v2[1]))
            new_mesh_triangles.extend([[midpoint, v3, v1], [midpoint, v3, v2]])
        else:
            raise ValueError("more than 3 refined neighbours detected")

    # number all the vertices
    vertex_map = {}
    for vertices in new_mesh_triangles:
        for vertex in vertices:
            if vertex not in vertex_map:
                vertex_map[vertex] = len(vertex_map)

    # build the mesh data arrays
    new_R = array([v[0] for v in vertex_map.keys()])
    new_z = array([v[1] for v in vertex_map.keys()])
    new_triangles = array(
        [[vertex_map[v] for v in verts] for verts in new_mesh_triangles], dtype=int64
    )

    return new_R, new_z, new_triangles


def remove_duplicate_vertices(R: ndarray, z: ndarray, triangles: ndarray) -> MeshData:
    R2 = R.copy()
    z2 = z.copy()
    # first, find duplicate vertices (including those which differ only by numerical error)
    not_duplicates = ones(R2.size, dtype=bool)
    indices = arange(R2.size, dtype=int64)
    for i in range(R2.size - 1):
        bools = (
            isclose(R2[i + 1 :], R2[i])
            & isclose(z2[i + 1 :], z2[i])
            & not_duplicates[i + 1 :]
        )
        if bools.any():
            R2[i + 1 :][bools] = R2[i]
            z2[i + 1 :][bools] = z2[i]
            not_duplicates[i + 1 :] &= ~bools
            indices[i + 1 :][bools] = i

    # build a new mesh out of the unique vertices
    unique_vals, inverse = unique(indices, return_inverse=True)
    new_R = R[unique_vals]
    new_z = z[unique_vals]
    new_triangles = inverse[triangles]

    # check if any of the triangles are now duplicates remove them
    new_triangles = unique(new_triangles, axis=0)
    new_triangles.sort(axis=1)
    return new_R, new_z, new_triangles


def mesh_generator(
    R_boundary: ndarray,
    z_boundary: ndarray,
    resolution: float,
    edge_resolution: float = None,
    edge_padding: float = 0.75,
    edge_max_area: float = 1.1,
    rotation: float = None,
) -> MeshData:
    """
    Generate a triangular mesh which fills the space inside a given boundary using a 2-stage
    process. First, a mesh of equilateral triangles is created which fills the space up to a
    chosen minimum distance from the boundary. An irregular mesh is then generated which fills
    the space between the central equilateral mesh and the boundary. The two meshes are then
    merged, and the resulting mesh is returned.

    :param R_boundary: \
        The major-radius values of the boundary as a 1D numpy array.

    :param z_boundary: \
        The z-height values of the boundary as a 1D numpy array.

    :param resolution: \
        The side-length of triangles in the central equilateral mesh.

    :param edge_resolution: \
        Sets the target area of triangles in the irregular edge mesh, which fills the
        space between the central equilateral mesh and the boundary. The `Triangle` C-code,
        which is used to generate the irregular mesh, will attempt to construct triangles
        with areas equal to that of an equilateral triangle with side length ``edge_resolution``.
        If not specified, the value passed as the ``resolution`` argument is used instead.

    :param edge_padding: \
        A multiplicative factor which defines the minimum allowed distance between a
        vertex in the central equilateral mesh and the boundary such that
        ``min_distance = edge_padding * resolution``. No vertices in the central equilateral
        mesh will be closer to the boundary than ``min_distance``.

    :param edge_max_area: \
        A multiplicative factor which sets the maximum allowed area of triangles in the
        irregular edge mesh, such that no triangle will have an area larger than
        ``edge_max_area`` times the target area set by the ``edge_resolution`` argument.

    :param rotation: \
        Angle (in radians) by which the orientations of triangles in the central
        equilateral mesh are rotated, relative to their default orientation.

    :return: \
        A tuple containing ``R_vert``, ``z_vert`` and ``triangles``.
        ``R_vert`` is the major-radius of the vertices as a 1D array. ``z_vert`` the is
        z-height of the vertices as a 1D array. ``triangles`` is a 2D array of integers
        of shape ``(N,3)`` specifying the indices of the vertices which form each triangle
        in the mesh, where ``N`` is the total number of triangles.
    """
    # build the central mesh
    central_R, central_z, central_triangles = build_central_mesh(
        R_boundary=R_boundary,
        z_boundary=z_boundary,
        resolution=resolution,
        padding_factor=edge_padding,
        rotation=rotation,
    )

    # find all boundaries on the central mesh
    boundaries = find_boundaries(central_triangles)
    # if there is more than one boundary, then there are multiple sub-meshes
    # pick the largest boundary, and discard any vertices which are outside of it
    boundaries = sorted(boundaries, key=lambda x: len(x))
    central_boundary = boundaries[-1]
    if len(boundaries) > 1:
        # turn the boundary into a polygon to test if points are inside
        poly = Polygon(x=central_R[central_boundary], y=central_z[central_boundary])

        max_dist = resolution * 1e-2
        trim_vertex = array(
            [
                not poly.is_inside(v) and poly.distance(v) > max_dist
                for v in zip(central_R, central_z)
            ]
        )
        # remove any vertices which are outside the boundary
        central_R, central_z, central_triangles = trim_vertices(
            central_R, central_z, central_triangles, trim_vertex
        )

        # re-calculate the boundary for the trimmed mesh
        boundaries = find_boundaries(central_triangles)
        # verify that there is now only one boundary
        assert len(boundaries) == 1
        central_boundary = boundaries[-1]

    # now we have the boundary, we can build the edge mesh using triangle.
    # prepare triangle inputs:
    edge_resolution = resolution if edge_resolution is None else edge_resolution
    eq_area = (edge_resolution**2) * 0.25 * sqrt(3)

    outer = array([R_boundary[:-1], z_boundary[:-1]]).T
    inner = array(
        [central_R[central_boundary[:-1]], central_z[central_boundary[:-1]]]
    ).T

    # fixme - for non-convex boundaries taking the mean doesn't always work
    voids = [[inner[:, 0].mean(), inner[:, 1].mean()]]

    edge_R, edge_z, edge_triangles = build_edge_mesh(
        outer_boundary=outer,
        inner_boundary=inner,
        void_markers=voids,
        max_area=eq_area * edge_max_area,
    )

    # combine the central and edge meshes
    R = concatenate([central_R, edge_R])
    z = concatenate([central_z, edge_z])
    triangles = concatenate(
        [central_triangles, edge_triangles + central_R.size], axis=0
    )
    R, z, triangles = remove_duplicate_vertices(R, z, triangles)

    # check that the final mesh has only one boundary
    assert len(find_boundaries(triangles)) == 1
    return R, z, triangles


def build_edge_mesh(
    inner_boundary: ndarray,
    outer_boundary: ndarray,
    void_markers: ndarray,
    max_area: float,
) -> MeshData:
    for b in [inner_boundary, outer_boundary]:
        assert b.ndim == 2 and b.shape[1] == 2
        assert b[0, 0] != b[-1, 0] or b[0, 1] != b[-1, 1]

    inner_segments = build_segments(inner_boundary.shape[0])
    outer_segments = build_segments(outer_boundary.shape[0])

    triangle_inputs = dict(
        vertices=vstack([outer_boundary, inner_boundary]),
        segments=connect_segments([outer_segments, inner_segments]),
        holes=void_markers,
    )

    options = f"QPBzpqa{max_area:.12f}"
    from triangle import triangulate

    triangle_outputs = triangulate(triangle_inputs, options)
    triangles = triangle_outputs["triangles"]
    vertices = triangle_outputs["vertices"]
    return vertices[:, 0], vertices[:, 1], triangles


def build_segments(n: int) -> ndarray:
    inds = arange(n)
    return vstack([inds, inds + 1]).T % n


def connect_segments(segments: list[ndarray]) -> ndarray:
    lengths = [s.shape[0] for s in segments]
    offsets = cumsum([0, *lengths[:-1]], dtype=int)
    return vstack([s + d for s, d in zip(segments, offsets)])
