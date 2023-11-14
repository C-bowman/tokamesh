from collections import defaultdict
from numpy import linspace, full, int64, arange, searchsorted, ndarray, zeros
from numpy import pi, sqrt, dot, array, cross, identity, tan


class BinaryTree:
    """
    divides the range specified by limits into n = 2**layers equal regions,
    and builds a binary tree which allows fast look-up of which of region
    contains a given value.

    :param int layers: number of layers that make up the tree
    :param limits: tuple of the lower and upper bounds of the look-up region.
    """

    def __init__(self, layers: int, limits: tuple):
        self.layers = layers
        self.nodes = 2**self.layers
        self.lims = limits
        self.edges = linspace(limits[0], limits[1], self.nodes + 1)
        self.mids = 0.5 * (self.edges[1:] + self.edges[:-1])

        self.indices = full(self.nodes + 2, fill_value=-1, dtype=int64)
        self.indices[1:-1] = arange(self.nodes)

    def lookup_index(self, values):
        return self.indices[searchsorted(self.edges, values)]


def build_edge_map(triangles: ndarray):
    """
    Generates various mappings to and from edges in the mesh.

    :param triangles: \
        A 2D numpy array of integers specifying the indices of the vertices which form
        each of the triangles in the mesh. The array must have shape ``(N,3)`` where
        ``N`` is the total number of triangles.

    :return: \
        A tuple containing ``triangle_edges``, ``edge_vertices`` and ``edge_map``.
        ``triangle_edges`` specifies the indices of the edges which make up each
        triangle as a 2D numpy array of shape ``(N,3)`` where ``N`` is the total
        number of triangles. ``edge_vertices`` specifies the indices of the vertices
        which make up each edge as a 2D numpy array of shape ``(M,2)`` where ``M``
        is the total number of edges. ``edge_map`` is a dictionary mapping the index
        of an edge to the indices of the triangles to which it belongs.
    """
    n_triangles = triangles.shape[0]
    triangle_edges = zeros([n_triangles, 3], dtype=int64)
    edge_indices = {}
    edge_map = defaultdict(list)
    for i in range(n_triangles):
        s1 = (
            min(triangles[i, 0], triangles[i, 1]),
            max(triangles[i, 0], triangles[i, 1]),
        )
        s2 = (
            min(triangles[i, 1], triangles[i, 2]),
            max(triangles[i, 1], triangles[i, 2]),
        )
        s3 = (
            min(triangles[i, 0], triangles[i, 2]),
            max(triangles[i, 0], triangles[i, 2]),
        )
        for j, edge in enumerate([s1, s2, s3]):
            if edge not in edge_indices:
                edge_indices[edge] = len(edge_indices)
            triangle_edges[i, j] = edge_indices[edge]
            edge_map[edge_indices[edge]].append(i)

    edge_vertices = zeros([len(edge_indices), 2], dtype=int64)
    for edge, i in edge_indices.items():
        edge_vertices[i, :] = [edge[0], edge[1]]

    return triangle_edges, edge_vertices, edge_map


class Camera:
    def __init__(
        self, position, direction, num_x=10, num_y=10, fov=40.0, max_distance=10.0
    ):
        self.u0 = position
        self.du = direction
        self.x_angles = linspace(-fov * pi / 360.0, fov * pi / 360.0, num_x)
        self.y_angles = linspace(-fov * pi / 360.0, fov * pi / 360.0, num_y)
        self.max_distance = max_distance

        # make sure the direction is normalised
        self.du /= sqrt(dot(self.du, self.du))

        # find the first perpendicular
        K = self.du[1] / self.du[0]
        b = 1.0 / sqrt(K**2 + 1.0)
        a = -K * b
        self.p1 = array([a, b, 0.0])

        # use cross-product to find second perpendicular
        self.p2 = cross(self.du, self.p1)

        # identity matrix
        self.I = identity(3)

        # calculate the ray directions
        tan_x = tan(self.x_angles)
        tan_y = tan(self.y_angles)
        norm = sqrt(1 + (tan_x**2)[:, None] + (tan_y**2)[None, :])
        v = (
            self.du[None, None, :]
            + tan_x[:, None, None] * self.p1[None, None, :]
            + tan_y[None, :, None] * self.p2[None, None, :]
        )
        self.ray_directions = v / norm[:, :, None]
        self.ray_directions.resize([num_x * num_y, 3])

        self.ray_ends = self.u0[None, :] + max_distance * self.ray_directions
        self.ray_starts = zeros(self.ray_ends.shape) + self.u0[None, :]

    def plot_rays(self, axis, points=200):
        dist = linspace(0, self.max_distance, points)
        positions = (
            self.u0[None, :, None]
            + self.ray_directions[:, :, None] * dist[None, None, :]
        )
        R = sqrt(positions[:, 0, :] ** 2 + positions[:, 1, :] ** 2).T
        z = positions[:, 2, :].T
        axis.plot(R, z)

    def project_rays(self, distance):
        positions = (
            self.u0[None, :, None]
            + self.ray_directions[:, :, None] * distance[None, None, :]
        )
        R = sqrt(positions[:, 0, :] ** 2 + positions[:, 1, :] ** 2).T
        z = positions[:, 2, :].T
        return R, z
