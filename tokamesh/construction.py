
from numpy import sqrt, ceil, sin, cos, arctan2, in1d, diff, minimum, maximum, unique
from numpy import array, zeros, linspace, arange, int64, concatenate, atleast_1d
from warnings import warn
from tokamesh.geometry import build_edge_map


def equilateral_mesh(x_range=(0,10), y_range=(0,5), scale=1.0, rotation=None, pivot=(0,0)):
    """
    Construct a mesh from equilateral triangles which fills a rectangular region.

    :param x_range: \
        A tuple in the form ``(x_min, x_max)`` specifying the range of the x-axis to cover with triangles.

    :param y_range: \
        A tuple in the form ``(y_min, y_max)`` specifying the range of the y-axis to cover with triangles.

    :param float scale: \
        The side-length of the triangles.

    :param rotation: \
        Angle (in radians) by which the mesh will be rotated.

    :param pivot: \
        Pivot point around which the rotation is applied.

    :return x_vert, y_vert, triangles: \
        ``x_vert`` is the x-position of the vertices as a 1D array. ``y_vert`` the is y-position
        of the vertices as a 1D array. ``triangles`` is a 2D array of integers of shape ``(N,3)``
        specifying the indices of the vertices which form each triangle in the mesh, where
        ``N`` is the total number of triangles.
    """
    # determine how many rows / columns of triangles to create
    N = int(ceil((x_range[1] - x_range[0])/scale))
    M = int(ceil((y_range[1] - y_range[0])/(scale*0.5*sqrt(3))))

    # create the vertices by producing a rectangular grid
    # and shifting every other row
    x_ax = linspace(0, N-1, N)*scale
    y_ax = linspace(0, M-1, M)*scale*0.5*sqrt(3)

    x = zeros([N,M])
    y = zeros([N,M])
    y[:,:] = y_ax[None,:] + y_range[0]
    x[:,:] = x_ax[:,None] + x_range[0]
    x[:,1::2] += 0.5*scale

    # rotate the vertices around a point if requested
    if rotation is not None:
        R = sqrt((x-pivot[0])**2 + (y-pivot[1])**2)
        theta = arctan2(y-pivot[1], x-pivot[0]) + rotation
        x = R*cos(theta) + pivot[0]
        y = R*sin(theta) + pivot[1]

    # divide up the grid into triangles
    triangle_inds = []
    for i in range(N-1):
        for j in range(M-1):
            v1 = M*i + j
            v2 = M*(i+1) + j
            v3 = M*i + j + 1
            v4 = M*(i+1) + j + 1

            if j % 2 == 0:
                triangle_inds.append([v1, v2, v3])
                triangle_inds.append([v2, v3, v4])
            else:
                triangle_inds.append([v1, v3, v4])
                triangle_inds.append([v1, v2, v4])

    return x.flatten(), y.flatten(), array(triangle_inds)




def trim_vertices(R, z, triangles, bools):
    """
    Removes chosen vertices (and any triangles containing those vertices) from a mesh.

    :param R: \
        The major radius of each mesh vertex as a 1D numpy array.

    :param z: \
        The z-height of each mesh vertex as a 1D numpy array.

    :param triangles: \
        A 2D numpy array of integers specifying the indices of the vertices which form
        each of the triangles in the mesh. The array must have shape ``(N,3)`` where `'N'`
        is the total number of triangles.

    :param bools: \
        A 1D array of boolean values corresponding to the vertices, which is ``True`` for
        any vertices which are to be removed from the mesh.

    :return R, z, triangles: \
        The ``R``, ``z`` and ``triangles`` arrays (defined as described above) with the
        specified vertices removed.
    """
    vert_inds = (~bools).nonzero()[0]
    tri_bools = in1d(triangles[:,0],vert_inds)
    tri_bools &= in1d(triangles[:,1],vert_inds)
    tri_bools &= in1d(triangles[:,2],vert_inds)
    tri_inds = tri_bools.nonzero()[0]
    index_converter = zeros(R.size, dtype=int64)
    index_converter[vert_inds] = arange(vert_inds.size)
    return R[vert_inds], z[vert_inds], index_converter[triangles[tri_inds,:]]




class Polygon(object):
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
        self.im = self.dx / self.dy
        self.c = self.y[:-1] - self.x[:-1]*self.dy/self.dx

        # pre-calculate the bounding rectangle of each edge for intersection testing
        self.x_upr = maximum(self.x[1:], self.x[:-1])
        self.x_lwr = minimum(self.x[1:], self.x[:-1])
        self.y_upr = maximum(self.y[1:], self.y[:-1])
        self.y_lwr = minimum(self.y[1:], self.y[:-1])

        # normalise the unit vectors
        self.lengths = sqrt(self.dx**2 + self.dy**2)
        self.dx /= self.lengths
        self.dy /= self.lengths

        self.zero_im = self.im == 0.

    def is_inside(self, v):
        x, y = v
        k = (y - self.c)*self.im

        limits_check = (self.y_lwr <= y) & (y <= self.y_upr) & (x <= self.x_upr)
        isec_check = ((self.x_lwr <= k) & (k <= self.x_upr) & (x <= k)) | self.zero_im

        intersections = (limits_check & isec_check).sum()
        if intersections % 2 == 0:
            return False
        else:
            return True

    def distance(self, v):
        x, y = v
        dx = x - self.x[:-1]
        dy = y - self.y[:-1]

        L = (dx*self.dx + dy*self.dy) / self.lengths
        D = dx*self.dy - dy*self.dx
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
        xpad = (xmax-xmin)*0.15
        ypad = (ymax-ymin)*0.15

        N = 200
        x_ax = linspace(xmin-xpad, xmax+xpad, N)
        y_ax = linspace(ymin-ypad, ymax+ypad, N)

        inside = zeros([N,N])
        distance = zeros([N,N])
        for i in range(N):
            for j in range(N):
                v = [x_ax[i], y_ax[j]]
                inside[i,j] = self.is_inside(v)
                distance[i,j] = self.distance(v)

        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(12,4))
        ax1 = fig.add_subplot(131)
        ax1.contourf(x_ax, y_ax, inside.T)
        ax1.plot(self.x, self.y, '.-', c='white', lw=2)
        ax1.set_title('point is inside polygon')

        ax2 = fig.add_subplot(132)
        ax2.contourf(x_ax, y_ax, distance.T, 100)
        ax2.plot(self.x, self.y, '.-', c='white', lw=2)
        ax2.set_title('distance from polygon')

        ax3 = fig.add_subplot(133)
        ax3.contourf(x_ax, y_ax, (distance*inside).T, 100)
        ax3.plot(self.x, self.y, '.-', c='white', lw=2)
        ax3.set_title('interior point distance from polygon')

        plt.tight_layout()
        plt.show()




def find_boundaries(triangles):
    """
    Find all the boundaries of a given mesh.

    :param triangles: \
        A 2D numpy array of integers specifying the indices of the vertices which form
        each of the triangles in the mesh. The array must have shape ``(N,3)`` where ``N`` is
        the total number of triangles.

    :return: \
        A list of 1D numpy arrays containing the indices of the vertices in each boundary.
    """
    # Construct a mapping from triangles to edges, and edges to vertices
    triangle_edges, edge_vertices = build_edge_map(triangles)
    # identify edges on the boundary by finding edges which are only part of one triangle
    unique_vals, counts = unique(triangle_edges, return_counts=True)
    boundary_edges_indices = (counts == 1).nonzero()[0]
    boundary_edges = edge_vertices[boundary_edges_indices, :]

    # now create a map between an edge, and the other edges to which it's connected
    boundary_connections = {}
    for i in range(boundary_edges.shape[0]):
        edges = ((boundary_edges[i, 0] == boundary_edges) | (boundary_edges[i, 1] == boundary_edges)).nonzero()[0]
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
            """
            [ find_boundaries warning ]
            >> The given mesh contains at least two sub-meshes which
            >> are connected by only one vertex. Currently, it is not
            >> guaranteed that this function will draw separate boundaries
            >> for each sub-mesh - this will be addressed in future update.
            """
        )

    # Now we need to convert the boundaries from edge indices to vertex indices
    vertex_boundaries = []
    for boundary in boundaries:
        # the order of the first two vertex indices needs to match the direction
        # in which the boundary is being traced.
        v1, v2 = edge_vertices[boundary[0],:]
        if v1 in edge_vertices[boundary[1],:]:
            vertex_boundary = [v2, v1]
        else:
            vertex_boundary = [v1, v2]

        # now loop over all the other edges and add the new vertex that appears
        for edge in boundary[1:]:
            v1, v2 = edge_vertices[edge,:]
            next_vertex = v1 if (v1 not in vertex_boundary) else v2
            vertex_boundary.append(next_vertex)

        vertex_boundaries.append(array(vertex_boundary))

    return vertex_boundaries




def build_central_mesh(R_boundary, z_boundary, scale, padding_factor=1.):
    """
    Generate an equilateral mesh which fills the space inside a given boundary,
    up to a chosen distance to the boundary edge.

    :param R_boundary: \
        The major-radius values of the boundary as a 1D numpy array.

    :param z_boundary: \
        The z-height values of the boundary as a 1D numpy array.

    :param scale: \
        The side-length of the equilateral triangles.

    :param padding_factor: \
        A multiplicative factor which defines the minimum distance to the boundary
        such that ``min_distance = padding_factor*scale``. No vertices in the returned
        mesh will be closer to the boundary than ``min_distance``.

    :return R_vert, z_vert, triangles: \
        ``R_vert`` is the major-radius of the vertices as a 1D array. ``z_vert`` the is
        z-height of the vertices as a 1D array. ``triangles`` is a 2D array of integers
        of shape ``(N,3)`` specifying the indices of the vertices which form each triangle
        in the mesh, where ``N`` is the total number of triangles.
    """
    poly = Polygon(R_boundary, z_boundary)

    r_range = (min(R_boundary) - 3 * scale, max(R_boundary) + 3 * scale)
    z_range = (min(z_boundary)-3*scale, max(z_boundary)+3*scale)

    R, z, triangles = equilateral_mesh(x_range=r_range, y_range=z_range, scale=scale)

    # remove all triangles which are too close too or inside walls
    bools = array([poly.is_inside(p)*poly.distance(p) < scale*padding_factor for p in zip(R,z)])

    return trim_vertices(R, z, triangles, bools)