from numpy import arange, array, concatenate, full, ones, tile, ndarray, zeros
from numpy import sqrt, diff
from scipy.sparse import csc_matrix, csr_matrix
from scipy.special import factorial
from scipy.linalg import solve
from itertools import product
from tokamesh.utilities import build_edge_map


def edge_difference_matrix(
    R: ndarray, z: ndarray, triangles: ndarray, normalised=False, sparse_format="csr"
):
    """
    Generates a sparse matrix which, when operating on a vector of field
    values at each vertex, produces a vector of the differences in those
    field values along each edge in the mesh.

    :param R: \
        The major radius of each mesh vertex as a 1D numpy array.

    :param z: \
        The z-height of each mesh vertex as a 1D numpy array.

    :param triangles: \
        A 2D numpy array of integers specifying the indices of the vertices which form
        each of the triangles in the mesh. The array must have shape ``(N,3)`` where ``N``
        is the total number of triangles.

    :param bool normalised: \
        If set as ``True``, the difference in the value across each edge is normalised by
        the length of the edge, yielding an estimate of the gradient along that edge.

    :param str sparse_format: \
        Specifies the sparse-matrix format to be used for the output. Pass ``'csr'`` for
        the ``scipy.sparse.csr_matrix`` format, and ``'csc'`` for the ``scipy.sparse.csc_matrix``
        format.
    """
    triangle_edges, edge_vertices, edge_map = build_edge_map(triangles)
    n_edges = edge_vertices.shape[0]
    shape = (n_edges, R.size)
    entry_vals = ones(2 * n_edges)

    if normalised:
        dR = diff(R[edge_vertices], axis=1)
        dz = diff(z[edge_vertices], axis=1)
        inv_distances = 1.0 / sqrt(dR**2 + dz**2)
        entry_vals[0::2] = inv_distances
        entry_vals[1::2] = -inv_distances
    else:
        entry_vals[1::2] = -1.0

    row_inds = tile(arange(n_edges), (1, 2)).T.flatten()
    col_inds = edge_vertices.T.flatten()
    sparse_format = sparse_format if sparse_format in ["csr", "csc"] else "csr"
    if sparse_format == "csr":
        return csr_matrix((entry_vals, (row_inds, col_inds)), shape=shape)
    elif sparse_format == "csc":
        return csc_matrix((entry_vals, (row_inds, col_inds)), shape=shape)


def umbrella_matrix(
    R: ndarray,
    z: ndarray,
    triangles: ndarray,
    ignore_boundary=True,
    inverse_distance_weighting=True,
    normalised=False,
    sparse_format="csr",
):
    """
    returns a sparse 'umbrella' matrix operator, which finds the difference between
    the value of every internal vertex value and the average value of the other
    vertices with which it is connected.

    :param R: \
        The major radius of each mesh vertex as a 1D numpy array.

    :param z: \
        The z-height of each mesh vertex as a 1D numpy array.

    :param triangles: \
        A 2D numpy array of integers specifying the indices of the vertices which form
        each of the triangles in the mesh. The array must have shape ``(N,3)`` where ``N``
        is the total number of triangles.

    :param bool ignore_boundary: \
        If set as ``True``, the operator will ignore vertices on any boundary of the mesh,
        instead returning a value of zero for those vertices.

    :param bool inverse_distance_weighting: \
        If set as ``True``, the sum over the neighbouring vertices is weighted by their
        inverse-distance to the central vertex. This tends to produce better results for
        irregular meshes with significant variation of edge-lengths.

    :param bool normalised: \
        If set as ``True`` the value produced by the operator for each vertex is normalised
        by the square of the mean distance between the neighbouring vertices and the central
        vertex.

    :param str sparse_format: \
        Specifies the sparse-matrix format to be used for the output. Pass ``'csr'`` for
        the ``scipy.sparse.csr_matrix`` format, and ``'csc'`` for the ``scipy.sparse.csc_matrix``
        format.
    """
    triangle_edges, edge_vertices, edge_map = build_edge_map(triangles)

    if ignore_boundary:
        boundary_edges = [i for i in range(len(edge_map)) if len(edge_map[i]) == 1]
        boundary_vertex_set = {edge_vertices[i, 0] for i in boundary_edges} | {
            edge_vertices[i, 1] for i in boundary_edges
        }
        verts = {i for i in range(R.size)} - boundary_vertex_set
    else:
        verts = range(R.size)

    row_inds_list = []
    col_inds_list = []
    entry_vals_list = []
    for i in verts:
        inds = ((edge_vertices[:, 0] == i) | (edge_vertices[:, 1] == i)).nonzero()[0]
        col_inds = [i]
        col_inds.extend([j for j in edge_vertices[inds, :].flatten() if i != j])
        col_inds = array(col_inds, dtype=int)
        distances = sqrt((R[i] - R[col_inds[1:]]) ** 2 + (z[i] - z[col_inds[1:]]) ** 2)
        row_inds = full(col_inds.size, fill_value=i, dtype=int)
        weights = full(col_inds.size, fill_value=-1.0)

        if inverse_distance_weighting:
            inv_distances = 1.0 / distances
            weights[1:] = inv_distances / inv_distances.sum()
        else:
            weights[1:] = 1.0 / (col_inds.size - 1.0)

        if normalised:
            weights /= 2.6 * distances.mean() ** 2

        row_inds_list.append(row_inds)
        col_inds_list.append(col_inds)
        entry_vals_list.append(weights)

    row_indices = concatenate(row_inds_list)
    col_indices = concatenate(col_inds_list)
    entry_values = concatenate(entry_vals_list)

    shape = (R.size, R.size)
    if sparse_format == "csr":
        return csr_matrix((entry_values, (row_indices, col_indices)), shape=shape)
    elif sparse_format == "csc":
        return csc_matrix((entry_values, (row_indices, col_indices)), shape=shape)


def parallel_derivative(
    R: ndarray, z: ndarray, index_grid: ndarray, order=2, sparse_format="csr"
):
    """
    Constructs a sparse matrix operator which estimates the derivative (with respect
    to poloidal distance) of a given set of field values at each vertex in direction
    parallel to the magnetic field.

    :param R: \
        The major radius of each mesh vertex as a 1D numpy array.

    :param z: \
        The z-height of each mesh vertex as a 1D numpy array.

    :param index_grid: \
        A 2D numpy array specifying the indices of triangular mesh vertices
        corresponding to each cell of the rectangular field-aligned grid.

    :param order: \
        The order of the derivative to be estimated. Must be either 1 or 2.

    :param str sparse_format: \
        Specifies the sparse-matrix format to be used for the output. Pass ``'csr'`` for
        the ``scipy.sparse.csr_matrix`` format, and ``'csc'`` for the ``scipy.sparse.csc_matrix``
        format.
    """
    shape = (3, index_grid.shape[0] * index_grid.shape[1])
    values = zeros(shape)
    i_indices = zeros(shape)
    j_indices = zeros(shape)

    prod = product(range(index_grid.shape[0]), range(index_grid.shape[1]))
    for k, (i, j) in enumerate(prod):
        if i == 0:
            inds = index_grid[:3, j]
            dl = sqrt(diff(R[inds]) ** 2 + diff(z[inds]) ** 2)
            pol_dist = array([0.0, dl[0], dl.sum()])
        elif i == index_grid.shape[0] - 1:
            inds = index_grid[-3:, j]
            dl = sqrt(diff(R[inds]) ** 2 + diff(z[inds]) ** 2)
            pol_dist = array([-dl.sum(), -dl[1], 0.0])
        else:
            inds = index_grid[i - 1 : i + 2, j]
            dl = sqrt(diff(R[inds]) ** 2 + diff(z[inds]) ** 2)
            pol_dist = array([-dl[0], 0.0, dl[1]])

        coeffs = get_fd_coeffs(pol_dist, order=order)

        values[:, k] = coeffs
        i_indices[:, k] = index_grid[i, j]
        j_indices[:, k] = inds

    shape = (R.size, R.size)
    indices = (i_indices.flatten(), j_indices.flatten())
    if sparse_format == "csr":
        return csr_matrix((values.flatten(), indices), shape=shape)
    elif sparse_format == "csc":
        return csc_matrix((values.flatten(), indices), shape=shape)


def perpendicular_derivative(
    R: ndarray, z: ndarray, index_grid: ndarray, order=2, sparse_format="csr"
):
    """
    Constructs a sparse matrix operator which estimates the derivative (with respect
    to poloidal distance) of a given set of field values at each vertex in direction
    perpendicular to the magnetic field.

    :param R: \
        The major radius of each mesh vertex as a 1D numpy array.

    :param z: \
        The z-height of each mesh vertex as a 1D numpy array.

    :param index_grid: \
        A 2D numpy array specifying the indices of triangular mesh vertices
        corresponding to each cell of the rectangular field-aligned grid.

    :param order: \
        The order of the derivative to be estimated. Must be either 1 or 2.

    :param str sparse_format: \
        Specifies the sparse-matrix format to be used for the output. Pass ``'csr'`` for
        the ``scipy.sparse.csr_matrix`` format, and ``'csc'`` for the ``scipy.sparse.csc_matrix``
        format.
    """
    shape = (3, index_grid.shape[0] * index_grid.shape[1])
    values = zeros(shape)
    i_indices = zeros(shape)
    j_indices = zeros(shape)

    prod = product(range(index_grid.shape[0]), range(index_grid.shape[1]))
    for k, (i, j) in enumerate(prod):
        if j == 0:
            inds = index_grid[i, :3]
            dl = sqrt(diff(R[inds]) ** 2 + diff(z[inds]) ** 2)
            perp_dist = array([0.0, dl[0], dl.sum()])
        elif j == index_grid.shape[1] - 1:
            inds = index_grid[i, -3:]
            dl = sqrt(diff(R[inds]) ** 2 + diff(z[inds]) ** 2)
            perp_dist = array([-dl.sum(), -dl[1], 0.0])
        else:
            inds = index_grid[i, j - 1 : j + 2]
            dl = sqrt(diff(R[inds]) ** 2 + diff(z[inds]) ** 2)
            perp_dist = array([-dl[0], 0.0, dl[1]])

        coeffs = get_fd_coeffs(perp_dist, order=order)

        values[:, k] = coeffs
        i_indices[:, k] = index_grid[i, j]
        j_indices[:, k] = inds

    shape = (R.size, R.size)
    indices = (i_indices.flatten(), j_indices.flatten())
    if sparse_format == "csr":
        return csr_matrix((values.flatten(), indices), shape=shape)
    elif sparse_format == "csc":
        return csc_matrix((values.flatten(), indices), shape=shape)


def get_fd_coeffs(points, order=1):
    # check validity of inputs
    if type(points) is not ndarray:
        points = array(points)
    n = len(points)
    if n <= order:
        raise ValueError(
            "The order of the derivative must be less than the number of points"
        )
    # build the linear system
    b = zeros(n)
    b[order] = factorial(order)
    A = ones([n, n])
    for i in range(1, n):
        A[i, :] = points**i
    # return the solution
    return solve(A, b)
