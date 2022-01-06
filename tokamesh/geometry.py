from numpy import sqrt, log, pi, tan, dot, cross, identity
from numpy import absolute, nan, isfinite, minimum, maximum
from numpy import array, ndarray, linspace, full, zeros, stack, savez, int64
from collections import defaultdict
from time import perf_counter
import sys


class BarycentricGeometryMatrix(object):
    """
    Class for calculating geometry matrices over triangular meshes using
    barycentric linear interpolation.

    :param R: \
        The major radius of each mesh vertex as a 1D numpy array.

    :param z: \
        The z-height of each mesh vertex as a 1D numpy array.

    :param triangles: \
        A 2D numpy array of integers specifying the indices of the vertices which form
        each of the triangles in the mesh. The array must have shape ``(N,3)`` where
        ``N`` is the total number of triangles.

    :param ray_origins: \
        The ``(x,y,z)`` position vectors of the origin of each ray (i.e. line-of-sight)
        as a 2D numpy array. The array must have shape ``(M,3)`` where ``M`` is the
        total number of rays.

    :param ray_ends: \
        The ``(x,y,z)`` position vectors of the end-points of each ray (i.e. line-of-sight)
        as a 2D numpy array. The array must have shape ``(M,3)`` where ``M`` is the
        total number of rays.
    """

    def __init__(self, R, z, triangles, ray_origins, ray_ends):
        # first check the validity of the data
        self.check_geometry_data(R, z, triangles, ray_origins, ray_ends)

        self.R = R
        self.z = z
        self.triangle_vertices = triangles
        self.n_vertices = self.R.size
        self.n_triangles = self.triangle_vertices.shape[0]
        self.GeomFacs = GeometryFactors()

        # calculate the ray data
        diffs = ray_ends - ray_origins
        self.lengths = sqrt((diffs ** 2).sum(axis=1))
        self.rays = diffs / self.lengths[:, None]
        self.pixels = ray_origins
        self.n_rays = self.lengths.size

        # coefficients for the quadratic representation of the ray radius
        self.q0 = self.pixels[:, 0] ** 2 + self.pixels[:, 1] ** 2
        self.q1 = 2 * (
            self.pixels[:, 0] * self.rays[:, 0] + self.pixels[:, 1] * self.rays[:, 1]
        )
        self.q2 = self.rays[:, 0] ** 2 + self.rays[:, 1] ** 2
        self.sqrt_q2 = sqrt(self.q2)

        # calculate terms used in the linear inequalities
        self.L_tan = -0.5 * self.q1 / self.q2  # distance of the tangency point
        self.R_tan_sqr = self.q0 + 0.5 * self.q1 * self.L_tan
        self.R_tan = sqrt(self.R_tan_sqr)  # major radius of the tangency point
        self.z_tan = (
            self.pixels[:, 2] + self.rays[:, 2] * self.L_tan
        )  # z-height of the tangency point
        self.m = self.rays[:, 2] / sqrt(
            self.q2
        )  # gradient of the hyperbola asymptote line

        # Construct a mapping from triangles to edges, and edges to vertices
        self.triangle_edges, self.edge_vertices, _ = build_edge_map(
            self.triangle_vertices
        )
        self.R_edges = self.R[self.edge_vertices]
        self.z_edges = self.z[self.edge_vertices]
        self.n_edges = self.edge_vertices.shape[0]

        # pre-calculate the properties of each edge
        self.R_edge_mid = self.R_edges.mean(axis=1)
        self.z_edge_mid = self.z_edges.mean(axis=1)
        self.edge_lengths = sqrt(
            (self.R_edges[:, 0] - self.R_edges[:, 1]) ** 2
            + (self.z_edges[:, 0] - self.z_edges[:, 1]) ** 2
        )
        self.edge_drn = zeros([self.n_edges, 2])
        self.edge_drn[:, 0] = self.R_edges[:, 1] - self.R_edges[:, 0]
        self.edge_drn[:, 1] = self.z_edges[:, 1] - self.z_edges[:, 0]
        self.edge_drn /= self.edge_lengths[:, None]

        # pre-calculate barycentric coordinate coefficients for each triangle
        R1, R2, R3 = [self.R[self.triangle_vertices[:, k]] for k in range(3)]
        z1, z2, z3 = [self.z[self.triangle_vertices[:, k]] for k in range(3)]
        self.area = 0.5 * ((z2 - z3) * (R1 - R3) + (R3 - R2) * (z1 - z3))
        self.lam1_coeffs = (
            0.5
            * stack([z2 - z3, R3 - R2, R2 * z3 - R3 * z2], axis=1)
            / self.area[:, None]
        )
        self.lam2_coeffs = (
            0.5
            * stack([z3 - z1, R1 - R3, R3 * z1 - R1 * z3], axis=1)
            / self.area[:, None]
        )

    def calculate(self, save_file=None):
        """
        Calculate the geometry matrix.

        :keyword str save_file: \
            A string specifying a file path to which the geometry matrix data will be
            saved using the numpy ``.npz`` format. If not specified, the geometry matrix
            data is still returned as a dictionary, but is not saved.

        :return: \
            The geometry matrix data as a dictionary of numpy arrays. The structure of
            the dictionary is as follows: ``entry_values`` is a 1D numpy array containing
            the values of all non-zero matrix entries. ``row_indices`` is a 1D numpy
            array containing the row-index of each of the non-zero entries. ``col_indices``
            is a 1D numpy array containing the column-index of each of the non-zero entries.
            ``shape`` is a 1D numpy array containing the dimensions of the matrix. The
            arrays defining the mesh are also stored as ``R``, ``z`` and ``triangles``.
        """
        # clear the geometry factors in case they contains data from a previous calculation
        self.GeomFacs.vertex_map.clear()
        # process the first triangle so we can estimate the run-time
        t_start = perf_counter()
        self.process_triangle(0)
        dt = perf_counter() - t_start

        # use the estimate to break the evaluation into groups
        group_size = max(int(1.0 / dt), 1)
        rem = (self.n_triangles - 1) % group_size
        ranges = zip(
            range(1, self.n_triangles, group_size),
            range(1 + group_size, self.n_triangles, group_size),
        )

        # calculate the contribution to the matrix for each triangle
        for start, end in ranges:
            [self.process_triangle(i) for i in range(start, end)]

            # print the progress
            f_complete = (end + 1) / self.n_triangles
            eta = int((perf_counter() - t_start) * (1 - f_complete) / f_complete)
            sys.stdout.write(
                f"\r >> Calculating geometry matrix:  [ {f_complete:.1%} complete   ETA: {eta} sec ]           "
            )
            sys.stdout.flush()

        # clean up any remaining triangles
        if rem != 0:
            [
                self.process_triangle(i)
                for i in range(self.n_triangles - rem, self.n_triangles)
            ]

        t_elapsed = perf_counter() - t_start
        mins, secs = divmod(t_elapsed, 60)
        hrs, mins = divmod(mins, 60)
        time_taken = "%d:%02d:%02d" % (hrs, mins, secs)
        sys.stdout.write(
            f"\r >> Calculating geometry matrix:  [ completed in {time_taken} sec ]           "
        )
        sys.stdout.flush()
        sys.stdout.write("\n")

        # convert the calculated matrix elements to a form appropriate for sparse matrices
        data_vals, vertex_inds, ray_inds = self.GeomFacs.get_sparse_matrix_data()

        data_dict = {
            "entry_values": data_vals,
            "row_indices": ray_inds,
            "col_indices": vertex_inds,
            "shape": array([self.n_rays, self.n_vertices]),
            "R": self.R,
            "z": self.z,
            "triangles": self.triangle_vertices,
        }

        # save the matrix data
        if save_file is not None:
            savez(save_file, **data_dict)

        return data_dict

    def inequality_checks(self, R, z):
        dz = z[:, None] - self.z_tan[None, :]
        mR = R[:, None] * self.m[None, :]
        R_check = R[:, None] > self.R_tan[None, :]
        t = self.m[None, :] * (R[:, None] - self.R_tan[None, :])
        rgn_A = (dz > -mR).all(axis=0)
        rgn_B = (dz < mR).all(axis=0)
        rgn_C = ((t < dz) & (dz < -t) & R_check).all(axis=0)
        rgn_D = (~R_check).all(axis=0)
        return ~(rgn_A | rgn_B | rgn_C | rgn_D)

    def edge_hyperbola_intersections(self, R0, z0, uR, uz, w):
        u_ratio = uR / uz
        alpha = R0 + (self.pixels[:, 2] - z0) * u_ratio
        beta = self.rays[:, 2] * u_ratio

        # calculate the quadratic coefficients
        a = self.q2 - beta ** 2
        b = self.q1 - 2 * alpha * beta
        c = self.q0 - alpha ** 2

        # use the descriminant to check for the existence of the roots
        D = b ** 2 - 4 * a * c
        i = (D >= 0).nonzero()

        # where roots exists, calculate them
        intersections = full([self.n_rays, 2], nan)
        sqrt_D = sqrt(D[i])
        twice_a = 2 * a[i]
        intersections[i, 0] = -(b[i] + sqrt_D) / twice_a
        intersections[i, 1] = -(b[i] - sqrt_D) / twice_a

        # convert the ray-distances of the intersections to side-displacements
        side_displacements = (
            intersections * self.rays[:, 2, None] + self.pixels[:, 2, None] - z0
        ) / uz
        # reject any intersections which don't occur on the edge itself
        invalid_intersections = absolute(side_displacements) > 0.5 * w
        intersections[invalid_intersections] = nan
        return intersections

    def horizontal_hyperbola_intersections(self, R0, z0, uR, uz, w):
        # find the intersections
        intersections = (z0 - self.pixels[:, 2]) / self.rays[:, 2]
        # convert the ray-distances of the intersections to side-displacements
        R_intersect = sqrt(self.q2 * (intersections - self.L_tan) ** 2 + self.R_tan_sqr)
        side_displacements = (R_intersect - R0) / uR
        # reject any intersections which don't occur on the edge itself
        invalid_intersections = absolute(side_displacements) > 0.5 * w
        intersections[invalid_intersections] = nan
        return intersections

    def process_triangle(self, tri):
        # a hyperbola can at most intersect a triangle six times, so we create space for this.
        intersections = zeros([self.n_rays, 6])
        # loop over each triangle edge and check for intersections
        edges = self.triangle_edges[tri, :]
        for j, edge in enumerate(edges):
            R0 = self.R_edge_mid[edge]
            z0 = self.z_edge_mid[edge]
            uR, uz = self.edge_drn[edge, :]
            w = self.edge_lengths[edge]
            if uz == 0.0:  # if the edge is horizontal, a simplified method can be used
                intersections[:, 2 * j] = self.horizontal_hyperbola_intersections(
                    R0, z0, uR, uz, w
                )
                intersections[:, 2 * j + 1] = nan
            else:  # else we need the general intersection calculation
                intersections[:, 2 * j : 2 * j + 2] = self.edge_hyperbola_intersections(
                    R0, z0, uR, uz, w
                )

        # clip all the intersections so that they lie in the allowed range
        maximum(intersections, 0.0, out=intersections)
        minimum(intersections, self.lengths[:, None], out=intersections)
        # now sort the intersections for each ray in order of distance
        intersections.sort(axis=1)

        # After sorting the intersections by distance along the ray, we now have (up to)
        # three pairs of distances where the ray enters and then leaves the triangle.
        # After the clipping operation, if any of these pairs contain the same value,
        # then that intersection occurs outside the allowed range and must be discarded.

        # loop over each of the three pairs:
        for j in range(3):
            equal = intersections[:, 2 * j] == intersections[:, 2 * j + 1]
            if equal.any():  # discard any pairs with equal distance values
                intersections[equal, 2 * j : 2 * j + 2] = nan

        # re-sort the intersections
        intersections.sort(axis=1)

        # check where valid intersections exist, and count how many there are per ray
        valid_intersections = isfinite(intersections)
        intersection_count = valid_intersections.sum(axis=1)

        # At this point, each ray should have an even number of intersections, if any
        # have an odd number then something has gone wrong, so raise an error.
        if (intersection_count % 2 == 1).any():
            raise ValueError("One or more rays has an odd number of intersections")

        max_intersections = intersection_count.max()
        for j in range(max_intersections // 2):
            indices = (intersection_count >= 2 * (j + 1)).nonzero()[0]
            # calculate the integrals of the barycentric coords over the intersection path
            L1_int, L2_int, L3_int = self.barycentric_coord_integral(
                l1=intersections[:, 2 * j],
                l2=intersections[:, 2 * j + 1],
                inds=indices,
                tri=tri,
            )

            # update the vertices with the integrals
            v1, v2, v3 = self.triangle_vertices[tri, :]
            self.GeomFacs.update_vertex(
                vertex_ind=v1, ray_indices=indices, integral_vals=L1_int
            )
            self.GeomFacs.update_vertex(
                vertex_ind=v2, ray_indices=indices, integral_vals=L2_int
            )
            self.GeomFacs.update_vertex(
                vertex_ind=v3, ray_indices=indices, integral_vals=L3_int
            )

    def barycentric_coord_integral(self, l1, l2, inds, tri):
        l1_slice = l1[inds]
        l2_slice = l2[inds]
        dl = l2_slice - l1_slice

        R_coeff = radius_hyperbolic_integral(
            l1=l1_slice,
            l2=l2_slice,
            l_tan=self.L_tan[inds],
            R_tan_sqr=self.R_tan_sqr[inds],
            sqrt_q2=self.sqrt_q2[inds],
        )

        z_coeff = self.pixels[inds, 2] * dl + 0.5 * self.rays[inds, 2] * (
            l2_slice ** 2 - l1_slice ** 2
        )
        lam1_int = (
            self.lam1_coeffs[tri, 0] * R_coeff
            + self.lam1_coeffs[tri, 1] * z_coeff
            + self.lam1_coeffs[tri, 2] * dl
        )
        lam2_int = (
            self.lam2_coeffs[tri, 0] * R_coeff
            + self.lam2_coeffs[tri, 1] * z_coeff
            + self.lam2_coeffs[tri, 2] * dl
        )
        lam3_int = dl - lam1_int - lam2_int
        return lam1_int, lam2_int, lam3_int

    @staticmethod
    def check_geometry_data(R, z, triangle_inds, ray_starts, ray_ends):
        """
        Check that all the data have the correct shapes / types
        """
        if not all(
            type(arg) is ndarray for arg in [R, z, triangle_inds, ray_starts, ray_ends]
        ):
            raise TypeError(
                """
                [ BarycentricGeometryMatrix error ]
                >> All arguments must be of type numpy.ndarray.
                """
            )

        if R.ndim != 1 or z.ndim != 1 or R.size != z.size:
            raise ValueError(
                """
                [ BarycentricGeometryMatrix error ]
                >> 'R' and 'z' arguments must be 1-dimensional arrays of equal length.
                """
            )

        if triangle_inds.ndim != 2 or triangle_inds.shape[1] != 3:
            raise ValueError(
                """
                [ BarycentricGeometryMatrix error ]
                >> 'triangle_inds' argument must be a 2-dimensional array of shape (N,3)
                >> where 'N' is the total number of triangles.
                """
            )

        if (
            ray_starts.ndim != 2
            or ray_ends.ndim != 2
            or ray_starts.shape[1] != 3
            or ray_ends.shape[1] != 3
            or ray_ends.shape[0] != ray_starts.shape[0]
        ):
            raise ValueError(
                """
                [ BarycentricGeometryMatrix error ]
                >> 'ray_starts' and 'ray_ends' arguments must be 2-dimensional arrays
                >> of shape (M,3), where 'M' is the total number of rays.
                """
            )


def radius_hyperbolic_integral(l1, l2, l_tan, R_tan_sqr, sqrt_q2):
    u1 = sqrt_q2 * (l1 - l_tan)
    u2 = sqrt_q2 * (l2 - l_tan)
    R1 = sqrt(u1 ** 2 + R_tan_sqr)
    R2 = sqrt(u2 ** 2 + R_tan_sqr)

    ratio = (u2 + R2) / (u1 + R1)
    return 0.5 * (u2 * R2 - u1 * R1 + log(ratio) * R_tan_sqr) / sqrt_q2


class GeometryFactors(object):
    def __init__(self):
        self.vertex_map = defaultdict(lambda: 0.0)

    def update_vertex(self, vertex_ind, ray_indices, integral_vals):
        for ray_idx, value in zip(ray_indices, integral_vals):
            self.vertex_map[(vertex_ind, ray_idx)] += value

    def get_sparse_matrix_data(self):
        vertex_inds = array([key[0] for key in self.vertex_map.keys()])
        ray_inds = array([key[1] for key in self.vertex_map.keys()])
        data_vals = array([v for v in self.vertex_map.values()])
        return data_vals, vertex_inds, ray_inds


def build_edge_map(triangles):
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


class Camera(object):
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
        b = 1.0 / sqrt(K ** 2 + 1.0)
        a = -K * b
        self.p1 = array([a, b, 0.0])

        # use cross-product to find second perpendicular
        self.p2 = cross(self.du, self.p1)

        # identity matrix
        self.I = identity(3)

        # calculate the ray directions
        tan_x = tan(self.x_angles)
        tan_y = tan(self.y_angles)
        norm = sqrt(1 + (tan_x ** 2)[:, None] + (tan_y ** 2)[None, :])
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
