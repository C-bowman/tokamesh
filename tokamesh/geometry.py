from numpy import sqrt, log
from numpy import absolute, nan, isfinite, minimum, maximum, isnan
from numpy import array, full, zeros, stack, savez, concatenate
from numpy import ndarray, finfo
from collections import defaultdict
from time import perf_counter
import sys

from tokamesh.utilities import build_edge_map


class BarycentricGeometryMatrix:
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

    def __init__(
        self,
        R: ndarray,
        z: ndarray,
        triangles: ndarray,
        ray_origins: ndarray,
        ray_ends: ndarray,
    ):
        # first check the validity of the data
        self.check_geometry_data(R, z, triangles, ray_origins, ray_ends)

        # store the mesh data
        self.R = R
        self.z = z
        self.triangle_vertices = triangles
        self.n_vertices = self.R.size
        self.n_triangles = self.triangle_vertices.shape[0]
        self.GeomFacs = GeometryFactors()

        # calculate the ray data
        diffs = ray_ends - ray_origins
        self.lengths = sqrt((diffs**2).sum(axis=1))
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
        self.L_tan = -0.5 * self.q1 / self.q2  # ray-distance of the tangency point
        self.R_tan_sqr = self.q0 + 0.5 * self.q1 * self.L_tan
        self.R_tan = sqrt(self.R_tan_sqr)  # major radius of the tangency point
        # z-height of the tangency point
        self.z_tan = self.pixels[:, 2] + self.rays[:, 2] * self.L_tan
        # gradient of the hyperbola asymptote line
        self.m = self.rays[:, 2] / sqrt(self.q2)

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

    def calculate(self, save_file=None) -> dict:
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
        group_size = max(min(int(1.0 / dt), (self.n_triangles - 1) // 4), 1)
        remainder = (self.n_triangles - 1) % group_size
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
        if remainder != 0:
            [
                self.process_triangle(i)
                for i in range(self.n_triangles - remainder, self.n_triangles)
            ]

        t_elapsed = perf_counter() - t_start
        mins, secs = divmod(t_elapsed, 60)
        hrs, mins = divmod(mins, 60)
        time_taken = f"{int(hrs)}:{int(mins):02d}:{int(secs):02d}"
        sys.stdout.write(
            f"\r >> Calculating geometry matrix:  [ completed in {time_taken} ]           "
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
        a = self.q2 - beta**2
        b = self.q1 - 2 * alpha * beta
        c = self.q0 - alpha**2

        # use the discriminant to check for the existence of the roots
        D = b**2 - 4 * a * c
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

    def process_triangle(self, tri_index: int):
        # a hyperbola can at most intersect a triangle six times, so we create space for this.
        intersections = zeros([self.n_rays, 6])
        # loop over each triangle edge and check for intersections
        edges = self.triangle_edges[tri_index, :]
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
            raise ValueError(
                f"""\n\n
                \r[ BarycentricGeometryMatrix error ]
                \r>> One or more rays has an odd number of intersections with
                \r>> triangle {tri_index}. This is typically caused by insufficient
                \r>> floating-point precision in the intersection calculations.
                """
            )

        max_intersections = intersection_count.max()
        for j in range(max_intersections // 2):
            indices = (intersection_count >= 2 * (j + 1)).nonzero()[0]
            # calculate the integrals of the barycentric coords over the intersection path
            L1_int, L2_int, L3_int = self.barycentric_coord_integral(
                l1=intersections[:, 2 * j],
                l2=intersections[:, 2 * j + 1],
                inds=indices,
                tri_index=tri_index,
            )

            # update the vertices with the integrals
            v1, v2, v3 = self.triangle_vertices[tri_index, :]
            self.GeomFacs.update_vertex(
                vertex_ind=v1, ray_indices=indices, integral_vals=L1_int
            )
            self.GeomFacs.update_vertex(
                vertex_ind=v2, ray_indices=indices, integral_vals=L2_int
            )
            self.GeomFacs.update_vertex(
                vertex_ind=v3, ray_indices=indices, integral_vals=L3_int
            )

    def barycentric_coord_integral(self, l1, l2, inds, tri_index: int):
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
            l2_slice**2 - l1_slice**2
        )
        lam1_int = (
            self.lam1_coeffs[tri_index, 0] * R_coeff
            + self.lam1_coeffs[tri_index, 1] * z_coeff
            + self.lam1_coeffs[tri_index, 2] * dl
        )
        lam2_int = (
            self.lam2_coeffs[tri_index, 0] * R_coeff
            + self.lam2_coeffs[tri_index, 1] * z_coeff
            + self.lam2_coeffs[tri_index, 2] * dl
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
                """\n\n
                \r[ BarycentricGeometryMatrix error ]
                \r>> All arguments must be of type numpy.ndarray.
                """
            )

        if R.ndim != 1 or z.ndim != 1 or R.size != z.size:
            raise ValueError(
                """\n\n
                \r[ BarycentricGeometryMatrix error ]
                \r>> 'R' and 'z' arguments must be 1-dimensional arrays of equal length.
                """
            )

        if triangle_inds.ndim != 2 or triangle_inds.shape[1] != 3:
            raise ValueError(
                """\n\n
                \r[ BarycentricGeometryMatrix error ]
                \r>> 'triangle_inds' argument must be a 2-dimensional array of shape (N,3)
                \r>> where 'N' is the total number of triangles.
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
                """\n\n
                \r[ BarycentricGeometryMatrix error ]
                \r>> 'ray_starts' and 'ray_ends' arguments must be 2-dimensional arrays
                \r>> of shape (M,3), where 'M' is the total number of rays.
                """
            )

        for tag, arr in [
            ("R", R),
            ("z", z),
            ("ray_starts", ray_starts),
            ("ray_ends", ray_ends),
        ]:
            float_precision = finfo(arr.dtype).precision
            if float_precision < 15:
                raise ValueError(
                    f"""\n\n
                    \r[ BarycentricGeometryMatrix error ]
                    \r>> The '{tag}' argument array has a data-type of {arr.dtype}
                    \r>> with a decimal precision of {float_precision}.
                    \r>> Arrays should use at least 64-bit floats, such that the
                    \r>> decimal precision is 15 or above.
                    """
                )


def radius_hyperbolic_integral(l1, l2, l_tan, R_tan_sqr, sqrt_q2):
    u1 = sqrt_q2 * (l1 - l_tan)
    u2 = sqrt_q2 * (l2 - l_tan)
    R1 = sqrt(u1**2 + R_tan_sqr)
    R2 = sqrt(u2**2 + R_tan_sqr)

    ratio = (u2 + R2) / (u1 + R1)
    return 0.5 * (u2 * R2 - u1 * R1 + log(ratio) * R_tan_sqr) / sqrt_q2


class GeometryFactors:
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


def linear_geometry_matrix(
    R: ndarray, ray_origins: ndarray, ray_ends: ndarray
) -> ndarray:
    """
    Calculates a geometry matrix using 1D linear-interpolation basis functions
    assuming that the emission varies only as a function of major-radius.

    :param R: \
        The major-radius position of each basis function in ascending order.

    :param ray_origins: \
        The ``(x,y,z)`` position vectors of the origin of each ray (i.e. line-of-sight)
        as a 2D numpy array. The array must have shape ``(M,3)`` where ``M`` is the
        total number of rays.

    :param ray_ends: \
        The ``(x,y,z)`` position vectors of the end-points of each ray (i.e. line-of-sight)
        as a 2D numpy array. The array must have shape ``(M,3)`` where ``M`` is the
        total number of rays.
    """
    # verify the inputs are all numpy arrays
    for name, var in [("R", R), ("ray_origins", ray_origins), ("ray_ends", ray_ends)]:
        if type(var) is not ndarray:
            raise TypeError(
                f"""\n
                \r[ linear_geometry_matrix error ]
                \r>> '{name}' argument must have type: 
                \r>> {ndarray}
                \r>> but instead has type:
                \r>> {type(var)}
                """
            )

    # check all shapes of the inputs
    n_points = R.size
    if R.ndim != 1 or R.size < 3:
        raise ValueError(
            f"""\n
            \r[ linear_geometry_matrix error ]
            \r>> 'R' argument must have one dimension and at least 3 elements,
            \r>> but instead has {R.ndim} dimensions and {R.size} elements.
            """
        )

    if (R[1:] - R[:-1] <= 0).any():
        raise ValueError(
            """\n
            \r[ linear_geometry_matrix error ]
            \r>> 'R' argument be sorted in ascending order, and contain only unique values.
            """
        )

    good_rays = (
        ray_origins.ndim == ray_ends.ndim == 2
        and ray_origins.shape[0] == ray_origins.shape[0]
        and ray_origins.shape[1] == ray_origins.shape[1] == 3
    )
    if not good_rays:
        raise ValueError(
            f"""\n
            \r[ linear_geometry_matrix error ]
            \r>> 'ray_origins' and 'ray_ends' must both have shape (N, 3)
            \r>> where 'N' is the number of rays. Instead, they have shapes
            \r>> {ray_origins.shape}, {ray_ends.shape}
            \r>> respectively.
            """
        )

    # calculate linear basis function coefficients
    grads = zeros([n_points - 1, 2])
    grads[:, 1] = 1.0 / (R[1:] - R[:-1])
    grads[:, 0] = -grads[:, 1]
    offsets = zeros([n_points - 1, 2])
    offsets[:, 0] = -grads[:, 0] * R[1:]
    offsets[:, 1] = -grads[:, 1] * R[:-1]

    # calculate the ray data
    diffs = ray_ends - ray_origins
    lengths = sqrt((diffs**2).sum(axis=1))
    rays = diffs / lengths[:, None]
    n_rays = lengths.size

    # coefficients for the quadratic representation of the ray radius
    q0 = ray_origins[:, 0] ** 2 + ray_origins[:, 1] ** 2
    q1 = 2 * (ray_origins[:, 0] * rays[:, 0] + ray_origins[:, 1] * rays[:, 1])
    q2 = rays[:, 0] ** 2 + rays[:, 1] ** 2
    sqrt_q2 = sqrt(q2)

    # pre-calculate quantities used in intersection and integral
    L_tan = -0.5 * q1 / q2  # ray-distance of the tangency point
    L_tan_sqr = L_tan**2
    R_tan_sqr = q0 + 0.5 * q1 * L_tan  # major radius of the tangency point squared

    # each possible pairing of ray and radius can have up to two intersections
    intersections = zeros([n_rays, n_points, 2])
    # solve for the intersections via quadratic formula
    c = q0[:, None] - R[None, :] ** 2
    discriminant = L_tan_sqr[:, None] - c / q2[:, None]
    discriminant[discriminant < 0] = nan
    roots = sqrt(discriminant)
    intersections[:, :, 0] = L_tan[:, None] - roots
    intersections[:, :, 1] = L_tan[:, None] + roots

    # clip all the intersections so that they lie in the allowed range
    maximum(intersections, 0.0, out=intersections)
    minimum(intersections, lengths[:, None, None], out=intersections)
    intersections[isnan(intersections)] = 0.0

    # now loop over each cell, and add the contribution to the geometry matrix
    G = zeros([n_rays, n_points])
    for cell in range(n_points - 1):
        cell_intersects = concatenate(
            [intersections[:, cell, :], intersections[:, cell + 1, :]], axis=1
        )
        cell_intersects.sort(axis=1)

        for i, j in [(0, 1), (2, 3)]:
            integral = radius_hyperbolic_integral(
                l1=cell_intersects[:, i],
                l2=cell_intersects[:, j],
                l_tan=L_tan,
                R_tan_sqr=R_tan_sqr,
                sqrt_q2=sqrt_q2,
            )
            dl = cell_intersects[:, j] - cell_intersects[:, i]
            G[:, cell] += integral * grads[cell, 0] + offsets[cell, 0] * dl
            G[:, cell + 1] += integral * grads[cell, 1] + offsets[cell, 1] * dl
    return G
