try:
    from importlib.metadata import version, PackageNotFoundError
except (ModuleNotFoundError, ImportError):
    from importlib_metadata import version, PackageNotFoundError

try:
    __version__ = version("tokamesh")
except PackageNotFoundError:
    from setuptools_scm import get_version

    __version__ = get_version(root="..", relative_to=__file__)

__all__ = ["__version__"]

from numpy import searchsorted, stack, log2, floor, unique, atleast_1d
from numpy import arange, linspace, int64, full, zeros, meshgrid, ndarray
from itertools import product
from tokamesh.intersection import edge_rectangle_intersection
from tokamesh.geometry import build_edge_map


class TriangularMesh(object):
    """
    Class for performing operations with a triangular mesh, such as
    interpolation and plotting.

    :param R: \
        The major radius of each mesh vertex as a 1D numpy array.

    :param z: \
        The z-height of each mesh vertex as a 1D numpy array.

    :param triangles: \
        A 2D numpy array of integers specifying the indices of the vertices which form
        each of the triangles in the mesh. The array must have shape ``(N,3)`` where
        ``N`` is the total number of triangles.
    """

    def __init__(self, R, z, triangles):

        for name, obj in [("R", R), ("z", z), ("triangles", triangles)]:
            if not isinstance(obj, ndarray):
                raise TypeError(
                    f"""\n
                    [ TriangularMesh error ]
                    >> The '{name}' argument of TriangularMesh should have type:
                    >> {ndarray}
                    >> but instead has type:
                    >> {type(obj)}
                    """
                )

        for name, obj in [("R", R), ("z", z)]:
            if obj.squeeze().ndim > 1:
                raise ValueError(
                    f"""\n
                    [ TriangularMesh error ]
                    >> The '{name}' argument of TriangularMesh should be
                    >> a 1D array, but given array has shape {obj.shape}.
                    """
                )

        if R.size != z.size:
            raise ValueError(
                f"""\n
                [ TriangularMesh error ]
                >> The 'R' and 'z' arguments of TriangularMesh should be
                >> of equal size, but given arrays have sizes {R.size} and {z.size}.
                """
            )

        if triangles.squeeze().ndim != 2 or triangles.squeeze().shape[1] != 3:
            raise ValueError(
                f"""\n
                [ TriangularMesh error ]
                >> The 'triangles' argument must have shape (num_triangles, 3)
                >> but given array has shape {triangles.shape}.
                """
            )

        self.R = R.squeeze()
        self.z = z.squeeze()
        self.triangle_vertices = triangles.squeeze()
        self.n_vertices = self.R.size
        self.n_triangles = self.triangle_vertices.shape[0]

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

        # Construct a mapping from triangles to edges, and edges to vertices
        self.triangle_edges, self.edge_vertices, _ = build_edge_map(
            self.triangle_vertices
        )
        self.R_edges = self.R[self.edge_vertices]
        self.z_edges = self.z[self.edge_vertices]
        self.n_edges = self.edge_vertices.shape[0]

        # store info about the bounds of the mesh
        self.R_limits = [self.R.min(), self.R.max()]
        self.z_limits = [self.z.min(), self.z.max()]

        self.build_binary_trees()

    def build_binary_trees(self):
        # we now divide the bounding rectangle of the mesh into
        # a rectangular grid, and create a mapping between each
        # grid cell and all triangles which intersect it.

        # find an appropriate depth for each tree
        R_extent = self.R[self.triangle_vertices].ptp(axis=1).mean()
        z_extent = self.z[self.triangle_vertices].ptp(axis=1).mean()
        R_depth = max(
            int(floor(log2((self.R_limits[1] - self.R_limits[0]) / R_extent))), 2
        )
        z_depth = max(
            int(floor(log2((self.z_limits[1] - self.z_limits[0]) / z_extent))), 2
        )
        # build binary trees for each axis
        self.R_tree = BinaryTree(R_depth, self.R_limits)
        self.z_tree = BinaryTree(z_depth, self.z_limits)

        # now build a map between rectangle centres and a list of
        # all triangles which intersect that rectangle
        self.tree_map = {}
        for i, j in product(range(self.R_tree.nodes), range(self.z_tree.nodes)):
            # limits of the rectangle
            R_lims = self.R_tree.edges[i : i + 2]
            z_lims = self.z_tree.edges[j : j + 2]
            # find all edges which intersect the rectangle
            edge_inds = edge_rectangle_intersection(
                R_lims, z_lims, self.R_edges, self.z_edges
            )
            edge_bools = zeros(self.n_edges, dtype=int64)
            edge_bools[edge_inds] = 1
            # use this to find which triangles intersect the rectangle
            triangle_bools = edge_bools[self.triangle_edges].any(axis=1)
            # add the indices of these triangles to the dict
            if triangle_bools.any():
                self.tree_map[(i, j)] = triangle_bools.nonzero()[0]

    def interpolate(self, R, z, vertex_values):
        """
        Given the values of a function at each vertex of the mesh, use barycentric
        interpolation to approximate the function at a chosen set of points. Any
        points which lie outside the mesh will be assigned a value of zero.

        :param R: \
            The major-radius of each interpolation point as a numpy array.

        :param z: \
            The z-height of each interpolation point as a numpy array.

        :param vertex_values: \
            The function value at each mesh vertex as a 1D numpy array.

        :return: \
            The interpolated function values as a numpy array.
        """
        if type(vertex_values) is not ndarray or vertex_values.ndim != 1:
            raise TypeError(
                """\n
                [ TriangularMesh error ]
                >> The 'vertex_values' argument of the TriangularMesh.interpolate
                >> method must have type numpy.ndarray, and have only one dimension.
                """
            )

        if vertex_values.size != self.n_vertices:
            raise ValueError(
                f"""\n
                [ TriangularMesh error ]
                >> The size of 'vertex_values' argument of the TriangularMesh.interpolate
                >> must be equal to the number of mesh vertices.
                >> The mesh has {self.n_vertices} vertices but given array is of size {vertex_values.size}.
                """
            )

        R_vals = atleast_1d(R)
        z_vals = atleast_1d(z)

        if R_vals.shape != z_vals.shape:
            raise ValueError(
                """\n
                [ TriangularMesh error ]
                >> The 'R' and 'z' arrays passed to the TriangularMesh.interpolate
                >> method are of inconsistent shapes - their shapes must be equal.
                """
            )

        input_shape = R_vals.shape
        if len(input_shape) > 1:
            R_vals = R_vals.flatten()
            z_vals = z_vals.flatten()

        # lookup sets of coordinates are in each grid cell
        unique_coords, slices, indices = self.grid_lookup(R_vals, z_vals)
        # loop over each unique grid coordinate
        interpolated_values = zeros(R_vals.size)
        for v, slc in zip(unique_coords, slices):
            # only need to proceed if the current coordinate contains triangles
            key = (v[0], v[1])
            if key in self.tree_map:
                # get triangles intersecting this cell
                search_triangles = self.tree_map[key]
                cell_indices = indices[slc]  # the indices of points inside this cell
                # get the barycentric coord values of each point, and the
                # index of the triangle which contains them
                coords, container_triangles = self.bary_coords(
                    R_vals[cell_indices], z_vals[cell_indices], search_triangles
                )
                # get the values of the vertices for the triangles which contain the points
                vals = vertex_values[self.triangle_vertices[container_triangles, :]]
                # take the dot-product of the coordinates and the vertex
                # values to get the interpolated value
                interpolated_values[cell_indices] = (coords * vals).sum(axis=1)
        if len(input_shape) > 1:
            interpolated_values.resize(input_shape)
        return interpolated_values

    def find_triangle(self, R, z):
        """
        Find the indices of the triangles which contain a given set of points.

        :param R: \
            The major-radius of each point as a numpy array.

        :param z: \
            The z-height of each point as a numpy array.

        :return: \
            The indices of the triangles which contain each point as numpy array.
            Any points which are not inside a triangle are given an index of -1.
        """
        R_vals = atleast_1d(R)
        z_vals = atleast_1d(z)

        if R_vals.shape != z_vals.shape:
            raise ValueError(
                """\n
                [ TriangularMesh error ]
                >> The 'R' and 'z' arrays passed to the TriangularMesh.interpolate
                >> method are of inconsistent shapes - their shapes must be equal.
                """
            )

        input_shape = R_vals.shape
        if len(input_shape) > 1:
            R_vals = R_vals.flatten()
            z_vals = z_vals.flatten()

        # lookup sets of coordinates are in each grid cell
        unique_coords, slices, indices = self.grid_lookup(R_vals, z_vals)
        # loop over each unique grid coordinate
        triangle_indices = full(R_vals.size, fill_value=-1, dtype=int)
        for v, slc in zip(unique_coords, slices):
            # only need to proceed if the current coordinate contains triangles
            key = (v[0], v[1])
            if key in self.tree_map:
                # get triangles intersecting this cell
                search_triangles = self.tree_map[key]
                cell_indices = indices[slc]  # the indices of points inside this cell
                # get the barycentric coord values of each point, and the
                # index of the triangle which contains them
                _, container_triangles = self.bary_coords(
                    R_vals[cell_indices], z_vals[cell_indices], search_triangles
                )
                triangle_indices[cell_indices] = container_triangles
        if len(input_shape) > 1:
            triangle_indices.resize(input_shape)
        return triangle_indices

    def grid_lookup(self, R, z):
        # first determine in which cell each point lies using the binary trees
        grid_coords = zeros([R.size, 2], dtype=int64)
        grid_coords[:, 0] = self.R_tree.lookup_index(R)
        grid_coords[:, 1] = self.z_tree.lookup_index(z)
        # find the set of unique grid coordinates
        unique_coords, inverse, counts = unique(
            grid_coords, axis=0, return_inverse=True, return_counts=True
        )
        # now create an array of indices which are ordered according
        # to which of the unique values they match
        indices = inverse.argsort()
        # build a list of slice objects which addresses those indices
        # which match each unique coordinate
        ranges = counts.cumsum()
        slices = [slice(0, ranges[0])]
        slices.extend([slice(*ranges[i : i + 2]) for i in range(ranges.size - 1)])
        return unique_coords, slices, indices

    def bary_coords(self, R, z, search_triangles):
        Q = stack([atleast_1d(R), atleast_1d(z), full(R.size, fill_value=1.0)], axis=0)
        lam1 = self.lam1_coeffs[search_triangles, :].dot(Q)
        lam2 = self.lam2_coeffs[search_triangles, :].dot(Q)
        lam3 = 1 - lam1 - lam2
        bools = (lam1 >= 0.0) & (lam2 >= 0.0) & (lam3 >= 0.0)
        i1, i2 = bools.nonzero()

        coords = zeros([R.size, 3])
        coords[i2, 0] = lam1[i1, i2]
        coords[i2, 1] = lam2[i1, i2]
        coords[i2, 2] = lam3[i1, i2]
        container_triangles = full(R.size, fill_value=-1)
        container_triangles[i2] = search_triangles[i1]
        return coords, container_triangles

    def draw(self, ax, **kwargs):
        """
        Draw the mesh using a given ``matplotlib.pyplot`` axis object.

        :param ax: \
            A ``matplotlib.pyplot`` axis object on which the mesh will be drawn by
            calling the 'plot' method of the object.

        :param kwargs: \
            Any valid keyword argument of ``matplotlib.pyplot.plot`` may be given in
            order to change the properties of the plot.
        """
        if ("color" not in kwargs) and ("c" not in kwargs):
            kwargs["color"] = "black"
        ax.plot(self.R_edges[0, :].T, self.z_edges[0, :].T, **kwargs)
        if "label" in kwargs:
            kwargs["label"] = None
        ax.plot(self.R_edges[1:, :].T, self.z_edges[1:, :].T, **kwargs)

    def get_field_image(self, vertex_values, shape=(150, 150), pad_fraction=0.01):
        """
        Given the value of a field at each mesh vertex, use interpolation to generate
        an image of the field across the whole mesh.

        :param vertex_values: \
            The value of the field being plotted at each vertex of the mesh as a 1D numpy array.

        :param shape: \
            A tuple of two integers specifying the dimensions of the image.

        :param pad_fraction: \
            The fraction of the mesh width/height used as padding to create a gap between
            the edge of the mesh and the edge of the plot.

        :return R_axis, z_axis, field_image: \
            ``R_axis`` is a 1D array of the major-radius value of each column of the image array.
            ``z_axis`` is a 1D array of the z-height value of each column of the image array.
            ``field_image`` is a 2D array of the interpolated field values. Any points outside
            the mesh are assigned a value of zero.
        """
        R_pad = (self.R_limits[1] - self.R_limits[0]) * pad_fraction
        z_pad = (self.R_limits[1] - self.R_limits[0]) * pad_fraction

        R_axis = linspace(self.R_limits[0] - R_pad, self.R_limits[1] + R_pad, shape[0])
        z_axis = linspace(self.z_limits[0] - z_pad, self.z_limits[1] + z_pad, shape[1])
        R_grid, z_grid = meshgrid(R_axis, z_axis)

        image = self.interpolate(
            R_grid.flatten(), z_grid.flatten(), vertex_values=vertex_values
        )
        image.resize((shape[1], shape[0]))
        return R_axis, z_axis, image.T


class BinaryTree:
    """
    divides the range specified by limits into n = 2**layers equal regions,
    and builds a binary tree which allows fast look-up of which of region
    contains a given value.

    :param int layers: number of layers that make up the tree
    :param limits: tuple of the lower and upper bounds of the look-up region.
    """

    def __init__(self, layers, limits):
        self.layers = layers
        self.nodes = 2 ** self.layers
        self.lims = limits
        self.edges = linspace(limits[0], limits[1], self.nodes + 1)
        self.mids = 0.5 * (self.edges[1:] + self.edges[:-1])

        self.indices = full(self.nodes + 2, fill_value=-1, dtype=int64)
        self.indices[1:-1] = arange(self.nodes)

    def lookup_index(self, values):
        return self.indices[searchsorted(self.edges, values)]
