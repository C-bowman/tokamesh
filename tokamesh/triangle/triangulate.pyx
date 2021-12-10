from tokamesh.triangle cimport ctriangulate
import numpy as np

cdef class TriangleOptions:
    cdef ctriangulate.triangulateio _c_triangulateio

    def __init__(self, double[::1] points=None, int[::1] segments=None, double[::1] holes=None):

        if points is not None:
            self._c_triangulateio.pointlist = &points[0]
            self._c_triangulateio.numberofpoints = points.shape[0] // 2
        else:
            self._c_triangulateio.pointlist = NULL
            self._c_triangulateio.numberofpoints = 0

        self._c_triangulateio.pointattributelist = NULL
        self._c_triangulateio.pointmarkerlist = NULL
        self._c_triangulateio.numberofpointattributes = 0

        if segments is not None:
            self._c_triangulateio.segmentlist = &segments[0]
            self._c_triangulateio.segmentmarkerlist = NULL
            self._c_triangulateio.numberofsegments = segments.shape[0] // 2
        else:
            self._c_triangulateio.segmentlist = NULL
            self._c_triangulateio.segmentmarkerlist = NULL
            self._c_triangulateio.numberofsegments = 0

        if holes is not None:
            self._c_triangulateio.holelist = &holes[0]
            self._c_triangulateio.numberofholes = holes.shape[0] // 2
        else:
            self._c_triangulateio.holelist = NULL
            self._c_triangulateio.numberofholes = 0

        self._c_triangulateio.trianglelist = NULL
        self._c_triangulateio.triangleattributelist = NULL
        self._c_triangulateio.trianglearealist = NULL
        self._c_triangulateio.neighborlist = NULL
        self._c_triangulateio.numberoftriangles = 0
        self._c_triangulateio.numberofcorners = 0
        self._c_triangulateio.numberoftriangleattributes = 0

        self._c_triangulateio.regionlist = NULL
        self._c_triangulateio.numberofregions = 0
        self._c_triangulateio.edgelist = NULL
        self._c_triangulateio.edgemarkerlist = NULL
        self._c_triangulateio.normlist = NULL
        self._c_triangulateio.numberofedges = 0

    @property
    def triangles(self):
        num_triangles = self._c_triangulateio.numberoftriangles
        return np.asarray(<int[:num_triangles * 3]> self._c_triangulateio.trianglelist).reshape((num_triangles, 3))

    @property
    def elements(self):
        num_elements = self._c_triangulateio.numberofpoints
        return np.asarray(<double[:num_elements * 2]> self._c_triangulateio.pointlist).reshape((num_elements, 2))


def interleave(array1, array2):
    """Interleave the two arrays into a 1D array"""
    return np.array((array1, array2)).T.flatten()


def triangulate(
    outer_boundary=None, inner_boundary=None, void_markers=None, max_area=0.0005
):
    """
    A Python interface for the 'Triangle' C-code which is packaged with Tokamesh.

    :param outer_boundary:
    :param inner_boundary:
    :param void_markers:
    :param max_area:
    :return:
    """

    points = interleave(*outer_boundary)

    if inner_boundary is None:
        n_cuts = 0
        segments = None
    else:
        inner_r = inner_boundary[0]
        inner_z = inner_boundary[1]
        n_cuts = len(inner_r) - 1

        inner_points_r = interleave(inner_r[:-1], inner_r[1:])
        inner_points_z = interleave(inner_z[:-1], inner_z[1:])
        inner_points = interleave(inner_points_r, inner_points_z)

        points = np.concatenate((points, inner_points))

        n_wall = len(outer_boundary[0])
        total = n_wall + (2 * n_cuts)
        segments = np.concatenate((
            interleave(
                np.arange(n_wall - 1),
                np.arange(1, n_wall),
            ),
            interleave(
                np.arange(n_wall, total - 1, 2),
                np.arange(n_wall + 1, total, 2),
            ),
        ), dtype=np.int32)

    if void_markers is not None:
        holes = interleave(*void_markers)
    else:
        holes = None

    input_options = TriangleOptions(points, segments, holes)

    # Initialise output
    output = TriangleOptions()

    # triangle flags:
    # - Q: quiet
    # - B: suppress boundary markers (we don't use them)
    # - z: number vertices from zero
    # - p: triangulate a planar straight line graph
    # - q: minimum angle of 20 degrees
    # - a: maximum area constraint
    triangle_switches = f"QPBzpqa{max_area:.12f}".encode()

    ctriangulate.triangulate(triangle_switches, &(input_options._c_triangulateio),
                             &(output._c_triangulateio), NULL)

    return output.elements, output.triangles
