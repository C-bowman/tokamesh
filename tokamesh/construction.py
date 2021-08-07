
from numpy import array, zeros, linspace, sqrt, ceil, sin, cos, arctan2, in1d, arange, int64


def equilateral_mesh(x_range=(0,10), y_range=(0,5), scale=1.0, rotation=None, pivot=(0,0)):
    """
    Construct a mesh from equilateral triangles which fills a rectangular region.

    :param x_range: \
        A tuple in the form `(x_min, x_max)` specifying the range of the x-axis to cover with triangles.

    :param y_range: \
        A tuple in the form `(y_min, y_max)` specifying the range of the y-axis to cover with triangles.

    :param float scale: \
        The side-length of the triangles.

    :param rotation: \
        Angle (in radians) by which the mesh will be rotated.

    :param pivot: \
        Pivot point around which the rotation is applied.

    :return x_vert, y_vert, triangles: \
        `x_vert` is the x-position of the vertices as a 1D array. `y_vert` the is y-position
        of the vertices as a 1D array. `triangles` is a 2D array of integers of shape `(N,3)`
        specifying the indices of the vertices which form each triangle in the mesh, where
        `N` is the total number of triangles.
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
        each of the triangles in the mesh. The array must have shape (N,3) where 'N' is
        the total number of triangles.

    :param bools: \
        A 1D array of boolean values corresponding to the vertices, which is `True` for
        any vertices which are to be removed from the mesh.

    :return R, z, triangles: \
        The `R`, `z` and `triangles` arrays (defined as described above) with the
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