from numpy import full, minimum, maximum, asanyarray, ndarray


def edge_rectangle_intersection(
    R_lims: tuple, z_lims: tuple, R_edges: ndarray, z_edges: ndarray
) -> ndarray:
    """
    Checks whether a given set of edges intersects an axis-aligned rectangle.
    Contact with a side or corner of the rectangle counts as an intersection.

    :param R_lims: \
        A tuple specifying the major radius at the left and right sides of the
        rectangle in the form ``(R_left, R_right)``.

    :param z_lims: \
        A tuple specifying the z-height at the bottom and top sides of the
        rectangle in the form ``(z_bottom, z_top)``.

    :param R_edges: \
        A 2D numpy array specifying the major-radius value at the ends of each
        edge. The array must have shape ``(N, 2)`` where ``N`` is the total number
        of edges.

    :param z_edges: \
        A 2D numpy array specifying the z-height value at the ends of each
        edge. The array must have shape ``(N, 2)`` where ``N`` is the total number
        of edges.

    :return intersections: \
        An array containing the indices of any edges which intersect
        the specified rectangle.
    """

    def check_input_array(array, array_name):
        new_array = asanyarray(array)
        if new_array.shape == (2,):
            new_array = new_array.reshape((1, 2))
        if len(new_array.shape) != 2:
            raise ValueError(
                f"Wrong shape for input {array_name}: expected (N, 2), got {new_array.shape}"
            )
        return new_array

    R_edges = check_input_array(R_edges, "R_edges")
    z_edges = check_input_array(z_edges, "z_edges")

    # first rule out the majority of edges in the mesh
    right_check = (R_edges > R_lims[1]).all(axis=1)
    left_check = (R_edges < R_lims[0]).all(axis=1)
    top_check = (z_edges > z_lims[1]).all(axis=1)
    bottom_check = (z_edges < z_lims[0]).all(axis=1)
    i = (~(right_check | left_check | top_check | bottom_check)).nonzero()[0]

    # Clip each segment's parameter interval [0, 1] against both coordinate bounds.
    t_min = full(i.size, 0.0)
    t_max = full(i.size, 1.0)
    for edges, limits in ((R_edges, R_lims), (z_edges, z_lims)):
        coords = edges[i, :].astype(float, copy=False)
        delta = coords[:, 1] - coords[:, 0]
        # Parallel segments already passed the bounding-box check for this axis.
        moving = delta != 0.0
        t0 = (limits[0] - coords[moving, 0]) / delta[moving]
        t1 = (limits[1] - coords[moving, 0]) / delta[moving]
        t_min[moving] = maximum(t_min[moving], minimum(t0, t1))
        t_max[moving] = minimum(t_max[moving], maximum(t0, t1))

    return i[t_min <= t_max]
