from numpy import full, nan, asanyarray, ndarray


def edge_rectangle_intersection(
    R_lims: tuple, z_lims: tuple, R_edges: ndarray, z_edges: ndarray
) -> ndarray:
    """
    Checks whether a given set of edges intersects a axis-aligned rectangle.

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

    # extract data for the edges which might intersect
    R = R_edges[i, :]
    z = z_edges[i, :]
    dR = R[:, 1] - R[:, 0]
    dz = z[:, 1] - z[:, 0]
    nonzeros = ((dR != 0.0) & (dz != 0.0)).nonzero()[0]
    edge_grads = full(R.shape[0], fill_value=nan)
    edge_grads[nonzeros] = dz[nonzeros] / dR[nonzeros]
    edge_const = z[:, 0] - edge_grads * R[:, 0]
    # first we check to see if any of the points are inside the rectangle
    inside_check = (
        (R_lims[0] < R) & (R_lims[1] > R) & (z_lims[0] < z) & (z_lims[1] > z)
    ).any(axis=1)

    # now check for intersections with the rectangle's left side
    left_z = edge_grads * R_lims[0] + edge_const
    left_split_check = (R - R_lims[0]).prod(axis=1) <= 0.0
    left_intersect = (z_lims[0] < left_z) & (z_lims[1] > left_z) & left_split_check

    # now the right side
    right_z = edge_grads * R_lims[1] + edge_const
    right_split_check = (R - R_lims[1]).prod(axis=1) <= 0.0
    right_intersect = (z_lims[0] < right_z) & (z_lims[1] > right_z) & right_split_check

    # now the bottom side
    bottom_R = (z_lims[0] - edge_const) / edge_grads
    bottom_split_check = (z - z_lims[0]).prod(axis=1) <= 0.0
    bottom_intersect = (
        (R_lims[0] < bottom_R) & (R_lims[1] > bottom_R) & bottom_split_check
    )

    # now the top side
    top_R = (z_lims[1] - edge_const) / edge_grads
    top_split_check = (z - z_lims[1]).prod(axis=1) <= 0.0
    top_intersect = (R_lims[0] < top_R) & (R_lims[1] > top_R) & top_split_check

    # if any of the conditions are met, then there is an intersection
    intersections = (
        inside_check
        | left_intersect
        | right_intersect
        | bottom_intersect
        | top_intersect
    )
    return i[intersections]
