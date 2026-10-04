from tokamesh.intersection import edge_rectangle_intersection

import numpy as np
import pytest


@pytest.mark.parametrize(
    "R_edges, z_edges",
    [
        ((0, 1), (2, 2)),  # Top, horizontal
        ((0.5, 0.5), (2, 3)),  # Top, vertical
        ((2, 3), (0.5, 0.5)),  # Right, horizontal
        ((2, 2), (0, 1)),  # Right, vertical
        ((1, 0), (-1, -1)),  # Bottom, horizontal
        ((0.5, 0.5), (-3, -2)),  # Bottom, vertical
        ((-1, -1), (1, 0)),  # Left, horizontal
        ((-3, -2), (0.5, 0.5)),  # Left, vertical
        ((-1, 0.25), (0.25, 2)),  # Overlapping bounds, but misses the top-left corner
        ((0.75, 2), (-1, 0.75)),  # Overlapping bounds, but misses the bottom-right corner
    ],
)
def test_edge_rectangle_intersection_on_axis_square_no_intersection(R_edges, z_edges):
    """Check various lines that don't intersect with the unit square"""

    intersections = edge_rectangle_intersection((0, 1), (0, 1), R_edges, z_edges)

    assert intersections.size == 0


def test_edge_rectangle_intersection_on_axis_square_no_intersection_multiple():
    """Check various lines that don't intersect with the unit square. Same
    as the previous test, but in one call
    """

    R_edges = np.array(
        ((0, 1), (0.5, 0.5), (2, 3), (2, 2), (1, 0), (0.5, 0.5), (-1, -1), (-3, -2))
    )
    z_edges = np.array(
        ((2, 2), (2, 3), (0.5, 0.5), (0, 1), (-1, -1), (-3, -2), (1, 0), (0.5, 0.5))
    )

    intersections = edge_rectangle_intersection((0, 1), (0, 1), R_edges, z_edges)

    assert intersections.size == 0


@pytest.mark.parametrize(
    "R_edges, z_edges, expected_intersections",
    [
        ((0.5, 0.5), (0.5, 1.5), [0]),  # Top
        ((0.5, 1.5), (0.5, 0.5), [0]),  # Right
        ((0.25, 1.1), (1.1, 0.25), [0]),  # Top and right
        ((0.5, 0.5), (-3, 0.75), [0]),  # Bottom
        ((-1, 0.75), (0.25, 0.25), [0]),  # Left
        ((-0.1, 0.9), (0.9, -0.1), [0]),  # Bottom and left
        ((-1, 2), (0.5, 0.5), [0]),  # Horizontal crossing, both endpoints outside
        ((0.5, 0.5), (-1, 2), [0]),  # Vertical crossing, both endpoints outside
        ((-1, 2), (-1, 2), [0]),  # Through opposite corners
        ((-1, 2), (2, -1), [0]),  # Through the other opposite corners
        ((-1, 2), (0, 0), [0]),  # Along the bottom boundary
        ((-1, 2), (1, 1), [0]),  # Along the top boundary
        ((0, 0), (-1, 2), [0]),  # Along the left boundary
        ((1, 1), (-1, 2), [0]),  # Along the right boundary
        ((-1, 1), (1, -1), [0]),  # Touches only the bottom-left corner
        ((0, 2), (2, 0), [0]),  # Touches only the top-right corner
        ((-1, 0), (0.5, 0.5), [0]),  # Endpoint on a side
        ((-1, 0), (-1, 0), [0]),  # Endpoint on a corner
        ((0.25, 0.75), (0.25, 0.75), [0]),  # Entirely inside
        ((0.5, 0.5), (0.5, 0.5), [0]),  # Zero-length edge inside
        ((0, 0), (0, 0), [0]),  # Zero-length edge on a corner
    ],
)
@pytest.mark.parametrize("reverse", [False, True])
def test_edge_rectangle_intersection_on_axis_square(
    R_edges, z_edges, expected_intersections, reverse
):
    """Check various lines that intersect with the unit square."""
    if reverse:
        R_edges = R_edges[::-1]
        z_edges = z_edges[::-1]

    intersections = edge_rectangle_intersection((0, 1), (0, 1), R_edges, z_edges)

    assert intersections.size == 1
    assert np.array_equal(intersections, expected_intersections)


def test_edge_rectangle_intersection_on_axis_square_multiple():
    """Check various lines that intersect with the unit square"""

    R_edges = np.array(
        (
            (0.5, 0.5),
            (0.5, 1.5),
            (0.25, 1.1),
            (0.5, 0.5),
            (-1, 0.75),
            (-0.1, 0.9),
            (0.5, 0.5),
        )
    )

    z_edges = np.array(
        (
            (0.5, 1.5),
            (0.5, 0.5),
            (1.1, 0.25),
            (-3, 0.75),
            (0.25, 0.25),
            (-1, 2),
            (0.9, -0.1),
        )
    )

    expected_intersections = np.arange(len(R_edges))

    intersections = edge_rectangle_intersection((0, 1), (0, 1), R_edges, z_edges)

    assert np.array_equal(intersections, expected_intersections)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64, np.uint64])
def test_edge_rectangle_intersection_mixed_edges(dtype):
    R_edges = np.array([(0, 4), (4, 5), (2, 2), (0, 1), (4, 0)], dtype=dtype)
    z_edges = np.array([(2, 2), (4, 5), (0, 4), (2, 4), (4, 0)], dtype=dtype)

    with np.errstate(divide="raise", invalid="raise", over="raise"):
        intersections = edge_rectangle_intersection((1, 3), (1, 3), R_edges, z_edges)

    assert np.array_equal(intersections, [0, 2, 4])


def test_edge_rectangle_intersection_empty():
    edges = np.empty((0, 2))
    intersections = edge_rectangle_intersection((0, 1), (0, 1), edges, edges)

    assert intersections.shape == (0,)
    assert np.issubdtype(intersections.dtype, np.integer)


def test_edge_rectangle_intersection_bad_inputs():
    with pytest.raises(ValueError):
        edge_rectangle_intersection((0, 1), (0, 1), 1, (0, 1))
    with pytest.raises(ValueError):
        edge_rectangle_intersection((0, 1), (0, 1), (0, 1), 1)
