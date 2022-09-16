import numpy as np
from tokamesh.construction import equilateral_mesh, build_central_mesh, mesh_generator
from tokamesh.construction import find_boundaries, refine_mesh, rotate, trim_vertices
from tokamesh.construction import remove_duplicate_vertices, Polygon
import pytest


def test_equilateral_mesh():
    x_min, x_max = 0, 1
    y_min, y_max = 2, 4
    resolution = 0.1
    x, y, triangles = equilateral_mesh((x_min, x_max), (y_min, y_max), resolution)

    assert np.all(x >= x_min)
    assert np.all(x <= x_max)
    assert np.all(y >= y_min)
    assert np.all(y <= y_max)

    expected_num_points = (1 + ((x_max - x_min) // resolution)) * (
        1 + ((y_max - y_min) // (resolution * np.sqrt(3) / 2))
    )
    assert len(x) == expected_num_points


def test_equilateral_mesh_rotated():
    x_min, x_max = 0, 1
    y_min, y_max = 2, 4
    resolution = 0.1
    x_rot, y_rot, _ = equilateral_mesh(
        (x_min, x_max), (y_min, y_max), resolution, rotation=np.pi / 2
    )
    x, y, _ = equilateral_mesh((x_min, x_max), (y_min, y_max), resolution, rotation=0.0)

    # Rotated pi/2 so x <=> y
    assert np.allclose(y, -x_rot)
    assert np.allclose(x, y_rot)

    expected_num_points = (1 + ((y_max - y_min) // resolution)) * (
        1 + ((x_max - x_min) // (resolution * np.sqrt(3) / 2))
    )
    assert len(x) == expected_num_points


def test_remove_duplicate_vertices():
    # create a square with two triangles
    target_R = np.array([1.0, 1.0, 2.0, 2.0])
    target_z = np.array([1.0, 2.0, 2.0, 1.0])
    target_triangles = np.array([[0, 1, 3], [1, 2, 3]])
    # now add a duplicate vertex and an extra triangle
    R = np.array([1.0, 1.0, 2.0, 2.0, 0.999999])
    z = np.array([1.0, 2.0, 2.0, 1.0, 2.000001])
    triangles = np.array([[0, 1, 3], [4, 2, 3], [1, 2, 3]])
    # remove the duplicates
    new_R, new_z, new_triangles = remove_duplicate_vertices(
        R=R, z=z, triangles=triangles
    )
    # check the shapes agree with the targets
    assert new_R.shape == target_R.shape
    assert new_z.shape == target_z.shape
    assert new_triangles.shape == target_triangles.shape
    # check the values agree with the targets
    assert (new_R == target_R).all()
    assert (new_z == target_z).all()
    assert (new_triangles == target_triangles).all()


def test_polygon_is_inside():
    # create a non-convex polygon
    x = np.array([1.0, 1.0, 3.0, 3.0, 2.5])
    y = np.array([1.0, 1.5, 2.0, 0.0, 1.25])
    P = Polygon(x=x, y=y)

    # first check points which should be inside
    assert P.is_inside([2.75, 1.0]) is True
    assert P.is_inside([2.0, 1.25]) is True
    assert P.is_inside([2.75, 1.25]) is True
    assert P.is_inside([2.5, 1.5]) is True

    # now check points which should be outside
    assert P.is_inside([2.0, 1.0]) is False
    assert P.is_inside([0.0, 1.0]) is False
    assert P.is_inside([3.0, 3.0]) is False
    assert P.is_inside([2.0, 2.0]) is False


def test_polygon_distance():
    x = np.array([0.0, 1.0, 1.0, 0.0])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    square = Polygon(x=x, y=y)

    assert np.isclose(square.distance((2.0, 0.5)), 1.0)
    assert np.isclose(square.distance((0.5, 2.0)), 1.0)
    assert np.isclose(square.distance((-1.0, 0.5)), 1.0)
    assert np.isclose(square.distance((0.5, -1.0)), 1.0)
    assert np.isclose(square.distance((0.5, 0.5)), 0.5)
    assert np.isclose(square.distance((2.0, 2.0)), np.sqrt(2.0))

    height = np.sqrt(3.0) / 2.0
    triangle = Polygon([0.0, 1.0, 0.5], [0.0, 0.0, height])

    assert np.isclose(triangle.distance([1.5, height]), height)
    assert np.isclose(triangle.distance([-0.5, height]), height)
    assert np.isclose(triangle.distance([0.5, -height]), height)


@pytest.mark.parametrize(
    "R, z, angle, pivot, expected",
    [
        (1, 0, np.pi, (0, 0), (-1, 0)),
        (0, 1, np.pi, (0, 0), (0, -1)),
        (1, 0, np.pi / 2, (0, 0), (0, 1)),
        (0, 1, np.pi / 2, (0, 0), (-1, 0)),
        (1, 0, 2 * np.pi, (0, 0), (1, 0)),
        (1, 1, np.pi, (2, 2), (3, 3)),
    ],
)
def test_rotate(R, z, angle, pivot, expected):
    answer = rotate(R, z, angle, pivot)
    assert np.allclose(answer, expected)


def assert_lists_are_equal(result, expected):
    assert len(result) == len(expected)
    for r, e in zip(result, expected):
        assert np.array_equal(r, e)


def test_find_boundary_one_triangle():
    triangles = np.array(((1, 2, 3),))
    boundaries = find_boundaries(triangles)
    expected = [np.array((1, 2, 3, 3))]

    assert_lists_are_equal(boundaries, expected)


def test_find_boundary_two_disconnected_triangle():
    triangles = np.array(((1, 2, 3), (4, 5, 6)))
    boundaries = find_boundaries(triangles)
    expected = [np.array((1, 2, 3, 3)), np.array((4, 5, 6, 6))]

    assert_lists_are_equal(boundaries, expected)


def test_find_boundary_multiple_triangles():
    triangles = np.array(
        [
            [0, 1, 2],
            [1, 2, 3],
            [2, 3, 5],
            [3, 4, 5],
            [4, 5, 6],
            [0, 2, 7],
            [2, 5, 8],
            [2, 7, 8],
            [5, 6, 9],
            [5, 8, 9],
            [7, 8, 10],
            [8, 9, 11],
            [8, 10, 11],
        ]
    )

    boundaries = find_boundaries(triangles)
    expected = [np.array((0, 1, 3, 4, 6, 9, 11, 10, 7, 7))]

    assert_lists_are_equal(boundaries, expected)


def test_find_boundary_multiple_triangles_inner_boundary():
    # Same as above but missing (3, 6, 9)
    triangles = np.array(
        [
            [0, 1, 2],
            [1, 2, 3],
            [2, 3, 5],
            [3, 4, 5],
            [4, 5, 6],
            [0, 2, 7],
            [2, 7, 8],
            [5, 6, 9],
            [5, 8, 9],
            [7, 8, 10],
            [8, 9, 11],
            [8, 10, 11],
        ]
    )

    boundaries = find_boundaries(triangles)
    expected = [np.array((0, 1, 3, 4, 6, 9, 11, 10, 7, 7)), np.array((5, 2, 8, 8))]

    assert_lists_are_equal(boundaries, expected)


def test_trim_vertices_one_triangle():
    R = np.array((0, 1, 0))
    z = np.array((0, 0, 1))
    triangles = np.array(((0, 1, 2),))

    trimmed_R, trimmed_z, trimmed_triangles = trim_vertices(
        R, z, triangles, np.array([False, True, False])
    )

    assert np.array_equal(trimmed_R, np.array((0, 0)))
    assert np.array_equal(trimmed_z, np.array((0, 1)))
    assert np.array_equal(trimmed_triangles, np.empty((0, 3)))


def test_trim_vertices_multiple_triangles():
    R = np.array((0.0, 0.5, 1.0, 1.5, 2.5, 2.0, 3.0, 0.5, 1.5, 2.5, 1.0, 2.0))
    z = np.array((2.0, 3.0, 2.0, 3.0, 3.0, 2.0, 2.0, 1.0, 1.0, 1.0, 0.0, 0.0))

    triangles = np.array(
        [
            [0, 1, 2],
            [1, 2, 3],
            [2, 3, 5],
            [3, 4, 5],
            [4, 5, 6],
            [0, 2, 7],
            [2, 5, 8],
            [2, 7, 8],
            [5, 6, 9],
            [5, 8, 9],
            [7, 8, 10],
            [8, 9, 11],
            [8, 10, 11],
        ]
    )

    trim_bools = np.array(
        [True, True, False, True, True, False, True, True, False, True, True, True]
    )

    trimmed_R, trimmed_z, trimmed_triangles = trim_vertices(R, z, triangles, trim_bools)

    expected_R = np.array((1.0, 2.0, 1.5))
    expected_z = np.array((2.0, 2.0, 1.0))
    expected_triangles = np.array(((0, 1, 2),))

    assert np.array_equal(trimmed_R, expected_R)
    assert np.array_equal(trimmed_z, expected_z)
    assert np.array_equal(trimmed_triangles, expected_triangles)


def test_build_central_mesh():
    x_min = 0.0
    x_max = 1.0
    y_min = 2.0
    y_max = 4.0

    R = np.array([x_min, x_max, x_min, x_max])
    z = np.array([y_min, y_min, y_max, y_max])
    resolution = 0.1
    x, y, triangles = build_central_mesh(R, z, resolution)

    assert np.all(x >= x_min + resolution)
    assert np.all(x <= x_max - resolution)
    assert np.all(y >= y_min + resolution)
    assert np.all(y <= y_max - resolution)

    expected_num_points = (1 + ((x_max - x_min) // (resolution))) * (
        1 + ((y_max - y_min) // (resolution * np.sqrt(3) / 2))
    )
    assert len(x) < expected_num_points


def test_build_central_mesh_rotated():
    x_min = 0.0
    x_max = 1.0
    y_min = 2.0
    y_max = 4.0

    R = np.array([x_min, x_max, x_min, x_max])
    z = np.array([y_min, y_min, y_max, y_max])
    resolution = 0.1
    x, y, triangles = build_central_mesh(R, z, resolution, rotation=np.pi / 4)

    assert np.all(x >= x_min + resolution)
    assert np.all(x <= x_max - resolution)
    assert np.all(y >= y_min + resolution)
    assert np.all(y <= y_max - resolution)

    expected_num_points = (1 + ((x_max - x_min) // (resolution))) * (
        1 + ((y_max - y_min) // (resolution * np.sqrt(3) / 2))
    )
    assert len(x) < expected_num_points


def test_refine_mesh():
    height = np.sqrt(3.0) / 2.0
    R0 = np.array([0.0, 1.0, 0.5])
    Z0 = np.array([0.0, 0.0, height])
    triangles0 = np.array(([0, 1, 2],))

    R1, Z1, triangles1 = refine_mesh(R0, Z0, triangles0, np.array([True]))

    expected_R1 = np.array([0.0, 0.5, 0.25, 1.0, 0.75, 0.5])
    expected_Z1 = np.array([0.0, 0.0, height / 2, 0.0, height / 2, height])
    expected_triangles1 = np.array([[0, 1, 2], [3, 1, 4], [5, 4, 2], [1, 4, 2]])

    assert np.allclose(R1, expected_R1)
    assert np.allclose(Z1, expected_Z1)
    assert np.allclose(triangles1, expected_triangles1)

    # Do a second level of refinement
    refines1 = np.zeros(triangles1.shape, dtype=bool)
    refines1[0] = True
    R2, Z2, triangles2 = refine_mesh(R1, Z1, triangles1, refines1)

    expected_R2 = np.array([0.0, 0.25, 0.125, 0.5, 0.375, 0.25, 1.0, 0.75, 0.5])
    expected_Z2 = np.array(
        [0.0, 0.0, 0.21650635, 0.0, 0.21650635, 0.4330127, 0.0, 0.4330127, 0.8660254]
    )
    expected_triangles2 = np.array(
        [
            [0, 1, 2],
            [3, 1, 4],
            [5, 4, 2],
            [1, 4, 2],
            [6, 3, 7],
            [8, 7, 5],
            [4, 7, 3],
            [4, 7, 5],
        ]
    )
    assert np.allclose(R2, expected_R2)
    assert np.allclose(Z2, expected_Z2)
    assert np.allclose(triangles2, expected_triangles2)


def test_refine_mesh_one_neighbour():
    height = np.sqrt(3.0) / 2.0
    R0 = np.array([0.0, 1.0, 0.5, 1.5, 2.0])
    Z0 = np.array([0.0, 0.0, height, height, 0.0])
    t0 = np.array(([0, 1, 2], [1, 2, 3], [1, 3, 4]))

    # Refine requiring neighbour to be bisected
    R1, Z1, t1 = refine_mesh(R0, Z0, t0, np.array([True, False, False]))

    expected_R1 = np.array([0.0, 0.5, 0.25, 1.0, 0.75, 0.5, 1.5, 2.0])

    expected_Z1 = np.array([0.0, 0.0, height / 2, 0.0, height / 2, height, height, 0.0])

    expected_t1 = np.array(
        [[0, 1, 2], [3, 1, 4], [5, 4, 2], [1, 4, 2], [4, 6, 3], [4, 6, 5], [3, 6, 7]]
    )

    assert np.allclose(R1, expected_R1)
    assert np.allclose(Z1, expected_Z1)
    assert np.allclose(t1, expected_t1)


def test_refine_mesh_two_neighbours():
    height = np.sqrt(3.0) / 2.0
    R0 = np.array([0.0, 1.0, 0.5, 1.5, 2.0])
    Z0 = np.array([0.0, 0.0, height, height, 0.0])
    t0 = np.array(([0, 1, 2], [1, 2, 3], [1, 3, 4]))

    # Refine requiring neighbour to be trisected
    R1, Z1, t1 = refine_mesh(R0, Z0, t0, np.array([True, False, True]))

    expected_R1 = np.array([0.0, 0.5, 0.25, 1.0, 0.75, 0.5, 1.25, 1.5, 1.5, 1.75, 2.0])
    expected_Z1 = np.array(
        [
            0.0,
            0.0,
            height / 2,
            0.0,
            height / 2,
            height,
            height / 2,
            height,
            0.0,
            height / 2,
            0.0,
        ]
    )
    expected_t1 = np.array(
        [
            [0, 1, 2],
            [3, 1, 4],
            [5, 4, 2],
            [1, 4, 2],
            [4, 6, 3],
            [4, 5, 7],
            [4, 6, 7],
            [3, 6, 8],
            [7, 6, 9],
            [10, 9, 8],
            [6, 9, 8],
        ]
    )
    assert np.allclose(R1, expected_R1)
    assert np.allclose(Z1, expected_Z1)
    assert np.allclose(t1, expected_t1)


def test_mesh_generator():
    height = np.sqrt(3.0) / 2.0
    triangle = Polygon([0.0, 1.0, 0.5], [0.0, 0.0, height])

    x, y, t = mesh_generator(triangle.x, triangle.y, resolution=0.1)

    # Check for duplicate points
    points = np.column_stack((x, y))
    unique_points = np.unique(points, axis=0)
    assert len(points) == len(unique_points)

    # Check for duplicate triangles
    unique_triangles = np.unique(t, axis=0)
    assert len(t) == len(unique_triangles)
