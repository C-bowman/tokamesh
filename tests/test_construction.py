
from numpy import array
from tokamesh.construction import remove_duplicate_vertices, Polygon
import pytest


def test_remove_duplicate_vertices():
    # create a square with two triangles
    target_R = array([1., 1., 2., 2.])
    target_z = array([1., 2., 2., 1.])
    target_triangles = array([[0,1,3],[1,2,3]])
    # now add a duplicate vertex and an extra triangle
    R = array([1., 1., 2., 2., 0.999999])
    z = array([1., 2., 2., 1., 2.000001])
    triangles = array([[0,1,3],[4,2,3],[1,2,3]])
    # remove the duplicates
    new_R, new_z, new_triangles = remove_duplicate_vertices(R=R, z=z, triangles=triangles)
    # check the shapes agree with the targets
    assert new_R.shape == target_R.shape
    assert new_z.shape == target_z.shape
    assert new_triangles.shape == target_triangles.shape
    # check the values agree with the targets
    assert (new_R == target_R).all()
    assert (new_z == target_z).all()
    assert (new_triangles == target_triangles).all()


def test_polygon():
    # create a non-convex polygon
    x = array([1., 1., 3., 3., 2.5])
    y = array([1., 1.5, 2., 0., 1.25])
    P = Polygon(x=x, y=y)

    # first check points which should be inside
    assert P.is_inside([2.75, 1.]) is True
    assert P.is_inside([2., 1.25]) is True
    assert P.is_inside([2.75, 1.25]) is True
    assert P.is_inside([2.5, 1.5]) is True

    # now check points which should be outside
    assert P.is_inside([2., 1.]) is False
    assert P.is_inside([0., 1.]) is False
    assert P.is_inside([3., 3.]) is False
    assert P.is_inside([2., 2.]) is False
