
from numpy import arange, array, loadtxt, sqrt
from numpy import int as npint
from os.path import isfile
import subprocess
import os
triangle_dir = os.path.abspath(__file__)[:-11]


def run_triangle(outer_boundary=None, inner_boundary=None, void_markers=None, max_area=None):
    """
    A Python interface for the 'Triangle' C-code which is packaged with Tokamesh.

    :param outer_boundary:
    :param inner_boundary:
    :param void_markers:
    :param max_area:
    :return:
    """

    # first check to see if the triangle executable exists in the given location
    if not isfile(triangle_dir + 'triangle'):
        print(' # triangle executable not found - attempting compile from source')

        # if not, check if the source file exists
        if not isfile(triangle_dir + 'triangle.c'):
            raise FileNotFoundError('source code file triangle.c not found')
        else:
            # if it does exist, try to compile using gcc
            subprocess.call('gcc -O -o ' + triangle_dir + 'triangle ' + triangle_dir + 'triangle.c -lm', shell=True)

    inputfile = triangle_dir + 'tri_input'

    # extract wall data
    wall_r = outer_boundary[0]
    wall_z = outer_boundary[1]
    n_wall = len(wall_r)

    if inner_boundary is not None:
        inner_r = inner_boundary[0]
        inner_z = inner_boundary[1]
        n_cuts = len(inner_r)-1
    else:
        n_cuts = 0

    if void_markers is not None:
        holes_r = void_markers[0]
        holes_z = void_markers[1]

    # convert interior boundary data to a series of cuts
    # Write the wall data to a triangle input file
    f = open(inputfile + '.poly', 'w')
    f.write(str(n_wall + n_cuts*2) + '  2  1  0\n')
    for i in arange(n_wall):
        f.write('{:<3d}{:10.4f}{:10.4f}\n'.format(i + 1, wall_r[i], wall_z[i]))


    if inner_boundary is not None:
        ctr = len(wall_r)
        for i in range(n_cuts):
            cut_start_r = inner_r[i]
            cut_start_z = inner_z[i]
            cut_end_r = inner_r[i+1]
            cut_end_z = inner_z[i+1]

            f.write('{:<3d}{:10.4f}{:10.4f}\n'.format(ctr + i + 1, cut_start_r, cut_start_z))
            ctr = ctr + 1
            f.write('{:<3d}{:10.4f}{:10.4f}\n'.format(ctr + i + 1, cut_end_r, cut_end_z))

        # write the polygon connection to the input file
        f.write(str(n_wall + n_cuts - 1) + ' 0\n')
        for i in arange(n_wall-1):
            f.write('{:<3d}{:4d}{:4d}\n'.format(i + 1, i + 1, i + 2))

        # write the cut lines to the input file
        ctr = n_wall - 1
        for i in arange(n_cuts):
            f.write('{:<3d}{:4d}{:4d}\n'.format(ctr + i + 1, ctr + i + 2, ctr + i + 3))
            ctr = ctr + 1


    if void_markers is not None:
        # Write the holes
        f.write('{:<2d}\n'.format(len(holes_r)))
        for i in arange(len(holes_r)):
            f.write('{:<2d}{:10.4f}{:10.4f}\n'.format(i + 1, holes_r[i], holes_z[i]))

    f.close()

    if max_area is None:
        area = str(0.0005)
    else:
        area = format(max_area, '.12f')

    # Call triangle
    if triangle_dir is None:
        subprocess.call('triangle -pqa' + area + ' ' + inputfile)
    else:
        subprocess.call(triangle_dir + 'triangle -pqa' + area + ' ' + inputfile, shell=True)

    elefile = inputfile + '.1.ele'
    nodefile = inputfile + '.1.node'

    elements = array(loadtxt(elefile, skiprows=1), dtype=npint)
    nodes = loadtxt(nodefile, skiprows=1)

    tri_x = nodes[:, 1]
    tri_y = nodes[:, 2]
    tri_nodes = elements[:, 1:4] - 1

    # After reading the data back in from triangle, exact numeric value of
    # vertices can be changed via truncation. Due to this, we may need to
    # adjust their value to match the existing central mesh edge vertices.
    b_x = array(inner_r[:-1])  # get the boundary coords - these are the 'correct' values
    b_y = array(inner_z[:-1])

    # loop over all vertices in the edge mesh
    cutoff = sqrt(max_area / (0.25*sqrt(3))) * 0.01
    for i in range(len(tri_x)):
        dist = sqrt((tri_x[i] - b_x)**2 + (tri_y[i] - b_y)**2)
        k = dist.argmin()
        if 0. < dist.min() < cutoff:
            tri_x[i] = b_x[k]
            tri_y[i] = b_y[k]

    return tri_x, tri_y, tri_nodes




def read_triangle_output(directory=None, inputfile='tri_input'):
    elefile = inputfile + '.1.ele'
    nodefile = inputfile + '.1.node'

    elements = array(loadtxt(directory + elefile, skiprows=1), dtype=npint)
    nodes = loadtxt(directory + nodefile, skiprows=1)

    tri_x = nodes[:, 1]
    tri_y = nodes[:, 2]
    tri_nodes = elements[:, 1:4] - 1
    return tri_x, tri_y, tri_nodes
