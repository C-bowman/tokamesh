
from numpy import array, zeros, linspace, sqrt, ceil, sin, cos, arctan2


def equilateral_mesh(x_range=(0,10), y_range=(0,5), scale=1.0, rotation=None, pivot=(0,0)):
    """
    Construct a mesh from equilateral triangles which fills a rectangular region.

    :param x_range: \


    :param y_range: \


    :param float scale: \
        The side-length of the triangles.

    :param rotation: \
        Angle (in radians) by which the mesh will be rotated.

    :param pivot: \
        Pivot point around which the rotation is applied.

    :return:
    """
    N = int(ceil((x_range[1] - x_range[0])/scale))
    M = int(ceil((y_range[1] - y_range[0])/(scale*0.5*sqrt(3))))

    x_ax = linspace(0, N-1, N)*scale
    y_ax = linspace(0, M-1, M)*scale*0.5*sqrt(3)

    x = zeros([N,M])
    y = zeros([N,M])
    y[:,:] = y_ax[None,:] + y_range[0]
    x[:,:] = x_ax[:,None] + x_range[0]
    x[:,1::2] += 0.5*scale

    if rotation is not None:
        R = sqrt((x-pivot[0])**2 + (y-pivot[1])**2)
        theta = arctan2(y-pivot[1], x-pivot[0]) + rotation
        x = R*cos(theta) + pivot[0]
        y = R*sin(theta) + pivot[1]

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
