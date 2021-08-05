
from numpy import array, sqrt, exp
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix

from tokamesh.construction import equilateral_mesh
from tokamesh.geometry import BarycentricGeometryMatrix, Camera

# build a simple equilateral mesh
R, z, triangles = equilateral_mesh(
    x_range=(0.1,1.5),
    y_range=(-1.2,0.8),
    scale=0.05
)

# define a test function which sets the emission value in (R,z)
def emission_func(R, z):
    r = 10*sqrt((R-0.85)**2 + (z-0.1)**2)
    return exp(-0.5*r**4)

# evaluate the emission at each vertex of the mesh
emission = emission_func(R, z)

# define a camera to image the emission
pixels = 150
cam = Camera(
    position=array([1.2,1.4,0.5]),
    direction=array([-1.,-0.5,-0.3]),
    fov=30.,
    max_distance=4.,
    num_x=pixels, num_y=pixels)

# use the mesh and camera information to calculate a geometry matrix
BGM = BarycentricGeometryMatrix(
    R=R,
    z=z,
    triangles=triangles,
    ray_origins=cam.ray_starts,
    ray_ends=cam.ray_ends
)

matrix_data = BGM.calculate()
# extract the data and build a sparse matrix
entry_values = matrix_data['entry_values']
row_values = matrix_data['row_indices']
col_values = matrix_data['col_indices']
shape = matrix_data['shape']

G = csc_matrix((entry_values, (row_values,col_values)), shape=shape)

# predict the pixel brightness by taking the product of the geometry
# matrix with the vector of emission values at each vertex
pixel_brightness = G.dot(emission)
# re-shape the pixel brightness into an image
brightness_image = pixel_brightness.reshape([pixels,pixels])

# plot the predicted image
plt.figure(figsize=(6,6))
plt.imshow(brightness_image.T)
plt.tight_layout()
plt.show()
