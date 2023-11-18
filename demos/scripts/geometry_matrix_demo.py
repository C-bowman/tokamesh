from numpy import array, sqrt, exp
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix

from tokamesh import TriangularMesh
from tokamesh.construction import equilateral_mesh
from tokamesh.geometry import BarycentricGeometryMatrix
from tokamesh.utilities import Camera

# build a simple equilateral mesh
R, z, triangles = equilateral_mesh(
    R_range=(0.3, 1.5), z_range=(-0.5, 0.5), resolution=0.03
)


# define a test function which sets the emission value in (R,z)
def emission_func(R, z):
    w = 0.05
    r = sqrt((R - 0.8) ** 2 + (z + 0.1) ** 2)
    return exp(-0.5 * (r / w) ** 4) + 0.5 * exp(-0.5 * ((r - 0.3) / w) ** 2)


# evaluate the emission at each vertex of the mesh
emission = emission_func(R, z)

# define a camera to image the emission
pixels = 150
cam = Camera(
    position=array([1.6, 1.8, 1.25]),
    direction=array([-1.0, -0.7, -0.8]),
    fov=40.0,
    max_distance=4.0,
    num_x=pixels,
    num_y=pixels,
)

# use the mesh and camera information to calculate a geometry matrix
BGM = BarycentricGeometryMatrix(
    R=R, z=z, triangles=triangles, ray_origins=cam.ray_starts, ray_ends=cam.ray_ends
)

matrix_data = BGM.calculate()

# extract the data and build a sparse matrix
entry_values = matrix_data["entry_values"]
row_values = matrix_data["row_indices"]
col_values = matrix_data["col_indices"]
shape = matrix_data["shape"]
G = csc_matrix((entry_values, (row_values, col_values)), shape=shape)

# predict the pixel brightness by taking the product of the geometry
# matrix with the vector of emission values at each vertex
pixel_brightness = G @ emission
# re-shape the pixel brightness into an image
brightness_image = pixel_brightness.reshape([pixels, pixels])

# plot the predicted image
fig = plt.figure(figsize=(9, 4))
# get an image of the emission on the mesh
mesh = TriangularMesh(R=R, z=z, triangles=triangles)
emission_R, emission_z, emission_image = mesh.get_field_image(vertex_values=emission)
ax1 = fig.add_subplot(121)
ax1.contourf(emission_R, emission_z, emission_image.T, 100)
ax1.set_title("Example emission function on the mesh")
ax1.set_xlabel("major radius (m)")
ax1.set_ylabel("z-height (m)")

ax2 = fig.add_subplot(122)
ax2.imshow(brightness_image.T)
ax2.axis("equal")
ax2.set_title("Camera image prediction")
ax2.axis("off")

plt.tight_layout()
plt.show()
