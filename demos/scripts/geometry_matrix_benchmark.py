from numpy import linspace, array, zeros, sqrt, sinc
from scipy.sparse import csc_matrix
from numpy.random import multivariate_normal
from scipy.integrate import simps
import matplotlib.pyplot as plt

from tokamesh.construction import equilateral_mesh
from tokamesh import TriangularMesh
from tokamesh.geometry import BarycentricGeometryMatrix
from tokamesh.utilities import Camera

"""
This script demonstrates that line-integrals over rays through a field
defined by barycentric interpolation on a mesh are calculated correctly
by geometry matrices produced by the BarycentricGeometryMatrix class.
"""

# build a basic mesh out of equilateral triangles
R, z, triangles = equilateral_mesh(resolution=0.01, R_range=(0, 0.2), z_range=(0, 0.2))
mesh = TriangularMesh(R, z, triangles)

# generate a random test field using a gaussian process
distance = sqrt((R[:, None] - R[None, :]) ** 2 + (z[:, None] - z[None, :]) ** 2)
scale = 0.04
covariance = sinc(distance / scale) ** 2
field = multivariate_normal(zeros(R.size), covariance)
field -= field.min()

# Generate an image of the field on the mesh
R_axis, z_axis, field_image = mesh.get_field_image(field, shape=(200, 150))

# plot the test field
from matplotlib import colormaps

cmap = colormaps["viridis"]

fig = plt.figure(figsize=(7, 7))
ax1 = fig.add_subplot(111)
ax1.contourf(R_axis, z_axis, field_image.T, 30)
ax1.set_facecolor(cmap(0.0))
mesh.draw(ax1, lw=0.25, color="white")
ax1.axis("equal")
ax1.set_title("Gaussian random field on the mesh")
plt.tight_layout()
plt.show()


# generate a synthetic camera to image the field
cam_position = array([0.17, 0.19, 0.18])
cam_direction = array([-0.1, -0.1, -0.06])
cam = Camera(
    position=cam_position, direction=cam_direction, max_distance=1.0, num_x=5, num_y=15
)

# calculate the geometry matrix data
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

# get the geometry matrix prediction of the line-integrals
matrix_integrals = G @ field

# manually calculate the line integrals for comparison
L = linspace(0, 0.5, 3000)  # build a distance axis for the integrals
# get the position of each ray at each distance
R_projection, z_projection = cam.project_rays(L)
# directly integrate along each ray
direct_integrals = zeros(R_projection.shape[1])
for i in range(R_projection.shape[1]):
    samples = mesh.interpolate(
        R_projection[:, i], z_projection[:, i], vertex_values=field
    )
    direct_integrals[i] = simps(samples, x=L)


# plot the results
fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(211)
ax1.plot(direct_integrals, "-", label="brute-force result", c="red")
ax1.plot(
    matrix_integrals,
    label="geometry matrix result",
    marker="o",
    markerfacecolor="none",
    ls="none",
    markeredgewidth=2,
    c="C0",
)
ax1.set_xlabel("Pixel-ray number")
ax1.set_ylabel("line-integral prediction")
ax1.set_xlim([-2, 77])
ax1.grid()
ax1.legend()

abs_frac_diff = abs(matrix_integrals / direct_integrals - 1)
mean_afd = abs_frac_diff.mean()
ax2 = fig.add_subplot(212)
ax2.plot(abs_frac_diff, c="green", alpha=0.5)
ax2.plot(abs_frac_diff, ".", c="green", markersize=4)
ax2.plot(
    [-10, 85],
    [mean_afd, mean_afd],
    ls="dashed",
    c="red",
    lw=2,
    label="{:.3%} mean absolute fractional difference".format(mean_afd),
)
ax2.set_ylim([1e-5, 1e-2])
ax2.set_yscale("log")
ax2.set_xlim([-1, 75])
ax2.set_xlabel("Pixel-ray number")
ax2.set_ylabel("absolute fractional difference")
ax2.grid()
ax2.legend()

plt.tight_layout()
plt.show()
