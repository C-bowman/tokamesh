Background
~~~~~~~~~~

A geometry matrix is a linear model which predicts the brightnesses that
line-integrated instruments (e.g. cameras, spectrometers) would measure
when imaging a given distribution of emission.

The geometry matrix element can be derived as follows: To predict the
measured brightness of the :math:`i`'th line-of-sight :math:`b_i` we
multiply the emission :math:`\mathcal{E}(x, y, z)` by a 'sensitivity'
function :math:`\mathcal{S}_i (x, y, z)` which describes what fraction
of emission at any point in space is measured as brightness by the
instrument, and intgrate this product over all space so that

.. math::
   b_i = \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} \int_{-\infty}^{\infty}
   \mathcal{E}(x, y, z) \mathcal{S}_i (x, y, z) \,\mathrm{d}x \,\mathrm{d}y \,\mathrm{d}z .

Next, we assume that the emission function is toroidally symmetric,
such that it depends only on the major radius :math:`R = \sqrt{x^2 + y^2}`
and :math:`z`. Additionally, we assume the emission can be expressed as
a weighted-sum of 2D basis functions :math:`\phi_j (R,z)` such that

.. math::
   \mathcal{E}(R, z) = \sum_{j} x_j \phi_j (R,z)

where :math:`x_j` are the basis function weights. We may now re-write :math:`b_i` as

.. math::
   b_i = \sum_j G_{ij} x_j

where :math:`G_{ij}` is the geometry matrix element given by

.. math::
   G_{ij} = \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} \int_{-\infty}^{\infty}
   \phi_j (R,z) \mathcal{S}_i (x, y, z) \,\mathrm{d}x \,\mathrm{d}y \,\mathrm{d}z .

After calculating the geometry matrix :math:`\mathbf{G}`, the vector of brightness
predictions :math:`\mathbf{b}` can be obtained through a single matrix-vector
product such that :math:`\mathbf{b = Gx}`.

However, defining a 3D sensitivity function for each line-of-sight is complicated,
and computing the 3D integral for all elements of the geometry matrix is very
expensive for instruments with a large number of lines-of-sight, such as a camera.

Instead, a further approximation is made, where we assume that each line-of-sight
collects emission only along a single line :math:`\ell_i`. This allows :math:`G_{ij}`
to be re-written as a line-integral through the basis functions so that

.. math::
   G_{ij} = \int \phi_j (R,z) \,\mathrm{d}\ell_i.


Choice of basis functions
-------------------------

When using a triangular mesh to represent the solution of a tomography problem,
the typical approach is to assume that the emission inside each triangle is
constant. This is equivalent to zeroth-order interpolation, and leads to the
following set of basis functions:

.. math::
   \phi_i (R,z) =
   \begin{cases}
   1       & \quad \text{if point } (R,z) \text{ is inside triangle } i \\
   0       & \quad \text{otherwise}
   \end{cases}

Tokamesh instead uses first-order (barycentric) interpolation to define the emission
inside each triangle. In this approach, the basis-function weights :math:`x_i` become
the emissivity :math:`\mathcal{E}_i` at each vertex. The emissivity inside a triangle
made up of vertices :math:`i`, :math:`j` and :math:`k` is given by the plane defined
by the three points :math:`(R_i, z_i, \mathcal{E}_i), (R_j, z_j, \mathcal{E}_j), (R_k, z_k, \mathcal{E}_k)`.

This leads to the following 'barycentric' basis functions:

.. math::
   \phi_i (R,z) =
   \begin{cases}
   \lambda_i (R,z)       & \quad \text{if point } (R,z) \text{ is inside a triangle containing vertex } i \\
   0       & \quad \text{otherwise}
   \end{cases}

where :math:`\lambda_i (R,z)` is the barycentric coordinate for vertex :math:`i` given by

.. math::
   \lambda_i (R,z) =
   \frac{ (z_j - z_k)(R - R_k) + (R_k - R_j)(z - z_k) }{(z_j - z_k)(R_i - R_k) + (R_k - R_j)(z_i - z_k)}

and :math:`(R_i, z_i), (R_j, z_j), (R_k, z_k)` are the positions of vertices :math:`i`, :math:`j` and :math:`k`
respectively.