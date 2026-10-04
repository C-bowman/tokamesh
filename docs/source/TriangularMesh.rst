TriangularMesh utility class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``TriangularMesh`` class allows for fast interpolation on triangular meshes.

Spatial lookup uses uniform axis grids with constant-time cell indexing per
coordinate. Grid-cell boundaries belong to the cell on their left; the minimum
axis bound remains excluded and the maximum included, preserving the existing
lookup convention.


.. autoclass:: tokamesh.TriangularMesh
   :members: interpolate, build_interpolator_matrix, find_triangle, plot_field, get_field_image, draw, save, load


Uniform grid lookup
-------------------

.. autoclass:: tokamesh.utilities.UniformGridLookup
   :members: lookup_index