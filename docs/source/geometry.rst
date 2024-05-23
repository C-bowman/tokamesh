Calculating geometry matrices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Geometry matrix calculation is handled by the ``calculate_geometry_matrix`` class
from the ``tokamesh.geometry`` module.

Example code for geometry matrix calculation can be found in the
`geometry matrix jupyter notebook demo <https://github.com/C-bowman/tokamesh/blob/main/demos/geometry_matrix_demo.ipynb>`_.

.. autofunction:: tokamesh.geometry.calculate_geometry_matrix


.. autoclass:: tokamesh.geometry.GeometryMatrix
   :members: build_sparse_array, save, load, entry_values, row_indices, col_indices, matrix_shape, R_vertices, z_vertices, triangle_vertices