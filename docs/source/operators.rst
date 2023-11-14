Mesh Operators
~~~~~~~~~~~~~~

The ``tokamesh.operators`` module provides functions for constructing
linear operators (i.e. matrices) which act on a vector of field values
at each mesh vertex to compute a new vector.

.. autofunction:: tokamesh.operators.edge_difference_matrix

.. autofunction:: tokamesh.operators.umbrella_matrix

.. autofunction:: tokamesh.operators.parallel_derivative

.. autofunction:: tokamesh.operators.perpendicular_derivative