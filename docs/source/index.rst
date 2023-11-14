Tokamesh
========
Tokamesh is a Python package which provides tools for constructing meshes and geometry matrices
used in tomographic inversion problems in toroidal fusion-energy devices.

Please use the `issue tracker <https://github.com/C-bowman/tokamesh/issues>`_ to report any issues,
or to request features/improvements.

Tokamesh is available from `PyPI <https://pypi.org/project/tokamesh/>`_, so can
be easily installed using `pip <https://pip.pypa.io/en/stable/>`_: as follows:

.. code-block:: bash

   pip install tokamesh

If pip is not available, you can clone from the GitHub `source repository <https://github.com/C-bowman/tokamesh>`_
or download the files from `PyPI <https://pypi.org/project/tokamesh/>`_ directly.


.. toctree::
   :maxdepth: 1
   :caption: Geometry matrices

   geometry_background
   geometry


.. toctree::
   :maxdepth: 1
   :caption: Triangular meshes

   construction
   construction_refinement
   operators

.. toctree::
   :maxdepth: 1
   :caption: Interpolation and plotting

   TriangularMesh