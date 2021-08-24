
# Tokamesh

[![Documentation Status](https://readthedocs.org/projects/tokamesh/badge/?version=latest)](https://tokamesh.readthedocs.io/en/latest/?badge=latest)

Tokamesh is a Python package which provides tools for constructing meshes 
and geometry matrices used in tomographic inversion problems in toroidal
fusion-energy devices such as [MAST-U](https://ccfe.ukaea.uk/research/mast-upgrade/).

### Features
 - **Advanced geometry matrix calculation**
   - Tokamesh constructs geometry matrices using barycentric linear-interpolation rather
     than the typical zeroth-order interpolation. This allows for accurate tomographic
     inversions with a significantly lower number of basis functions.
     ![geo matrix example](https://i.imgur.com/tqElYG3.png)
     <br><br>
 - **Tomography-optimised mesh construction**
   - Tokamesh provides tools to create meshes that are optimised for tomography problems,
     e.g. local-refinement of triangles to increase mesh density in areas where it is
     needed without greatly increasing the size of the mesh.
     ![Example mesh](https://i.imgur.com/lNGVnaY.png)
 
### Jupyter notebook demos
Jupyter notebooks are available which demonstrate
[mesh construction](https://github.com/C-bowman/tokamesh/blob/main/demos/mesh_construction_demo.ipynb)
and [geometry matrix calculation](https://github.com/C-bowman/tokamesh/blob/main/demos/geometry_matrix_demo.ipynb).

### Installation
Tokamesh is available from [PyPI](https://pypi.org/project/tokamesh/), 
so can be easily installed using [pip](https://pip.pypa.io/en/stable/) as follows:
```bash
pip install tokamesh
```

### Documentation
The package documentation is available at [tokamesh.readthedocs.io](https://tokamesh.readthedocs.io/en/latest/).