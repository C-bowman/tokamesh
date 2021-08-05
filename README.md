
# Tokamesh
Tokamesh is a Python package which provides tools for constructing meshes 
and geometry matrices used in tomographic inversion problems in toroidal
fusion-energy devices such as [MAST-U](https://ccfe.ukaea.uk/research/mast-upgrade/).

## Features
 - **Advanced geometry matrix calculation**
   - Tokamesh constructs geometry matrices using barycentric linear-interpolation rather
     than the typical zeroth-order interpolation. This allows for accurate tomographic
     inversions with a significantly lower number of basis functions.