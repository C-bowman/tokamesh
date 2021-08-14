
# Tokamesh
Tokamesh is a Python package which provides tools for constructing meshes 
and geometry matrices used in tomographic inversion problems in toroidal
fusion-energy devices such as [MAST-U](https://ccfe.ukaea.uk/research/mast-upgrade/).

### Features
 - **Advanced geometry matrix calculation**
   - Tokamesh constructs geometry matrices using barycentric linear-interpolation rather
     than the typical zeroth-order interpolation. This allows for accurate tomographic
     inversions with a significantly lower number of basis functions.
     <p style="text-align:center;"><img width="700" alt="4" src="https://i.imgur.com/DU5UeLD.png"></p>
     <br>
 - **Tomography-optimised mesh construction**
   - Tokamesh provides tools to create meshes that are optimised for tomography problems,
     e.g. local-refinement of triangles to increase mesh density in areas where it is
     needed without greatly increasing the size of the mesh.
   <p style="text-align:center;"><img width="600" alt="4" src="https://i.imgur.com/lbdZJbY.png"></p>

### Documentation
ReadTheDocs site coming soon!