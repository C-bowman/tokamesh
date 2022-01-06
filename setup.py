from setuptools import setup, Extension
from Cython.Build import cythonize
import sys

libraries = []
if sys.platform == "linux":
    libraries.append("m")


ext_modules = [
    Extension(
        "tokamesh.triangle.triangulate",
        sources=["tokamesh/triangle/triangulate.pyx", "tokamesh/triangle/triangle.c"],
        libraries=libraries,
        define_macros=[("REAL", "double"), ("TRILIBRARY", None), ("NO_TIMER", None)],
    )
]

setup(ext_modules=cythonize(ext_modules, gdb_debug=True, language_level=3))
