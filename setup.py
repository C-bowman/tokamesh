from setuptools import setup, Extension
from Cython.Build import cythonize

ext_modules = [
    Extension(
        "tokamesh.triangle.triangulate",
        sources=["tokamesh/triangle/triangulate.pyx", "tokamesh/triangle/triangle.c"],
        libraries=["m"],
        define_macros=[("REAL", "double"), ("TRILIBRARY", None), ("NO_TIMER", None)],
    )
]

setup(ext_modules=cythonize(ext_modules, gdb_debug=True, language_level=3))
