from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="tokamesh",
    version="0.1.0",
    author="Chris Bowman",
    author_email="chris.bowman.physics@gmail.com",
    description="Python tools for constructing meshes and geometry matrices used in tomography problems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/C-bowman/tokamesh",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
