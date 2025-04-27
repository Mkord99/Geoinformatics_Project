"""
Setup script for EOFtoolkit.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="eoftoolkit",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A toolkit for EOF (Empirical Orthogonal Function) analysis of NetCDF data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/eoftoolkit",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "Topic :: Scientific/Engineering :: Hydrology",
        "Topic :: Scientific/Engineering :: Oceanography",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "scipy>=1.5.0",
        "matplotlib>=3.3.0",
        "pandas>=1.0.0",
        "netCDF4>=1.5.0",
        "basemap>=1.2.0",
        "pyproj>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
            "flake8",
            "black",
            "sphinx",
            "sphinx_rtd_theme",
        ],
    },
)