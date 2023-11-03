# setup.py
"""
cd Bin/CalculationCode/lib
python setup.py build_ext --inplace
"""
from setuptools import setup
from Cython.Build import cythonize
import numpy


setup(
    ext_modules=cythonize("boussinesq_eq1d.pyx"),
    include_dirs=[numpy.get_include()],
)
