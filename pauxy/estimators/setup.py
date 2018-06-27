from distutils.core import setup
from Cython.Build import cythonize

setup(name="ueg_kernels", ext_modules=cythonize('ueg_kernels.pyx'),)
