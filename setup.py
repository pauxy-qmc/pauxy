from Cython.Build import cythonize
import numpy
from setuptools import find_packages, setup
from setuptools.extension import Extension
import sys

extensions = [
        Extension("pauxy.estimators.ueg_kernels",
                  ["pauxy/estimators/ueg_kernels.pyx"],
		  include_dirs=[numpy.get_include()])
        ]

setup(
    name='pauxy',
    author='PAUXY developers',
    url='http://github.com/fdmalone/pauxy',
    packages=find_packages(exclude=['examples', 'docs', 'tests', 'tools', 'setup.py']),
    license='Lesser GPL v2.1',
    description='Python Implementations of Auxilliary Field QMC algorithms',
    python_requires=">=3.6.0",
    long_description=open('README.rst').read(),
    ext_modules = cythonize(extensions, include_path=[numpy.get_include()],
                            compiler_directives={'language_level':
                                                 sys.version_info[0]})
)
