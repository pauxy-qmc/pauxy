from setuptools import find_packages, setup

setup(
    name='pauxy',
    version='0.1',
    author='Fionn Malone',
    url='http://github.com/fdmalone/pauxy',
    packages=find_packages(exclude=['examples', 'docs', 'tests', 'tools', 'setup.py']),
    license='Lesser GPL v2.1',
    description='Python Implementations of Auxilliary Field QMC algorithms',
    long_description=open('README.rst').read(),
    requires=['numpy (>=0.19.1)', 'pandas (>=0.20)',
	      'scipy (>=1.13.3)', 'h5py (>=2.7.1)'],
)
