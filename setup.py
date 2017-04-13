from distutils.core import setup

setup(
    name='afqmcpy',
    version='0.1',
    author='Fionn Malone',
    packages=('afqmcpy',),
    license='GNU',
    description='Simple AFQMC Code',
    long_description=open('README.rst').read(),
    requires=['numpy', 'pandas (>= 0.13)',],
)
