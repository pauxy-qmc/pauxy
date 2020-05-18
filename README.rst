=====
PAUXY
=====

PAUXY is a collection of **P**\ ython implementations of **AUX**\ illiar\ **Y** field
quantum Monte Carlo algorithms with a focus on simplicity rather than speed.

.. image:: https://travis-ci.org/pauxy-qmc/pauxy.svg?branch=master
    :target: https://travis-ci.org/pauxy-qmc/pauxy

.. image:: http://readthedocs.org/projects/pauxy/badge/?version=latest
    :target: http://pauxy.readthedocs.io/en/latest/?badge=latest

.. image:: http://img.shields.io/badge/License-LGPL%20v2.1-blue.svg
    :target: http://github.com/fdmalone/pauxy/blob/master/LICENSE

Features
--------
PAUXY can currently:

- estimate ground state properties of real (ab-initio) and model (Hubbard + UEG) systems.
- perform phaseless and constrained path AFQMC.
- calculate expectation values and correlation functions using back propagation.
- calculate imaginary time correlation functions.
- perform simple data analysis.

Installation
------------

Clone the repository

::

    $ git clone https://github.com/pauxy-qmc/pauxy.git

and run the following in the top-level pauxy directory

::

    $ python setup.py build_ext --inplace

You may also need to set your PYTHONPATH appropriately.

Requirements
------------

* python (>= 3.6)
* numpy (>= 0.19.1)
* scipy (>= 1.13.3)
* h5py (>= 2.7.1)
* mpi4py (>= 3.0.1)
* cython (>= 0.29.2)

To run the tests you will need pytest and pandas.
To perform error analysis you will also need `pyblock <https://github.com/jsspencer/pyblock>`


Running the Test Suite
----------------------

Pauxy contains unit tests and short deterministic longer tests of full calculations.
To the tests you can do:

::

    $ pytest

In the main repository.

.. image:: https://travis-ci.org/pauxy-qmc/pauxy.svg?branch=master
    :target: https://travis-ci.org/pauxy-qmc/pauxy

Documentation
-------------

Notes on the underlying theory as well as documentation and tutorials are available at
`readthedocs <https://pauxy.readthedocs.org>`_.

.. image:: http://readthedocs.org/projects/pauxy/badge/?version=latest
    :target: http://pauxy.readthedocs.io/en/latest/?badge=latest
