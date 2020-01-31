=====
PAUXY
=====

PAUXY is a collection of **P**\ ython implementations of **AUX**\ illiar\ **Y** field
quantum Monte Carlo algorithms with a focus on simplicity rather than speed.

.. image:: https://travis-ci.org/fdmalone/pauxy.svg?branch=master
    :target: https://travis-ci.org/fdmalone/pauxy

.. image:: http://readthedocs.org/projects/pauxy/badge/?version=latest
    :target: http://pauxy.readthedocs.io/en/latest/?badge=latest

.. image:: http://img.shields.io/badge/License-LGPL%20v2.1-blue.svg
    :target: http://github.com/fdmalone/pauxy/blob/master/LICENSE

Features
--------
PAUXY can currently:

- estimate ground state properties of model systems (Hubbard models and generic
  systems defined by (real) FCIDUMPs).
- perform phaseless, constrained path and free projection AFQMC using open ended random
  walks.
- calculate expectation values and correlation functions using back propagation.
- calculate imaginary time correlation functions.
- control the sign problem using a variety of trial wavefunctions including free-electron,
  UHF and GHF, all in single- or multi-determinant form.
- perform simple data analysis.

Installation
------------

Clone the repository

::

    $ git clone https://github.com/fdmalone/pauxy.git

and run the following in the top-level pauxy directory

::

    $ python setup.py build_ext --inplace

You will also need to set your PYTHONPATH appropriately.

Requirements
------------

* python (>= 3.6)
* numpy (>= 0.19.1)
* scipy (>= 1.13.3)
* h5py (>= 2.7.1)
* mpi4py (>= 3.0.1)
* cython (>= 0.29.2)

To run the tests you will need pytest and pandas.  To perform error analysis you will also
need `pyblock <https://github.com/jsspencer/pyblock>`


Running the Test Suite
----------------------

Pauxy contains unit tests and short deterministic longer tests of full calculations.
To the tests you can do:

::

    $ pytest

These tests should all pass.

.. image:: https://travis-ci.org/fdmalone/pauxy.svg?branch=master
    :target: https://travis-ci.org/fdmalone/pauxy

Documentation
-------------

Notes on the underlying theory as well as documentation and tutorials are available at
`readthedocs <https://pauxy.readthedocs.org>`_.

.. image:: http://readthedocs.org/projects/pauxy/badge/?version=latest
    :target: http://pauxy.readthedocs.io/en/latest/?badge=latest
