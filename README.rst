=====
PAUXY
=====

PAUXY is a collection of **P**\ ython implementations of **AUX**\ illiar\ **Y** field
quantum Monte Carlo algorithms with a focus on simplicity and transparency rather than
speed.

Features
--------
Pauxy can currently do:

- estimation of ground state properties of model systems (Hubbard model or a generic
  system via an FCIDUMP).
- phaseless, constrained path and free projection AFQMC. 
- calculate expectation values and correlation functions using back propagation.
- calculate imaginary time correlation functions.
- control the sign problem using a variety of trial wavefunctions including free-electron,
  UHF and GHF, all in single- or multi-determinant form.

Installation
------------

Clone the repository

::

    $ git clone https://github.com/fdmalone/pauxy.git

and set the PYTHONPATH appropriately.

Requirements
------------

* python (>= 3.6)
* numpy (>= 0.19.1)
* scipy (>= 1.13.3)
* h5py (>= 2.7.1)
* matplotlib (optional)
* mpi4py (optional)

In addition, if you want to run the test suite you'll need to get
`testcode <https://github.com/jsspencer/testcode>`_.

Running the Test Suite
----------------------

First change to the test directory and run

::

    $ ~/path/to/testcode/bin/testcode.py

If python3 is not your default python interpreter then run

::

    $ ~/path/to/testcode/bin/testcode.py -p 1 --user-option pauxy launch_parallel python3

Currently only serial tests exist.

Documentation
-------------

Notes on the underlying theory as well as documentation and tutorials are available at
`readthedocs <https://pauxy.readthedocs.org>`_.

.. image:: http://readthedocs.org/projects/pauxy/badge/?version=latest
    :target: http://pauxy.readthedocs.io/en/latest/?badge=latest

LICENSE
-------
GPL v2.1
