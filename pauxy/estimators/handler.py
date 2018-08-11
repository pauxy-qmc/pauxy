"""Routines and classes for estimation of observables."""

from __future__ import print_function

import copy
import h5py
import numpy
import os
import scipy.linalg
import time
import warnings
from pauxy.estimators.back_propagation import BackPropagation
from pauxy.estimators.mixed import Mixed
from pauxy.estimators.itcf import ITCF


class Estimators(object):
    """Container for qmc estimates of observables.

    Parameters
    ----------
    estimates : dict
        input options detailing which estimators to calculate. By default only
        mixed estimates will be calculated.
    root : bool
        True if on root/master processor.
    qmc : :class:`pauxy.state.QMCOpts` object.
        Container for qmc input options.
    system : :class:`pauxy.hubbard.Hubbard` / system object in general.
        Container for model input options.
    trial : :class:`pauxy.trial_wavefunction.X' object
        Trial wavefunction class.
    BT2 : :class:`numpy.ndarray`
        One body propagator.
    verbose : bool
        If true we print out additional setup information.

    Attributes
    ----------
    h5f : :class:`h5py.File`
        Output file object.
    estimates : dict
        Dictionary of estimator objects.
    back_propagation : bool
        True if doing back propagation, specified in estimates dict.
    nbp : int
        Number of back propagation steps.
    nprop_tot : int
        Total number of auxiliary field configurations we store / use for back
        propagation and itcf calculation.
    calc_itcf : bool
        True if calculating imaginary time correlation functions (ITCFs).
    """

    def __init__(self, estimates, root, qmc, system, trial, BT2, verbose=False):
        if root:
            index = estimates.get('index', 0)
            self.h5f_name = estimates.get('filename', None)
            if self.h5f_name is None:
                overwrite = estimates.get('overwrite', True)
                self.h5f_name = 'estimates.%s.h5' % index
                while os.path.isfile(self.h5f_name) and not overwrite:
                    index = int(self.h5f_name.split('.')[1])
                    index = index + 1
                    self.h5f_name = 'estimates.%s.h5' % index
            self.h5f = h5py.File(self.h5f_name, 'w')
        else:
            self.h5f = None
        # Sub-members:
        # 1. Back-propagation
        mixed = estimates.get('mixed', {})
        self.estimators = {}
        dtype = complex
        self.estimators['mixed'] = Mixed(mixed, root, self.h5f,
                                         qmc, trial, dtype)
        bp = estimates.get('back_propagated', None)
        self.back_propagation = bp is not None
        if self.back_propagation:
            self.estimators['back_prop'] = BackPropagation(bp, root, self.h5f,
                                                           qmc, system, trial,
                                                           dtype, BT2)
            self.nprop_tot = self.estimators['back_prop'].nmax
            self.nbp = self.estimators['back_prop'].nmax
        else:
            self.nprop_tot = 1
            self.nbp = 1
        # 2. Imaginary time correlation functions.
        itcf = estimates.get('itcf', None)
        self.calc_itcf = itcf is not None
        if self.calc_itcf:
            self.estimators['itcf'] = ITCF(itcf, qmc, trial, root, self.h5f,
                                           system.nbasis, dtype,
                                           self.nprop_tot, BT2)
            self.nprop_tot = self.estimators['itcf'].nprop_tot

    def print_step(self, comm, nprocs, step, nmeasure):
        """Print QMC estimates.

        Parameters
        ----------
        comm :
            MPI communicator.
        nprocs : int
            Number of processors.
        step : int
            Current iteration number.
        nmeasure : int
            Number of steps between measurements.
        """
        for k, e in self.estimators.items():
            e.print_step(comm, nprocs, step, nmeasure)
        if comm.Get_rank() == 0:
            self.h5f.flush()

    def update(self, system, qmc, trial, psi, step, free_projection=False):
        """Update estimators

        Parameters
        ----------
        system : system object in general.
            Container for model input options.
        qmc : :class:`pauxy.state.QMCOpts` object.
            Container for qmc input options.
        trial : :class:`pauxy.trial_wavefunction.X' object
            Trial wavefunction class.
        psi : :class:`pauxy.walkers.Walkers` object
            CPMC wavefunction.
        step : int
            Current simulation step
        free_projection : bool
            True if doing free projection.
        """
        for k, e in self.estimators.items():
            e.update(system, qmc, trial, psi, step, free_projection)
