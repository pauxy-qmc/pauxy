import numpy as np
import scipy.linalg
import copy
import afqmcpy.estimators
import afqmcpy.trial_wave_function as twfn

# Worthwhile overloading / having real and complex walker classes (Hermitian
# conjugate)?
class Walker:

    def __init__(self, nw, system, trial, index):
        self.weight = nw
        self.phi = copy.deepcopy(trial.psi)
        self.inv_ovlp = [0, 0]
        self.inverse_overlap(trial.psi, system.nup)
        self.G = [0, 0]
        self.greens_function(trial, system.nup)
        self.ot = 1.0
        self.E_L = afqmcpy.estimators.local_energy(system, self.G)[0].real
        # walkers overlap at time tau before backpropagation occurs
        self.ot_bp = 1.0
        # walkers weight at time tau before backpropagation occurs
        self.weight_bp = nw
        # walkers auxiliary field configuration in back propagation interval
        self.field_config = np.zeros(shape=(system.nbasis), dtype=int)

    def inverse_overlap(self, trial, nup):
        self.inv_ovlp[0] = scipy.linalg.inv((trial[:,:nup].conj()).T.dot(self.phi[:,:nup]))
        self.inv_ovlp[1] = scipy.linalg.inv((trial[:,nup:].conj()).T.dot(self.phi[:,nup:]))

    def update_inverse_overlap(self, trial, vtup, vtdown, nup, i):
        self.inv_ovlp[0] = afqmcpy.utils.sherman_morrison(self.inv_ovlp[0],
                                                          trial.psi[:,:nup].T[:,i],
                                                          vtup)
        self.inv_ovlp[1] = afqmcpy.utils.sherman_morrison(self.inv_ovlp[1],
                                                          trial.psi[:,nup:].T[:,i],
                                                          vtdown)

    def calc_otrial(self, trial):
        # The importance function, i.e. <phi_T|phi>. We do 1 over this because
        # inv_ovlp stores the inverse overlap matrix for ease when updating the
        # green's function.
        return 1.0/(scipy.linalg.det(self.inv_ovlp[0])*scipy.linalg.det(self.inv_ovlp[1]))

    def update_overlap(self, probs, xi, coeffs):
        self.ot = 2 * self.ot * probs[xi]

    def reortho(self, nup):
        (self.phi[:,:nup], Rup) = scipy.linalg.qr(self.phi[:,:nup], mode='economic')
        (self.phi[:,nup:], Rdown) = scipy.linalg.qr(self.phi[:,nup:], mode='economic')
        signs_up = np.diag(np.sign(np.diag(Rup)))
        signs_down = np.diag(np.sign(np.diag(Rdown)))
        self.phi[:,:nup] = self.phi[:,:nup].dot(signs_up)
        self.phi[:,nup:] = self.phi[:,nup:].dot(signs_down)
        detR = (scipy.linalg.det(signs_up.dot(Rup))*scipy.linalg.det(signs_down.dot(Rdown)))
        self.ot = self.ot / detR
        return detR

    def greens_function(self, trial, nup):
        self.G[0] = (
            (self.phi[:,:nup].dot(self.inv_ovlp[0]).dot(trial.psi[:,:nup].conj().T)).T
        )
        self.G[1] = (
            (self.phi[:,nup:].dot(self.inv_ovlp[1]).dot(trial.psi[:,nup:].conj().T)).T
        )

    def local_energy(self, system):
        return afqmcpy.estimators.local_energy(system, self.G)


class MultiDetWalker:
    '''Essentially just some wrappers around Walker class.'''

    def __init__(self, nw, system, trial, index=0):
        self.weight = nw
        self.phi = copy.deepcopy(trial.psi[index])
        # This stores an array of overlap matrices with the various elements of
        # the trial wavefunction.
        up_shape = (trial.ndets, system.nup, system.nup)
        down_shape = (trial.ndets, system.ndown, system.ndown)
        self.inv_ovlp = [np.zeros(shape=(up_shape)),
                         np.zeros(shape=(down_shape))]
        self.inverse_overlap(trial.psi, system.nup)
        # Green's functions for various elements of the trial wavefunction.
        self.Gi = np.zeros(shape=(trial.ndets, 2, system.nbasis,
                              system.nbasis))
        # Should be nfields per basis * ndets.
        # Todo: update this for the continuous HS trasnform case.
        self.R = np.zeros(shape=(trial.ndets, 2, 2))
        # Actual green's function contracted over determinant index in Gi above.
        # i.e., <psi_T|c_i^d c_j|phi>
        self.G = np.zeros(shape=(2, system.nbasis, system.nbasis))
        self.ots = np.zeros(shape=(2,trial.ndets))
        # Contains overlaps of the current walker with the trial wavefunction.
        self.ot = self.calc_otrial(trial)
        self.greens_function(trial, system.nup)
        self.E_L = afqmcpy.estimators.local_energy(system, self.G)[0].real
        G2 = afqmcpy.estimators.gab(trial.psi[0][:,:system.nup],
                                    trial.psi[0][:,:system.nup])
        self.index = index
        self.field_config = np.zeros(shape=(system.nbasis), dtype=int)

    def inverse_overlap(self, trial, nup):
        for (indx, t) in enumerate(trial):
            self.inv_ovlp[0][indx,:,:] = (
                scipy.linalg.inv((t[:,:nup].conj()).T.dot(self.phi[:,:nup]))
            )
            self.inv_ovlp[1][indx,:,:] = (
                scipy.linalg.inv((t[:,nup:].conj()).T.dot(self.phi[:,nup:]))
            )

    def calc_otrial(self, trial):
        # The importance function, i.e. <phi_T|phi>. We do 1 over this because
        # inv_ovlp stores the inverse overlap matrix for ease when updating the
        # green's function.
        # The trial wavefunctions coefficients should be complex conjugated
        # on initialisation!
        # This looks wrong for the UHF case - no spin considerations here.
        ot = 0.0
        for (ix, c) in enumerate(trial.coeffs):
            deto_up = 1.0 / scipy.linalg.det(self.inv_ovlp[0][ix,:,:])
            deto_down = 1.0 / scipy.linalg.det(self.inv_ovlp[1][ix,:,:])
            self.ots[0, ix] = deto_up
            self.ots[1, ix] = deto_down
            ot += c * deto_up * deto_down
        return ot

    def update_overlap(self, probs, xi, coeffs):
        # Update each component's overlap and the total overlap.
        # The trial wavefunctions coeficients should be included in ots?
        self.ots = np.einsum('ji,ij->ij',self.R[:,xi,:],self.ots)
        self.ot = sum(coeffs*self.ots[0,:]*self.ots[1,:])

    def reortho(self, nup):
        # We assume that our walker is still block diagonal in the spin basis.
        (self.phi[:,:nup], Rup) = scipy.linalg.qr(self.phi[:,:nup], mode='economic')
        (self.phi[:,nup:], Rdown) = scipy.linalg.qr(self.phi[:,nup:], mode='economic')
        # Enforce a positive diagonal for the overlap.
        signs_up = np.diag(np.sign(np.diag(Rup)))
        signs_down = np.diag(np.sign(np.diag(Rdown)))
        self.phi[:,:nup] = self.phi[:,:nup].dot(signs_up)
        self.phi[:,nup:] = self.phi[:,nup:].dot(signs_down)
        # Todo: R is upper triangular.
        detR_up = scipy.linalg.det(signs_up.dot(Rup))
        detR_down = scipy.linalg.det(signs_down.dot(Rdown))
        self.ots[0] = self.ots[0] / detR_up
        self.ots[1] = self.ots[1] / detR_down
        self.ot = self.ot / (detR_up*detR_down)

    def greens_function(self, trial, nup):
        for (ix, t) in enumerate(trial.psi):
            # construct "local" green's functions for each component of psi_T
            self.Gi[ix,0,:,:] = (
                (self.phi[:,:nup].dot(self.inv_ovlp[0][ix]).dot(t[:,:nup].conj().T)).T
            )
            self.Gi[ix,1,:,:] = (
                (self.phi[:,nup:].dot(self.inv_ovlp[1][ix]).dot(t[:,nup:].conj().T)).T
            )
        denom = np.einsum('j,ij->i',trial.coeffs, self.ots)
        self.G = np.einsum('i,ijkl,ji->jkl', trial.coeffs, self.Gi, self.ots)
        self.G[0] = self.G[0]/denom[0]
        self.G[1] = self.G[1]/denom[1]

    def update_inverse_overlap(self, trial, vtup, vtdown, nup, i):
        for (ix, t) in enumerate(trial.psi):
            self.inv_ovlp[0][ix] = (
                afqmcpy.utils.sherman_morrison(self.inv_ovlp[0][ix],
                                               t[:,:nup].T[:,i], vtup)
            )
            self.inv_ovlp[1][ix] = (
                afqmcpy.utils.sherman_morrison(self.inv_ovlp[1][ix],
                    t[:,nup:].T[:,i], vtdown)
            )
    def local_energy(self, system):
        return afqmcpy.estimators.local_energy_multi_det(system,
                                                         self.Gi,
                                                         self.weights)

class MultiGHFWalker:
    '''Essentially just some wrappers around Walker class.'''

    def __init__(self, nw, system, trial, index=0):
        self.weight = nw
        # Initialise to a particular free electron slater determinant rather
        # than GHF.
        if trial.read_init is not None:
            orbs = twfn.read_fortran_complex_numbers(trial.read_init)
            self.phi = orbs.reshape((2*system.nbasis, system.ne), order='F')
        else:
            self.phi = np.zeros(shape=(2*system.nbasis,system.ne),
                                dtype=trial.psi.dtype)
            tmp = afqmcpy.trial_wave_function.FreeElectron(system,
                                     trial.psi.dtype==complex, {})
            self.phi[:system.nbasis,:system.nup] = tmp.psi[:,:system.nup]
            self.phi[system.nbasis:,system.nup:] = tmp.psi[:,system.nup:]
        # This stores an array of overlap matrices with the various elements of
        # the trial wavefunction.
        self.inv_ovlp = np.zeros(shape=(trial.ndets, system.ne, system.ne),
                                 dtype=self.phi.dtype)
        self.weights = np.zeros(trial.ndets, dtype=trial.psi.dtype)
        ovlp = (np.dot(trial.psi[0].conj().T, self.phi))
        self.inverse_overlap(trial.psi, system.nup)
        # Green's functions for various elements of the trial wavefunction.
        self.Gi = np.zeros(shape=(trial.ndets, 2*system.nbasis,
                           2*system.nbasis), dtype=self.phi.dtype)
        # Should be nfields per basis * ndets.
        # Todo: update this for the continuous HS trasnform case.
        self.R = np.zeros(shape=(trial.ndets, 2), dtype=self.phi.dtype)
        # Actual green's function contracted over determinant index in Gi above.
        # i.e., <psi_T|c_i^d c_j|phi>
        self.G = np.zeros(shape=(2*system.nbasis, 2*system.nbasis),
                          dtype=self.phi.dtype)
        self.ots = np.zeros(trial.ndets, dtype=self.phi.dtype)
        # Contains overlaps of the current walker with the trial wavefunction.
        self.ot = self.calc_otrial(trial)
        self.greens_function(trial, system.nup)
        self.E_L = afqmcpy.estimators.local_energy_ghf(system, self.Gi,
                                                       self.weights,
                                                       sum(self.weights))[0].real
        self.field_config = np.zeros(shape=(system.nbasis), dtype=int)
        self.nb = system.nbasis

    def inverse_overlap(self, trial, nup):
        for (indx, t) in enumerate(trial):
            self.inv_ovlp[indx,:,:] = (
                scipy.linalg.inv((t.conj()).T.dot(self.phi))
            )

    def calc_otrial(self, trial):
        # The importance function, i.e. <phi_T|phi>. We do 1 over this because
        # inv_ovlp stores the inverse overlap matrix for ease when updating the
        # green's function.
        # The trial wavefunctions coefficients should be complex conjugated
        # on initialisation!
        for (ix, inv) in enumerate(self.inv_ovlp):
            self.ots[ix] = 1.0 / scipy.linalg.det(inv)
            self.weights[ix] = trial.coeffs[ix] * self.ots[ix]
        return sum(self.weights)

    def update_overlap(self, probs, xi, coeffs):
        # Update each component's overlap and the total overlap.
        # The trial wavefunctions coeficients should be included in ots?
        self.ots = self.R[:,xi] * self.ots
        self.weights = coeffs * self.ots
        # The overlap should have the phaseless constraint imposed on it
        self.ot = 2.0 * self.ot * probs[xi]

    def reortho(self, nup):
        # We assume that our walker is still block diagonal in the spin basis.
        (self.phi[:self.nb,:nup], Rup) = scipy.linalg.qr(self.phi[:self.nb,:nup], mode='economic')
        (self.phi[self.nb:,nup:], Rdown) = scipy.linalg.qr(self.phi[self.nb:,nup:], mode='economic')
        # Enforce a positive diagonal for the overlap.
        signs_up = np.diag(np.sign(np.diag(Rup)))
        signs_down = np.diag(np.sign(np.diag(Rdown)))
        self.phi[:self.nb,:nup] = self.phi[:self.nb,:nup].dot(signs_up)
        self.phi[self.nb:,nup:] = self.phi[self.nb:,nup:].dot(signs_down)
        # Todo: R is upper triangular.
        detR = (scipy.linalg.det(signs_up.dot(Rup))*scipy.linalg.det(signs_down.dot(Rdown)))
        self.ots = self.ots / detR
        self.ot = self.ot / detR

    def greens_function(self, trial, nup):
        for (ix, t) in enumerate(trial.psi):
            # construct "local" green's functions for each component of psi_T
            self.Gi[ix,:,:] = (
                (self.phi.dot(self.inv_ovlp[ix]).dot(t.conj().T)).T
            )
        denom = sum(self.weights)
        self.G = np.einsum('i,ijk->jk', self.weights, self.Gi) / denom

    def update_inverse_overlap(self, trial, vtup, vtdown, nup, i):
        for (indx, t) in enumerate(trial.psi):
            self.inv_ovlp[indx,:,:] = (
                scipy.linalg.inv((t.conj()).T.dot(self.phi))
            )

    def local_energy(self, system):
        return afqmcpy.estimators.local_energy_ghf(system, self.Gi,
                                                   self.weights, self.ot)
    # def update_inverse_overlap(self, trial, vtup, vtdown, nup, i):
        # for (ix, t) in enumerate(trial.psi):
            # self.inv_ovlp[ix] = (
                # afqmcpy.utils.sherman_morrison(self.inv_ovlp[ix],
                                               # t[:,:nup].T[:,i], vtup)
            # )
