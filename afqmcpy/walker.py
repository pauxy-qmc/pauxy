import numpy as np
import scipy.linalg
import copy
import afqmcpy.estimators

# Worthwhile overloading / having real and complex walker classes (Hermitian
# conjugate)?
class Walker:

    def __init__(self, nw, system, trial, index):
        self.weight = nw
        self.phi = copy.deepcopy(trial)
        self.inv_ovlp = [0, 0]
        self.inverse_overlap(trial, system.nup)
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

    def calc_otrial(self, trial):
        # The importance function, i.e. <phi_T|phi>. We do 1 over this because
        # inv_ovlp stores the inverse overlap matrix for ease when updating the
        # green's function.
        return 1.0/(scipy.linalg.det(self.inv_ovlp[0])*scipy.linalg.det(self.inv_ovlp[1]))

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
        self.G[0] = np.dot(np.dot(self.phi[:,:nup], self.inv_ovlp[0]), (trial[:,:nup].conj()).T).T
        self.G[1] = np.dot(np.dot(self.phi[:,nup:], self.inv_ovlp[1]), (trial[:,nup:].conj()).T).T


class MultiDetWalker:
    '''Essentially just some wrappers around Walker class.'''

    def __init__(self, nw, system, trial, index):
        self.weight = nw
        # ?
        self.phi = copy.deepcopy(trial[random.randint(0,ndets),:,:,:])
        # This contains the inverse of the walkers determinant with ALL of the
        # elements of the trial wavefunction
        self.inv_ovlp = numpy.zeros(numpy.shape(trial))
        self.inverse_overlap(trial)
        # Green's functions for ALL elements of the trial function
        self.Gi = numpy.zeros(numpy.shape(trial))
        # Actual green's function contracted over determinant index in Gi above.
        # i.e., <psi_T|c_i^d c_j|phi>
        self.G = numpy.zeros(numpy.shape(trial)[1:])
        self.greens_function(trial)
        self.ots = numpy.zeros(numpy.shape(trial)[0])
        self.ot = self.calc_otrial(trial)
        self.E_L = afqmcpy.estimators.local_energy_multi_det(system, self.G)[0].real
        self.index = index

    def inverse_overlap(self, trial):
        for (indx, t) in enumerate(trial):
            self.inv_ovlp[indx,0,:,:] = scipy.linalg.inv((t[0,:,:].conj()).T.dot(self.phi[0,:,:]))
            self.inv_ovlp[indx,1,:,:] = scipy.linalg.inv((t[1,:,:].conj()).T.dot(self.phi[1,:,:]))

    def calc_otrial(self, trial):
        # The importance function, i.e. <phi_T|phi>. We do 1 over this because
        # inv_ovlp stores the inverse overlap matrix for ease when updating the
        # green's function.
        for (indx, inv_ovlp) in enumerate(self.inv_ovlp):
            # The trial wavefunctions coefficients should be complex conjugated
            # on initialisation!
            # store the individual overlaps for use later when constructing
            # green's function.
            self.ots[indx] = trial.coeff[indx]/(scipy.linalg.det(inv_ovlp[indx,0,:,:])*scipy.linalg.det(inv_ovlp[indx,1,:,:]))
        return otrial

    def reortho(self):
        (self.phi, R) = [list(t)
                         for t in zip(*[scipy.linalg.qr(p, mode='economic')
                         for p in self.phi])]
        signs_up = np.diag(np.sign(np.diag(R[0])))
        signs_down = np.diag(np.sign(np.diag(R[1])))
        self.phi[0] = self.phi[0].dot(signs_up)
        self.phi[1] = self.phi[1].dot(signs_down)
        detR = (scipy.linalg.det(signs_up.dot(R[0]))*scipy.linalg.det(signs_down.dot(R[1])))
        self.ot = self.ot / detR

    def reortho_free(self):
        (self.phi, R) = [list(t)
                         for t in zip(*[scipy.linalg.qr(p, mode='economic')
                         for p in self.phi])]
        signs_up = np.diag(np.sign(np.diag(R[0])))
        signs_down = np.diag(np.sign(np.diag(R[1])))
        self.phi[0] = self.phi[0].dot(signs_up)
        self.phi[1] = self.phi[1].dot(signs_down)
        detR = (scipy.linalg.det(signs_up.dot(R[0]))*scipy.linalg.det(signs_down.dot(R[1])))
        self.ot = self.ot / detR
        self.weight = self.weight * detR

    def greens_function(self, trial):
        # Look into einsum, keep it simple for the moment.
        for (indx, t) in enumerate(trial):
            # construct "local" green's functions for each component of psi_T
            self.Gs[index,0,:,:] = numpy.multi_dot(self.phi[0], self.inv_ovlp[index,0,:,:], trial.conj().T).T
            self.Gs[index,1,:,:] = numpy.multi_dot(self.phi[1], self.inv_ovlp[index,1,:,:], trial.conj().T).T
        self.G = numpy.einsum('i,ijkl,i->jkl', trial.coeffs, self.Gs, self.ots) / self.ot
