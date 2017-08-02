import numpy as np
import scipy.linalg
import copy
import afqmcpy.estimators

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

    def update_overlap(self, trial, vtup, vtdown, nup, i):
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
            np.dot(np.dot(self.phi[:,:nup],self.inv_ovlp[0]),(trial.psi[:,:nup].conj()).T).T
        )
        self.G[1] = (
            np.dot(np.dot(self.phi[:,nup:],self.inv_ovlp[1]),(trial.psi[:,nup:].conj()).T).T
        )


class MultiDetWalker:
    '''Essentially just some wrappers around Walker class.'''

    def __init__(self, nw, system, trial, index=0):
        self.weight = nw
        self.phi = copy.deepcopy(trial.psi[index])
        # This stores an array of overlap matrices with the various elements of
        # the trial wavefunction.
        up_shape = (trial.psi.shape[0], system.nup, system.nup)
        down_shape = (trial.psi.shape[0], system.ndown, system.ndown)
        self.inv_ovlp = [np.zeros(shape=(up_shape)),
                         np.zeros(shape=(down_shape))]
        self.inverse_overlap(trial.psi, system.nup)
        # Green's functions for various elements of the trial wavefunction.
        self.Gi = np.zeros(shape=(trial.psi.shape[0], 2, system.nbasis,
                              system.nbasis))
        # Actual green's function contracted over determinant index in Gi above.
        # i.e., <psi_T|c_i^d c_j|phi>
        self.G = np.zeros(shape=(2, system.nbasis, system.nbasis))
        self.ots = np.zeros(trial.ndets)
        # Contains overlaps of the current walker with the trial wavefunction.
        self.ot = self.calc_otrial(trial)
        self.greens_function(trial, system.nup)
        self.E_L = afqmcpy.estimators.local_energy(system, self.G)[0].real
        G2 = afqmcpy.estimators.gab(trial.psi[0][:,:system.nup],
                                    trial.psi[0][:,:system.nup])
        print (self.E_L, afqmcpy.estimators.local_energy(system, [G2,G2])[0].real)
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
        for (ix, c) in enumerate(trial.coeffs):
            dup = 1.0 / scipy.linalg.det(self.inv_ovlp[0][ix,:,:])
            ddown = 1.0 / scipy.linalg.det(self.inv_ovlp[1][ix,:,:])
            self.ots[ix] = c * dup * ddown
        return (sum(self.ots))

    def reortho(self, nup):
        (self.phi[:,:nup], Rup) = scipy.linalg.qr(self.phi[:,:nup], mode='economic')
        (self.phi[:,nup:], Rdown) = scipy.linalg.qr(self.phi[:,nup:], mode='economic')
        # Enforce a positive diagonal for the overlap.
        signs_up = np.diag(np.sign(np.diag(Rup)))
        signs_down = np.diag(np.sign(np.diag(Rdown)))
        self.phi[:,:nup] = self.phi[:,:nup].dot(signs_up)
        self.phi[:,nup:] = self.phi[:,nup:].dot(signs_down)
        # Todo: R is upper triangular.
        detR = (scipy.linalg.det(signs_up.dot(Rup))*scipy.linalg.det(signs_down.dot(Rdown)))
        self.ots = self.ots / detR
        self.ot = self.ot / detR

    def greens_function(self, trial, nup):
        for (ix, t) in enumerate(trial.psi):
            # construct "local" green's functions for each component of psi_T
            self.Gi[ix,0,:,:] = (
                    self.phi[:,:nup].dot(self.inv_ovlp[0][ix]).dot(t[:,:nup].conj().T)
            )
            self.Gi[ix,1,:,:] = (
                    self.phi[:,nup:].dot(self.inv_ovlp[1][ix]).dot(t[:,nup:].conj().T)
            )
        self.G = np.einsum('i,ijkl,i->jkl', trial.coeffs, self.Gi, self.ots) / self.ot

    def update_overlap(self, trial, vtup, vtdown, nup, i):
        for (ix, t) in enumerate(trial.psi):
            self.inv_ovlp[0][ix] = (
                afqmcpy.utils.sherman_morrison(self.inv_ovlp[0][ix],
                                               t[:,:nup].T[:,i], vtup)
            )
            self.inv_ovlp[1][ix] = (
                afqmcpy.utils.sherman_morrison(self.inv_ovlp[1][ix],
                                                                  t[:,nup:].T[:,i],
                                                                  vtdown)
            )
