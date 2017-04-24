import hamiltonian
import numpy as np
import scipy.linalg
import walker
import estimators

def do_qmc(state, interactive=False):

    est = []
    (eigs, eigv) = scipy.linalg.eig(state.system.T)
    psi_trial = np.array([eigv[:,:state.system.nup], eigv[:,:state.system.ndown]])
    idx = eigs.argsort()[::-1]
    eigs = eigs[idx]
    eigv = eigv[:,idx]
    psi = [walker.Walker(1, state.system, psi_trial) for w in range(0, state.nwalkers)]
    E_T = estimators.local_energy(state.system, psi[0])
    for step in range(0, state.nsteps):
        elocal = 0.0
        nw = 0.0
        for w in psi:
            w.prop_t2(state.projectors.bt2, psi_trial)
            if w.weight > 0:
                w.prop_v(state.auxf, state.system.nbasis, psi_trial)
            # print w.weight
            if w.weight > 0:
                w.prop_t2(state.projectors.bt2, psi_trial)
            # print w.weight
            if step%state.nmeasure == 0:
                w.reortho()
            w.weight = w.weight * np.exp(state.dt*E_T)
            elocal += w.weight * estimators.local_energy(state.system, w)
            nw += w.weight
        E_T = elocal / nw
        if step%state.nmeasure == 0:
            if interactive:
                est.append(elocal/(state.nmeasure*nw))
            else:
                print (("%5d     %10.8e    %10.8e")%(step, elocal, nw))

    if interactive:
        return est

