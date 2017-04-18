import hamiltonian
import numpy as np
import scipy.linalg
import walker
import estimators

def do_qmc(state, interactive=False):

    nw = 10
    elocal = 0
    est = []
    (eigs, eigv) = scipy.linalg.eig(state.system.T)
    psi = [walker.Walker(1, state.system, eigv) for w in range(0, nw)]
    psi_trial = [psi[0].phi[0], psi[0].phi[1]]
    for step in range(0, state.nsteps):
        for w in psi:
            w.prop_t2(state.projectors.bt2)
            w.prop_v(state.auxf, state.system.nbasis)
            w.prop_t2(state.projectors.bt2)
            w.overlap(psi_trial)
            # if step%state.nmeasure == 0:
                # w.reortho()
            elocal += estimators.local_energy(state.system, w, psi_trial)
        if step%state.nmeasure == 0:
            if interactive:
                est.append(elocal/(state.nmeasure*nw))
            else:
                print (("%5d     %10.8e")%(step, elocal/(state.nmeasure*nw)))
            elocal = 0.0


    if interactive:
        return est

