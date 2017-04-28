import numpy as np
import scipy.linalg
import walker
import estimators
import pop_control
import trial_wave_function

def do_qmc(state, interactive=False):

    est = []
    psi_trial = trial_wave_function.free_electron(state.system)
    psi = [walker.Walker(1, state.system, psi_trial) for w in range(0, state.nwalkers)]
    E_T = estimators.local_energy(state.system, psi[0])
    estimators.header()
    elocal = 0
    total_weight = 0
    for step in range(0, state.nsteps):
        for w in psi:
            if w.weight > 0:
                w.prop_t2(state.projectors.bt2, psi_trial)
            if w.weight > 0:
                w.prop_v(state.auxf, state.system.nbasis, psi_trial)
            if w.weight > 0:
                w.prop_t2(state.projectors.bt2, psi_trial)
            w.weight = w.weight * np.exp(state.dt*(E_T-state.cfac))
            elocal += w.weight * estimators.local_energy(state.system, w)
            total_weight += w.weight
            if step%state.nmeasure == 0:
                w.reortho()
        if step%state.nmeasure == 0:
            if interactive:
                est.append(elocal/(state.nmeasure*total_weight))
            else:
                print (("%9d %10.8e %10.8e %10.8e")%(step, total_weight/state.nmeasure,
                        elocal/state.nmeasure,np.exp(state.dt*(E_T-state.cfac)) ))
            pop_control.comb(psi, state.nwalkers)
            E_T = elocal / total_weight
            elocal = 0.0
            nw = 0.0

    if interactive:
        return est

