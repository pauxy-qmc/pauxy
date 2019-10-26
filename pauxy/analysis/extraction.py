import pandas as pd
import numpy
import json
import h5py
from pauxy.utils.misc import get_from_dict


def extract_data(filename, group, estimator, raw=False):
    fp = get_param(filename, ['propagators', 'free_projection'])
    with h5py.File(filename, 'r') as fh5:
        dsets = list(fh5[group][estimator].keys())
        data = numpy.array([fh5[group][estimator][d][:] for d in dsets])
        if 'rdm' in estimator or raw:
            return data
        else:
            header = fh5[group]['headers'][:]
            header = numpy.array([h.decode('utf-8') for h in header])
            df = pd.DataFrame(data)
            df.columns = header
            if not fp:
                df = df.apply(numpy.real)
            return df

def extract_mixed_estimates(filename, skip=0):
    return extract_data(filename, 'basic', 'energies')[skip:]

def extract_bp_estimates(filename, skip=0):
    return extract_data(filename, 'back_propagated', 'energies')[skip:]

def extract_rdm(filename, est_type='back_propagated', rdm_type='one_rdm'):
    rdmtot = []
    nzero = -1
    one_rdm = extract_data(filename, est_type, rdm_type)
    denom = extract_data(filename, est_type, 'denominator', raw=True)
    fp = get_param(filename, ['propagators','free_projection'])
    if fp:
        return (one_rdm, denom)
    else:
        return one_rdm / denom[:,None,None]

def set_info(frame, md):
    system = md.get('system')
    qmc = md.get('qmc')
    propg = md.get('propagators')
    trial = md.get('trial')
    ncols = len(frame.columns)
    frame['dt'] = qmc.get('dt')
    frame['nwalkers'] = qmc.get('ntot_walkers')
    frame['free_projection'] = propg.get('free_projection')
    beta = qmc.get('beta')
    bp = md['estimators']['estimators'].get('back_prop')
    if bp is not None:
        frame['tau_bp'] = bp['tau_bp']
    if beta is not None:
        frame['beta'] = beta
        mu = system.get('mu')
        if mu is not None:
            frame['mu'] = system.get('mu')
        frame['mu_T'] = trial.get('mu')
        frame['Nav_T'] = trial.get('nav')
    else:
        frame['E_T'] = trial.get('energy')
    if system['name'] == "UEG":
        frame['rs'] = system.get('rs')
        frame['ecut'] = system.get('ecut')
        frame['nup'] = system.get('nup')
        frame['ndown'] = system['ndown']
    elif system['name'] == "Hubbard":
        frame['U'] = system.get('U')
        frame['nx'] = system.get('nx')
        frame['ny'] = system.get('ny')
    elif system['name'] == "Generic":
        ints = system.get('integral_file')
        if ints is not None:
            frame['integrals'] = ints
        chol = system.get('threshold')
        if chol is not None:
            frame['cholesky_treshold'] = chol
        frame['nup'] = system.get('nup')
        frame['ndown'] = system.get('ndown')
        frame['nbasis'] = system.get('nbasis', 0)
    return list(frame.columns[ncols:])

def get_metadata(filename):
    with h5py.File(filename, 'r') as fh5:
        metadata = json.loads(fh5['metadata'][()])
    return metadata

def get_param(filename, param):
    md = get_metadata(filename)
    return get_from_dict(md, param)

def get_sys_param(filename, param):
    return get_param(filename ['system', param])


# TODO : FDM FIX.
# def analysed_itcf(filename, elements, spin, order, kspace):
    # data = h5py.File(filename, 'r')
    # md = json.loads(data['metadata'][:][0])
    # dt = md['qmc']['dt']
    # mode = md['estimators']['estimators']['itcf']['mode']
    # stack_size = md['psi']['stack_size']
    # convert = {'up': 0, 'down': 1, 'greater': 0, 'lesser': 1}
    # if kspace:
        # gf = data['kspace_itcf'][:]
        # gf_err = data['kspace_itcf_err'][:]
    # else:
        # gf = data['real_itcf'][:]
        # gf_err = data['real_itcf_err'][:]
    # tau = stack_size * dt * numpy.arange(0,gf.shape[0])
    # isp = convert[spin]
    # it = convert[order]
    # results = pd.DataFrame()
    # results['tau'] = tau
    # # note that the interpretation of elements necessarily changes if we
    # # didn't store the full green's function.
    # if mode == 'full':
        # name = 'G_'+order+'_spin_'+spin+'_%s%s'%(elements[0],elements[1])
        # results[name] = gf[:,isp,it,elements[0],elements[1]]
        # results[name+'_err'] = gf_err[:,isp,it,elements[0],elements[1]]
    # else:
        # name = 'G_'+order+'_spin_'+spin+'_%s%s'%(elements[0],elements[0])
        # results[name] = gf[:,isp,it,elements[0]]
        # results[name+'_err'] = gf_err[:,isp,it,elements[0]]

    # return results
