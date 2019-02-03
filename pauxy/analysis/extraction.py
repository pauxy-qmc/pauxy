import pandas as pd
import numpy
import json
import h5py

def extract_hdf5_data_sets(files):
    data = []
    for f in files:
        try:
            data.append(extract_hdf5(f))
        except OSError:
            print("Error reading %f."%f)

    return data

def extract_mixed_estimates(filename, skip=0):
    data = h5py.File(filename, 'r')
    metadata = json.loads(data['metadata'][:][0])
    basic = data['mixed_estimates/energies'][:]
    headers = data['mixed_estimates/headers'][:]
    basic = pd.DataFrame(basic)
    basic.columns = headers
    nzero = numpy.nonzero(basic['Weight'].values)[0][-1]
    return (basic[skip:nzero])

def extract_rdm(filename, skip, est_type='back_propagated'):
    data = h5py.File(filename, 'r')
    metadata = json.loads(data['metadata'][:][0])
    bpe = data[est_type+'_estimates/energies'][:]
    headers = data[est_type+'_estimates/headers'][:]
    est_data = pd.DataFrame(bpe)
    est_data.columns = headers
    nzero = numpy.nonzero(est_data['E'].values)[0][-1]
    try:
        rdm = data[est_type+'_estimates/single_particle_greens_function'][:]
        weights = est_data['Weight'].values.real
        # if len(rdm.shape) == 3:
            # # GHF format
            # w = weights[skip:nzero,None,None]
        # else:
            # # UHF format
            # w = weights[skip:nzero,None,None,None]
        return rdm[skip:nzero]
    except KeyError:
        return None

def extract_hdf5(filename):
    data = h5py.File(filename, 'r')
    metadata = json.loads(data['metadata'][:][0])
    estimates = metadata.get('estimators').get('estimators')
    basic = data['mixed_estimates/energies'][:]
    headers = data['mixed_estimates/headers'][:]
    basic = pd.DataFrame(basic)
    basic.columns = headers
    if estimates is not None:
        if estimates.get('mixed').get('rdm'):
            mixed_rdm = data['mixed_estimates/single_particle_greens_function'][:]
        else:
            mixed_rdm = None
        bp = estimates.get('back_prop')
        if bp is not None:
            bpe = data['back_propagated_estimates/energies'][:]
            headers = data['back_propagated_estimates/headers'][:]
            bp_data = pd.DataFrame(bpe)
            bp_data.columns = headers
            if bp['rdm']:
                bp_rdm = data['back_propagated_estimates/single_particle_greens_function'][:]
            else:
                bp_rdm = None
        else:
            bp_data = None
            bp_rdm = None
        itcf_info = estimates.get('itcf')
        if itcf_info is not None:
            itcf = data['single_particle_greens_function/real_space'][:]
            if itcf_info['kspace']:
                kspace_itcf = data['single_particle_greens_function/k_space'][:]
            else:
                kspace_itcf = None
        else:
            itcf = None
            kspace_itcf = None

    return (metadata, basic, bp_data, itcf, kspace_itcf, mixed_rdm, bp_rdm)

def extract_hdf5_simple(filename):
    data = h5py.File(filename, 'r')
    metadata = json.loads(data['metadata'][:][0])
    estimates = metadata.get('estimators').get('estimators')
    basic = data['mixed_estimates/energies'][:]
    headers = data['mixed_estimates/headers'][:]
    basic = pd.DataFrame(basic)
    basic.columns = headers
    return (metadata, basic)

def extract_test_data_hdf5(filename):
    (md, data, bp, itcf, kitcf, mrdm, bprdm) = extract_hdf5(filename)
    if (bp is not None):
        data = bp
    if mrdm is not None:
        mrdm = mrdm[abs(mrdm) > 1e-10].flatten()
        data = pd.DataFrame({'G', mrdm})
    if bprdm is not None:
        bprdm = bprdm[abs(bprdm) > 1e-10].flatten()
        data = pd.DataFrame({'G': bprdm})
    if itcf is not None:
        itcf = itcf[abs(itcf) > 1e-10].flatten()
        data = pd.DataFrame(itcf)
    return data.apply(numpy.real)[::8].to_dict(orient='list')

def analysed_itcf(filename, elements, spin, order, kspace):
    data = h5py.File(filename, 'r')
    md = json.loads(data['metadata'][:][0])
    dt = md['qmc']['dt']
    tmax = md['estimators']['estimators']['itcf']['tmax']
    tau = numpy.arange(0, tmax+1e-8, dt)
    mode = md['estimators']['estimators']['itcf']['mode']
    convert = {'up': 0, 'down': 1, 'greater': 0, 'lesser': 1}
    if kspace:
        gf = data['kspace_itcf'][:]
        gf_err = data['kspace_itcf_err'][:]
    else:
        gf = data['real_itcf'][:]
        gf_err = data['real_itcf_err'][:]
    isp = convert[spin]
    it = convert[order]
    results = pd.DataFrame()
    results['tau'] = tau
    # note that the interpretation of elements necessarily changes if we
    # didn't store the full green's function.
    if mode == 'full':
        name = 'G_'+order+'_spin_'+spin+'_%s%s'%(elements[0],elements[1])
        results[name] = gf[:,isp,it,elements[0],elements[1]]
        results[name+'_err'] = gf_err[:,isp,it,elements[0],elements[1]]
    else:
        name = 'G_'+order+'_spin_'+spin+'_%s%s'%(elements[0],elements[0])
        results[name] = gf[:,isp,it,elements[0]]
        results[name+'_err'] = gf_err[:,isp,it,elements[0]]

    return results

def analysed_energies(filename, name):
    data = h5py.File(filename, 'r')
    md = json.loads(data['metadata'][:][0])
    dt = md['qmc']['dt']
    output = data[name+'/estimates'][:]
    columns = data[name+'/headers'][:]
    results = pd.DataFrame(output, columns=columns)

    return results

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
    if beta is not None:
        frame['beta'] = beta
        frame['mu'] = trial.get('mu')
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
        frame['name'] = system.get('integral_file').split('.')[2].split('/')[1]
        frame['nfv'] = system.get('nfv')
        frame['nup'] = system.get('nup')
        frame['ndown'] = system.get('ndown')
        frame['ncore'] = system.get('ncore')
        frame['nbasis'] = system.get('nbasis')
        frame['cholesky_treshold'] = system.get('threshold')
    return list(frame.columns[ncols:])

def get_metadata(filename):
    with h5py.File(filename, 'r') as fh5:
        metadata = json.loads(fh5['metadata'][:][0])
    return metadata
