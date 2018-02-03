import pandas as pd
import numpy
import json
import h5py
import pauxy.analysis.utils

def extract_hdf5_data_sets(files):

    data =  [extract_hdf5(f) for f in files]

    return data

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

def extract_test_data_hdf5(filename):
    (md, data, bp, itcf, kitcf, mrdm, bprdm) = extract_hdf5(filename)
    if (bp is not None):
        data = bp
    elif mrdm is not None:
        data = mrdm
    elif bprdm is not None:
        data = pmrdm
    elif itcf is not None:
        data = pd.DataFrame(itcf.reshape(itcf.shape[0], -1))
        if kitcf is not None:
            data.append(kitcf.reshape(kitcf.shape[0], -1))
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

def correlation_function(filename, name, iy):
    data = h5py.File(filename, 'r')
    md = json.loads(data['metadata'][:][0])
    nx = int(md['system']['nx'])
    ny = int(md['system']['ny'])
    output = data[name+'/correlation'][:]
    columns = ['hole', 'hole_err', 'spin', 'spin_err']
    h, herr = analysis.utils.get_strip(output[0], output[1], iy[0], nx, ny)
    s, serr = analysis.utils.get_strip(output[2], output[3],
                                       iy[0], nx, ny, True)
    results = pd.DataFrame({'hole': h, 'hole_err': herr,
                            'spin': s, 'spin_err': serr},
                           columns=columns)

    return results

def analysed_energies(filename, name):
    data = h5py.File(filename, 'r')
    md = json.loads(data['metadata'][:][0])
    dt = md['qmc']['dt']
    output = data[name+'/estimates'][:]
    columns = data[name+'/headers'][:]
    results = pd.DataFrame(output, columns=columns)

    return results
