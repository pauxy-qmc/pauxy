import h5py
import numpy
import scipy.linalg
import json
import time
import sys
from pauxy.systems.hubbard import Hubbard
from pauxy.estimators.utils import H5EstimatorHelper
from pauxy.trial_density_matrices.onebody import OneBody
from pauxy.walkers.stack import PropagatorStack
from pauxy.estimators.thermal import one_rdm_from_G
from pauxy.estimators.mixed import local_energy
from pauxy.utils.io import format_fixed_width_strings, format_fixed_width_floats
from pauxy.utils.misc import get_numeric_names, serialise
from pauxy.analysis.blocking import reblock_local_energy

system = Hubbard({'nx': 6, 'ny': 1, 'U': 4.0, 'mu': 2, 'nup': 3, 'ndown': 3})

beta = float(sys.argv[1])
charge_decomp = int(sys.argv[2])
isym =  int(sys.argv[3])
dt =  float(sys.argv[4])

nslice = int(round(beta/dt))
nsteps = 10
blocks = 1000
stack_size = min(nslice, 10)
recomp_freq = min(nslice, 10)
charge_decomp = bool(charge_decomp)

from utils import get_aux_fields
gamma, auxf, delta, aux_wfac = get_aux_fields(system, dt, charge_decomp)

stack = PropagatorStack(stack_size, nslice, system.nbasis, numpy.complex128, lowrank=False)
# stack_a = PropagatorStack(1, nslice, system.nbasis, numpy.complex128, lowrank=False)
fields = numpy.random.randint(0, 2, nslice*system.nbasis).reshape(nslice, system.nbasis)

from utils import get_one_body
BH1, BH1inv = get_one_body(system, dt)

from greens import (
        recompute_greens_function,
        propagate_greens_function,
        update_greens_function
        )
# G = recompute_greens_function(fields, stack, auxf, BH1, time_slice=nslice)
# P = one_rdm_from_G(G)
# e, t, v = local_energy(system, G)

filename = '{:s}.{:d}.h5'.format(('charge' if charge_decomp else 'spin'), isym)
metadata = {
        'system': serialise(system),
        'qmc': {
            'dt': dt,
            'blocks': blocks,
            'nsteps': nsteps,
            'beta': beta
            },
        'propagators': {
            'free_projection': True
            },
        'estimators': {}
        }
with h5py.File(filename, 'w') as fh5:
    fh5['basic/headers'] = numpy.array(['Iteration', 'ETotal', 'Nav', 'Sign']).astype('S')
    fh5['metadata'] = json.dumps(metadata, sort_keys=False, indent=4)
output = H5EstimatorHelper(filename, 'basic')

print(format_fixed_width_strings(['Iteration', 'ETotal', 'ETotal_imag', 'Nav',
    'Nav_imag', 'Sign', 'Time']))
sign = 1.0
nsites = system.nbasis

G = recompute_greens_function(fields, stack, auxf, BH1, time_slice=nslice,
                              from_scratch=True)
for block in range(blocks):
    etot_block = 0.0 + 0.0j
    nav_block = 0.0 + 0.0j
    sgn_block = 0.0 + 0.0j
    tblock = time.time()
    for step in range(nsteps):
        etot_slice = 0.0 + 0.0j
        nav_slice = 0.0 + 0.0j
        sgn_slice = 0.0 + 0.0j
        tstep = time.time()
        for islice in range(nslice):
            sgn_step = 0.0
            tslice = time.time()
            G = propagate_greens_function(G, fields[islice],
                                          BH1inv, BH1, auxf)
            for ibasis in range(system.nbasis):
                # Propose spin flip
                field = fields[islice,ibasis]
                if field == 0:
                    xi = 1
                else:
                    xi = 0
                ratio_up = 1.0 + (1.0-G[0,ibasis,ibasis])*delta[xi,0]
                ratio_dn = 1.0 + (1.0-G[1,ibasis,ibasis])*delta[xi,1]
                ratio = ratio_up*ratio_dn*aux_wfac[xi]
                sign *= numpy.sign(ratio)
                P = abs(ratio) / (1.0 + abs(ratio))
                r = numpy.random.random()
                if r < P:
                    # accept move
                    fields[islice,ibasis] = xi
                    # update equal time greens function
                    start = time.time()
                    update_greens_function(G, ibasis, xi, delta)
                    # update_greens_function(G_a, ibasis, xi, delta)
                    tupdate_gf = time.time() - start
            tslice = time.time() - tslice
            # Propagate GF
            P = one_rdm_from_G(G)
            e, t, v = local_energy(system, P)
            etot_slice += sign * e
            nav = P[0].trace() + P[1].trace()
            nav_slice += sign * nav
            sgn_slice += sign
            # G(t+1)
            if islice % recomp_freq == recomp_freq-1 and islice != 0:
                start = time.time()
                G = recompute_greens_function(fields, stack, auxf, BH1,
                                              time_slice=islice)
                tre_gf = time.time() - start
        tstep = time.time() - tstep
        etot_block += etot_slice / nslice
        # print(etot_block)
        nav_block += nav_slice / nslice
        sgn_block += sgn_slice / nslice
    output.push([block,etot_block/nsteps,
                 nav_block/(nsites*nsteps),
                 sgn_block/nsteps], 'energies')
    tblock = time.time() - tblock
    output.increment()
    outd = [block,
            etot_block.real/nsteps,
            etot_block.imag/nsteps,
            nav_block.real/(nsites*nsteps),
            nav_block.imag/(nsites*nsteps),
            sgn_block.real/nsteps,
            tblock]
    print(format_fixed_width_floats(outd))
