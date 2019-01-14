from pauxy.propagation.continuous import Continuous
# TODO: Fix this
from pauxy.propagation.utils import get_discrete_propagator

def get_propagator_driver(options, qmc, system, trial, verbose=False):
    hs = options['hubbard_stratonovich']
    if 'discrete' in hs:
        return get_discrete_propagator(options, qmc, system, trial, verbose)
    else:
        return Continuous(options, qmc, system, trial, verbose)
