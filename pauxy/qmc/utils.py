from pauxy.propagation.continuous import Continuous
from pauxy.propagation.discrete import Discrete 

def get_propagator_driver(options, qmc, system, trial, verbose=False):
    hs = options['hubbard_stratonovich']
    if 'discrete' in hs:
        return Discrete(options, qmc, system, trial, verbose)
    else:
        return Continuous(options, qmc, system, trial, verbose)
