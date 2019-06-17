import copy
import numpy
from pauxy.utils.io import get_input_value


class QMCOpts(object):
    r"""Input options and certain constants / parameters derived from them.

    Initialised from a dict containing the following options, not all of which
    are required.

    Parameters
    ----------
    method : string
        Which auxiliary field method are we using? Currently only CPMC is
        implemented.
    nwalkers : int
        Number of walkers to propagate in a simulation.
    dt : float
        Timestep.
    nsteps : int
        Total number of Monte Carlo steps to perform.
    nmeasure : int
        Frequency of energy measurements.
    nstblz : int
        Frequency of Gram-Schmidt orthogonalisation steps.
    npop_control : int
        Frequency of population control.
    temp : float
        Temperature. Currently not used.
    nequilibrate : int
        Number of steps used for equilibration phase. Only used to fix local
        energy bound when using phaseless approximation.
    importance_sampling : boolean
        Are we using importance sampling. Default True.
    hubbard_statonovich : string
        Which hubbard stratonovich transformation are we using. Currently the
        options are:

        - discrete : Use the discrete Hirsch spin transformation.
        - opt_continuous : Use the continuous transformation for the Hubbard
          model.
        - generic : Use the generic transformation. To be used with Generic
          system class.

    ffts : boolean
        Use FFTS to diagonalise the kinetic energy propagator? Default False.
        This may speed things up for larger lattices.

    Attributes
    ----------
    cplx : boolean
        Do we require complex wavefunctions?
    mf_shift : float
        Mean field shift for continuous Hubbard-Stratonovich transformation.
    iut_fac : complex float
        Stores i*(U*dt)**0.5 for continuous Hubbard-Stratonovich transformation.
    ut_fac : float
        Stores (U*dt) for continuous Hubbard-Stratonovich transformation.
    mf_nsq : float
        Stores M * mf_shift for continuous Hubbard-Stratonovich transformation.
    local_energy_bound : float
        Energy pound for continuous Hubbard-Stratonovich transformation.
    mean_local_energy : float
        Estimate for mean energy for continuous Hubbard-Stratonovich transformation.
    """

    def __init__(self, inputs, system, verbose=False):
        self.nwalkers = get_input_value(inputs, 'num_walkers',
                                        default=10, alias=['nwalkers'],
                                        verbose=verbose)
        self.dt = get_input_value(inputs, 'timestep', default=0.005,
                                  alias=['dt'], verbose=verbose)
        self.nsteps = get_input_value(inputs, 'num_steps',
                                      default=10, alias=['nsteps'],
                                      verbose=verbose)
        self.nmeasure = get_input_value(inputs, 'num_blocks',
                                        default=1000,
                                        alias=['nmeasure', 'blocks'],
                                        verbose=verbose)
        self.nstblz = get_input_value(inputs, 'stabilise_freq',
                                      default=10,
                                      alias=['nstabilise', 'reortho'],
                                      verbose=verbose)
        self.npop_control = get_input_value(inputs, 'pop_control_freq',
                                            default=10,
                                            alias=['npop_control'],
                                            verbose=verbose)
        self.nupdate_shift = get_input_value(inputs, 'update_shift_freq',
                                             default=10,
                                             alias=['nupdate_shift'],
                                             verbose=verbose)
        self.nequilibrate = get_input_value(inputs, 'num_equilibrate_steps',
                                            default=int(1.0/self.dt),
                                            alias=['nequilibrate'],
                                            verbose=verbose)
        self.beta = get_input_value(inputs, 'beta', default=None,
                                    verbose=verbose)
        self.beta_reduced = get_input_value(inputs, 'beta_reduced',
                                            default=None,
                                            verbose=verbose)
        self.rng_seed = get_input_value(inputs, 'rng_seed',
                                        default=None,
                                        alias=['random_seed', 'seed'],
                                        verbose=verbose)
