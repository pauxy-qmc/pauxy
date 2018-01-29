import numpy
import copy

class QMCOpts:
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

    def __init__(self, inputs, system):
        self.method = inputs.get('method', 'CPMC')
        self.nwalkers = inputs.get('nwalkers', None)
        self.dt = inputs.get('dt', None)
        self.nsteps = inputs.get('nsteps', None)
        self.nmeasure = inputs.get('nmeasure', 10)
        self.nstblz = inputs.get('nstabilise', 10)
        self.npop_control = inputs.get('npop_control', 10)
        self.nupdate_shift = inputs.get('nupdate_shift', 10)
        self.temp = inputs.get('temperature', None)
        self.nequilibrate = inputs.get('nequilibrate', int(1.0/self.dt))
        self.importance_sampling = inputs.get('importance_sampling', True)
        if self.importance_sampling:
            self.constraint = 'constrained'
        else:
            self.constraint = 'free'
        self.hubbard_stratonovich = inputs.get('hubbard_stratonovich',
                                                'discrete')
        self.ffts = inputs.get('kinetic_kspace', False)
        self.cplx = ('continuous' in self.hubbard_stratonovich
                     or system.ktwist.all() != None)
        self.exp_nmax = inputs.get('expansion_order', 6)
