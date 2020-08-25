from pauxy.dqmc_updates.hubbard import DiscreteHubbard
from pauxy.utils.io import get_input_value

def get_update_driver(system, dt, nslice, options={}, verbose=False):
    low_rank = get_input_value(options, 'low_rank', default=False,
                                alias=['low_rank_trick'],
                                verbose=verbose)
    stack_size = get_input_value(options, 'stack_size', default=1,
                                 alias=['stack'],
                                 verbose=verbose)
    if system.name == "Hubbard":
        dynamic_force = get_input_value(options, 'dynamic_force', default=False,
                                        alias=['force_bias'],
                                        verbose=verbose)
        charge_decomp = get_input_value(options, 'charge_decomp', default=False,
                                        alias=['charge', 'charge_decomposition'],
                                        verbose=verbose)
        single_site = get_input_value(options, 'single_site', default=True,
                                      alias=['single_site_update'],
                                      verbose=verbose)
        return DiscreteHubbard(system, dt, nslice,
                               stack_size=stack_size,
                               dynamic_force=dynamic_force,
                               low_rank=low_rank,
                               charge_decomp=charge_decomp,
                               single_site=single_site)
