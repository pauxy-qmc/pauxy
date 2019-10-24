Simple PYSCF Workflow
=====================

In this example we will go through the basic steps necessary to generate AFQMC input from
a pyscf scf calculation for a simple H10 chain.

The pyscf scf script is given below (scf.py in the current directory):

.. code-block:: python

    from pyscf import gto, scf, cc

    atom = gto.M(atom=[('H', 1.6*i, 0, 0) for i in range(0,10)],
                 basis='sto-6g',
                 verbose=4,
                 unit='Bohr')
    mf = scf.UHF(atom)
    mf.chkfile = 'scf.chk'
    mf.kernel()

The important point is to specify the `chkfile` option.

Once the scf converges we need to generate the wavefunction and integrals using the
`pyscf_to_pauxy.py` script found in `pauxy/tools/pyscf`.

.. code-block:: bash

    python /path/to/pauxy/tools/pyscf/pyscf_to_pauxy.py -i 'scf.chk'

You should find a file called `afqmc.h5` and pauxy input file `input.json` created from
information in `afqmc.h5`.

.. code-block:: json

    {
        "system": {
            "name": "Generic",
            "nup": 5,
            "ndown": 5,
            "integrals": "afqmc.h5"
        },
        "qmc": {
            "dt": 0.005,
            "nsteps": 5000,
            "nmeasure": 10,
            "nwalkers": 30,
            "pop_control": 1
        },
        "trial": {
            "filename": "afqmc.h5"
        }
    }

The input options should be carefully updated, with particular attention paid to the
timestep `dt` and the total number of walkers `nwalkers`.

Run the AFQMC calculation by:

.. code-block:: bash

    python /path/to/pauxy/bin/pauxy.py input.json

See the documentation for more input options and the converter:

.. code-block:: bash

    python /path/to/pauxy/tools/pyscf/pyscf_to_pauxy.py --help
