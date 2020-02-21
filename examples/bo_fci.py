import matplotlib.pyplot as plt
import itertools
from pauxy.systems.hubbard import Hubbard
from pauxy.systems.hubbard_holstein import HubbardHolstein
from pauxy.estimators.ci import simple_fci_bose_fermi, simple_fci, simple_lang_firsov, simple_lang_firsov_unitary
import scipy
import numpy
import scipy.sparse.linalg
import pandas as pd

l = 1.0
w0 = 1.0

x0 = 2.82842712
ndata = 100
X1 = numpy.linspace(start=x0-5.5,stop=x0+5.5, num=ndata)
X2 = numpy.linspace(start=x0-5.5,stop=x0+5.5, num=ndata)

df = pd.DataFrame()

energies = numpy.zeros((len(X1), len(X2)),dtype=numpy.float64)

for i1, x1 in enumerate(X1):
    for i2, x2 in enumerate(X2):
        options = {
        "name": "HubbardHolstein",
        "nup": 1,
        "ndown": 0,
        "nx": 2,
        "ny": 1,
        "U": 0.0,
        "t": 1.0,
        "w0": w0,
        "lambda": l,
        "xpbc" :True
        }
        system = HubbardHolstein (options, verbose=True)
        system0 = Hubbard (options, verbose=True)
        const = system.g * numpy.sqrt(2.0 * system.m * system.w0)
        Heph = - numpy.diag(numpy.array([x1, x2]) * const)
        Ttmp = system0.T.copy()
        system0.T[0] = Ttmp[0] + Heph
        system0.T[1] = Ttmp[1] + Heph

        (eig, evec), H = simple_fci(system0, hamil=True)

        Eph = 0.5 * system.m * system.w0 ** 2 * (x1**2 + x2**2)
        # print(eig[0])
        energies[i1,i2] = eig[0]+Eph
        # df0 = pd.DataFrame({"X1":x1, "X2":x2, 
        #                  "E":eig[0]+Eph})
        # df = df.append(df0)

# print(df.to_string(index=False))

import seaborn as sns 
# print(energies)
# print(X1)
# print(X2)
X1s = ["%3.2f"%x1 for x1 in X1]
X2s = ["%3.2f"%x2 for x2 in X2]

df = pd.DataFrame(energies, index=X1s, columns=X2s)
# print(df.to_string())
sns.heatmap(df, annot=False)
idx = numpy.argmin(energies)
print(energies.shape)
e = energies.ravel()
i1 = int(idx/ndata)
i2 = int(idx - i1 * ndata)
# print(i1,i2)
print(idx, e[idx], energies[i1,i2], X1[i1], X2[i2])
# exit()
# print(numpy.min(energies), energies[idx], idx)

plt.show()

# df = pd.read_clipboard()
# table = df.pivot('X1', 'X2', 'E')
# ax = sns.heatmap(table)
# ax = sns.heatmap(df)
# ax.invert_yaxis()
# print(table)
# plt.show()