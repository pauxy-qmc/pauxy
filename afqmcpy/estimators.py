import numpy as np

def greens_function(walker, trial):

    gup = np.dot(np.dot(walker.phi[0], walker.ovlp[0]), np.transpose(trial[0]))
    gdown = np.dot(np.dot(walker.phi[1], walker.ovlp[1]), np.transpose(trial[1]))

    return np.array([gup, gdown])

def local_energy(system, walker, trial):

    G = greens_function(walker, trial)

    ke = np.einsum('ij,ji', system.T, G[0]+G[1])
    pe = sum(system.U*G[0][i][i]*G[1][i][i] for i in range(0, system.nbasis))

    return ke + pe
