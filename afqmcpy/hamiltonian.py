import numpy as np

def kinetic(system):
    '''Kinetic part of the Hamiltonian

'''

    t = np.zeros((system.nbasis, system.nbasis))

    for i in range(0, system.nbasis):
        for j in range(0, system.nbasis):
            xy1 = decode_basis(system, i)
            xy2 = decode_basis(system, j)
            print i, j, xy1, xy2
            if (np.dot(xy1-xy2,xy1-xy2) == 1):
                t[i, j] = -system.t
    return t

def decode_basis(system, i):
    return np.array([i%system.nx, i/system.nx])
