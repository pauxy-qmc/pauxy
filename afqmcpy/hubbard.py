class Hubbard:

    def __init__(self, t, U, nup, ndown, nx, ny):

        self.t = t
        self.U = U
        self.nup = nup
        self.ndown = ndown
        self.ne = nup + ndown
        self.nx = nx
        self.ny = ny
        if ny > 1:
            self.nbasis = nx + ny
        else:
            self.nbasis = nx
