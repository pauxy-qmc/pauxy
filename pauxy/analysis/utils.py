import pauxy.hubbard

def get_strip(cfunc, cfunc_err, ix, nx, ny, stag=False):
    iy = [i for i in range(ny)]
    idx = [pauxy.hubbard.encode_basis(ix,i,nx) for i in iy]
    if stag:
        c = [((-1)**(ix+i))*cfunc[ib] for (i, ib) in zip(iy,idx)]
    else:
        c = [cfunc[ib] for ib in idx]
    cerr = [cfunc_err[ib] for ib in idx]
    return c, cerr
