'''Various useful routines maybe not appropriate elsewhere'''

import numpy

def sherman_morrison(Ainv, u, vt):
    '''Sherman-Morrison update of a matrix inverse

Parameters
----------
Ainv : numpy.ndarray
    Matrix inverse of A to be updated.
u : numpy.array
    column vector
vt : numpy.array
    transpose of row vector

Returns
-------
Ainv : numpy.ndarray
    Update inverse: :math:`(A + u x vT)^{-1}`
'''

    return (
        Ainv - (Ainv.dot(numpy.outer(u,vt)).dot(Ainv))/(1.0+vt.dot(Ainv).dot(u))
    )
