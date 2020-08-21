import numpy
import scipy.linalg

def get_aux_fields(system, dt, charge_decomp):
    if charge_decomp:
        gamma = numpy.arccosh(numpy.exp(-0.5*dt*system.U+0j))
        auxf = numpy.array([[numpy.exp(gamma), numpy.exp(gamma)],
                            [numpy.exp(-gamma), numpy.exp(-gamma)]])
        # if current field is +1 want to flip to -1, then delta[1] gives flip factor
        delta = numpy.array([[numpy.exp(2*gamma), numpy.exp(2*gamma)],
                             [numpy.exp(-2*gamma), numpy.exp(-2*gamma)]])
        delta -= 1.0
        # e^{-gamma x}
        aux_wfac = numpy.exp(0.5*dt*system.U) * numpy.array([numpy.exp(-2*gamma),
                                                             numpy.exp(2*gamma)])
        aux_wfac_ = numpy.exp(0.5*dt*system.U) * numpy.array([numpy.exp(-gamma),
                                                             numpy.exp(gamma)])
    else:
        gamma = numpy.arccosh(numpy.exp(0.5*dt*system.U))
        auxf = numpy.array([[numpy.exp(gamma), numpy.exp(-gamma)],
                            [numpy.exp(-gamma), numpy.exp(gamma)]])
        delta = numpy.array([[numpy.exp(2*gamma), numpy.exp(-2*gamma)],
                             [numpy.exp(-2*gamma), numpy.exp(2*gamma)]])
        delta -= 1.0
        aux_wfac = numpy.array([1.0, 1.0])

    auxf = auxf * numpy.exp(-0.5*dt*system.U)
    return gamma, auxf, delta, aux_wfac

def get_one_body(system, dt):
    I = numpy.eye(system.nbasis, dtype=numpy.complex128)
    BH1 = numpy.array([scipy.linalg.expm(-dt*(system.H1[0]-system.mu*I)),
                       scipy.linalg.expm(-dt*(system.H1[1]-system.mu*I))],
                       dtype=numpy.complex128)
    BH1inv = numpy.array([scipy.linalg.expm(dt*(system.H1[0]-system.mu*I)),
                          scipy.linalg.expm(dt*(system.H1[1]-system.mu*I))],
                          dtype=numpy.complex128)
    return BH1, BH1inv

