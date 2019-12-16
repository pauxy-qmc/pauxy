import cmath
import copy
import numpy
import math
import scipy.linalg

class HarmonicOscillator(object):
    def __init__(self, w, order, shift):
        self.w = w
        self.order = order
        # self.norm = (self.w / math.pi) ** 0.25 # not necessary but we just include...
        self.norm = 1.0 # not necessary but we just include...
        self.xavg = shift
        # self.eshift = self.xavg**2 * self.w**2 / 2.0
#-------------------------
    def value(self,X): # X : lattice configuration
        result = numpy.prod(self.norm * numpy.exp(- self.w / 2.0 * (X-self.xavg) * (X-self.xavg)))
        return result 
#-------------------------
    def gradient(self,X): # grad / value
        # grad = (-self.w * (X-self.xavg)) * self.value(X)
        grad = -self.w * (X-self.xavg)
        return grad
#-------------------------
    def laplacian(self,X): # laplacian / value
        # lap = self.w * self.w * (X-self.xavg) * (X-self.xavg) * self.value(X) - self.w * self.value(X)
        lap = self.w * self.w * (X-self.xavg) * (X-self.xavg) - self.w 
        return lap
#-------------------------
    def local_energy(self, X):

        nsites = X.shape[0]

        ke   = - 0.5 * numpy.sum(self.laplacian(X))
        pot  = 0.5 * self.w * self.w * numpy.sum(X * X)

        eloc = ke+pot - 0.5 * self.w * nsites # No zero-point energy

        return eloc
