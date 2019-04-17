#!/usr/bin/env python

import h5py
import numpy
import sys
from pauxy.analysis.extraction import extract_hdf5

data = extract_hdf5(sys.argv[1])
mixed = data[1]
mixed = mixed.apply(numpy.real)
mixed = mixed[mixed['Weight']>0]
print (mixed.to_string())
