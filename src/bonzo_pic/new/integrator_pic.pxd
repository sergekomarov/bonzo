# -*- coding: utf-8 -*-

from bnz.defs_cy cimport *
from bnz.coord.grid cimport *


#==========================================================================

# Integrator class.

cdef class BnzIntegrator:

  # Attributes.

  # Courant number
  cdef real cour

  # number of passes of current filter
  cdef int Nfilt

  # Functions.

  cdef void init_data(self)
