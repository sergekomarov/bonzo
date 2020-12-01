# -*- coding: utf-8 -*-

from bnz.defs_cy cimport *
from bnz.coord.grid cimport *


# Particle boundary conditions.

cdef class BnzSim

# particle BC function pointer
ctypedef void (*PrtBcFunc)(BnzSim)

cdef class ParticleBC:

  # BC flags
  cdef int bc_flags[3][2]

  # BC function pointers
  cdef PrtBcFunc prt_bc_funcs[3][2]

  # BC buffers
  IF MPI:
    cdef:
      real2d sendbuf, recvbuf
      ints recvbuf_size, sendbuf_size

cdef void apply_bc_prt(BnzSim)
