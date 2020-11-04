# -*- coding: utf-8 -*-

import numpy as np
cimport numpy as np
from cython.parallel import prange, parallel, threadid

from libc.stdio cimport printf
from libc.stdlib cimport malloc, calloc, free

# ADD LAPLACIANS

# ==============================================================================

cdef void cons2prim_3(real4d W, real4d U, ints lims[6], real gam) nogil:

  cdef:
    ints j,k,n
    real **U1
    real **W1

  # with nogil, parallel(num_threads=OMP_NT):

  U1 = <real**>calloc(NWAVES, sizeof(real*))
  W1 = <real**>calloc(NWAVES, sizeof(real*))

  for k in range(lims[4],lims[5]+1):
    for j in range(lims[2],lims[3]+1):

      for n in range(NWAVES):
        U1[n] = &(U[n,k,j,0])
        W1[n] = &(W[n,k,j,0])

      cons2prim_1(W1, U1, lims[0], lims[1], gam)

  free(U1)
  free(W1)


#===============================================================================

cdef void prim2cons_3(real4d U, real4d W, ints lims[6], real gam) nogil:

  cdef:
    ints j,k,n
    real **U1
    real **W1

  # with nogil, parallel(num_threads=OMP_NT):

  U1 = <real**>calloc(NWAVES, sizeof(real*))
  W1 = <real**>calloc(NWAVES, sizeof(real*))

  for k in range(lims[4],lims[5]+1):
    for j in range(lims[2],lims[3]+1):

      for n in range(NWAVES):
        U1[n] = &(U[n,k,j,0])
        W1[n] = &(W[n,k,j,0])

      prim2cons_1(U1, W1, lims[0], lims[1], gam)

  free(U1)
  free(W1)
