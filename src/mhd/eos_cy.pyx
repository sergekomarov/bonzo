# -*- coding: utf-8 -*-

import numpy as np
cimport numpy as np
from cython.parallel import prange, parallel, threadid

from libc.stdio cimport printf
from libc.stdlib cimport malloc, calloc, free

# ADD LAPLACIANS

# ==============================================================================

cdef void cons2prim_3(real4d w, real4d u, int lims[6], real gam) nogil:

  cdef:
    int j,k,n
    real **u1
    real **w1

  with nogil, parallel(num_threads=OMP_NT):
    
    u1 = <real**>calloc(NMODE, sizeof(real*))
    w1 = <real**>calloc(NMODE, sizeof(real*))

    for k in prange(lims[4],lims[5]+1, schedule='dynamic'):
      for j in range(lims[2],lims[3]+1):

        for n in range(NMODE):
          u1[n] = &(u[n,k,j,0])
          w1[n] = &(w[n,k,j,0])

        cons2prim_1(w1, u1, lims[0], lims[1], gam)

    free(u1)
    free(w1)


#===============================================================================

cdef void prim2cons_3(real4d u, real4d w, int lims[6], real gam) nogil:

  cdef:
    int j,k,n
    real **u1
    real **w1

  with nogil, parallel(num_threads=OMP_NT):

    u1 = <real**>calloc(NMODE, sizeof(real*))
    w1 = <real**>calloc(NMODE, sizeof(real*))

    for k in prange(lims[4],lims[5]+1, schedule='dynamic'):
      for j in range(lims[2],lims[3]+1):

        for n in range(NMODE):
          u1[n] = &(u[n,k,j,0])
          w1[n] = &(w[n,k,j,0])

        prim2cons_1(u1, w1, lims[0], lims[1], gam)

    free(u1)
    free(w1)
