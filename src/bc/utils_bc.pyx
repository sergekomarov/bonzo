# -*- coding: utf-8 -*-

import numpy as np
cimport numpy as np
from cython.parallel import prange, parallel

from libc.math cimport sqrt,floor,ceil,log,exp,sin,cos,pow,fabs,fmin,fmax


#----------------------------------------------------------------

cdef inline void copy_layer_x(real3d A,
                       ints ib, ints ib0, int width,
                       ints Ntot[3]) nogil:

  cdef ints j,k,g

  for k in range(Ntot[2]):
    for j in range(Ntot[1]):
      for g in range(width):

        A[k,j,ib+g] = A[k,j,ib0+g]

cdef inline void copy_layer_y(real3d A,
                       ints js, ints js0, int width,
                       ints Ntot[3]) nogil:

  cdef ints i,k,g

  for k in range(Ntot[2]):
    for g in range(width):
      for i in range(Ntot[0]):

        A[k,js+g,i] = A[k,js0+g,i]

cdef inline void copy_layer_z(real3d A,
                       ints ks, ints ks0, int width,
                       ints Ntot[3]) nogil:

  cdef ints i,j,g

  for g in range(width):
    for j in range(Ntot[1]):
      for i in range(Ntot[0]):

        A[ks+g,j,i] = A[ks0+g,j,i]


# ===========================================================================

cdef inline void copy_reflect_layer_x(real3d A, int sgn,
                       ints ib, ints ib0, int width,
                       ints Ntot[3]) nogil:

  cdef:
    ints j,k,g
    ints ib1 = ib+width-1

  for k in range(Ntot[2]):
    for j in range(Ntot[1]):
      for g in range(width):

        A[k,j,ib1-g] = sgn * A[k,j,ib0+g]

cdef inline void copy_reflect_layer_y(real3d A, int sgn,
                       ints js, ints js0, int width,
                       ints Ntot[3]) nogil:

  cdef:
    ints i,k,g
    ints js1 = js+width-1

  for k in range(Ntot[2]):
    for g in range(width):
      for i in range(Ntot[0]):

        A[k,js1-g,i] = sgn * A[k,js0+g,i]

cdef inline void copy_reflect_layer_z(real3d A, int sgn,
                       ints ks, ints ks0, int width,
                       ints Ntot[3]) nogil:

  cdef:
    ints i,j,g
    ints ks1 = ks+width-1

  for g in range(width):
    for j in range(Ntot[1]):
      for i in range(Ntot[0]):

        A[ks1-g,j,i] = sgn * A[ks0+g,j,i]


# ===========================================================================


cdef inline void copy_add_layer_x(real3d A,
                       ints ib, ints ib0, int width,
                       ints Ntot[3]) nogil:

  cdef ints j,k,g

  for k in range(Ntot[2]):
    for j in range(Ntot[1]):
      for g in range(width):
        A[k,j,ib+g] = A[k,j,ib+g] + A[k,j,ib0+g]

cdef inline void copy_add_layer_y(real3d A,
                       ints js, ints js0, int width,
                       ints Ntot[3]) nogil:

  cdef ints i,g,k

  for k in range(Ntot[2]):
    for g in range(width):
      for i in range(Ntot[0]):
        A[k,js+g,i] = A[k,js+g,i] + A[k,js0+g,i]

cdef inline void copy_add_layer_z(real3d A,
                       ints ks, ints ks0, int width,
                       ints Ntot[3]) nogil:

  cdef ints i,j,k,g

  for g in range(width):
    for j in range(Ntot[1]):
      for i in range(Ntot[0]):

        A[ks+g,j,i] = A[ks+g,j,i] + A[ks0+g,j,i]



# ===========================================================================

cdef inline void copy_add_reflect_layer_x(real3d A, int sgn,
                       ints ib, ints ib0, int width, ints Ntot[3]) nogil:

  cdef:
    ints i,j,k,g
    ints ib1 = ib+width-1

  for k in range(Ntot[2]):
    for j in range(Ntot[1]):
      for g in range(width):

        A[k,j,ib1-g] = A[k,j,ib1-g] + sgn*A[k,j,ib0+g]

cdef inline void copy_add_reflect_layer_y(real3d A, int sgn,
                       ints js, ints js0, int width, ints Ntot[3]) nogil:

  cdef:
    ints i,j,k,g
    ints js1 = js+width-1

  for k in range(Ntot[2]):
    for g in range(width):
      for i in range(Ntot[0]):

        A[k,js1-g,i] = A[k,js1-g,i] + sgn * A[k,js0+g,i]

cdef inline void copy_add_reflect_layer_z(real3d A, int sgn,
                       ints ks, ints ks0, int width, ints Ntot[3]) nogil:

  cdef:
    ints i,j,k,g
    ints ks1 = ks+width-1

  for g in range(width):
    for j in range(Ntot[1]):
      for i in range(Ntot[0]):

        A[ks1-g,j,i] = A[ks1-g,j,i] + sgn * A[ks0+g,j,i]



# ==========================================================================


cdef inline void set_layer_x(real3d A, double set2,
                       ints ib, int width, ints Ntot[3]) nogil:

  cdef ints i,j,k,g

  for k in range(Ntot[2]):
    for j in range(Ntot[1]):
      for g in range(width):

        A[k,j,ib+g] = set2

cdef inline void set_layer_y(real3d A, double set2,
                       ints js, int width, ints Ntot[3]) nogil:

  cdef ints i,j,k,g

  for k in range(Ntot[2]):
    for g in range(width):
      for i in range(Ntot[0]):

        A[k,js+g,i] = set2

cdef inline void set_layer_z(real3d A, double set2,
                       ints ks, int width, ints Ntot[3]) nogil:

  cdef ints i,j,k,g

  for g in range(width):
    for j in range(Ntot[1]):
      for i in range(Ntot[0]):

        A[ks+g,j,i] = set2



# ==========================================================================


cdef inline void prolong_x(real3d A, int LR,
                       ints ib0, int width, ints Ntot[3]) nogil:

  # LR=0: prolong to left, LR=1: to right
  cdef:
    ints i,j,k,g
    ints ib01 = ib0 + 1 - 2*LR
    ints ib1 = ib01 - (1-LR) * width + LR

  for k in range(Ntot[2]):
    for j in range(Ntot[1]):
      for g in range(width):

        A[k,j,ib1+g] = A[k,j,ib01]

cdef inline void prolong_y(real3d A, int LR,
                       ints jb0, int width, ints Ntot[3]) nogil:

  cdef:
    ints i,j,k,g
    ints jb01 = jb0 + 1 - 2*LR
    ints jb1 = jb01 - (1-LR) * width + LR

  for k in range(Ntot[2]):
    for g in range(width):
      for i in range(Ntot[0]):

        A[k,jb1+g,i] = A[k,jb01,i]

cdef inline void prolong_z(real3d A, int LR,
                       ints kb0, int width, ints Ntot[3]) nogil:

  cdef:
    ints i,j,k,g
    ints kb01 = kb0 + 1 - 2*LR
    ints kb1 = kb01 - (1-LR) * width + LR

  for g in range(width):
    for j in range(Ntot[1]):
      for i in range(Ntot[0]):

        A[kb1+g,j,i] = A[kb01,j,i]



# =========================================================

cdef inline void copy_layer_r_sph(real3d A, int sgn,
                       ints ib, ints ib0, int width,
                       ints Ntot[3], ints Nact[3]) nogil:

  cdef:
    ints j,k,j_,k_,g
    ints ib1 = ib+width-1
    ints Nph_pi = Nact[2]/2, Nth_pi = Nact[1]
    ints ng = (Ntot[0]-Nact[0])/2

  for k in range(Ntot[2]):

    IF D3D: k_ = (k-ng + Nph_pi) % Nact[2] + ng
    ELSE:   k_ = k

    for j in range(Ntot[1]):

      IF D2D: j_ = Nth_pi - j + 2*ng-1
      ELSE:   j_ = j

      for g in range(width):
        A[k,j,ib1-g] = sgn * A[k_,j_,ib0+g]

#---------------------------------------------------

cdef inline void copy_layer_r_cyl(real3d A, int sgn,
                       ints ib, ints ib0, int width,
                       ints Ntot[3], ints Nact[3]) nogil:

  cdef:
    ints j,j_,k,g
    ints ib1 = ib+width-1
    ints Nph_pi = Nact[1]/2
    ints ng = (Ntot[0]-Nact[0])/2

  for k in range(Ntot[2]):
    for j in range(Ntot[1]):

      IF D2D: j_ = (j-ng + Nph_pi) % Nact[1] + ng
      ELSE:   j_ = j

      for g in range(width):
        A[k,j,ib1-g] = sgn * A[k,j_,ib0+g]

#----------------------------------------------------

cdef inline void copy_layer_th_sph(real3d A, int sgn,
                       ints jb, ints jb0, int width,
                       ints Ntot[3], ints Nact[3]) nogil:

  cdef:
    ints i,k,k_,g
    ints jb1 = jb+width-1
    ints Nph_pi = Nact[2]/2
    ints ng = (Ntot[0]-Nact[0])/2

  for k in range(Ntot[2]):

    IF D3D: k_ = (k-ng + Nph_pi) % Nact[2] + ng
    ELSE:   k_ = k

    for i in range(Ntot[0]):
      for g in range(width):
        A[k,jb1-g,i] = sgn * A[k_, jb0+g, i]

# ---------------------------------------------------

cdef inline void copy_add_layer_r_sph(real3d A, int sgn,
                       ints ib, ints ib0, int width,
                       ints Ntot[3], ints Nact[3]) nogil:

  cdef:
    ints j,k,j_,k_,g
    ints ib1 = ib+width-1
    ints Nph_pi = Nact[2]/2, Nth_pi = Nact[1]
    ints ng = (Ntot[0]-Nact[0])/2

  for k in range(Ntot[2]):

    IF D3D: k_ = (k-ng + Nph_pi) % Nact[2] + ng
    ELSE:   k_ = k

    for j in range(Ntot[1]):

      IF D2D: j_ = Nth_pi - j + 2*ng-1
      ELSE:   j_ = j

      for g in range(width):
        A[k,j,ib1-g] += sgn * A[k_,j_,ib0+g]

#---------------------------------------------------

cdef inline void copy_add_layer_r_cyl(real3d A, int sgn,
                       ints ib, ints ib0, int width,
                       ints Ntot[3], ints Nact[3]) nogil:

  cdef:
    ints j,j_,k,g
    ints ib1 = ib+width-1
    ints Nph_pi = Nact[1]/2
    ints ng = (Ntot[0]-Nact[0])/2

  for k in range(Ntot[2]):
    for j in range(Ntot[1]):

      IF D2D: j_ = (j-ng + Nph_pi) % Nact[1] + ng
      ELSE:   j_ = j

      for g in range(width):
        A[k,j,ib1-g] += sgn * A[k,j_,ib0+g]

#----------------------------------------------------

cdef inline void copy_add_layer_th_sph(real3d A, int sgn,
                       ints jb, ints jb0, int width,
                       ints Ntot[3], ints Nact[3]) nogil:

  cdef:
    ints i,k,k_,g
    ints jb1 = jb+width-1
    ints Nph_pi = Nact[2]/2
    ints ng = (Ntot[0]-Nact[0])/2

  for k in range(Ntot[2]):

    IF D3D: k_ = (k-ng + Nph_pi) % Nact[2] + ng
    ELSE:   k_ = k

    for i in range(Ntot[0]):
      for g in range(width):
        A[k,jb1-g,i] += sgn * A[k_, jb0+g, i]



# ===========================================================================

IF MPI:

  cdef inline void pack(real3d A, real1d sendbuf, ints *offset, ints *lims,
                        int *pack_order, int sign):

    cdef:
      ints i,j,k
      ints i1,i2, j1,j2, k1,k2
      ints di,dj,dk

    if pack_order[0]==1:
      i1,i2 = lim[0],lim[1]+1
      di=1
    else:
      i1,i2 = lim[1],lim[0]-1
      di=-1

    if pack_order[1]==1:
      j1,j2 = lim[2],lim[3]+1
      dj=1
    else:
      j1,j2 = lim[3],lim[2]-1
      dj=-1

    if pack_order[2]==1:
      k1,k2 = lim[4],lim[5]+1
      dk=1
    else:
      k1,k2 = lim[5],lim[4]-1
      dk=-1

    for k in range(k1,k2):
      for j in range(j1,j2):
        for i in range(i1,i2):

          sendbuf[offset[0]] = sign*A[k,j,i]
          offset[0] = offset[0]+1


  cdef inline void unpack(real3d A, real1d recvbuf, ints *offset, ints *lims):

    cdef ints i,j,k

    for k in range(lims[4],lims[5]+1):
      for j in range(lims[2],lims[3]+1):
        for i in range(lims[0],lims[1]+1):

          A[k,j,i] = recvbuf[offset[0]]
          offset[0] = offset[0]+1


  cdef inline void unpack_add(real3d A, real1d recvbuf, ints *offset, ints *lims):

    cdef ints i,j,k

    for k in range(lims[4],lims[5]+1):
      for j in range(lims[2],lims[3]+1):
        for i in range(lims[0],lims[1]+1):

          A[k,j,i] = A[k,j,i] + recvbuf[offset[0]]
          offset[0] = offset[0]+1
