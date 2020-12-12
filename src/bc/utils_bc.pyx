# -*- coding: utf-8 -*-

import numpy as np
cimport numpy as np
from cython.parallel import prange, parallel

#----------------------------------------------------------------

cdef inline void copy_layer_x(real3d A,
                       int ib, int ib0, int width,
                       int Ntot[3]) nogil:

  cdef int j,k,g

  for k in range(Ntot[2]):
    for j in range(Ntot[1]):
      for g in range(width):

        A[k,j,ib+g] = A[k,j,ib0+g]

cdef inline void copy_layer_y(real3d A,
                       int js, int js0, int width,
                       int Ntot[3]) nogil:

  cdef int i,k,g

  for k in range(Ntot[2]):
    for g in range(width):
      for i in range(Ntot[0]):

        A[k,js+g,i] = A[k,js0+g,i]

cdef inline void copy_layer_z(real3d A,
                       int ks, int ks0, int width,
                       int Ntot[3]) nogil:

  cdef int i,j,g

  for g in range(width):
    for j in range(Ntot[1]):
      for i in range(Ntot[0]):

        A[ks+g,j,i] = A[ks0+g,j,i]


# ===========================================================================

cdef inline void copy_reflect_layer_x(real3d A, int sgn,
                       int ib, int ib0, int width,
                       int Ntot[3]) nogil:

  cdef:
    int j,k,g
    int ib1 = ib+width-1

  for k in range(Ntot[2]):
    for j in range(Ntot[1]):
      for g in range(width):

        A[k,j,ib1-g] = sgn * A[k,j,ib0+g]

cdef inline void copy_reflect_layer_y(real3d A, int sgn,
                       int js, int js0, int width,
                       int Ntot[3]) nogil:

  cdef:
    int i,k,g
    int js1 = js+width-1

  for k in range(Ntot[2]):
    for g in range(width):
      for i in range(Ntot[0]):

        A[k,js1-g,i] = sgn * A[k,js0+g,i]

cdef inline void copy_reflect_layer_z(real3d A, int sgn,
                       int ks, int ks0, int width,
                       int Ntot[3]) nogil:

  cdef:
    int i,j,g
    int ks1 = ks+width-1

  for g in range(width):
    for j in range(Ntot[1]):
      for i in range(Ntot[0]):

        A[ks1-g,j,i] = sgn * A[ks0+g,j,i]


# ===========================================================================


cdef inline void copy_add_layer_x(real3d A,
                       int ib, int ib0, int width,
                       int Ntot[3]) nogil:

  cdef int j,k,g

  for k in range(Ntot[2]):
    for j in range(Ntot[1]):
      for g in range(width):
        A[k,j,ib+g] = A[k,j,ib+g] + A[k,j,ib0+g]

cdef inline void copy_add_layer_y(real3d A,
                       int js, int js0, int width,
                       int Ntot[3]) nogil:

  cdef int i,g,k

  for k in range(Ntot[2]):
    for g in range(width):
      for i in range(Ntot[0]):
        A[k,js+g,i] = A[k,js+g,i] + A[k,js0+g,i]

cdef inline void copy_add_layer_z(real3d A,
                       int ks, int ks0, int width,
                       int Ntot[3]) nogil:

  cdef int i,j,k,g

  for g in range(width):
    for j in range(Ntot[1]):
      for i in range(Ntot[0]):

        A[ks+g,j,i] = A[ks+g,j,i] + A[ks0+g,j,i]



# ===========================================================================

cdef inline void copy_add_reflect_layer_x(real3d A, int sgn,
                       int ib, int ib0, int width, int Ntot[3]) nogil:

  cdef:
    int i,j,k,g
    int ib1 = ib+width-1

  for k in range(Ntot[2]):
    for j in range(Ntot[1]):
      for g in range(width):

        A[k,j,ib1-g] = A[k,j,ib1-g] + sgn*A[k,j,ib0+g]

cdef inline void copy_add_reflect_layer_y(real3d A, int sgn,
                       int js, int js0, int width, int Ntot[3]) nogil:

  cdef:
    int i,j,k,g
    int js1 = js+width-1

  for k in range(Ntot[2]):
    for g in range(width):
      for i in range(Ntot[0]):

        A[k,js1-g,i] = A[k,js1-g,i] + sgn * A[k,js0+g,i]

cdef inline void copy_add_reflect_layer_z(real3d A, int sgn,
                       int ks, int ks0, int width, int Ntot[3]) nogil:

  cdef:
    int i,j,k,g
    int ks1 = ks+width-1

  for g in range(width):
    for j in range(Ntot[1]):
      for i in range(Ntot[0]):

        A[ks1-g,j,i] = A[ks1-g,j,i] + sgn * A[ks0+g,j,i]



# ==========================================================================


cdef inline void set_layer_x(real3d A, double set2,
                       int ib, int width, int Ntot[3]) nogil:

  cdef int i,j,k,g

  for k in range(Ntot[2]):
    for j in range(Ntot[1]):
      for g in range(width):

        A[k,j,ib+g] = set2

cdef inline void set_layer_y(real3d A, double set2,
                       int js, int width, int Ntot[3]) nogil:

  cdef int i,j,k,g

  for k in range(Ntot[2]):
    for g in range(width):
      for i in range(Ntot[0]):

        A[k,js+g,i] = set2

cdef inline void set_layer_z(real3d A, double set2,
                       int ks, int width, int Ntot[3]) nogil:

  cdef int i,j,k,g

  for g in range(width):
    for j in range(Ntot[1]):
      for i in range(Ntot[0]):

        A[ks+g,j,i] = set2



# ==========================================================================


cdef inline void prolong_x(real3d A, int LR,
                       int ib0, int width, int Ntot[3]) nogil:

  # LR=0: prolong to left, LR=1: to right
  cdef:
    int i,j,k,g
    int ib01 = ib0 + 1 - 2*LR
    int ib1 = ib01 - (1-LR) * width + LR

  for k in range(Ntot[2]):
    for j in range(Ntot[1]):
      for g in range(width):

        A[k,j,ib1+g] = A[k,j,ib01]

cdef inline void prolong_y(real3d A, int LR,
                       int jb0, int width, int Ntot[3]) nogil:

  cdef:
    int i,j,k,g
    int jb01 = jb0 + 1 - 2*LR
    int jb1 = jb01 - (1-LR) * width + LR

  for k in range(Ntot[2]):
    for g in range(width):
      for i in range(Ntot[0]):

        A[k,jb1+g,i] = A[k,jb01,i]

cdef inline void prolong_z(real3d A, int LR,
                       int kb0, int width, int Ntot[3]) nogil:

  cdef:
    int i,j,k,g
    int kb01 = kb0 + 1 - 2*LR
    int kb1 = kb01 - (1-LR) * width + LR

  for g in range(width):
    for j in range(Ntot[1]):
      for i in range(Ntot[0]):

        A[kb1+g,j,i] = A[kb01,j,i]



# =========================================================

cdef inline void copy_layer_r_sph(real3d A, int sgn,
                       int ib, int ib0, int width,
                       int Ntot[3], int Nact[3]) nogil:

  cdef:
    int j,k,j_,k_,g
    int ib1 = ib+width-1
    int Nph_pi = Nact[2]/2, Nth_pi = Nact[1]
    int ng = (Ntot[0]-Nact[0])/2

  for k in range(Ntot[2]):

    # CHECK THIS
    IF D3D: k_ = (k-ng + Nph_pi) % Nact[2] + ng
    ELSE:   k_ = k

    for j in range(Ntot[1]):

      IF D2D: j_ = Nth_pi - j + 2*ng-1
      ELSE:   j_ = j

      for g in range(width):
        A[k,j,ib1-g] = sgn * A[k_,j_,ib0+g]

#---------------------------------------------------

cdef inline void copy_layer_r_cyl(real3d A, int sgn,
                       int ib, int ib0, int width,
                       int Ntot[3], int Nact[3]) nogil:

  cdef:
    int j,j_,k,g
    int ib1 = ib+width-1
    int Nph_pi = Nact[1]/2
    int ng = (Ntot[0]-Nact[0])/2

  for k in range(Ntot[2]):
    for j in range(Ntot[1]):

      # CHECK THIS
      IF D2D: j_ = (j-ng + Nph_pi) % Nact[1] + ng
      ELSE:   j_ = j

      for g in range(width):
        A[k,j,ib1-g] = sgn * A[k,j_,ib0+g]

#----------------------------------------------------

cdef inline void copy_layer_th_sph(real3d A, int sgn,
                       int jb, int jb0, int width,
                       int Ntot[3], int Nact[3]) nogil:

  cdef:
    int i,k,k_,g
    int jb1 = jb+width-1
    int Nph_pi = Nact[2]/2
    int ng = (Ntot[0]-Nact[0])/2

  for k in range(Ntot[2]):

    # CHECK THIS
    IF D3D: k_ = (k-ng + Nph_pi) % Nact[2] + ng
    ELSE:   k_ = k

    for i in range(Ntot[0]):
      for g in range(width):
        A[k,jb1-g,i] = sgn * A[k_, jb0+g, i]

# ---------------------------------------------------

cdef inline void copy_add_layer_r_sph(real3d A, int sgn,
                       int ib, int ib0, int width,
                       int Ntot[3], int Nact[3]) nogil:

  cdef:
    int j,k,j_,k_,g
    int ib1 = ib+width-1
    int Nph_pi = Nact[2]/2, Nth_pi = Nact[1]
    int ng = (Ntot[0]-Nact[0])/2

  for k in range(Ntot[2]):

    # CHECK THIS
    IF D3D: k_ = (k-ng + Nph_pi) % Nact[2] + ng
    ELSE:   k_ = k

    for j in range(Ntot[1]):

      IF D2D: j_ = Nth_pi - j + 2*ng-1
      ELSE:   j_ = j

      for g in range(width):
        A[k,j,ib1-g] += sgn * A[k_,j_,ib0+g]

#---------------------------------------------------

cdef inline void copy_add_layer_r_cyl(real3d A, int sgn,
                       int ib, int ib0, int width,
                       int Ntot[3], int Nact[3]) nogil:

  cdef:
    int j,j_,k,g
    int ib1 = ib+width-1
    int Nph_pi = Nact[1]/2
    int ng = (Ntot[0]-Nact[0])/2

  for k in range(Ntot[2]):
    for j in range(Ntot[1]):

      # CHECK THIS
      IF D2D: j_ = (j-ng + Nph_pi) % Nact[1] + ng
      ELSE:   j_ = j

      for g in range(width):
        A[k,j,ib1-g] += sgn * A[k,j_,ib0+g]

#----------------------------------------------------

cdef inline void copy_add_layer_th_sph(real3d A, int sgn,
                       int jb, int jb0, int width,
                       int Ntot[3], int Nact[3]) nogil:

  cdef:
    int i,k,k_,g
    int jb1 = jb+width-1
    int Nph_pi = Nact[2]/2
    int ng = (Ntot[0]-Nact[0])/2

  for k in range(Ntot[2]):

    # CHECK THIS
    IF D3D: k_ = (k-ng + Nph_pi) % Nact[2] + ng
    ELSE:   k_ = k

    for i in range(Ntot[0]):
      for g in range(width):
        A[k,jb1-g,i] += sgn * A[k_, jb0+g, i]



# ===========================================================================

IF MPI:

  cdef inline void pack(real3d A, real1d sendbuf, long *offset, int *lims,
                        int *pack_order, int sign):

    cdef:
      int i,j,k
      int i1,i2, j1,j2, k1,k2
      int di,dj,dk

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

    for k in range(k1,k2,dk):
      for j in range(j1,j2,dj):
        for i in range(i1,i2,di):

          sendbuf[offset[0]] = sign*A[k,j,i]
          offset[0] = offset[0]+1


  cdef inline void unpack(real3d A, real1d recvbuf, long *offset, int *lims):

    cdef int i,j,k

    for k in range(lims[4],lims[5]+1):
      for j in range(lims[2],lims[3]+1):
        for i in range(lims[0],lims[1]+1):

          A[k,j,i] = recvbuf[offset[0]]
          offset[0] = offset[0]+1


  cdef inline void unpack_add(real3d A, real1d recvbuf, long *offset, int *lims):

    cdef int i,j,k

    for k in range(lims[4],lims[5]+1):
      for j in range(lims[2],lims[3]+1):
        for i in range(lims[0],lims[1]+1):

          A[k,j,i] = A[k,j,i] + recvbuf[offset[0]]
          offset[0] = offset[0]+1
