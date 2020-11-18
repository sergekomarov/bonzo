# -*- coding: utf-8 -*-

from mpi4py import MPI as mpi
from mpi4py cimport MPI as mpi

import numpy as np
cimport numpy as np

from libc.math cimport sqrt,floor,ceil,log,exp,sin,cos,pow,fabs,fmin,fmax

IF SPREC:
  np_real = np.float32
ELSE:
  np_real = np.float64


# ===============================================================

# Initialization of geometric factors and arrays of coefficients.

cdef void set_geometry(GridCoord *gc):

  cdef ints *Ntot = gc.Ntot

  # Set volume coordinates, cell volumes and face areas.

  cdef:
    real rc,rca,dr, hm,hp
    real **lf = gc.lf
    real **dlf = gc.dlf

  # auxilary coefficients to calculate cell lengths, areas and volumes
  gc.d2r = <real*>calloc(Ntot[0], sizeof(real))

  # auxilary coefficients to calculate geometric source terms
  gc.rinv_mean  = <real*>calloc(Ntot[0],sizeof(real))
  gc.src_coeff1 = <real*>calloc(Ntot[0],sizeof(real))

  for i in range(Ntot[0]):

    rc  = 0.5*(lf[0][i]+lf[0][i+1])
    rca = fabs(rc)
    dr = dlf[0][i]

    hm = 3. - 0.5*dr/rca
    hp = 3. + 0.5*dr/rca

    gc.lv[0][i] = rc + dr**2 / (12.*rc)
    gc.d2r[i] = rca*dr
    gc.rinv_mean[i] = 1./rca
    gc.src_coeff1[i] = 0.5*gc.rinv_mean[i]**2

    gc.hm_ratio[0][i] = (hm+1.)/(hp-1.)
    gc.hp_ratio[0][i] = (hp+1.)/(hm-1.)

  for n in range(1,3):
    for j in range(Ntot[n]):
      gc.lv[n][j] = 0.5*(lf[n][j] + lf[n][j+1])
      gc.hm_ratio[n][j] = 2.
      gc.hp_ratio[n][j] = 2.

  for n in range(3):
    for i in range(1,Ntot[n]):
      gc.dlv[n][i] = gc.lv[n][i] - gc.lv[n][i-1]
      gc.dlv_inv[n][i] = 1./gc.dlv[n][i]


# ====================================================================

cdef inline real get_edge_len_x_(GridCoord *gc, ints i, ints j, ints k) nogil:

  return gc.dlf[0][i]

cdef inline real get_edge_len_y_(GridCoord *gc, ints i, ints j, ints k) nogil:

  return gc.lf[0][i] * gc.dlf[1][j]

cdef inline real get_edge_len_z_(GridCoord *gc, ints i, ints j, ints k) nogil:

  return gc.dlf[2][k]

# ----------------------------------------------------

cdef inline real get_centr_len_x_(GridCoord *gc, ints i, ints j, ints k) nogil:

  return gc.dlv[0][i]

cdef inline real get_centr_len_y_(GridCoord *gc, ints i, ints j, ints k) nogil:

  return gc.lv[0][i] * gc.dlv[1][j]

cdef inline real get_centr_len_z_(GridCoord *gc, ints i, ints j, ints k) nogil:

  return gc.dlv[2][k]

# -----------------------------------------------------

cdef inline real get_face_area_x_(GridCoord *gc, ints i, ints j, ints k) nogil:

  return gc.lf[0][i] * gc.dlf[1][j] * gc.dlf[2][k]

cdef inline real get_face_area_y_(GridCoord *gc, ints i, ints j, ints k) nogil:

  return gc.dlf[0][i] * gc.dlf[2][k]

cdef inline real get_face_area_z_(GridCoord *gc, ints i, ints j, ints k) nogil:

  return gc.d2r[i] * gc.dlf[1][j]

# -----------------------------------------------------

cdef inline real get_cell_vol_(GridCoord *gc, ints i, ints j, ints k) nogil:

  return gc.d2r[i] * gc.dlf[1][j] * gc.dlf[2][k]
