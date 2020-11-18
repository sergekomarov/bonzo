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

  for n in range(3):
    for i in range(gc.Ntot[n]):

      gc.lv[n][i] = 0.5 * (gc.lf[n][i] + gc.lf[n][i+1])
      gc.hm_ratio[n][i] = 2.
      gc.hp_ratio[n][i] = 2.

  for n in range(3):
    for i in range(1,gc.Ntot[n]):
      gc.dlv[n][i] = gc.lv[n][i] - gc.lv[n][i-1]
      gc.dlv_inv[n][i] = 1./gc.dlv[n][i]


# ====================================================================

cdef inline real get_edge_len_x_(GridCoord *gc, ints i, ints j, ints k) nogil:

  return gc.dlf[0][i]

cdef inline real get_edge_len_y_(GridCoord *gc, ints i, ints j, ints k) nogil:

  return gc.dlf[1][j]

cdef inline real get_edge_len_z_(GridCoord *gc, ints i, ints j, ints k) nogil:

  return gc.dlf[2][k]

# ----------------------------------------------------

cdef inline real get_centr_len_x_(GridCoord *gc, ints i, ints j, ints k) nogil:

  return gc.dlv[0][i]

cdef inline real get_centr_len_y_(GridCoord *gc, ints i, ints j, ints k) nogil:

  return gc.dlv[1][j]

cdef inline real get_centr_len_z_(GridCoord *gc, ints i, ints j, ints k) nogil:

  return gc.dlv[2][k]

# -----------------------------------------------------

cdef inline real get_face_area_x_(GridCoord *gc, ints i, ints j, ints k) nogil:

  return gc.dlf[1][j] * gc.dlf[2][k]

cdef inline real get_face_area_y_(GridCoord *gc, ints i, ints j, ints k) nogil:

  return gc.dlf[0][i] * gc.dlf[2][k]

cdef inline real get_face_area_z_(GridCoord *gc, ints i, ints j, ints k) nogil:

  return gc.dlf[0][i] * gc.dlf[1][j]

# -----------------------------------------------------

cdef inline real get_cell_vol_(GridCoord *gc, ints i, ints j, ints k) nogil:

  return gc.dlf[0][i] * gc.dlf[1][j] * gc.dlf[2][k]
