# -*- coding: utf-8 -*-

import numpy as np
cimport numpy as np

from libc.stdlib cimport malloc, calloc, free

IF SPREC:
  np_real = np.float32
ELSE:
  np_real = np.float64


cdef void init_geom_data_(GridCoord *gc):
  return


cdef void set_geometry_(GridCoord *gc):

  # Initialization of volume coordinates and various geometric factors.

  cdef int i,j,k,n

  for n in range(3):
    for i in range(gc.Ntot[n]):

      gc.lv[n][i] = 0.5 * (gc.lf[n][i] + gc.lf[n][i+1])
      gc.hm_ratio[n][i] = 2.
      gc.hp_ratio[n][i] = 2.

  for i in range(gc.Ntot[0]):
    gc.syxf[i] = 1.
    gc.syxv[i] = 1.
    gc.szxf[i] = 1.
    gc.szxv[i] = 1.
  for j in range(gc.Ntot[1]):
    gc.szyf[j] = 1.
    gc.szyv[j] = 1.

  for n in range(3):
    for i in range(1,gc.Ntot[n]):
      gc.dlv[n][i] = gc.lv[n][i] - gc.lv[n][i-1]
      gc.dlv_inv[n][i] = 1./gc.dlv[n][i]


cdef void free_geom_data_(GridCoord* gc):
  return


cdef void add_geom_src_terms_(real4d u, real4d w,
                              real4d fx, real4d fy, real4d fz, GridCoord *gc,
                              int *lims, real dt) nogil:
  return

# ---------------------------------------------------------------------------

cdef inline real get_edge_len_x_(GridCoord *gc, int i, int j, int k) nogil:

  return gc.dlf[0][i]

cdef inline real get_edge_len_y_(GridCoord *gc, int i, int j, int k) nogil:

  return gc.dlf[1][j]

cdef inline real get_edge_len_z_(GridCoord *gc, int i, int j, int k) nogil:

  return gc.dlf[2][k]

# ---------------------------------------------------------------------------

cdef inline real get_centr_len_x_(GridCoord *gc, int i, int j, int k) nogil:

  return gc.dlv[0][i]

cdef inline real get_centr_len_y_(GridCoord *gc, int i, int j, int k) nogil:

  return gc.dlv[1][j]

cdef inline real get_centr_len_z_(GridCoord *gc, int i, int j, int k) nogil:

  return gc.dlv[2][k]

# ---------------------------------------------------------------------------

cdef inline real get_face_area_x_(GridCoord *gc, int i, int j, int k) nogil:

  return gc.dlf[1][j] * gc.dlf[2][k]

cdef inline real get_face_area_y_(GridCoord *gc, int i, int j, int k) nogil:

  return gc.dlf[0][i] * gc.dlf[2][k]

cdef inline real get_face_area_z_(GridCoord *gc, int i, int j, int k) nogil:

  return gc.dlf[0][i] * gc.dlf[1][j]

# ---------------------------------------------------------------------------

cdef inline real get_cell_vol_(GridCoord *gc, int i, int j, int k) nogil:

  return gc.dlf[0][i] * gc.dlf[1][j] * gc.dlf[2][k]
