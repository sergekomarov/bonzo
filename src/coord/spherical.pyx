# -*- coding: utf-8 -*-

from mpi4py import MPI as mpi
from mpi4py cimport MPI as mpi

import numpy as np
cimport numpy as np

import sys

from libc.math cimport sqrt,floor,ceil,log,exp,sin,cos,pow,fabs,fmin,fmax
from utils cimport calloc_2d_array, calloc_3d_array, calloc_4d_array, calloc_from_memview_4d
from utils cimport maxi, mini, sqr

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
    real dsth,dcth,dth

  cdef:
    real **lf = gc.lf
    real **dlf = gc.dlf

  # auxilary coefficients to calculate cell lengths, areas and volumes
  gc.d2r      = <real*>calloc(Ntot[0],  sizeof(real))
  gc.d3r      = <real*>calloc(Ntot[0],  sizeof(real))
  gc.sin_thc  = <real*>calloc(Ntot[1],  sizeof(real))
  gc.sin_thf  = <real*>calloc(Ntot[1]+1,sizeof(real))
  gc.dcos_thf = <real*>calloc(Ntot[2],  sizeof(real))

  # auxilary coefficients to calculate geometric source terms
  gc.rinv_mean  = <real*>calloc(Ntot[0],sizeof(real))
  gc.src_coeff1 = <real*>calloc(Ntot[0],sizeof(real))
  gc.src_coeff2 = <real**>calloc_2d_array(Ntot[1],Ntot[0],sizeof(real))

  cdef:
    np.ndarray[real, ndim=1] sth = np.zeros(Ntot[1]+1, dtype=np_real)
    np.ndarray[real, ndim=1] cth = np.zeros(Ntot[1]+1, dtype=np_real)

  for i in range(Ntot[0]):

    rc  = 0.5*(lf[0][i] + lf[0][i+1])
    rca = fabs(rc)
    dr = dlf[0][i]

    hm = 3. + 2.*dr * (-10.*rca + dr) / ( 20.*rc**2 + dr**2)
    hp = 3. + 2.*dr * ( 10.*rca + dr) / ( 20.*rc**2 + dr**2)

    # volume r coordinate
    gc.lv[0][i] = rc + 2.*rc*dr**2 / (12.*rc**2 + dr**2)
    #3./4 * (lf[0][i+1]**4 - lf[0][i]**4) / (lf[0][i+1]**3 - lf[0][i]**3)

    gc.d2r[i] = rca*dr
    gc.d3r[i] = dr * (rc*2 + dr**2 / 12.)

    gc.rinv_mean[i]  = gc.d2r[i] / gc.d3r[i]
    gc.src_coeff1[i] = 0.5 * gc.rinv_mean[i] / rc**2

    gc.hm_ratio[0][i] = (hm+1.)/(hp-1.)
    gc.hp_ratio[0][i] = (hp+1.)/(hm-1.)

  for j in range(Ntot[1]+1):
    sth[j] = sin(lf[1][j])
    cth[j] = cos(lf[1][j])

  for j in range(Ntot[1]):

    dsth = sth[j] - sth[j+1]
    dcth = cth[j] - cth[j+1]
    dth = dlf[j]

    # signs?
    hm = - dth * (dsth + dth * cth[j])   / (dth * (sth[j] + sth[j+1]) - 2*dcth)
    hp =   dth * (dsth + dth * cth[j+1]) / (dth * (sth[j] + sth[j+1]) - 2*dcth)

    gc.lv[1][j] = ( (lf[1][j]   * cth[j]   - sth[j]
                   - lf[1][j+1] * cth[j+1] + sth[j+1]) / dcth )

    gc.sin_thf[j]  = fabs(sth)
    gc.sin_thv[j]  = fabs(sin(gc.lv[1][j]))
    gc.dcos_thf[j] = fabs(dcth)

    gc.hm_ratio[1][j] = (hm+1.)/(hp-1.)
    gc.hp_ratio[1][j] = (hp+1.)/(hm-1.)

    for i in range(Ntot[0]):
      # <cot(theta)/r>
      gc.src_coeff2[j][i] = gc.rinv_mean[i] * (fabs(sth[j+1]) - fabs(sth[j])) / fabs(dcth)

  for k in range(Ntot[2]):
    gc.lv[2][k] = 0.5 * (lf[2][k] + lf[2][k+1])
    gc.hm_ratio[2][k] = 2.
    gc.hp_ratio[2][k] = 2.

  # Calculate distances between cell centers and their inverse.

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

  return gc.lf[0][i] * gc.sin_thf[j] * dlf[2][k]

# ----------------------------------------------------

cdef inline real get_centr_len_x_(GridCoord *gc, ints i, ints j, ints k) nogil:

  return gc.dlv[0][i]

cdef inline real get_centr_len_y_(GridCoord *gc, ints i, ints j, ints k) nogil:

  return gc.lv[0][i] * gc.dlv[1][j]

cdef inline real get_centr_len_z_(GridCoord *gc, ints i, ints j, ints k) nogil:

  return gc.lv[0][i] * gc.sin_thc[j] * gc.dlv[2][k]

# -----------------------------------------------------

cdef inline real get_face_area_x_(GridCoord *gc, ints i, ints j, ints k) nogil:

  return sqr(gc.lf[0][i]) * gc.dlf[0][i] * gc.dcos_thf[j] * gc.dlf[2][k]

cdef inline real get_face_area_y_(GridCoord *gc, ints i, ints j, ints k) nogil:

  return gc.d2r[i] * gc.sin_thf[j] * gc.dlf[2][k]

cdef inline real get_face_area_z_(GridCoord *gc, ints i, ints j, ints k) nogil:

  return gc.d2r[i] * gc.dlf[1][j]

# -----------------------------------------------------

cdef inline real get_cell_vol_(GridCoord *gc, ints i, ints j, ints k) nogil:

  return gc.d3r[i] * gc.dcos_thf[j] * gc.dlf[2][k]
