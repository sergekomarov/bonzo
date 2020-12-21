# -*- coding: utf-8 -*-

import numpy as np
cimport numpy as np

from libc.stdlib cimport malloc, calloc, free
from cython.parallel import prange, parallel

IF SPREC:
  np_real = np.float32
ELSE:
  np_real = np.float64


cdef void init_geom_data_(GridCoord *gc):

  # auxilary coefficients to calculate cell lengths, areas and volumes
  gc.d2r = <real*>calloc(gc.Ntot[0], sizeof(real))

  # auxilary coefficients to calculate geometric source terms
  gc.rinv_mean  = <real*>calloc(gc.Ntot[0],sizeof(real))
  gc.src_coeff1 = <real*>calloc(gc.Ntot[0],sizeof(real))


cdef void set_geometry_(GridCoord *gc):

  # Initialization of volume coordinates and various geometric factors.

  cdef:
    int i,j,k,n
    real rc,rca,dr, hm,hp

  for i in range(gc.Ntot[0]):

    rc  = 0.5*(gc.lf[0][i] + gc.lf[0][i+1])
    rca = FABS(rc)
    dr = gc.dlf[0][i]

    hm = 3. - 0.5*dr/rca
    hp = 3. + 0.5*dr/rca

    gc.lv[0][i] = rc + dr**2 / (12.*rc)
    gc.d2r[i] = rca*dr
    gc.rinv_mean[i] = 1./rca
    gc.src_coeff1[i] = 0.5*gc.rinv_mean[i]**2

    gc.hm_ratio[0][i] = (hm+1.)/(hp-1.)
    gc.hp_ratio[0][i] = (hp+1.)/(hm-1.)

    gc.syxf[i] = 1./gc.lf[0][i]
    gc.syxv[i] = 1./gc.lv[0][i]
    gc.szxf[i] = 1.
    gc.szxv[i] = 1.

  for j in range(gc.Ntot[1]):
    gc.szyf[j] = 1.
    gc.szyv[j] = 1.

  for n in range(1,3):
    for j in range(gc.Ntot[n]):
      gc.lv[n][j] = 0.5*(gc.lf[n][j] + gc.lf[n][j+1])
      gc.hm_ratio[n][j] = 2.
      gc.hp_ratio[n][j] = 2.

  for n in range(3):
    for i in range(1,gc.Ntot[n]):
      gc.dlv[n][i] = gc.lv[n][i] - gc.lv[n][i-1]
      gc.dlv_inv[n][i] = 1./gc.dlv[n][i]


cdef void free_geom_data_(GridCoord* gc):
  free(gc.rinv_mean)
  free(gc.d2r)
  free(gc.src_coeff1)


# ---------------------------------------------------------------------------

cdef void add_geom_src_terms_(real4d u, real4d w,
                              real4d fx, real4d fy, real4d fz, GridCoord *gc,
                              int *lims, real dt) nogil:

  # Add geometric source terms associated with the Godunov flux divergence.

  cdef:
    int i,j,k
    real mpp, mtt, mtp, rp2,rm2, rp,rm
    real b2h, by2,bz2,bybz, a

  for k in prange(lims[4], lims[5]+1, num_threads=OMP_NT, nogil=True, schedule='dynamic'):
    for j in range(lims[2], lims[3]+1):
      for i in range(lims[0], lims[1]+1):

        mpp = w[RHO,k,j,i]*SQR(w[VY,k,j,i]) + w[PR,k,j,i]

        IF TWOTEMP:
          mpp += w[PE,k,j,i]

        IF MFIELD:
          by2 = SQR(w[BY,k,j,i])
          b2h = 0.5 * (SQR(w[BX,k,j,i]) + SQR(w[BZ,k,j,i]) + by2)
          a = 1.
        IF CGL:
          a += 1.5 * (w[PR,k,j,i] - w[PPD,k,j,i]) / b2h
        IF MFIELD:
          mpp += b2h - a*by2

        rp = gc.lf[0][i+1]
        rm = gc.lf[0][i]

        # add radial momentum source term at second order
        u[MX,k,j,i] += dt * gc.rinv_mean[i] * mpp
        #u[MX,k,j,i] += - dt * 0.5*rinv * (fy[MY,k,j,i+1] + fy[MY,k,j,i])

        # this expression is exact (-> exact conservation of angular momentum):
        u[MY,k,j,i] -= dt * gc.src_coeff1[i] * (rp * fx[MY,k,j,i+1] + rm * fx[MY,k,j,i])



# ----------------------------------------------------------------------------

cdef inline real get_edge_len_x_(GridCoord *gc, int i, int j, int k) nogil:

  return gc.dlf[0][i]

cdef inline real get_edge_len_y_(GridCoord *gc, int i, int j, int k) nogil:

  return gc.lf[0][i] * gc.dlf[1][j]

cdef inline real get_edge_len_z_(GridCoord *gc, int i, int j, int k) nogil:

  return gc.dlf[2][k]

# ----------------------------------------------------------------------------

cdef inline real get_centr_len_x_(GridCoord *gc, int i, int j, int k) nogil:

  return gc.dlv[0][i]

cdef inline real get_centr_len_y_(GridCoord *gc, int i, int j, int k) nogil:

  return gc.lv[0][i] * gc.dlv[1][j]

cdef inline real get_centr_len_z_(GridCoord *gc, int i, int j, int k) nogil:

  return gc.dlv[2][k]

# ----------------------------------------------------------------------------

cdef inline real get_face_area_x_(GridCoord *gc, int i, int j, int k) nogil:

  return gc.lf[0][i] * gc.dlf[1][j] * gc.dlf[2][k]

cdef inline real get_face_area_y_(GridCoord *gc, int i, int j, int k) nogil:

  return gc.dlf[0][i] * gc.dlf[2][k]

cdef inline real get_face_area_z_(GridCoord *gc, int i, int j, int k) nogil:

  return gc.d2r[i] * gc.dlf[1][j]

# ----------------------------------------------------------------------------

cdef inline real get_cell_vol_(GridCoord *gc, int i, int j, int k) nogil:

  return gc.d2r[i] * gc.dlf[1][j] * gc.dlf[2][k]
