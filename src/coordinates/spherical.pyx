# -*- coding: utf-8 -*-

import numpy as np
cimport numpy as np

from cython.parallel import prange, parallel
from libc.stdlib cimport malloc, calloc, free
from bnz.util cimport calloc_2d_array, free_2d_array

IF SPREC:
  np_real = np.float32
ELSE:
  np_real = np.float64


cdef void init_geom_data_(GridCoord *gc):

  # auxilary coefficients to calculate cell lengths, areas and volumes
  gc.d2r      = <real*>calloc(gc.Ntot[0],  sizeof(real))
  gc.d3r      = <real*>calloc(gc.Ntot[0],  sizeof(real))
  gc.sin_thv  = <real*>calloc(gc.Ntot[1],  sizeof(real))
  gc.sin_thf  = <real*>calloc(gc.Ntot[1]+1,sizeof(real))
  gc.dcos_thf = <real*>calloc(gc.Ntot[1],  sizeof(real))

  # auxilary coefficients to calculate geometric source terms
  gc.rinv_mean  = <real*>calloc(gc.Ntot[0],sizeof(real))
  gc.src_coeff1 = <real*>calloc(gc.Ntot[0],sizeof(real))
  gc.src_coeff2 = <real**>calloc_2d_array(gc.Ntot[1],Ntot[0],sizeof(real))


cdef void set_geometry_(GridCoord *gc):

  # Initialization of volume coordinates and various geometric factors.

  cdef:
    int i,j,k,n
    real rc,rca,dr, hm,hp
    real dsth,dcth,dth

  cdef int *Ntot = gc.Ntot

  cdef:
    np.ndarray[real, ndim=1] sth = np.zeros(Ntot[1]+1, dtype=np_real)
    np.ndarray[real, ndim=1] cth = np.zeros(Ntot[1]+1, dtype=np_real)

  for i in range(Ntot[0]):

    rc  = 0.5*(gc.lf[0][i] + gc.lf[0][i+1])
    rca = FABS(rc)
    dr = gc.dlf[0][i]

    # hm = 3. + 2.*dr * (-10.*rca + dr) / ( 20.*rc**2 + dr**2)
    # hp = 3. + 2.*dr * ( 10.*rca + dr) / ( 20.*rc**2 + dr**2)

    # volume r coordinate
    gc.lv[0][i] = rc + 2.*rc*dr**2 / (12.*rc**2 + dr**2)
    #3./4 * (lf[0][i+1]**4 - lf[0][i]**4) / (lf[0][i+1]**3 - lf[0][i]**3)

    gc.d2r[i] = rca*dr
    gc.d3r[i] = dr * (rc**2 + dr**2 / 12.)

    gc.rinv_mean[i]  = gc.d2r[i] / gc.d3r[i]
    gc.src_coeff1[i] = 0.5 * gc.rinv_mean[i] / rc**2

    # gc.hm_ratio[0][i] = (hm+1.)/(hp-1.)
    # gc.hp_ratio[0][i] = (hp+1.)/(hm-1.)

    gc.syxf[i] = 1./gc.lf[0][i]
    gc.syxv[i] = 1./gc.lv[0][i]

    gc.szxf[i] = 1./gc.lf[0][i]
    gc.szxv[i] = 1./gc.lv[0][i]

  for j in range(Ntot[1]+1):
    sth[j] = SIN(gc.lf[1][j])
    cth[j] = COS(gc.lf[1][j])

  for j in range(Ntot[1]):

    dsth = sth[j] - sth[j+1]
    dcth = cth[j] - cth[j+1]
    dth  = gc.dlf[1][j]

    # signs?
    # hm = - dth * (dsth + dth * cth[j])   / (dth * (sth[j] + sth[j+1]) - 2*dcth)
    # hp =   dth * (dsth + dth * cth[j+1]) / (dth * (sth[j] + sth[j+1]) - 2*dcth)
    #
    # gc.hm_ratio[1][j] = (hm+1.)/(hp-1.)
    # gc.hp_ratio[1][j] = (hp+1.)/(hm-1.)

    gc.lv[1][j] = ( (gc.lf[1][j]   * cth[j]   - sth[j]
                   - gc.lf[1][j+1] * cth[j+1] + sth[j+1]) / dcth )

    gc.sin_thf[j]  = FABS(sth[j])
    gc.sin_thv[j]  = FABS(SIN(gc.lv[1][j]))
    gc.dcos_thf[j] = FABS(dcth)

    gc.szyf[j] = 1./gc.sin_thf[j]
    gc.szyv[j] = 1./gc.sin_thv[j]

    for i in range(Ntot[0]):
      # <cot(theta)/r>
      gc.src_coeff2[j][i] = gc.rinv_mean[i] * (FABS(sth[j+1]) - FABS(sth[j])) / FABS(dcth)

  for k in range(Ntot[2]):
    gc.lv[2][k] = 0.5 * (gc.lf[2][k] + gc.lf[2][k+1])
    # gc.hm_ratio[2][k] = 2.
    # gc.hp_ratio[2][k] = 2.

  # Calculate distances between cell centers and their inverse.

  for n in range(3):
    for i in range(1,Ntot[n]):
      gc.dlv[n][i] = gc.lv[n][i] - gc.lv[n][i-1]
      gc.dlv_inv[n][i] = 1./gc.dlv[n][i]


cdef void free_geom_data_(GridCoord* gc):
  free(gc.rinv_mean)
  free(gc.d2r)
  free(gc.d3r)
  free(gc.sin_thf)
  free(gc.sin_thc)
  free(gc.dcos_thf)
  free(gc.src_coeff1)
  free_2d_array(gc.src_coeff2)


# ---------------------------------------------------------------------------

cdef void add_geom_src_terms_(real4d u, real4d w,
                              real4d fx, real4d fy, real4d fz, GridCoord *gc,
                              int *lims, real dt) nogil:
  cdef:
    int i,j,k
    real mpp, mtt, mtp, rp2,rm2, rp,rm
    real b2h, by2,bz2,bybz, a

  for k in prange(lims[4], lims[5]+1, num_threads=OMP_NT, nogil=True, schedule='dynamic'):
    for j in range(lims[2], lims[3]+1):
      for i in range(lims[0], lims[1]+1):

        mtt = w[RHO,k,j,i]*SQR(w[VY,k,j,i]) + w[PR,k,j,i]
        mpp = w[RHO,k,j,i]*SQR(w[VZ,k,j,i]) + w[PR,k,j,i]
        mtp = w[RHO,k,j,i]*w[VY,k,j,i]*w[VZ,k,j,i]

        IF TWOTEMP:
          mtt += w[PE,k,j,i]
          mpp += w[PE,k,j,i]

        IF MFIELD:
          by2  = SQR(w[BY,k,j,i])
          bz2  = SQR(w[BZ,k,j,i])
          bybz = w[BY,k,j,i] * w[BZ,k,j,i]
          b2h  = 0.5 * (SQR(w[BX,k,j,i]) + by2 + bz2)
          a = 1.
        IF CGL:
          a += 1.5 * (w[PR,k,j,i] - w[PPD,k,j,i]) / b2h
        IF MFIELD:
          mtt += b2h - a*by2
          mpp += b2h - a*bz2
          mtp += - a*bybz

        rp2 = SQR(gc.lf[0][i+1])
        rm2 = SQR(gc.lf[0][i])

        u[MX,k,j,i] += dt * gc.rinv_mean[i] * (mtt + mpp)

        u[MY,k,j,i] -= dt * gc.src_coeff1[i] * (rp2 * fx[MY,k,j,i+1] + rm2 * fx[MY,k,j,i])
        u[MZ,k,j,i] -= dt * gc.src_coeff1[i] * (rp2 * fx[MZ,k,j,i+1] + rm2 * fx[MZ,k,j,i])

        u[MY,k,j,i] += dt * gc.src_coeff2[j][i] * mpp
        u[MZ,k,j,i] -= dt * gc.src_coeff2[j][i] * mtp

  return


# ===========================================================================

cdef inline real get_edge_len_x_(GridCoord *gc, int i, int j, int k) nogil:

  return gc.dlf[0][i]

cdef inline real get_edge_len_y_(GridCoord *gc, int i, int j, int k) nogil:

  return gc.lf[0][i] * gc.dlf[1][j]

cdef inline real get_edge_len_z_(GridCoord *gc, int i, int j, int k) nogil:

  return gc.lf[0][i] * gc.sin_thf[j] * gc.dlf[2][k]

# --------------------------------------------------------------------------

cdef inline real get_centr_len_x_(GridCoord *gc, int i, int j, int k) nogil:

  return gc.dlv[0][i]

cdef inline real get_centr_len_y_(GridCoord *gc, int i, int j, int k) nogil:

  return gc.lv[0][i] * gc.dlv[1][j]

cdef inline real get_centr_len_z_(GridCoord *gc, int i, int j, int k) nogil:

  return gc.lv[0][i] * gc.sin_thv[j] * gc.dlv[2][k]

# --------------------------------------------------------------------------

cdef inline real get_cell_width_x_(GridCoord *gc, int i, int j, int k) nogil:

  return gc.dlf[0][i]

cdef inline real get_cell_width_y_(GridCoord *gc, int i, int j, int k) nogil:

  return gc.lv[0][i] * gc.dlf[1][j]

cdef inline real get_cell_width_z_(GridCoord *gc, int i, int j, int k) nogil:

  return gc.lv[0][i] * gc.sin_thv[j] * gc.dlf[2][k]

# --------------------------------------------------------------------------

cdef inline real get_face_area_x_(GridCoord *gc, int i, int j, int k) nogil:

  return SQR(gc.lf[0][i]) * gc.dcos_thf[j] * gc.dlf[2][k]

cdef inline real get_face_area_y_(GridCoord *gc, int i, int j, int k) nogil:

  return gc.d2r[i] * gc.sin_thf[j] * gc.dlf[2][k]

cdef inline real get_face_area_z_(GridCoord *gc, int i, int j, int k) nogil:

  return gc.d2r[i] * gc.dlf[1][j]

# --------------------------------------------------------------------------

cdef inline real get_cell_vol_(GridCoord *gc, int i, int j, int k) nogil:

  return gc.d3r[i] * gc.dcos_thf[j] * gc.dlf[2][k]
