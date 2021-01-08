# -*- coding: utf-8 -*-

import numpy as np
cimport numpy as np
from cython.parallel import prange, parallel, threadid

from libc.stdlib cimport malloc, calloc, free
from libc.stdio cimport printf

from bnz.coordinates.coord cimport get_face_area_x, get_face_area_y, get_face_area_z
from bnz.coordinates.coord cimport get_edge_len_x, get_edge_len_y, get_edge_len_z
from bnz.coordinates.coord cimport get_centr_len_x, get_centr_len_y, get_centr_len_z


cdef void advance_b(real4d b1, real4d b0, real4d ee,
                    GridCoord *gc, int *lims, real dt) nogil:

  # Advance face-centered magnetic field via constrained transport.

  cdef:
    int i,j,k, ip1,jp1,kp1
    real dtda, dsp,dsm

  for k in range(lims[4],lims[5]+1):
    for j in range(lims[2],lims[3]+1):
      for i in range(lims[0],lims[1]+2):

        dtda = dt / get_face_area_x(gc,i,j,k)

        b1[0,k,j,i] = b0[0,k,j,i]

        IF D2D:

          dsm = get_edge_len_z(gc,i,j,  k)
          dsp = get_edge_len_z(gc,i,j+1,k)

          b1[0,k,j,i] -= dtda * (dsp * ee[2,k,j+1,i] - dsm * ee[2,k,j,i])

        IF D3D:

          dsm = get_edge_len_y(gc,i,j,k)
          dsp = get_edge_len_y(gc,i,j,k+1)

          b1[0,k,j,i] += dtda * (dsp * ee[1,k+1,j,i] - dsm * ee[1,k,j,i])

  # can be parallelized if swap j and k
  for k in range(lims[4],lims[5]+1):
    for j in range(lims[2],lims[3]+1):

      # exclude poles theta=0, theta=pi
      if gc.geom==CG_SPH:
        if gc.lf[1][j]==0.:
          continue
        if gc.lf[1][j]==B_PI:
          continue

      for i in range(lims[0],lims[1]+1):

        dtda = dt / get_face_area_y(gc,i,j,k)

        dsm = get_edge_len_z(gc,i,  j,k)
        dsp = get_edge_len_z(gc,i+1,j,k)

        b1[1,k,j,i] = b0[1,k,j,i] + dtda * (dsp * ee[2,k,j,i+1] - dsm * ee[2,k,j,i])

        IF D3D:

          dsm = get_edge_len_x(gc,i,j,k)
          dsp = get_edge_len_x(gc,i,j,k+1)

          b1[1,k,j,i] -= dtda * (dsp * ee[0,k+1,j,i] - dsm * ee[0,k,j,i])


  # for k in prange(lims[4],lims[5]+2, num_threads=OMP_NT, nogil=True, schedule='dynamic'):
  for k in range(lims[4],lims[5]+2):
    for j in range(lims[2],lims[3]+1):
      for i in range(lims[0],lims[1]+1):

        dtda = dt / get_face_area_z(gc,i,j,k)

        dsm = get_edge_len_y(gc,i,  j,k)
        dsp = get_edge_len_y(gc,i+1,j,k)

        b1[2,k,j,i] = b0[2,k,j,i] - dtda * (dsp * ee[1,k,j,i+1] - dsm * ee[1,k,j,i])

        IF D2D:

          dsm = get_edge_len_x(gc,i,j, k)
          dsp = get_edge_len_x(gc,i,j+1,k)

          b1[2,k,j,i] += dtda * (dsp * ee[0,k,j+1,i] - dsm * ee[0,k,j,i])


# ---------------------------------------------------------------------------

cdef void interp_bc(real4d u, real4d b, GridCoord *gc, int *lims) nogil:

  # Interpolate magnetic field from cell faces to cell centers.

  cdef int i,j,k
  cdef:
    real *lf = gc.lf
    real *lv = gc.lv
    real *dlfi = gc.dlf_inv

  for k in range(lims[4],lims[5]+1):
    for j in range(lims[2],lims[3]+1):
      for i in range(lims[0],lims[1]+1):

        u[BX,k,j,i] = ( (lv[0][i]   - lf[0][i]) * b[0,k,j,i]
                      + (lf[0][i+1] - lv[0][i]) * b[0,k,j,i+1] ) * dlfi[0][i]

        IF D2D:
          u[BY,k,j,i] = ( (lv[1][j]   - lf[1][j]) * b[1,k,j,  i]
                        + (lf[1][j+1] - lv[1][j]) * b[1,k,j+1,i] ) * dlfi[1][j]
        ELSE:
          u[BY,k,j,i] = b[1,k,j,i]

        IF D3D:
          u[BZ,k,j,i] = ( (lv[2][k]   - lf[2][k]) * b[2,k,  j,i]
                        + (lf[2][k+1] - lv[2][k]) * b[2,k+1,j,i] ) * dlfi[2][k]
        ELSE:
          u[BZ,k,j,i] = b[2,k,j,i]


# ---------------------------------------------------------------------------

cdef void ec_from_prim(real4d ec, real4d w, int *lims) nogil:

  # Calculate electric field at cell centers from primitive variables.

  cdef:
    int i,j,k
    int il,iu, jl,ju, kl,ku

  il,iu = lims[0]-1,lims[1]+1
  jl,ju = lims[2]-1,lims[3]+1
  kl,ku = lims[4]-1,lims[5]+1
  IF not D2D: jl,ju=0,0
  IF not D3D: kl,ku=0,0

  for k in prange(kl,ku+1, num_threads=OMP_NT, nogil=True, schedule='dynamic'):
    for j in range(jl,ju+1):
      for i in range(il,iu+1):

        ec[0,k,j,i] = -(w[VY,k,j,i]*w[BZ,k,j,i] - w[VZ,k,j,i]*w[BY,k,j,i])
        ec[1,k,j,i] =  (w[VX,k,j,i]*w[BZ,k,j,i] - w[VZ,k,j,i]*w[BX,k,j,i])
        ec[2,k,j,i] = -(w[VX,k,j,i]*w[BY,k,j,i] - w[VY,k,j,i]*w[BX,k,j,i])


# ---------------------------------------------------------------------------

cdef void interp_ee1(real4d ee, real4d ec,
                     real4d fx, real4d fy, real4d fz, int *lims) nogil:

  # Interpolate electric field from cell centers to edge centers
  # using cell-centered E-field and face-centered Godunov fluxes.

  cdef:
    int k,j,i
    int il,iu, jl,ju, kl,ku

  il,iu = lims[0],lims[1]+1
  jl,ju = lims[2],lims[3]+1
  kl,ku = lims[4],lims[5]+1
  IF not D2D: jl,ju=0,0
  IF not D3D: kl,ku=0,0

  IF D3D and D2D:

    for k in range(kl, ku+1):
      for j in range(jl, ju+1):
        for i in range(il, iu+1):

          #j -> j-1/2
          #m -> m-1/2
          ee[0,k,j,i] = ( 0.5*(- fy[BZ,k-1,j,i] - fy[BZ,k,j,i]
                               + fz[BY,k,j-1,i] + fz[BY,k,j,i])
                - 0.25*(ec[0,k,  j-1,i] + ec[0,k,  j,i]
                      + ec[0,k-1,j-1,i] + ec[0,k-1,j,i]) )

          #m -> m-1/2
          #i -> i-1/2
          ee[1,k,j,i] = ( 0.5*( - fz[BX,k,j,i-1] - fz[BX,k,j,i]
                                + fx[BZ,k-1,j,i] + fx[BZ,k,j,i])
                - 0.25*(ec[1,k,  j,i-1] + ec[1,k,  j,i]
                      + ec[1,k-1,j,i-1] + ec[1,k-1,j,i]) )

          #i -> i-1/2
          #j -> j-1/2
          ee[2,k,j,i] = ( 0.5*(- fx[BY,k,j-1,i] - fx[BY,k,j,i]
                               + fy[BX,k,j,i-1] + fy[BX,k,j,i])
                - 0.25*(ec[2,k,j,  i-1] + ec[2,k,j,  i]
                      + ec[2,k,j-1,i-1] + ec[2,k,j-1,i]) )

  ELIF D2D:

    for j in range(jl, ju+1):
      for i in range(il, iu+1):

        ee[0,0,j,i] = - fy[BZ,0,j,i]
        ee[1,0,j,i] =   fx[BZ,0,j,i]

        ee[2,0,j,i] = ( 0.5*(- fx[BY,0,j-1,i] - fx[BY,0,j,i]
                             + fy[BX,0,j,i-1] + fy[BX,0,j,i])
              - 0.25*(ec[2,0,j,  i-1] + ec[2,0,j,  i]
                    + ec[2,0,j-1,i-1] + ec[2,0,j-1,i]) )

  ELSE:

    for i in range(il,iu+1):

      ee[0,0,0,i] = - ec[0, 0,0,i]
      ee[1,0,0,i] =   fx[BZ,0,0,i]
      ee[2,0,0,i] = - fx[BY,0,0,i]



# ----------------------------------------------------------------

cdef inline real get_ct_weight(real frho, real rho_l, real rho_r,
                               real ds, real dt) nogil:

  cdef real v_c = 1e3 * frho * dt / (FMIN(rho_l,rho_r) * ds)
  return 0.5 + FMAX(-0.5, FMIN(0.5, v_c))


cdef void interp_ee2(real4d ee, real4d ec,
                     real4d fx, real4d fy, real4d fz, int *lims,
                     real3d rho, real dt) nogil:

  cdef:
    int k,j,i
    int il,iu, jl,ju, kl,ku
    real dedx_14, dedy_14, dedz_14
    real dedx_34, dedy_34, dedz_34
    real dsx,dsy,dsz
    real wgtx, wgty, wgtz

  il,iu = lims[0],lims[1]+1
  jl,ju = lims[2],lims[3]+1
  kl,ku = lims[4],lims[5]+1
  IF not D2D: jl,ju=0,0
  IF not D3D: kl,ku=0,0


  # ---- Ez ----

  IF not D2D: ju -= 1

  for k in range(kl, ku):
    for j in range(jl, ju+1):
      for i in range(il, iu+1):

        IF D2D:

          dsx = get_centr_len_x(gc,i,j,k)
          wgtx = get_ct_weight(fx[RHO,k,j,i], rho[k,j,i-1], rho[k,j,i], dsx, dt)

          dedy_14 = ( wgtx  * (ec[2,k,j,i-1] - fy[BX,k,j,i-1])
                + (1.-wgtx) * (ec[2,k,j,i]   - fy[BX,k,j,i]) )

          dedy_34 = ( wgtx  * (fy[BX,k,j,i-1] - ec[2,k,j-1,i-1])
                + (1.-wgtx) * (fy[BX,k,j,i]   - ec[2,k,j-1,i]) )

          dsy = get_centr_len_y(gc,i,j,k)
          wgty = get_ct_weight(fy[RHO,k,j,i], rho[k,j-1,i], rho[k,j,i], dsy, dt)

          dedx_14 = ( wgty  * (ec[2,k,j-1,i] + fx[BY,k,j-1,i])
                + (1.-wgty) * (ec[2,k,j,  i] + fx[BY,k,j,  i]) )

          dedx_34 = ( wgty * (-fx[BY,k,j-1,i] - ec[2,k,j-1,i-1])
                + (1.-wgty) *(-fx[BY,k,j,  i] - ec[2,k,j,  i-1]) )

          #i -> i-1/2; j -> j-1/2
          ee[2,k,j,i] = 0.25*(- fx[BY,k,j-1,i] - fx[BY,k,j,i]
                              + fy[BX,k,j,i-1] + fy[BX,k,j,i]
                      + dedy_34 - dedy_14 + dedx_34 - dedx_14)

        ELSE:
          ee[2,k,j,i] = -fx[BY,k,j,i]


  # ---- Ex ----

  IF not D3D: ku -= 1

  for k in range(kl, ku+1):
    for j in range(jl, ju+1):
      for i in range(il, iu):

        IF D2D and D3D:

          dsy = get_centr_len_y(gc,i,j,k)
          wgty = get_ct_weight(fy[RHO,k,j,i], rho[k,j-1,i], rho[k,j,i], dsy, dt)

          dedz_14 = ( wgty  * (ec[0,k,j-1,i] - fz[BY,k,j-1,i])
                + (1.-wgty) * (ec[0,k,j,  i] - fz[BY,k,j,  i]) )

          dedz_34 = ( wgty  * (fz[BY,k,j-1,i] - ec[0,k,  j-1,i])
                + (1.-wgty) * (fz[BY,k,j,  i] - ec[0,k-1,j,  i]) )

          dsz = get_centr_len_z(gc,i,j,k)
          wgtz = get_ct_weight(fz[RHO,k,j,i], rho[k-1,j,i], rho[k,j,i], dsz, dt)

          dedy_14 = ( wgtz  * (ec[0,k-1,j,i] + fy[BZ,k-1,j,i])
                + (1.-wgtz) * (ec[0,k,  j,i] + fy[BZ,k,  j,i]) )

          dedy_34 = ( wgtz *  (- fy[BZ,k-1,j,i] - ec[0,k,j-1,i])
                + (1.-wgtz) * (- fy[BZ,k,  j,i] - ec[0,k,j-1,i]) )

          #j -> j-1/2; m -> m-1/2
          ee[0,k,j,i] = 0.25*(- fy[BZ,k-1,j,i] - fy[BZ,k,j,i]
                              + fz[BY,k,j-1,i] + fz[BY,k,j,i]
                    + dedz_34 - dedz_14 + dedy_34 - dedy_14)

        ELIF D2D:
          ee[0,k,j,i] = -fy[BZ,k,j,i]
        ELSE:
          ee[0,k,j,i] = ec[0,k,j,i]


  # ---- Ey ----

  IF not D2D: ju += 1

  for k in range(kl, ku+1):
    for j in range(jl, ju):
      for i in range(il, iu+1):

        IF D3D:

          dsz = get_centr_len_z(gc,i,j,k)
          wgtz = get_ct_weight(fz[RHO,k,j,i], rho[k-1,j,i], rho[k,j,i], dsz, dt)

          dedx_14 = ( wgtz  * (ec[1,k-1,j,i] - fx[BZ,k-1,j,i])
                + (1.-wgtz) * (ec[1,k,  j,i] - fx[BZ,k,  j,i]) )

          dedx_34 = ( wgtz  * (fx[BZ,k-1,j,i] - ec[1,k-1,j,i-1])
                + (1.-wgtz) * (fx[BZ,k,  j,i] - ec[1,k,  j,i-1]) )

          dsx = get_centr_len_x(gc,i,j,k)
          wgtx = get_ct_weight(fx[RHO,k,j,i], rho[k,j,i-1], rho[k,j,i], dsx, dt)

          dedz_14 = ( wgtx  * (ec[1,k,j,i-1] + fz[BX,k,j,i-1])
                + (1.-wgtx) * (ec[1,k,j,i]   + fz[BX,k,j,i]) )

          dedz_34 = ( wgtx  * (-fz[BX,k,j,i-1] - ec[1,k-1,j,i-1])
                + (1.-wgtx) * (-fz[BX,k,j,i]   - ec[1,k-1,j,i]) )

          #m -> m-1/2
          #i -> i-1/2
          ee[1,k,j,i] = 0.25*( - fz[BX,k,  j,i-1] - fz[BX,k,j,i]
                               + fx[BZ,k-1,j,i]   + fx[BZ,k,j,i]
                  + dedx_34 - dedx_14 + dedz_34 - dedz_14)

        ELSE:
          ee[1,k,j,i] = fx[BZ,k,j,i]
