# -*- coding: utf-8 -*-

from mpi4py import MPI as mpi
from mpi4py cimport MPI as mpi

import numpy as np
cimport numpy as np

import sys

from libc.math cimport sqrt,floor,ceil,log,exp,sin,cos,pow,fabs,fmin,fmax
from utils cimport calloc_2d_array
from utils cimport maxi, mini, sqr

from user_grid import set_user_coord_x, set_user_coord_y, set_user_coord_z

IF USE_CYLINDRICAL:
  from cylindrical cimport *
ELIF USE_SPHERICAL:
  from spherical cimport *
ELSE:
  from cartesian cimport *

IF SPREC:
  np_real = np.float32
ELSE:
  np_real = np.float64


# ============================================================

# Coordinate initialization for MHD.

cdef void init_coordinates(BnzGrid grid):

  set_spacings(grid)
  set_geometry(grid.coord)


# ============================================================

# Coordinate initialization for MHD.

cdef void set_spacings(BnzGrid grid):

  cdef:
    GridCoord *gc = &(grid.coord)
    GridBc gbc = grid.bc

  cdef:

    ints *pos = gc.pos

    ints *Ntot = gc.Ntot
    ints *Nact = gc.Nact
    ints *Ntot_glob = gc.Ntot_glob
    ints *Nact_glob = gc.Nact_glob

    real *lmin = gc.lmin
    real *lmax = gc.lmax


  # Set the scale of coordinate axes.

  x_scale = read_param("computation","x_scale",'s',usr_dir)
  if x_scale=='uni':
    gc.scale[0] = CS_UNI
  elif x_scale=='log':
    gc.scale[0] = CS_LOG
  elif x_scale=='usr':
    gc.scale[0] = CS_USR

  IF D2D:
    y_scale = read_param("computation","y_scale",'s',usr_dir)
    if y_scale=='uni':
      gc.scale[1] = CS_UNI
    elif y_scale=='log':
      gc.scale[1] = CS_LOG
    elif y_scale=='usr':
      gc.scale[1] = CS_USR

  IF D3D:
    z_scale =  read_param("computation","z_scale",'s',usr_dir)
    if z_scale=='uni':
      gc.scale[2] = CS_UNI
    elif z_scale=='log':
      gc.scale[2] = CS_LOG
    elif z_scale=='usr':
      gc.scale[2] = CS_USR

  # -------------------------------------------------------------

  # Set coordinates of cell faces (outflow BC assumed by default).

  cdef ints Ntot_max = maxi(maxi(Ntot[0],Ntot[1]),Ntot[2])

  cdef:
    **lf = gc.lf
    **dlf = gc.dlf
    **dlf_inv = gc.dlf_inv

  # cell faces
  lf  = <real**>calloc_2d_array(3, Ntot_max+1, sizeof(real))
  # cell spacings
  dlf = <real**>calloc_2d_array(3, Ntot_max, sizeof(real))
  # inverse cell spacings
  dlf_inv = <real**>calloc_2d_array(3, Ntot_max, sizeof(real))

  # Set user cell spacings.

  if gc.scale[0]==CS_USR:
    set_user_coord_x(gc)
  if gc.scale[1]==CS_USR:
    set_user_coord_y(gc)
  if gc.scale[2]==CS_USR:
    set_user_coord_z(gc)

  cdef:
    real dx,dxi,a, lmin_log, l1, dlf_
    ints n, i,i_,j,k,iglob, ind1=0, ind2=0

  # Set uniform or logarithmic cell spacings.

  for n in range(3):

    if Ntot[n]==1:
      lf[n][0]=lmin[n]
      lf[n][1]=lmax[n]
      continue

    if n==0:
      ind1=gc.i1
      ind2=gc.i2
    elif n==1:
      ind1=gc.j1
      ind2=gc.j2
    elif n==2:
      ind1=gc.k1
      ind2=gc.k2

    # Uniform axis.

    if gc.scale[n]==CS_UNI:

      dx = (lmax[n] - lmin[n]) / Nact_glob[n]
      l1 = pos[n] * Nact[n] * dx

      for i in range(Ntot[n]):
        lf[n][i] = l1 + (i-ind1)*dx
      lf[n][i+1] = lf[n][i] + dx

    # Logarithmic axis.

    elif gc.scale[n]==CS_LOG:

      if gbc.bc_flags[n][0]==0 or gbc.bc_flags[n][1]==0:
        sys.exit('periodic BC in {}-direction cannot be used with logarithmic cell spacing'.format(n))
      if gc.lmin[n]<=0:
        sys.exit('lmin[{}]<=0 is incompatible with logarithmic cell spacing'.format(n))

      a = (lmax[n]/lmin[n])**(1./Nact_glob[n])

      for i in range(Ntot[n]):
        iglob = i + pos[n]*Nact[n] - ind1
        lf[n][i] = lmin[n] * a**iglob
      lf[n][i+1] = a * lf[n][i]

      # if lmin[n]==0.: lmin_log = lmax[n]/Nact_glob[n]
      # else: lmin_log = lmin[n]
      #
      # a = (lmax[n]/lmin_log)**(1./Nact_glob[n])
      #
      # if lmin[n]==0 and pos[n]==0:
      #
      #   for i in range(ind1-1,Ntot[n]):
      #     lf[n][i] = (i-ind1)*lmin_log
      #
      #   lf[n][ind1]=0.
      #
      #   for i in range(ind1+1,Ntot[n]):
      #     lf[n][i]  = lmin_log * a**(i-ind1-1)
      #   lf[n][i+1]  = a * lf[n][i]
      #
      # else:
      #
      #   for i in range(Ntot[n]):
      #     iglob = i + pos[n]*Nact[n]
      #     lf[n][i]  = lmin_log * a**(iglob-ind1)
      #   lf[n][i+1]  = a * lf[n][i]

    # Reset coordinates in ghost cells as approproate for non-outflow BC.

    if gbc.bc_flags[n][0]==0:
      for i in range(ind1-1,-1,-1):
        i_ = ind2-i+(ind1-1)
        dlf_ = lf[n][i_+1] - lf[n][i_]
        lf[n][i] = lf[n][i+1] - dlf_

    if gbc.bc_flags[n][1]==0:
      for i in range(ind2+1,Ntot[n]):
        i_ = ind1+i-(ind2+1)
        dlf_ = lf[n][i_+1] - lf[n][i_]
        lf[n][i+1] = lf[n][i] + dlf_

    if gbc.bc_flags[n][0]==2:
      for i in range(ind1-1,-1,-1):
        i_ = ind1-i+(ind1-1)
        dlf_ = lf[n][i_+1] - lf[n][i_]
        lf[n][i] = lf[n][i+1] - dlf_

    if gbc.bc_flags[n][1]==2:
      for i in range(ind2+1,Ntot[n]):
        i_ = ind2-i+(ind2+1)
        dlf_ = lf[n][i_+1] - lf[n][i_]
        lf[n][i+1] = lf[n][i] + dlf_

  # Calculate cell spacings and their inverse.

  for n in range(3):
    for i in range(Ntot[n]):
      dlf[n][i] = lf[n][i+1] - lf[n][i]
      dlf_inv[n][i] = 1./dlf[n][i]

  # For user-specified spacings, it is user responsibility to set spacings
  # in ghost cells as appropriate for ouflow (default) or user BC

  # allocate space for volume coordinates
  gc.lv = <real**>calloc_2d_array(3, Ntot_max, sizeof(real))
  gc.dlv = <real**>calloc_2d_array(3, Ntot_max, sizeof(real))
  gc.dlv_inv = <real**>calloc_2d_array(3, Ntot_max, sizeof(real))

  # allocate space for coefficients used in parabolic reconstruction
  gc.hm_ratio = <real*>calloc_2d_array(3,Ntot_max, sizeof(real))
  gc.hp_ratio = <real*>calloc_2d_array(3,Ntot_max, sizeof(real))


# ======================================================================

cdef void add_geom_src_terms(GridCoord *gc, real4d w, real4d u,
                real4d fx, real4d fy, real4d fz, ints *lims, real dt) nogil:

  cdef:
    ints i,j,k
    real mpp, mtt, mtp, rp2,rm2, rp,rm
    real b2h, by2,bz2,bybz, a

  if gc.geom==CG_CYL:

    for k in range(lims[4], lims[5]+1):
      for j in range(lims[2], lims[3]+1):
        for i in range(lims[0], lims[1]+1):

          mpp = w[RHO,k,j,i]*sqr(w[VY,k,j,i]) + w[PR,k,j,i]

          IF TWOTEMP:
            mpp += w[PE,k,j,i]

          IF MFIELD:
            by2 = sqr(w[BY,k,j,i])
            b2h = 0.5 * (sqr(w[BX,k,j,i]) + sqr(w[BZ,k,j,i]) + by2)
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

  elif gc.geom==CG_SPH:

    for k in range(lims[4], lims[5]+1):
      for j in range(lims[2], lims[3]+1):
        for i in range(lims[0], lims[1]+1):

          mtt = w[RHO,k,j,i]*sqr(w[VY,k,j,i]) + w[PR,k,j,i]
          mpp = w[RHO,k,j,i]*sqr(w[VZ,k,j,i]) + w[PR,k,j,i]
          mtp = w[RHO,k,j,i]*w[VY,k,j,i]*w[VZ,k,j,i]

          IF TWOTEMP:
            mtt += w[PE,k,j,i]
            mpp += w[PE,k,j,i]

          IF MFIELD:
            by2  = sqr(w[BY,k,j,i])
            bz2  = sqr(w[BZ,k,j,i])
            bybz = w[BY,k,j,i] * w[BZ,k,j,i]
            b2h  = 0.5 * (sqr(w[BX,k,j,i]) + by2 + bz2)
            a = 1.
          IF CGL:
            a += 1.5 * (w[PR,k,j,i] - w[PPD,k,j,i]) / b2h
          IF MFIELD:
            mtt += b2h - a*by2
            mpp += b2h - a*bz2
            mtp += - a*bybz

          rp2 = sqr(gc.lf[0][i+1])
          rm2 = sqr(gc.lf[0][i])

          u[MX,k,j,i] += dt * gc.rinv_mean[i] * (mtt + mpp)

          u[MY,k,j,i] -= dt * gc.src_coeff1[i] * (rp2 * fx[MY,k,j,i+1] + rm2 * fx[MY,k,j,i])
          u[MZ,k,j,i] -= dt * gc.src_coeff1[i] * (rp2 * fx[MZ,k,j,i+1] + rm2 * fx[MZ,k,j,i])

          u[MY,k,j,i] += dt * gc.src_coeff2[j][i] * mpp
          u[MZ,k,j,i] -= dt * gc.src_coeff2[j][i] * mtp

  return



# =====================================================================

# Subtract/add the Laplacian of an array without copying.

cdef void add_laplacian(GridCoord *gc, real4d a, int sgn) nogil:

  cdef ints i,j,k, n

  # for n in range(NMODE):
  #   for k in range(gc.Ntot[2]):
  #     for j in range(gc.Ntot[1]):
  #       for i in range(gc.Ntot[0]):
  #         lapl  = gc.dlf_inv[0] * (a[n,k,j,i-1] + 2*a[n,k,j,i] + a[n,k,j,i+1])
  #         lapl += gc.dlf_inv[1] * (a[n,k,j-1,i] + 2*a[n,k,j,i] + a[n,k,j+1,i])
  #         lapl += gc.dlf_inv[2] * (a[n,k-1,j,i] + 2*a[n,k,j,i] + a[n,k+1,j,i])

  cdef:
    ints Nx = gc.Ntot[0]
    ints Ny = gc.Ntot[1]
    ints Nz = gc.Ntot[2]

  cdef:
    real h0 = 1.-sgn*1./12
    real h1 = sgn*1./24

  cdef:
    real **tmp1 = gc.scratch[]
    real **tmp2 = gc.scratch[]

  cdef real am1,a0,ap1,ap2

  for n in range(NMODE):

    for k in range(Nz):
      for j in range(Ny):

        tmp2[0,0] = a[n,k,j,0]

        for i in range(1,Nx-2,2):

          tmp1[0,0] = h1*(a[n,k,j,i-1] + a[n,k,j,i+1]) + h0*a[n,k,j,i]
          a[n,k,j,i-1] = tmp2[0,0]

          tmp2[0,0] = h1*(a[n,k,j,i] + a[n,k,j,i+2]) + h0*a[n,k,j,i+1]
          a[n,k,j,i] = tmp1[0,0]

        i = i+2

        if i==Nx-2:
          a[n,k,j,i] = h1*(a[n,k,j,i-1] + a[n,k,j,i+1]) + h0*a[n,k,j,i]

        a[n,k,j,i-1] = tmp2[0,0]


    # -------------------------------------------------------------------

    IF D2D:

      for k in range(Nz):

        for i in range(Nx):

          tmp2[0,i] = a[n,k,0,i]

        for j in range(1,Ny-2,2):

          for i in range(Nx):

            am1,a0,ap1,ap2 = a[n,k,j-1,i], a[n,k,j,i], a[n,k,j+1,i], a[n,k,j+2,i]

            tmp1[0,i] = h1 * (am1 + ap1) + h0 * a0
            a[n,k,j-1,i] = tmp2[0,i]

            tmp2[0,i] = h1 * (a0  + ap2) + h0 * ap1
            a[n,k,j,i] = tmp1[0,i]

        j = j+2

        if j==Ny-2:

          for i in range(Nx):
            a[n,k,j,i] = h1*(a[n,k,j-1,i] + a[n,k,j+1,i]) + h0*a[n,k,j,i]

        for i in range(Nx):
          a[n,k,j-1,i] = tmp2[0,i]

    # ----------------------------------------------------------------------

    IF D3D:

      for j in range(Ny):
        for i in range(Nx):

          tmp2[j,i] = a[n,0,j,i]

      for k in range(1,Nz-2,2):

        for j in range(Ny):
          for i in range(Nx):

            am1,a0,ap1,ap2 = a[n,k-1,j,i], a[n,k,j,i], a[n,k+1,j,i], a[n,k+2,j,i]

            tmp1[j,i] = h1*(am1 + ap1) + h0*a0
            a[n,k-1,j,i] = tmp2[j,i]

            tmp2[j,i] = h0*(a0  + ap2) + h0*ap1
            a[n,k,j,i] = tmp1[j,i]

      k = k+2

      if k==Nz-2:

        for j in range(Ny):
          for i in range(Nx):

            a[n,k,j,i] = h1*(a[n,k-1,j,i] + a[n,k+1,j,i]) + h0*a[n,k,j,i]

        for j in range(Ny):
          for i in range(Nx):
            a[n,k-1,j,i] = tmp2[j,i]


# ====================================================================

cdef inline real get_edge_len_x(GridCoord *gc, ints i, ints j, ints k) nogil:
  return get_edge_len_x_(gc, i,j,k)

cdef inline real get_edge_len_y(GridCoord *gc, ints i, ints j, ints k) nogil:
  return get_edge_len_y_(gc, i,j,k)

cdef inline real get_edge_len_z(GridCoord *gc, ints i, ints j, ints k) nogil:
  return get_edge_len_z_(gc, i,j,k)

# ----------------------------------------------------

cdef inline real get_centr_len_x(GridCoord *gc, ints i, ints j, ints k) nogil:
  return get_centr_len_x_(gc, i,j,k)

cdef inline real get_centr_len_y(GridCoord *gc, ints i, ints j, ints k) nogil:
  return get_centr_len_y_(gc, i,j,k)

cdef inline real get_centr_len_z(GridCoord *gc, ints i, ints j, ints k) nogil:
  return get_centr_len_z_(gc, i,j,k)

# -----------------------------------------------------

cdef inline real get_face_area_x(GridCoord *gc, ints i, ints j, ints k) nogil:
  return get_face_area_x_(gc, i,j,k)

cdef inline real get_face_area_y(GridCoord *gc, ints i, ints j, ints k) nogil:
  return get_face_area_y_(gc, i,j,k)

cdef inline real get_face_area_z(GridCoord *gc, ints i, ints j, ints k) nogil:
  return get_face_area_z_(gc, i,j,k)

# -----------------------------------------------------

cdef inline real get_cell_vol(GridCoord *gc, ints i, ints j, ints k) nogil:
  return get_cell_vol_(gc, i,j,k)
