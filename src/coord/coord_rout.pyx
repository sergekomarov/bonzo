# -*- coding: utf-8 -*-

from mpi4py import MPI as mpi
from mpi4py cimport MPI as mpi

import numpy as np
cimport numpy as np

import sys

from libc.math cimport sqrt,floor,ceil,log,exp,sin,cos,pow,fabs,fmin,fmax
from utils cimport calloc_2d_array, calloc_3d_array, calloc_4d_array, calloc_from_memview_4d
from utils cimport maxi, mini, sqr

from user_grid import set_user_coord_x, set_user_coord_y, set_user_coord_z

IF SPREC:
  np_real = np.float32
ELSE:
  np_real = np.float64


# ============================================================

# Coordinate initialization for MHD.

cdef void init_coord(BnzGrid grid):

  cdef:
    GridCoord *gc = &(grid.coord)
    GridBC gbc = grid.bc

  cdef:

    ints *pos = gc.pos

    ints *Ntot = gc.Ntot
    ints *Nact = gc.Nact
    ints *Ntot_glob = gc.Ntot_glob
    ints *Nact_glob = gc.Nact_glob

    real *lmin = gc.lmin
    real *lmax = gc.lmax

  # Set global coordinate limits.

  lmin[0] = read_param("computation","xmin",'f',usr_dir)
  lmax[0] = read_param("computation","xmax",'f',usr_dir)

  IF D2D:
    lmin[1] = read_param("computation","ymin",'f',usr_dir)
    lmax[1] = read_param("computation","ymax",'f',usr_dir)

  IF D3D:
    lmin[2] = read_param("computation","zmin",'f',usr_dir)
    lmax[2] = read_param("computation","zmax",'f',usr_dir)

  # Set coordinate geometry.

  geom = read_param("computation","geometry",'s',usr_dir)
  if geom=='car':
    gc.coord_geom=CG_CAR
  elif geom=='cyl':
    gc.coord_geom==CG_CYL
  elif geom=='sph':
    gc.coord_geom=CG_SPH

  # Round boundaries to their limits in cylindrical or spherical coordinates.

  IF D2D:
    if gc.coord_geom==CG_CYL and fabs(lmax[1] - 2*M_PI)<=1e-2:
      lmax[1] = 2*M_PI
    if gc.coord_geom==CG_SPH and fabs(lmax[1] -   M_PI)<=1e-2:
      lmax[1] = M_PI
  IF D3D:
    if gc.coord_geom==CG_SPH and fabs(lmax[2]-2*M_PI)<=1e-2:
      lmax[2] = 2*M_PI

  # enforce full range of angular coordinates (this can/should be changed)
  # if gc.coord_geom==CG_SPH:
  #   IF D2D:
  #     lmin[1] = 0
  #     lmax[1] = M_PI
  #   IF D3D:
  #     lmin[2] = 0
  #     lmax[2] = 2*M_PI
  # elif gc.coord_geom==CG_CYL:
  #   IF D2D:
  #     lmin[1] = 0
  #     lmax[1] = 2*M_PI

  # Reset BCs as appropriate for non-Cartesian geometries.

  if gc.coord_geom==CG_SPH or gc.coord_geom==CG_CYL:
    if lmin[0]==0.: gbc.bc_flags[0][0] = 2

  IF D2D:
    if gc.coord_geom==CG_CYL:
      if lmin[1]==0. and lmax[1]==2*M_PI:
        gbc.bc_flags[1][0] = 0
        gbc.bc_flags[1][1] = 0
    if gc.coord_geom==CG_SPH:
      if lmin[1]==0.: gbc.bc_flags[1][0] = 2
      if lmax[1]==M_PI: gbc.bc_flags[1][1] = 2
  IF D3D:
    if gc.coord_geom==CG_SPH:
      if lmin[2]==0. and lmax[2]==2*M_PI:
        gbc.bc_flags[2][0] = 0
        gbc.bc_flags[2][1] = 0

  # Set the scale of coordinate axes.

  x_scale = read_param("computation","x_scale",'s',usr_dir)
  if x_scale=='uni':
    gc.coord_scale[0] = CS_UNI
  elif x_scale=='log':
    gc.coord_scale[0] = CS_LOG
  elif x_scale=='usr':
    gc.coord_scale[0] = CS_USR

  IF D2D:
    y_scale = read_param("computation","y_scale",'s',usr_dir)
    if y_scale=='uni':
      gc.coord_scale[1] = CS_UNI
    elif y_scale=='log':
      gc.coord_scale[1] = CS_LOG
    elif y_scale=='usr':
      gc.coord_scale[1] = CS_USR

  IF D3D:
    z_scale =  read_param("computation","z_scale",'s',usr_dir)
    if z_scale=='uni':
      gc.coord_scale[2] = CS_UNI
    elif z_scale=='log':
      gc.coord_scale[2] = CS_LOG
    elif z_scale=='usr':
      gc.coord_scale[2] = CS_USR


  # -------------------------------------------------------------

  # Set coordinates of cell faces (outflow BC assumed by default).

  cdef ints Ntot_max = maxi(maxi(Ntot[0],Ntot[1]),Ntot[2])

  cdef:
    **lf = gc.lf
    **dlf = gc.dlf
    **dlf_inv = gc.dlf_inv

  # cell faces
  lf   = <real**>calloc_2d_array(3, Ntot_max+1, sizeof(real))
  # cell spacings
  dlf  = <real**>calloc_2d_array(3, Ntot_max, sizeof(real))
  # inverse cell spacings
  dlf_inv = <real**>calloc_2d_array(3, Ntot_max, sizeof(real))

  # set user coordinates
  if gc.coord_scale[0]==CS_USR:
    set_user_coord_x(gc)
  if gc.coord_scale[1]==CS_USR:
    set_user_coord_y(gc)
  if gc.coord_scale[2]==CS_USR:
    set_user_coord_z(gc)

  cdef:
    real dx,dxi,a, lmin_log, l1, dlf_
    ints n, i,j,k,iglob, ind1=0, ind2=0

  for n in range(3):

    IF not D2D:
      if n==1:
        if gc.coord_geom==CG_SPH:
          lf[1][0]=0.
          lf[1][1]=M_PI
        elif gc.coord_geom==CG_CYL:
          lf[1][0]=0.
          lf[1][1]=2*M_PI
        else:
          lf[1][0]=0.
          lf[1][1]=1.
        dlf[1][0]=lf[1][1]-lf[1][0]
        continue

    IF not D3D:
      if n==2:
        if gc.coord_geom==CG_SPH:
          lf[2][0]=0.
          lf[2][1]=2*M_PI
        else:
          lf[2][0]=0.
          lf[2][1]=1.
        dlf[2][0]=lf[2][1]-lf[2][0]
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

    if gc.coord_scale[n]==CS_UNI:

      dx = (lmax[n] - lmin[n]) / Nact_glob[n]
      l1 = pos[n] * Nact[n] * dx

      for i in range(Ntot[n]):
        lf[n][i] = l1 + (i-ind1)*dx
      lf[n][i+1] = lf[n][i] + dx

    # Logarithmic axis.

    elif gc.coord_scale[n]==CS_LOG:

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
        dlf_ = lf[n][ind2-i+(ind1-1)+1] - lf[n][ind2-i+(ind1-1)]
        lf[n][i] = lf[n][i+1] - dlf_

    if gbc.bc_flags[n][1]==0:
      for i in range(ind2+1,Ntot[n]):
        dlf_ = lf[n][ind1+i-(ind2+1)+1] - lf[n][ind1+i-(ind2+1)]
        lf[n][i+1] = lf[n][i] + dlf_

    if gbc.bc_flags[n][0]==2:
      for i in range(ind1-1,-1,-1):
        dlf_ = lf[n][ind1-i+(ind1-1)+1] - lf[n][ind1-i+(ind1-1)]
        lf[n][i] = lf[n][i+1] - dlf_

    if gbc.bc_flags[n][1]==2:
      for i in range(ind2+1,Ntot[n]):
        dlf_ = lf[n][ind2-i+(ind2+1)+1] - lf[n][ind2-i+(ind2+1)]
        lf[n][i+1] = lf[n][i] + dlf_


    # Calculate cell spacings and their inverse.
    for i in range(Ntot[n]):
      dlf[n][i] = lf[n][i+1] - lf[n][i]
      dlf_inv[n][i] = 1./dlf[n][i]

    # For user-specified spacings, it is user responsibility to set spacings
    # in ghost cells as appropriate for ouflow (default) or user BC



  # ---------------------------------------------------

  # Set volume coordinates, cell volumes and face areas.

  cdef:
    real rc,dr
    real dsth,dcth,dth
    real hm,hp
    real1d d2r,d3r, sth,cth

  cdef:

    real  **lv = gc.lv
    real  **dlv = gc.dlv
    real  **dlv_inv = gc.dlv_inv

    real  ***dv = gc.dv
    real ****da = gc.da
    real ****ds = gc.ds

    real   *rinv_mean = gc.rinv_mean
    real  *src_coeff1 = gc.src_coeff1
    real **src_coeff2 = gc.src_coeff2

    real **hm_ratio = gc.hm_ratio
    real **hp_ratio = gc.hp_ratio

  # volume coordinates
  lv = <real**>calloc_2d_array(3, Ntot_max, sizeof(real))
  dlv = <real**>calloc_2d_array(3, Ntot_max, sizeof(real))
  dlv_inv = <real**>calloc_2d_array(3, Ntot_max, sizeof(real))

  # inverse cell volumes
  dv_inv = <real***>calloc_3d_array(Ntot[2],Ntot[1],Ntot[0],sizeof(real))

  # areas of cell faces in all directions
  da = <real****>calloc_4d_array(3,Ntot[2],Ntot[1],Ntot[0], sizeof(real))

  # lengths of cell edges in all directions
  ds = <real****>calloc_4d_array(3,Ntot[2],Ntot[1],Ntot[0], sizeof(real))

  # auxilary coefficients to calculate geometric source terms
  if gc.coord_geom==CG_CYL or gc.coord_geom==CG_SPH:
    rinv_mean  = <real*>calloc(Ntot[0],sizeof(real))
    src_coeff1 = <real*>calloc(Ntot[0],sizeof(real))
  if gc.coord_geom==CG_SPH:
    src_coeff2 = <real*>calloc_2d_array(Ntot[1],Ntot[0],sizeof(real))

  # coefficients used in parabolic reconstruction
  hm_ratio = <real*>calloc_2d_array(3,Ntot_max, sizeof(real))
  hp_ratio = <real*>calloc_2d_array(3,Ntot_max, sizeof(real))


  if gc.coord_geom==CG_CAR:

    for n in range(3):
      for i in range(Ntot[n]):

        lv[n][i] = 0.5 * (lf[n][i] + lf[n][i+1])
        hm_ratio[n][i] = 2.
        hp_ratio[n][i] = 2.

    for k in range(Ntot[2]):
      for j in range(Ntot[1]):
        for i in range(Ntot[0]):

          ds[0][k][j][i] = dlf[0][i]
          ds[1][k][j][i] = dlf[1][i]
          ds[2][k][j][i] = dlf[2][i]

          da[0][k][j][i] = dlf[1][j] * dlf[2][k]
          da[1][k][j][i] = dlf[0][i] * dlf[2][k]
          da[2][k][j][i] = dlf[0][i] * dlf[1][j]

          dv_inv[k][j][i] = 1. / (dlf[0][i] * dlf[1][j] * dlf[2][k])


  elif gc.coord_geom==CG_CYL:

    d2r = np.zeros(Ntot[0], dtype=np_real)

    for i in range(Ntot[0]):
      rc  = 0.5*(lf[0][i]+lf[0][i+1])
      dr = dlf[0][i]

      d2r[i] = rc*dr

      lv[0][i] = rc + dr**2 / (12.*rc)

      rinv_mean[i] = 1./rc
      src_coeff1[i] = 0.5*rinv_mean[i]**2

      hm = 3. - 0.5*dr/rc
      hp = 3. + 0.5*dr/rc
      hm_ratio[0][i] = (hm+1.)/(hp-1.)
      hp_ratio[0][i] = (hp+1.)/(hm-1.)

    for n in range(1,3):
      for j in range(Ntot[n]):
        lv[n][j] = 0.5*(lf[n][j] + lf[n][j+1])
        hm_ratio[n][j] = 2.
        hp_ratio[n][j] = 2.

    for k in range(Ntot[2]):
      for j in range(Ntot[1]):
        for i in range(Ntot[0]):

          ds[0][k][j][i] = dlf[0][i]
          ds[1][k][j][i] = lf[0][i] * dlf[1][i]
          ds[2][k][j][i] = dlf[2][i]

          da[0][k][j][i] = lf[0][i] * dlf[1][j] * dlf[2][k]
          da[1][k][j][i] = dlf[0][i] * dlf[2][k]
          da[2][k][j][i] = d2r[i] * dlf[1][j]

          dv_inv[k][j][i] = 1. / ( d2r[i] * dlf[1][j] * dlf[2][k] )


  elif gc.coord_geom==CG_SPH:

    d2r = np.zeros(Ntot[0], dtype=np_real)
    d3r = np.zeros(Ntot[0], dtype=np_real)

    sth = np.zeros(Ntot[1]+1, dtype=np_real)
    cth = np.zeros(Ntot[1]+1, dtype=np_real)

    for i in range(Ntot[0]):

      rc  = 0.5*(lf[0][i]+lf[0][i+1])
      dr = dlf[0][i]

      d2r[i] = rc*dr
      d3r[i] = dr * (rc*2 + dr**2 / 12.)
      #3./4 * (lf[0][i+1]**4 - lf[0][i]**4) / (lf[0][i+1]**3 - lf[0][i]**3)

      lv[0][i] = rc + 2.*rc*dr**2 / (12.*rc**2 + dr**2)

      rinv_mean[i]  = d2r / d3r
      src_coeff1[i] = 0.5 * rinv_mean[i] / rc**2

      hm = 3. + 2.*dr * (-10.*rc + dr) / (-20.*r**2 + dr**2)
      hp = 3. + 2.*dr * (-10.*rc + dr) / ( 20.*r**2 + dr**2)
      hm_ratio[0][i] = (hm+1.)/(hp-1.)
      hp_ratio[0][i] = (hp+1.)/(hm-1.)

    for j in range(Ntot[1]+1):
      sth[j] = sin(lf[1][j])
      cth[j] = cos(lf[1][j])

    for j in range(Ntot[1]):

      lv[1][j] = ( (lf[1][j]   * cth[j]   - sth[j]
                  - lf[1][j+1] * cth[j+1] + sth[j+1]) / (cth[j] - cth[j+1]) )

      dsth = sth[j] - sth[j+1]
      dcth = cth[j] - cth[j+1]
      dth = dlf[j]
      hm = - dth * (dsth + dth * cth[j])   / (dth * (sth[j] + sth[j+1]) - 2*dcth)
      hp =   dth * (dsth + dth * cth[j+1]) / (dth * (sth[j] + sth[j+1]) - 2*dcth)
      hm_ratio[1][j] = (hm+1.)/(hp-1.)
      hp_ratio[1][j] = (hp+1.)/(hm-1.)

    for j in range(Ntot[1]):
      for i in range(Ntot[0]):
        # <cot(theta)/r>
        src_coeff2[j][i] = rinv_mean[i] * (sth[j+1] - sth[j]) / (cth[j] - cth[j+1])

    for k in range(Ntot[2]):
      lv[2][k] = 0.5 * (lf[2][k] + lf[2][k+1])
      hm_ratio[2][k] = 2.
      hp_ratio[2][k] = 2.

    for k in range(Ntot[2]):
      for j in range(Ntot[1]):
        for i in range(Ntot[0]):

        ds[0][k][j][i] = dlf[0][i]
        ds[1][k][j][i] = lf[0][i] * dlf[1][j]
        ds[2][k][j][i] = lf[0][i] * sth[j] * dlf[2][k]

        da[0][k][j][i] = lf[0][i]**2 * (cth[j] - cth[j+1]) * dlf[2][k]
        da[1][k][j][i] = d2r[i] * sth[j]  * dlf[2][k]
        da[2][k][j][i] = d2r[i] * dlf[1][j]

        dv_inv[k][j][i]   = 1. / (d3r[i] * (cth[j] - cth[j+1]) * dlf[2][k])


  # Calculate cell spacings and their inverse.

  for n in range(3):
    for i in range(Ntot[n]):
      dlv[n][i] = lv[n][i] - lv[n][i-1]
      dlv_inv[n][i] = 1./dlv[n][i]



# ======================================================================

cdef void add_geom_src_terms(GridCoord *gc, real4d W, real4d U,
                real4d Fx, real4d Fy, real4d Fz, ints *lims, real dt) nogil:

  cdef:
    ints i,j,k
    real mpp, mtt, mtp, rp2,rm2, rp,rm
    real b2h, by2,bz2,bybz, a=1.

  # ADD CGL TERMS

  if gc.coord_geom==CG_CYL:

    for k in range(lims[4], lims[5]+1):
      for j in range(lims[2], lims[3]+1):
        for i in range(lims[0], lims[1]+1):

          mpp = W[RHO,k,j,i]*sqr(W[VY,k,j,i]) + W[PR,k,j,i]

          IF TWOTEMP:
            mpp += W[PE,k,j,i]

          IF MFIELD:
            by2 = sqr(W[BY,k,j,i])
            b2h = 0.5 * (sqr(W[BX,k,j,i]) + sqr(W[BZ,k,j,i]) + by2)
          IF CGL:
            a = 1. + 1.5 * (W[PR,k,j,i] - W[PPD,k,j,i]) / b2h
          IF MFIELD:
            mpp += b2h - a*by2

          rp = gc.lf[0][i+1]
          rm = gc.lf[0][i]

          # add radial momentum source term at second order
          U[MX,k,j,i] += dt * gc.rinv_mean[i] * mpp
          #U[MX,k,j,i] += - dt * 0.5*rinv * (Fy[MY,k,j,i+1] + Fy[MY,k,j,i])

          # this expression is exact (-> exact conservation of angular momentum):
          U[MY,k,j,i] -= dt * gc.src_coeff1[i] * (rp * Fx[MY,k,j,i+1] + rm * Fx[MY,k,j,i])

  elif gc.coord_geom==CG_SPH:

    for k in range(lims[4], lims[5]+1):
      for j in range(lims[2], lims[3]+1):
        for i in range(lims[0], lims[1]+1):

          mtt = W[RHO,k,j,i]*sqr(W[VY,k,j,i]) + W[PR,k,j,i]
          mpp = W[RHO,k,j,i]*sqr(W[VZ,k,j,i]) + W[PR,k,j,i]
          mtp = W[RHO,k,j,i]*W[VY,k,j,i]*W[VZ,k,j,i]

          IF TWOTEMP:
            mtt += W[PE,k,j,i]
            mpp += W[PE,k,j,i]

          IF MFIELD:
            by2  = sqr(W[BY,k,j,i])
            bz2  = sqr(W[BZ,k,j,i])
            bybz = W[BY,k,j,i] * W[BZ,k,j,i]
            b2h  = 0.5 * (sqr(W[BX,k,j,i]) + by2 + bz2)
          IF CGL:
            a = 1. + 1.5 * (W[PR,k,j,i] - W[PPD,k,j,i]) / b2h
          IF MFIELD:
            mtt += b2h - a*by2
            mpp += b2h - a*bz2
            mtp += - a*bybz

          rp2 = sqr(gc.lf[0][i+1])
          rm2 = sqr(gc.lf[0][i])

          U[MX,k,j,i] += dt * gc.rinv_mean[i] * (mtt + mpp)

          U[MY,k,j,i] -= dt * gc.src_coeff1[i] * (rp2 * Fx[MY,k,j,i+1] + rm2 * Fx[MY,k,j,i])
          U[MZ,k,j,i] -= dt * gc.src_coeff1[i] * (rp2 * Fx[MZ,k,j,i+1] + rm2 * Fx[MZ,k,j,i])

          U[MY,k,j,i] += dt * gc.src_coeff2[j][i] * mpp
          U[MZ,k,j,i] -= dt * gc.src_coeff2[j][i] * mtp

  return



# =====================================================================

# Subtract/add the Laplacian of an array without copying.

cdef void add_laplacian(GridCoord *gc, real4d A, int sgn) nogil:

  cdef ints i,j,k, n

  # for n in range(NWAVES):
  #   for k in range(gc.Ntot[2]):
  #     for j in range(gc.Ntot[1]):
  #       for i in range(gc.Ntot[0]):
  #         lapl  = gc.dlf_inv[0] * (A[n,k,j,i-1] + 2*A[n,k,j,i] + A[n,k,j,i+1])
  #         lapl += gc.dlf_inv[1] * (A[n,k,j-1,i] + 2*A[n,k,j,i] + A[n,k,j+1,i])
  #         lapl += gc.dlf_inv[2] * (A[n,k-1,j,i] + 2*A[n,k,j,i] + A[n,k+1,j,i])

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

  cdef real Am1,A0,Ap1,Ap2

  for n in range(NWAVES):

    for k in range(Nz):
      for j in range(Ny):

        tmp2[0,0] = A[n,k,j,0]

        for i in range(1,Nx-2,2):

          tmp1[0,0] = h1*(A[n,k,j,i-1] + A[n,k,j,i+1]) + h0*A[n,k,j,i]
          A[n,k,j,i-1] = tmp2[0,0]

          tmp2[0,0] = h1*(A[n,k,j,i] + A[n,k,j,i+2]) + h0*A[n,k,j,i+1]
          A[n,k,j,i] = tmp1[0,0]

        i = i+2

        if i==Nx-2:
          A[n,k,j,i] = h1*(A[n,k,j,i-1] + A[n,k,j,i+1]) + h0*A[n,k,j,i]

        A[n,k,j,i-1] = tmp2[0,0]


    # -------------------------------------------------------------------

    IF D2D:

      for k in range(Nz):

        for i in range(Nx):

          tmp2[0,i] = A[n,k,0,i]

        for j in range(1,Ny-2,2):

          for i in range(Nx):

            Am1,A0,Ap1,Ap2 = A[n,k,j-1,i], A[n,k,j,i], A[n,k,j+1,i], A[n,k,j+2,i]

            tmp1[0,i] = h1 * (Am1 + Ap1) + h0 * A0
            A[n,k,j-1,i] = tmp2[0,i]

            tmp2[0,i] = h1 * (A0  + Ap2) + h0 * Ap1
            A[n,k,j,i] = tmp1[0,i]

        j = j+2

        if j==Ny-2:

          for i in range(Nx):
            A[n,k,j,i] = h1*(A[n,k,j-1,i] + A[n,k,j+1,i]) + h0*A[n,k,j,i]

        for i in range(Nx):
          A[n,k,j-1,i] = tmp2[0,i]

    # ----------------------------------------------------------------------

    IF D3D:

      for j in range(Ny):
        for i in range(Nx):

          tmp2[j,i] = A[n,0,j,i]

      for k in range(1,Nz-2,2):

        for j in range(Ny):
          for i in range(Nx):

            Am1,A0,Ap1,Ap2 = A[n,k-1,j,i], A[n,k,j,i], A[n,k+1,j,i], A[n,k+2,j,i]

            tmp1[j,i] = h1*(Am1 + Ap1) + h0*A0
            A[n,k-1,j,i] = tmp2[j,i]

            tmp2[j,i] = h0*(A0  + Ap2) + h0*Ap1
            A[n,k,j,i] = tmp1[j,i]

      k = k+2

      if k==Nz-2:

        for j in range(Ny):
          for i in range(Nx):

            A[n,k,j,i] = h1*(A[n,k-1,j,i] + A[n,k+1,j,i]) + h0*A[n,k,j,i]

        for j in range(Ny):
          for i in range(Nx):
            A[n,k-1,j,i] = tmp2[j,i]




# Convert cell indices to global physical coordinates.
# Cell indices count all cells.

# cdef inline void lind2gcrd(real *x, real *y, real *z,
#                            ints i, ints j, ints k, GridParams gp) nogil:
#
#   IF MPI:
#     x[0] = (gp.pos[0] * gp.Nact[0] + i-gp.i1 + 0.5) * gp.dl[0]
#     y[0] = (gp.pos[1] * gp.Nact[1] + j-gp.j1 + 0.5) * gp.dl[1]
#     z[0] = (gp.pos[2] * gp.Nact[2] + k-gp.k1 + 0.5) * gp.dl[2]
#   ELSE:
#     x[0] = (i-gp.i1 + 0.5) * gp.dl[0]
#     y[0] = (j-gp.j1 + 0.5) * gp.dl[1]
#     z[0] = (k-gp.k1 + 0.5) * gp.dl[2]
#
#
# cdef inline void lind2gcrd_x(real *x, ints i, GridParams gp) nogil:
#
#   IF MPI:
#     x[0] = (gp.pos[0] * gp.Nact[0] + i-gp.i1 + 0.5) * gp.dl[0]
#   ELSE:
#     x[0] = (i-gp.i1 + 0.5) * gp.dl[0]
#
# cdef inline void lind2gcrd_y(real *y, ints j, GridParams gp) nogil:
#
#   IF MPI:
#     y[0] = (gp.pos[1] * gp.Nact[1] + j-gp.j1 + 0.5) * gp.dl[1]
#   ELSE:
#     y[0] = (j-gp.j1 + 0.5) * gp.dl[1]
#
# cdef inline void lind2gcrd_z(real *z, ints k, GridParams gp) nogil:
#
#   IF MPI:
#     z[0] = (gp.pos[2] * gp.Nact[2] + k-gp.k1 + 0.5) * gp.dl[2]
#   ELSE:
#     z[0] = (k-gp.k1 + 0.5) * gp.dl[2]
#
#
#
# # Convert local physical coordinates to global coordinates.
#
# cdef inline void lcrd2gcrd(
#         real *xglob, real *yglob, real *zglob,
#         real xloc,  real yloc,  real zloc,
#         GridParams gp) nogil:
#
#   IF MPI:
#     xglob[0] = gp.pos[0] * gp.L[0] + xloc
#     yglob[0] = gp.pos[1] * gp.L[1] + yloc
#     zglob[0] = gp.pos[2] * gp.L[2] + zloc
#   ELSE:
#     xglob[0] = xloc
#     yglob[0] = yloc
#     zglob[0] = zloc
#
#
# cdef inline void lcrd2gcrd_x(real *xglob, real xloc, GridParams gp) nogil:
#
#   IF MPI:
#     xglob[0] = gp.pos[0] * gp.L[0] + xloc
#   ELSE:
#     xglob[0] = xloc
#
# cdef inline void lcrd2gcrd_y(real *yglob, real yloc, GridParams gp) nogil:
#
#   IF MPI:
#     yglob[0] = gp.pos[1] * gp.L[1] + yloc
#   ELSE:
#     yglob[0] = yloc
#
# cdef inline void lcrd2gcrd_z(real *zglob, real zloc, GridParams gp) nogil:
#
#   IF MPI:
#     zglob[0] = gp.pos[2] * gp.L[2] + zloc
#   ELSE:
#     zglob[0] = zloc
#
#
# # Convert local index to global index.
#
# cdef inline void lind2gind(
#         ints *iglob, ints *jglob, ints *kglob,
#         ints iloc, ints jloc, ints kloc,
#         GridParams gp) nogil:
#
#   IF MPI:
#     iglob[0] = gp.pos[0] * gp.Nact[0] + iloc
#     jglob[0] = gp.pos[1] * gp.Nact[1] + jloc
#     kglob[0] = gp.pos[2] * gp.Nact[2] + kloc
#   ELSE:
#     iglob[0] = iloc
#     jglob[0] = jloc
#     kglob[0] = kloc
#
#
# cdef inline void lind2gind_x(ints *iglob, ints iloc, GridParams gp) nogil:
#
#   IF MPI:
#     iglob[0] = gp.pos[0] * gp.Nact[0] + iloc
#   ELSE:
#     iglob[0] = iloc
#
# cdef inline void lind2gind_y(ints *jglob, ints jloc, GridParams gp) nogil:
#
#   IF MPI:
#     jglob[0] = gp.pos[1] * gp.Nact[1] + jloc
#   ELSE:
#     jglob[0] = jloc
#
# cdef inline void lind2gind_z(ints *kglob, ints kloc, GridParams gp) nogil:
#
#   IF MPI:
#     kglob[0] = gp.pos[2] * gp.Nact[2] + kloc
#   ELSE:
#     kglob[0] = kloc
