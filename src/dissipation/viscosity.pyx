# -*- coding: utf-8 -*-

IF MPI:
  from mpi4py import MPI as mpi
  from mpi4py cimport MPI as mpi

import numpy as np
cimport numpy as np
from cython.parallel import prange, parallel, threadid
from libc.stdio cimport printf

from bnz.util cimport print_root
cimport util_diffusion as utdiff


cdef void diffuse(BnzDiffusion diff, BnzGrid grid, real dt):
  # only need integr to apply BCs

  cdef:
    GridCoord *gc = grid.coord
    real4d prim = grid.data.prim
    BnzIntegr integr = diff.integr
    DiffScratch scr = diff.scratch

    real dt_hyp = dt
    real dt_diff
    real a
    int k,j,i, n,m, s, ngh

  cdef int il,iu, jl,ju, kl,ku

  il, iu = 2, gc.Ntot[0]-3
  IF D2D: jl, ju = 2, gc.Ntot[1]-3
  ELSE: jl, ju = 0, 0
  IF D3D: kl, ku = 2, gc.Ntot[2]-3
  ELSE: kl, ku = 0, 0

  cdef:
    # scratch STS-arrays
    real4d vel0=scr.vel0, velm1=scr.velm1
    real4d dvel0=scr.dvel0, dvel=scr.dvel
    real2d sts_coeff

  # cdef int uniflat_geom = 0
  # if gc.geom==CG_CAR and gc.scale[0]==CS_UNI and gc.scale[1]==CS_UNI and gc.scale[2]==CS_UNI:
  #   uniflat_geom=1

  # primitive vars have already been calculated for all diffusion processes

  # apply BC to all variables (including face-centered magnetic field)
  grid.apply_grid_bc(integr, np.arange(RHO,VX,VY,VZ))

  # calculate diffusive time step
  dt_diff = get_dt(prim, gc, diff)

  # calculate number of STS iterations
  s = diff.get_nsts(dt_hyp, dt_diff)
  print_root("\nviscosity, N_STS=%d ... ", s)

  # calculate STS coefficients
  sts_coeff = diff.get_sts_coeff(s)

  ngh = gc.ng/2

  # do STS iterations
  for m in range(1,s+1):

    # if uniflat_geom:
    apply_diff_oper_uniflat(dvel, prim, gc, diff)
    # else:
    #   apply_diff_oper(dvel, prim, gc, diff)

    if m==1:
      for n in range(3):
        for k in range(kl,ku+1):
          for j in range(jl,ju+1):
            for i in range(il,iu+1):

              vel0[ n,k,j,i] = prim[VX+n,k,j,i]
              dvel0[n,k,j,i] = dvel[n,k,j,i]

    # do STS iteration
    for n in range(3):
      for k in range(kl,ku+1):
        for j in range(jl,ju+1):
          for i in range(il,iu+1):

            a = prim[VX+n,k,j,i]
            prim[VX+n,k,j,i] =      ( sts_coeff[m,MU]   *  prim[VX+n,k,j,i] +
                                      sts_coeff[m,NU]   * velm1[n,k,j,i]
             + (1 - sts_coeff[m,MU] - sts_coeff[m,NU])  *  vel0[n,k,j,i]
                          + dt_hyp * (sts_coeff[m,MUT]  *  dvel[n,k,j,i]
                                    + sts_coeff[m,GAMT] * dvel0[n,k,j,i]) )
            velm1[n,k,j,i] = a


    # need 2 layers of ghost cells to do 1 iteration
    if m%ngh==0 and m!=s:
      grid.apply_grid_bc(integr, np.asarray([VX,VY,VZ]))

  # if uniflat_geom:
  update_nrg_uniflat(prim, vel0, gc, diff, dt)
  # else:
  #   update_nrg(prim, vel0, gc, diff, dt)


# -------------------------------------------------------------------------

cdef void apply_diff_oper_uniflat(real3d dvel, real4d w,
                          GridCoord *gc, BnzDiffusion diff):

  # Apply Navier-Stokes viscosity matrix operator to velocity.

  cdef:

    int i,j,k,n
    real one12th = 1./12

  cdef:
    real dxi = gc.dlf_inv[0][0]
    real dyi = gc.dlf_inv[1][0]
    real dzi = gc.dlf_inv[2][0]
    real dxi2 = dxi*dxi
    real dyi2 = dyi*dyi
    real dzi2 = dzi*dzi

  cdef:
    int il,iu, jl,ju, kl,ku
    int ill,iuu, jll,juu, kll,kuu

  ill, iuu = 2, gc.Ntot[0]-3
  IF D2D: jll, juu = 2, gc.Ntot[1]-3
  ELSE: jll, juu = 0, 0
  IF D3D: kll, kuu = 2, gc.Ntot[2]-3
  ELSE: kll, kuu = 0, 0

  il, iu = 1, gc.Ntot[0]-2
  IF D2D: jl, ju = 1, gc.Ntot[1]-2
  ELSE: jl, ju = 0, 0
  IF D3D: kl, ku = 1, gc.Ntot[2]-2
  ELSE: kl, ku = 0, 0

  # scratch array to store velocity divergence
  cdef real3d div = diff.scr.div


  for k in range(kl, ku+1):
    for j in range(jl, ju+1):
      for i in range(il, iu+1):

        div[k,j,i] = dxi * (w[VX,k,j,i+1] - w[VX,k,j,i-1])
        IF D2D:
          div[k,j,i] = div[k,j,i] + dyi * (w[VY,k,j+1,i] - w[VY,k,j-1,i])
        IF D3D:
          div[k,j,i] = div[k,j,i] + dzi * (w[VZ,k+1,j,i] - w[VZ,k-1,j,i])

        div[k,j,i] = one12th * div[k,j,i]

  for n in range(3):
    for k in range(kll, kuu+1):
      for j in range(jll, juu+1):
        for i in range(ill, iuu+1):

          dvel[n,k,j,i] = dxi2 * (w[VX+n,k,j,i-1] - 2.*w[VX+n,k,j,i] + w[VX+n,k,j,i+1])
          IF D2D:
            dvel[n,k,j,i] = dvel[n,k,j,i] + dyi2 * (w[VX+n,k,j-1,i] - 2.*w[VX+n,k,j,i] + w[VX+n,k,j+1,i])
          IF D3D:
            dvel[n,k,j,i] = dvel[n,k,j,i] + dzi2 * (w[VX+n,k-1,j,i] - 2.*w[VX+n,k,j,i] + w[VX+n,k+1,j,i])

  for k in range(kll, kuu+1):
    for j in range(jll, juu+1):
      for i in range(ill, iuu+1):

        dvel[0,k,j,i] = dvel[0,k,j,i] + dxi * (div[k,j,i+1] - div[k,j,i-1])

  IF D2D:
    for k in range(kll, kuu+1):
      for j in range(jll, juu+1):
        for i in range(ill, iuu+1):

          dvel[1,k,j,i] = dvel[1,k,j,i] + dyi * (div[k,j+1,i] - div[k,j-1,i])

  IF D3D:
    for k in range(kll, kuu+1):
      for j in range(jll, juu+1):
        for i in range(ill, iuu+1):

          dvel[2,k,j,i] = dvel[2,k,j,i] + dzi * (div[k+1,j,i] - div[k-1,j,i])

  for n in range(3):
    for k in range(kll, kuu+1):
      for j in range(jll, juu+1):
        for i in range(ill, iuu+1):

          dvel[n,k,j,i] = diff.mu * dvel[n,k,j,i] / w[RHO,k,j,i]


# --------------------------------------------------------------------------

cdef void update_nrg_uniflat(real4d w, real4d vel0, GridCoord *gc,
                             BnzDiffusion diff, real dt):

  # Update energy due to viscous fluxes.

  cdef:

    int i,j,k
    real de,dp,ek,ek0,dek
    int i1 = gc.i1, i2=gc.i2
    int j1 = gc.j1, j2=gc.j2
    int k1 = gc.k1, k2=gc.k2
    int i1_1,i2_1, j1_1,j2_1, k1_1,k2_1

    real dxi=gc.dlf_inv[0][0]
    real dyi=gc.dlf_inv[1][0]
    real dzi=gc.dlf_inv[2][0]

    real gamm1 = diff.gam-1
    real one6th = 1./6
    real dtmu = dt * diff.mu
    real tx,ty,tz

    # local scratch arrays
    real3d div  = diff.scr.div   # velocity divergence
    real4d flx = diff.scr.dvel  # viscous fluxes

  i1_1 = gc.i1-1
  i2_1 = gc.i2+1
  IF D2D:
    j1_1 = gc.j1-1
    j2_1 = gc.j2+1
  ELSE:
    j1_1,j2_1 = 0,0
  IF D3D:
    k1_1 = gc.k1-1
    k2_1 = gc.k2+1
  ELSE:
    k1_1,k2_1 = 0,0


  for k in range(k1_1, k2_1+1):
    for j in range(j1_1, j2_1+1):
      for i in range(i1_1, i2_1+1):
        div[k,j,i] = dxi*(w[VX,k,j,i+1] - w[VX,k,j,i-1])

  IF D2D:
    for k in range(k1_1, k2_1+1):
      for j in range(j1_1, j2_1+1):
        for i in range(i1_1, i2_1+1):
          div[k,j,i] = div[k,j,i] + dyi*(w[VY,k,j+1,i] - w[VY,k,j-1,i])

  IF D3D:
    for k in range(k1_1, k2_1+1):
      for j in range(j1_1, j2_1+1):
        for i in range(i1_1, i2_1+1):
          div[k,j,i] = div[k,j,i] + dzi*(w[VZ,k+1,j,i] - w[VZ,k-1,j,i])

  for k in range(k1_1, k2_1+1):
    for j in range(j1_1, j2_1+1):
      for i in range(i1_1, i2_1+1):
        div[k,j,i] = one6th * div[k,j,i]

  IF D2D and D3D:

    for k in range(k1, k2+2):#, nogil=True, num_threads=OMP_NT, schedule='dynamic'):
      for j in range(j1, j2+2):
        for i in range(i1, i2+2):

          # X flux

          tx = 2.*dxi * (w[VX,k,j,i] - w[VX,k,j,i-1]) - (div[k,j,i-1] + div[k,j,i])
          ty =  ( dxi * (w[VY,k,j,i] - w[VY,k,j,i-1])
                       + 0.25*dyi * (w[VX,k,j+1,i-1] - w[VX,k,j-1,i-1]
                                   + w[VX,k,j+1,i] -   w[VX,k,j-1,i]) )
          tz = ( dxi * (w[VZ,k,j,i] - w[VZ,k,j,i-1])
                       + 0.25*dzi * (w[VX,k+1,j,i-1] - w[VX,k-1,j,i-1]
                                   + w[VX,k+1,j,i]  -  w[VX,k-1,j,i]) )

          flx[0,k,j,i] = 0.5*( (w[VX,k,j,i-1] + w[VX,k,j,i]) * tx +
                               (w[VY,k,j,i-1] + w[VY,k,j,i]) * ty +
                               (w[VZ,k,j,i-1] + w[VZ,k,j,i]) * tz )

          # Y flux

          tx = ( dyi * (w[VX,k,j,i] - w[VX,k,j-1,i])
                       + 0.25*dxi * (w[VY,k,j-1,i+1] - w[VY,k,j-1,i-1]
                                   + w[VY,k,j,i+1] -   w[VY,k,j,i-1]) )
          ty = 2.*dyi * (w[VY,k,j,i] - w[VY,k,j-1,i]) - (div[k,j-1,i] + div[k,j,i])
          tz =  ( dyi * (w[VZ,k,j,i] - w[VZ,k,j-1,i])
                       + 0.25*dzi * (w[VY,k+1,j-1,i] - w[VY,k-1,j-1,i]
                                   + w[VY,k+1,j,i] -   w[VY,k-1,j,i]) )

          flx[1,k,j,i] = 0.5*( (w[VX,k,j-1,i] + w[VX,k,j,i]) * tx +
                               (w[VY,k,j-1,i] + w[VY,k,j,i]) * ty +
                               (w[VZ,k,j-1,i] + w[VZ,k,j,i]) * tz )

          # Z flux

          tx = ( dzi * (w[VX,k,j,i] - w[VX,k-1,j,i])
                       + 0.25*dxi * (w[VZ,k-1,j,i+1] - w[VZ,k-1,j,i-1]
                                   + w[VZ,k,j,i+1] -   w[VZ,k,j,i-1]) )
          ty = ( dzi * (w[VY,k,j,i] - w[VY,k-1,j,i])
                       + 0.25*dyi * (w[VZ,k-1,j+1,i] - w[VZ,k-1,j-1,i]
                                   + w[VZ,k,j+1,i]   - w[VZ,k,j-1,i]) )
          tz = 2.*dzi * (w[VZ,k,j,i] - w[VZ,k-1,j,i]) - (div[k-1,j,i] + div[k,j,i])

          flx[2,k,j,i] = 0.5*( (w[VX,k-1,j,i] + w[VX,k,j,i]) * tx +
                               (w[VY,k-1,j,i] + w[VY,k,j,i]) * ty +
                               (w[VZ,k-1,j,i] + w[VZ,k,j,i]) * tz )

  ELIF D2D:

    for k in range(k1, k2+2):#, nogil=True,num_threads=OMP_NT, schedule='dynamic'):
      for j in range(j1, j2+2):
        for i in range(i1, i2+1):

          # X flux

          tx = 2.*dxi * (w[VX,k,j,i] - w[VX,k,j,i-1]) - (div[k,j,i-1] + div[k,j,i])
          ty =  ( dxi * (w[VY,k,j,i] - w[VY,k,j,i-1])
                  + 0.25*dyi * (w[VX,k,j+1,i-1]- w[VX,k,j-1,i-1]
                              + w[VX,k,j+1,i] -  w[VX,k,j-1,i]) )
          tz =   dxi * (w[VZ,k,j,i] - w[VZ,k,j,i-1])

          flx[0,k,j,i] = 0.5*( (w[VX,k,j,i-1] + w[VX,k,j,i]) * tx +
                               (w[VY,k,j,i-1] + w[VY,k,j,i]) * ty +
                               (w[VZ,k,j,i-1] + w[VZ,k,j,i]) * tz )

          # Y flux

          tx = ( dyi * (w[VX,k,j,i] - w[VX,k,j-1,i])
                    + 0.25*dxi * (w[VY,k,j-1,i+1]- w[VY,k,j-1,i-1]
                                + w[VY,k,j,i+1] -  w[VY,k,j,i-1]) )
          ty = 2.*dyi * (w[VY,k,j,i] - w[VY,k,j-1,i]) - (div[k,j-1,i] + div[k,j,i])
          tz =    dyi * (w[VZ,k,j,i] - w[VZ,k,j-1,i])

          flx[1,k,j,i] = 0.5*( (w[VX,k,j-1,i] + w[VX,k,j,i]) * tx +
                               (w[VY,k,j-1,i] + w[VY,k,j,i]) * ty +
                               (w[VZ,k,j-1,i] + w[VZ,k,j,i]) * tz )

  ELSE:

    for k in range(k1, k2+2):#, nogil=True, num_threads=OMP_NT, schedule='dynamic'):
      for j in range(j1, j2+1):
        for i in range(i1, i2+1):

          # X flux

          tx = 2.*dxi * (w[VX,k,j,i] - w[VX,k,j,i-1]) - (div[k,j,i-1] + div[k,j,i])
          ty =    dxi * (w[VY,k,j,i] - w[VY,k,j,i-1])
          tz =    dxi * (w[VZ,k,j,i] - w[VZ,k,j,i-1])

          flx[0,k,j,i] = 0.5*( (w[VX,k,j,i-1] + w[VX,k,j,i]) * tx +
                               (w[VY,k,j,i-1] + w[VY,k,j,i]) * ty +
                               (w[VZ,k,j,i-1] + w[VZ,k,j,i]) * tz )

  for k in range(k1, k2+1):
    for j in range(j1, j2+1):
      for i in range(i1, i2+1):

        de = dxi * (flx[0,k,j,i+1] - flx[0,k,j,i])
        IF D2D: de = de + dyi * (flx[1,k,j+1,i] - flx[1,k,j,i])
        IF D3D: de = de + dzi * (flx[2,k+1,j,i] - flx[2,k,j,i])
        de = dtmu * de

        ek0 = 0.5*(SQR(vel0[0,k,j,i]) + SQR(vel0[1, k,j,i]) + SQR(vel0[2, k,j,i])) * w[RHO,k,j,i]
        dek = 0.5*(SQR(  w[VX,k,j,i]) + SQR(   w[VY,k,j,i]) + SQR(   w[VZ,k,j,i])) * w[RHO,k,j,i] - ek0
        dp = gamm1 * (de - dek)

        w[PR,k,j,i] = w[PR,k,j,i] + dp
        IF CGL: w[PPD,k,j,i] = w[PPD,k,j,i] + dp


# -----------------------------------------------------------------------------

cdef real get_dt(real4d w, GridCoord *gc, BnzDiffusion diff):

  # Calculate diffusive timestep.

  cdef:
    int i,j,k, n
    int id
    real rhoi, ad, ad_dl2, ad_dl2_max = 0.
    # double dvx=0, dvy=0, dvz=0, dvi=0

  cdef:
    real ad_dl2_max_loc[OMP_NT]
    real *dxfi = gc.dlf_inv[0]
    real *dyfi = gc.dlf_inv[1]
    real *dzfi = gc.dlf_inv[2]

  IF MPI:
    cdef:
      double[::1] var     = np.empty(1, dtype='f8')
      double[::1] var_max = np.empty(1, dtype='f8')

  with nogil, parallel(num_threads=OMP_NT):

    id = threadid()

    for k in prange(gc.k1, gc.k2+1, schedule='dynamic'):
      for j in range(gc.j1, gc.j2+1):
        for i in range(gc.i1, gc.i2+1):

          rhoi = 1./w[RHO,k,j,i]

          for n in range(3):

            ad = diff.mu * rhoi

            # dvx         = w[VX,k,j,i+1] - w[VX,k,j,i-1]
            # IF D2D: dvy = w[VY,k,j+1,i] - w[VY,k,j-1,i]
            # IF D3D: dvz = w[VZ,k+1,j,i] - w[VZ,k-1,j,i]
            #
            # dvi = 1./SQRT(dvx*dvx + dvy*dvy + dvz*dvz + SMALL_NUM)

            ad_dl2 = SQR(dxfi[i]) * ad #* FABS(dvx*dvi)
            IF D2D: ad_dl2 = ad_dl2 + SQR(dyfi[j]) * ad #* FABS(dvy*dvi)
            IF D3D: ad_dl2 = ad_dl2 + SQR(dzfi[k]) * ad #* FABS(dvz*dvi)

            if ad_dl2 > ad_dl2_max_loc[id]: ad_dl2_max_loc[id] = ad_dl2

  for i in range(OMP_NT):
    if ad_dl2_max_loc[i] > ad_dl2_max: ad_dl2_max = ad_dl2_max_loc[i]

  IF MPI:
    var[0] = ad_dl2_max
    mpi.COMM_WORLD.Allreduce(var, var_max, op=mpi.MAX)
    ad_dl2_max = var_max[0]

  return diff.cour_diff / ad_dl2_max
