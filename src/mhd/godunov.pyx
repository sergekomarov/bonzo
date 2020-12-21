# -*- coding: utf-8 -*-

import numpy as np
cimport numpy as np
from cython.parallel import prange, parallel, threadid

from libc.math cimport isnan
from libc.stdlib cimport calloc, free
from libc.stdio cimport printf

from bnz.util cimport print_root, timediff
from bnz.util cimport calloc_2d_array, free_2d_array, swap_array_ptrs

from bnz.coordinates.coord cimport get_cell_vol, get_face_area_x
from bnz.coordinates.coord cimport get_face_area_y, get_face_area_z
from bnz.integrate.integrator cimport reconstr_const, reconstr_linear, reconstr_weno, reconstr_parab
from bnz.integrate.integrator cimport ReconstrFunc


cdef void godunov_fluxes(real4d flx_x, real4d flx_y, real4d flx_z,
                         real4d prim, real4d bfld, GridCoord *gc, int *lims,
                         BnzIntegr integr, int order):

  # Calculate Godunov fluxes from conserved variables.

  cdef:
    int i,j,k,n,d, n1,d1
    timeval tstart, tstart_step, tstop

  cdef:
    # array of primitive variables used as input to reconstruction
    real ***prim1
    # Godunov flux
    real **flx1
    # interface magnetic field
    real *bfld1
    # left/right interface states
    real **wl
    real **wl_
    real **wr
    # scratch array used in reconstruction
    real ***rcn_scr
  cdef:
    int il,iu, jl,ju, kl,ku
    int ib,ie, jb,je, kb,ke

  # these are the limits of the resulting hydro array after it will have been
  # advanced in time with the fluxes
  il,iu, jl,ju, kl,ku = lims[0],lims[1], lims[2],lims[3], lims[4],lims[5]

  cdef:
    # swap velocity and magnetic field components in cyclic order
    int1d varsx = swap_mhd_vec(XAX)
    int1d varsy = swap_mhd_vec(YAX)
    int1d varsz = swap_mhd_vec(ZAX)

  # set reconstruction function based on the selected method and order
  cdef ReconstrFunc reconstr_func = get_reconstr_func(order, integr)

  cdef:
    int thread_id
    int nt = OMP_NT
  IF not D2D and not D3D: nt = 1  # don't use OpenMP in 1D


  with nogil, parallel(num_threads=nt):

    thread_id = threadid()

    wl  = integr.scratch.wl[thread_id]
    wl_ = integr.scratch.wl_[thread_id]
    wr  = integr.scratch.wr[thread_id]
    rcn_scr = integr.scratch.w_rcn[thread_id]

    prim1 = <real ***>calloc_2d_array(2*order+1, NMODE, sizeof(real*))
    flx1 = <real **>calloc(NMODE, sizeof(real*))
    bfld1 = NULL

    # print_root(rank, "\nGodunov fluxes along X... ")
    # gettimeofday(&tstart, NULL)

    # Godunov fluxes in x-direction.

    kb=kl-1
    ke=ku+1

    jb=jl-1
    je=ju+1

    IF not D2D: jb,je=0,0
    IF not D3D: kb,ke=0,0

    for k in prange(kb,ke+1, schedule='dynamic'):
      for j in range(jb,je+1):

        # shallow slice of the array of primitive variables
        # 'order' sets the size of the interpolation stencil = 2*order+1
        for d in range(2*order+1):
          for n in range(NMODE):
            prim1[d][n] = &prim[varsx[n],k,j,0] + d-order

        # reconstruction sets wl[i+1], wr[i] => need to start from il-1
        reconstr_func(wl, wr, prim1, rcn_scr, gc,
                      XAX, il-1,iu+1, j,k,
                      integr.char_proj, integr.gam)

        # shallow slice of x-flux array
        for n in range(NMODE): flx1[n] = &flx_x[varsx[n],k,j,0]

        # calulate Godunov x-flux
        IF MFIELD: bfld1 = &(bfld[0,k,j,0])
        integr.rsolver_func(flx1, wl,wr, bfld1, il,iu+1, integr.gam)

    # gettimeofday(&tstop, NULL)
    # print_root(rank, "%.1f ms\n", timediff(tstart,tstop))


    IF D2D:

      # Godunov fluxes in y-direction.

      ib=il-1
      ie=iu+1

      kb=kl-1
      ke=ku+1
      IF not D3D: kb,ke=0,0

      # print_root(rank, "Godunov fluxes along Y... ")
      # gettimeofday(&tstart, NULL)

      for k in prange(kb,ke+1, schedule='dynamic'):

        # reconstruction of y-interfaces along x-direction
        # sets wl[j+1], wr[j] => need to calculate jl-1 first

        for d in range(2*order+1):
          for n in range(NMODE):
            prim1[d][n] = &prim[varsy[n], k, jl-1 + d-order, 0]

        reconstr_func(wl, wr, prim1, rcn_scr, gc,
                      YAX, ib,ie, jl-1,k,
                      integr.char_proj, integr.gam)

        for j in range(jl,ju+2):

          for d in range(2*order+1):
            for n in range(NMODE):
              prim1[d][n] = &prim[varsy[n], k, j+d-order, 0]

          reconstr_func(wl_, wr, prim1, rcn_scr,
                        YAX, ib,ie, j,k,
                        integr.char_proj, integr.gam)

          for n in range(NMODE): flx1[n] = &flx_y[varsy[n],k,j,0]

          IF MFIELD: bfld1 = &(bfld[1,k,j,0])
          integr.rsolver_func(flx1, wl,wr, bfld1, ib,ie, integr.gam)

          swap_array_ptrs(wl,wl_)

      # gettimeofday(&tstop, NULL)
      # print_root(rank, "%.1f ms\n", timediff(tstart,tstop))


    IF D3D:

      # Godunov fluxes in z-direction.

      ib=il-1
      ie=iu+1

      jb=jl-1
      je=ju+1

      # print_root(rank, "Godunov fluxes along Z... ")
      # gettimeofday(&tstart, NULL)

      for j in prange(jb,je+1, schedule='dynamic'):

        for d in range(2*order+1):
          for n in range(NMODE):
            prim1[d][n] = &prim[varsz[n], kl-1 + d-order, j, 0]

        reconstr_func(wl, wr, prim1, rcn_scr, gc,
                      ZAX, ib,ie, j,kl-1,
                      integr.char_proj, integr.gam)

        for k in range(kl,ku+2):

          for d in range(2*order+1):
            for n in range(NMODE):
              prim1[d][n] = &prim[varsz[n], k+d-order, j, 0]

          reconstr_func(wl_,wr, prim1, rcn_scr, gc,
                        ZAX, ib,ie, j,k,
                        integr.char_proj, integr.gam)

          for n in range(NMODE): flx1[n] = &flx_z[varsz[n],k,j,0]

          IF MFIELD: bfld1 = &(bfld[2,k,j,0])
          integr.rsolver_func(flx1, wl,wr, bfld1, ib,ie, integr.gam)

          swap_array_ptrs(wl,wl_)

        # gettimeofday(&tstop, NULL)
        # print_root(rank, "%.1f ms\n", timediff(tstart,tstop))

      free_2d_array(prim1)
      free(flx1)


# ----------------------------------------------------------------------------

cdef void advance_hydro(real4d u1, real4d u0, real4d fx, real4d fy, real4d fz,
                        GridCoord *gc, int *lims, real dt) nogil:

  # Advance cell-centered hydrodynamic variables using Godunov fluxes.

  cdef:
    int i,j,k,n
    real dtdv, dam,dap

  cdef int Nvar_hydro
  IF MFIELD: Nvar_hydro = NMODE-3
  ELSE: Nvar_hydro = NMODE

  for n in range(Nvar_hydro):
    for k in range(lims[4],lims[5]+1):
      for j in range(lims[2],lims[3]+1):
        for i in range(lims[0],lims[1]+1):

          dtdv = dt/get_cell_vol(gc,i,j,k)

          dam = get_face_area_x(gc,i,  j,k)
          dap = get_face_area_x(gc,i+1,j,k)

          u1[n,k,j,i] = u0[n,k,j,i] - dtdv * (dap * fx[n,k,j,i+1]
                                            - dam * fx[n,k,j,i])
          IF D2D:

            dam = get_face_area_y(gc,i,j,  k)
            dap = get_face_area_y(gc,i,j+1,k)

            u1[n,k,j,i] -= dtdv * (dap * fy[n,k,j+1,i] - dam * fy[n,k,j,i])

          IF D3D:

            dam = get_face_area_z(gc,i,j,k)
            dap = get_face_area_z(gc,i,j,k+1)

            u1[n,k,j,i] -= dtdv * (dap * fz[n,k+1,j,i] - dam * fz[n,k,j,i])



# --------------------------------------------------------------------

# cdef void apply_pressure_floor(real4d U, int lims[6],
#                                double p_flr, double gam) nogil:
#
#   cdef:
#     int i,j,k
#     real rhoi, p,pe, ek, em=0.
#     real gamm1 = gam-1, gamm1i = 1./gamm1
#
#   for k in range(lims[4],lims[5]+1):
#     for j in range(lims[2],lims[3]+1):
#       for i in range(lims[0],lims[1]+1):
#
#         if U[RHO,k,j,i]<0.:
#           printf('rho = %f; apply density floor\n', U[RHO,k,j,i])
#           U[RHO,k,j,i] = p_flr
#
#         rhoi = 1. / U[RHO,k,j,i]
#
#         ek = 0.5*rhoi * (U[MX,k,j,i]**2 + U[MY,k,j,i]**2 + U[MZ,k,j,i]**2)
#         IF MFIELD: em = 0.5 * (U[BX,k,j,i]**2 + U[BY,k,j,i]**2 + U[BZ,k,j,i]**2)
#
#         p = gamm1 * (U[EN,k,j,i] - ek - em)
#
#         IF TWOTEMP:
#           pe = pow(U[RHO,k,j,i], gam) * exp(U[SE,k,j,i] * rhoi)
#           p = p - pe
#
#         if p<0.:
#           printf('p = %f; apply pressure floor\n', p)
#           p = p_flr
#
#         IF TWOTEMP: p = p + pe
#
#         U[EN,k,j,i] = gamm1i * p + ek + em



# ---------------------------------------------------------------------------

cdef int1d swap_mhd_vec(int ax):

  cdef:
    int n
    int Nvar_hydro
    int[::1] vars=np.zeros(NMODE,np.intp)

  IF MFIELD: Nvar_hydro = NMODE-3
  ELSE: Nvar_hydro = NMODE


  vars[RHO] = RHO
  for n in range(4,Nvar_hydro):
    vars[n] = n

  if ax==0:
    vars[VX], vars[VY], vars[VZ] = VX,VY,VZ
    IF MFIELD: vars[BX], vars[BY], vars[BZ] = BX,BY,BZ

  elif ax==1:
    vars[VX], vars[VY], vars[VZ] = VY,VZ,VX
    IF MFIELD: vars[BX], vars[BY], vars[BZ] = BY,BZ,BX

  elif ax==2:
    vars[VX], vars[VY], vars[VZ] = VZ,VX,VY
    IF MFIELD: vars[BX], vars[BY], vars[BZ] = BZ,BX,BY

  return vars


# --------------------------------------------------------------------

cdef ReconstrFunc get_reconstr_func(int order, BnzIntegr integr) nogil:

  if order==0:
    return reconstr_const

  elif order==1:

    if integr.reconstr_order==1:
      return integr.reconstr_func
    else:
      # default first-order reconstruction
      return reconstr_linear

  elif order==2:
    return integr.reconstr_func
