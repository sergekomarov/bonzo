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
from bnz.coordinates.coord cimport get_cell_width_x, get_cell_width_y, get_cell_width_z


cdef void diffuse(BnzDiffusion diff, BnzGrid grid, real dt):
  # only need integr to apply BCs

  # Evolve electron temperature by super-time-stepping.

  cdef:
    GridCoord *gc = grid.coord
    real4d prim = grid.data.prim
    real4d bfld = grid.data.bfld
    BnzIntegr integr = diff.integr
    DiffScratch scr = diff.scratch

    real dt_hyp = dt
    real dt_diff
    real a
    int k,j,i, n, s

  cdef int il,iu, jl,ju, kl,ku

  il, iu = 1, gc.Ntot[0]-2
  IF D2D: jl, ju = 1, gc.Ntot[1]-2
  ELSE: jl, ju = 0, 0
  IF D3D: kl, ku = 1, gc.Ntot[2]-2
  ELSE: kl, ku = 0, 0

  cdef:
    # scratch STS-arrays
    real3d pres0=scr.temp0, presm1=scr.tempm1
    real3d dtemp0=scr.dtemp0, dtemp=scr.dtemp
    real2d sts_coeff

  # which pressure variable to update
  cdef int PIND
  IF TWOTEMP: PIND=PE
  ELSE: PIND=PR

  # primitive vars have already been calculated for all diffusion processes

  # apply BC to all variables (including face-centered magnetic field)
  grid.apply_grid_bc(integr, np.arange(NVAR))

  # calculate diffusive time step
  dt_diff = get_dt(prim, gc, diff)

  # calculate number of STS iterations
  s = diff.get_nsts(dt_hyp, dt_diff)
  print_root("\nelectron thermal conduction, N_STS=%d ... ", s)

  # calculate STS coefficients
  sts_coeff = diff.get_sts_coeff(s)

  # do STS iterations
  for n in range(1,s+1):

    # apply diffusion operator to temperature (the result is written to dtemp)
    apply_diff_oper(dtemp, prim,bfld, gc, diff)

    # save initial pressure and mtemp
    if n==1:
      for k in range(kl,ku+1):
        for j in range(jl,ju+1):
          for i in range(il,iu+1):

            pres0[k,j,i] = prim[PIND,k,j,i]
            dtemp0[k,j,i] = dtemp[k,j,i]

    # do STS iteration
    for k in range(kl,ku+1):
      for j in range(jl,ju+1):
        for i in range(il,iu+1):

          a = prim[PIND,k,j,i]
          prim[PIND,k,j,i] =       ( sts_coeff[n,MU]   *   prim[PIND,k,j,i] +
                                     sts_coeff[n,NU]   * presm1[k,j,i]
           + (1. - sts_coeff[n,MU] - sts_coeff[n,NU])  *  pres0[k,j,i]
                         + dt_hyp * (sts_coeff[n,MUT]  *  dtemp[k,j,i]
                                   + sts_coeff[n,GAMT] * dtemp0[k,j,i]) )
          presm1[k,j,i] = a

    # need 1 layer of ghost cells to do 1 iteration
    if n%gc.ng==0 and n!=s:
      # only apply BC to electron pressure
      grid.apply_grid_bc(integr, np.asarray([PIND]))

    # for k in range(kl,ku+1):
    #   for j in range(jl,ju+1):
    #     for i in range(il,iu+1):
    #       if prim[PIND,k,j,i]<0.:
    #         print('P={} at {} {} {}, step n={}'.format(prim[PIND,k,j,i],i,j,k,n))
            # prim[PIND,k,j,i] = 1e-2

  # update perpendicular ion pressure if needed
  IF CGL and not TWOTEMP:
    for k in range(gc.k1, gc.k2+1):
      for j in range(gc.j1, gc.j2+1):
        for i in range(gc.i1, gc.i2+1):
          prim[PPD,k,j,i] = prim[PPD,k,j,i] + prim[PIND,k,j,i]-pres0[k,j,i]


# --------------------------------------------------------------------------

cdef void apply_diff_oper(real3d dtemp, real4d prim, real4d bfld,
                          GridCoord *gc, BnzDiffusion diff):

  # Apply thermal conduction matrix operator to temperature.

  cdef:
    int i,j,k
    real bxh,byh,bzh,bhi, bxh1,byh1,bzh1
    real dTdx,dTdy,dTdz, dTds, kappah
    real h, Tl,Tr,Tc,rhoc,mom,qsat
    real a

  cdef:
    DiffScratch scr = diff.scratch
    real3d temp = scr.temp
    real3d fx=scr.fx_diff, fy=scr.fy_diff, fz=scr.fz_diff
    real3d kappa = scr.kappa_par
    real3d kappa_mag = scr.kappa_mag

  cdef int il,iu, jl,ju, kl,ku

  il, iu = 1, gc.Ntot[0]-2
  IF D2D: jl, ju = 1, gc.Ntot[1]-2
  ELSE: jl, ju = 0, 0
  IF D3D: kl, ku = 1, gc.Ntot[2]-2
  ELSE: kl, ku = 0, 0

  cdef real gamm1 = diff.gam-1.


  for k in range(gc.Ntot[2]):
    for j in range(gc.Ntot[1]):
      for i in range(gc.Ntot[0]):

        temp[k,j,i] = prim[PR,k,j,i] / prim[RHO,k,j,i]
        kappa[k,j,i] = diff.kappa0 #* POW(temp[k,j,i], 2.5)

  for k in range(kl, ku+1):#, nogil=True, num_threads=OMP_NT, schedule='dynamic'):
    dTdx,dTdy,dTdz,dTds = 0.,0.,0.,0.
    for j in range(jl, ju+1):
      for i in range(il, iu+2):

        kappah = ( 2. * kappa[k,j,i-1] * kappa[k,j,i]
                     / (kappa[k,j,i-1] + kappa[k,j,i] + SMALL_NUM) )

        dTdx = gc.dlv_inv[0][i] * (temp[k,j,i]-temp[k,j,i-1])

        if diff.thcond_type==TC_ANISO:

          IF MFIELD:

            bxh = bfld[0,k,j,i]
            byh = 0.5*(prim[BY,k,j,i-1] + prim[BY,k,j,i])
            bzh = 0.5*(prim[BZ,k,j,i-1] + prim[BZ,k,j,i])
            bhi = 1./SQRT(bxh*bxh + byh*byh + bzh*bzh + SMALL_NUM)
            bxh1,byh1,bzh1 = bxh*bhi, byh*bhi, bzh*bhi

            # cosine of angle between heat flux and X
            h = FABS(bxh1)

            IF D2D: dTdy = utdiff.tr_grad_lim_yx(temp, i,j,k, gc)
            IF D3D: dTdz = utdiff.tr_grad_lim_zx(temp, i,j,k, gc)

            fx[k,j,i] = - kappah * bxh1 * (bxh1 * dTdx + byh1 * dTdy + bzh1 * dTdz)

            # printf('%f %f %f %f %f %f\n', bxh1,byh1,bzh1, dtdy,dtdz, fx[k,j,i])

          ELSE: pass

        elif diff.thcond_type==TC_ISO:

          IF D2D: dTdy = utdiff.tr_grad_yx(temp, i,j,k, gc)
          IF D3D: dTdz = utdiff.tr_grad_zx(temp, i,j,k, gc)
          dTds = SQRT(dTdx*dTdx + dTdy*dTdy + dTdz*dTdz + SMALL_NUM)

          # cosine of angle between heat flux and X
          h = FABS(dTdx / dTds)

          fx[k,j,i] = - kappah * dTdx

        # saturation

        if diff.sat_hfe:

          Tl,Tr = temp[k,j,i-1],temp[k,j,i]

          Tc = 0.5*(Tl+Tr)
          rhoc = 0.5*(prim[RHO,k,j,i-1]+prim[RHO,k,j,i])

          mom = 0.5*h * rhoc * SQRT(0.67*Tc)

          qsat = Tl * mom if fx[k,j,i]>0. else Tr * mom

          fx[k,j,i] = fx[k,j,i] * qsat / (FABS(fx[k,j,i]) + qsat + SMALL_NUM)


  IF D2D:

    for k in range(kl, ku+1):#, nogil=True, num_threads=OMP_NT, schedule='dynamic'):
      dTdx,dTdy,dTdz,dTds = 0.,0.,0.,0.
      for j in range(jl, ju+2):
        for i in range(il, iu+1):

          kappah = ( 2. * kappa[k,j-1,i] * kappa[k,j,i]
                       / (kappa[k,j-1,i] + kappa[k,j,i] + SMALL_NUM) )

          dTdy = gc.dlv_inv[1][j] * (temp[k,j,i]-temp[k,j-1,i])

          if diff.thcond_type==TC_ANISO:

            IF MFIELD:

              byh = bfld[1,k,j,i]
              bxh = 0.5*(prim[BX,k,j-1,i] + prim[BX,k,j,i])
              bzh = 0.5*(prim[BZ,k,j-1,i] + prim[BZ,k,j,i])
              bhi = 1./SQRT(bxh*bxh + byh*byh + bzh*bzh + SMALL_NUM)
              bxh1,byh1,bzh1 = bxh*bhi, byh*bhi, bzh*bhi

              # cosine of angle between heat flux and X
              h = FABS(byh1)

              dTdx = utdiff.tr_grad_lim_xy(temp, i,j,k, gc)
              IF D3D: dTdz = utdiff.tr_grad_lim_zy(temp, i,j,k, gc)

              fy[k,j,i] = - kappah * byh1 * (bxh1*dTdx + byh1*dTdy + bzh1*dTdz)

            ELSE: pass

          elif diff.thcond_type==TC_ISO:

            dTdx = utdiff.tr_grad_xy(temp, i,j,k, gc)
            IF D3D: dTdz = utdiff.tr_grad_zy(temp, i,j,k, gc)
            dTds = SQRT(dTdx*dTdx + dTdy*dTdy + dTdz*dTdz + SMALL_NUM)

            # cosine of angle between heat flux and X
            h = FABS(dTdy / dTds)

            fy[k,j,i] = - kappah * dTdy

          # saturation

          if diff.sat_hfe:

            Tr = temp[k,j,i]
            Tl = temp[k,j-1,i]

            Tc = 0.5*(Tl+Tr)
            rhoc = 0.5*(prim[RHO,k,j-1,i] + prim[RHO,k,j,i])

            mom = 0.5*h * rhoc * SQRT(0.67*Tc)

            qsat = Tl * mom if fy[k,j,i]>0. else Tr * mom

            fy[k,j,i] = fy[k,j,i] * qsat / (FABS(fy[k,j,i]) + qsat + SMALL_NUM)


  IF D3D:

    for k in range(kl, ku+2):#, nogil=True, num_threads=OMP_NT, schedule='dynamic'):
      dTdx,dTdy,dTdz,dTdr = 0.,0.,0.,0.
      for j in range(jl, ju+1):
        for i in range(il, iu+1):

          kappah = ( 2. * kappa[k-1,j,i] * kappa[k,j,i]
                       / (kappa[k-1,j,i] + kappa[k,j,i] + SMALL_NUM) )

          dTdz = gc.dlv_inv[2][k] * (temp[k,j,i]-temp[k-1,j,i])

          if diff.thcond_type==TC_ANISO:

            IF MFIELD:

              bzh = bfld[2,k,j,i]
              bxh = 0.5*(prim[BX,k-1,j,i] + prim[BX,k,j,i])
              byh = 0.5*(prim[BY,k-1,j,i] + prim[BY,k,j,i])
              bhi = 1./SQRT(bxh*bxh + byh*byh + bzh*bzh + SMALL_NUM)
              bxh1,byh1,bzh1 = bxh*bhi, byh*bhi, bzh*bhi

              # cosine of angle between heat flux and X
              h = FABS(bzh1)

              dTdx = utdiff.tr_grad_lim_xz(temp, i,j,k, gc)
              IF D2D: dTdy = utdiff.tr_grad_lim_yz(temp, i,j,k, gc)

              fz[k,j,i] = - kappah * bzh1 * (bxh1*dTdx + byh1*dTdy + bzh1*dTdz)

            ELSE: pass

          elif diff.thcond_type==TC_ISO:

            dTdx = utdiff.tr_grad_xz(temp, i,j,k, gc)
            IF D2D: dTdy = utdiff.tr_grad_yz(temp, i,j,k, gc)
            dTds = SQRT(dTdx*dTdx + dTdy*dTdy + dTdz*dTdz + SMALL_NUM)

            # cosine of angle between heat flux and X
            h = FABS(dTdz / dTds)

            fz[k,j,i] = - kappah * dTdz

          # saturation

          if diff.sat_hfe:

            Tl = temp[k-1,j,i]
            Tr = temp[k,j,i]
            Tc = 0.5*(Tl+Tr)
            rhoc = 0.5*(prim[RHO,k-1,j,i] + prim[RHO,k,j,i])
            mom = 0.5*h * rhoc * SQRT(0.67*Tc)

            qsat = Tl * mom if fz[k,j,i]>0. else Tr * mom

            fz[k,j,i] = fz[k,j,i] * qsat / (FABS(fz[k,j,i]) + qsat + SMALL_NUM)


  for k in range(kl, ku+1):
    for j in range(jl, ju+1):
      for i in range(il, iu+1):

        dtemp[k,j,i] = - (fx[k,j,i+1] - fx[k,j,i]) * gamm1 * gc.dlf_inv[0][i]
        IF D2D: dtemp[k,j,i] -= (fy[k,j+1,i] - fy[k,j,i]) * gamm1 * gc.dlf_inv[1][j]
        IF D3D: dtemp[k,j,i] -= (fz[k+1,j,i] - fz[k,j,i]) * gamm1 * gc.dlf_inv[2][k]


# cdef void set_thconductivity(real3d kappa, real4d W, GridParams gp):
#
#   cdef:
#     int i,j,k
#
#   for k in range(gc.Ntot[2]):
#     for j in range(gc.Ntot[1]):
#       for i in range(gc.Ntot[0]):
#         kappa[k,j,i] =



# ------------------------------------------------------------------------------

cdef real get_dt(real4d prim, GridCoord *gc, BnzDiffusion diff):

  # Calculate diffusive timestep.

  cdef:
    int k,j,i
    int id
    # real dTdx,dTdy,dTdz,dTdsi, b2i
    real ad, ad_dl2, ad_dl2_max = 0.
    real dsx,dsy,dsz

  cdef:
    real gamm1 = diff.gam-1.
    real T1, rho1, _temp

  cdef:
    # real3d temp = diff.scratch.temp
    real d_dl2_max_loc[OMP_NT]

    # real *dxvi = gc.dlv_inv[0]
    # real *dyvi = gc.dlv_inv[1]
    # real *dzvi = gc.dlv_inv[2]

  IF MPI:
    cdef:
      double[::1] var     = np.empty(1, dtype='f8')
      double[::1] var_max = np.empty(1, dtype='f8')


  with nogil, parallel(num_threads=OMP_NT):

  #   for k in prange(gc.Ntot[2], schedule='dynamic'):
  #     for j in range(gc.Ntot[1]):
  #       for i in range(gc.Ntot[0]):
  #
  #         IF not TWOTEMP: temp[k,j,i] = prim[PR,k,j,i] / prim[RHO,k,j,i]
  #         ELSE: temp[k,j,i] = prim[PE,k,j,i] / prim[RHO,k,j,i]

    id = threadid()

    for k in range(gc.k1, gc.k2+1):#, schedule='dynamic'):
      # dTdx,dTdy,dTdz,dTdsi = 0,0,0,0
      for j in range(gc.j1, gc.j2+1):
        for i in range(gc.i1, gc.i2+1):

          # if prim[RHO,k,j,i] < 0.8: rho1 = 0.8
          # else: rho1 = prim[RHO,k,j,i]
          # if temp[k,j,i] > 1.6: T1 = 1.6
          # else: T1 = temp[k,j,i]
          #
          # ad = diff.kappa0 / rho1 * gamm1 * SQR(T1) * SQRT(FABS(T1))

          # if diff.scratch.kappa_mag[k,j,i]<0.:
          #   ad = 1e-20
          # else:
          #   ad = diff.kappa0 / prim[RHO,k,j,i] * gamm1 * SQR(temp[k,j,i]) * SQRT(FABS(temp[k,j,i]))

          # if prim[RHO,k,j,i] < 0.8 or temp[k,j,i] > 1.6: ad=SMALL_NUM

          dsx = get_cell_width_x(gc,i,j,k)
          dsy = get_cell_width_y(gc,i,j,k)
          dsz = get_cell_width_z(gc,i,j,k)

          IF not TWOTEMP: _temp = prim[PR,k,j,i] / prim[RHO,k,j,i]
          ELSE: _temp = prim[PE,k,j,i] / prim[RHO,k,j,i]

          ad = diff.kappa0 / prim[RHO,k,j,i] * gamm1 #* SQR(_temp) * SQRT(FABS(_temp))

          ad_dl2 = ad/SQR(dsx) #* FABS(dTdx*dTdsi)
          IF D2D: ad_dl2 = ad_dl2 + ad/SQR(dsy) #* FABS(dTdy*dTdsi)
          IF D3D: ad_dl2 = ad_dl2 + ad/SQR(dsz) #* FABS(dTdz*dTdsi)

          if ad_dl2 > ad_dl2_max_loc[id]: ad_dl2_max_loc[id] = ad_dl2

          # dTdx   = 0.5*((temp[k,j,i+1] - temp[k,j,i]  ) * dxvi[i+1]
          #             + (temp[k,j,i]   - temp[k,j,i-1]) * dxvi[i])
          # IF D2D:
          #   dTdy = 0.5*((temp[k,j+1,i] - temp[k,j,i]  ) * dyvi[j+1]
          #             + (temp[k,j,i]   - temp[k,j-1,i]) * dyvi[j]  ) * gc.syxv[i]
          # IF D3D:
          #   dTdz = 0.5*((temp[k+1,j,i] - temp[k,j,i]  ) * dzvi[k+1]
          #             + (temp[k,j,i]   - temp[k-1,j,i]) * dzvi[k]  ) * gc.szxv[i] * gc.szyv[j]
          #
          # dTdsi = 1. / SQRT(SQR(dTdx) + SQR(dTdy) + SQR(dTdz) + SMALL_NUM)

          # if diff.thcond_type==TC_ANISO:
          #
          #   IF MFIELD:
          #
          #     b2i = 1./(SQR(prim[BX,k,j,i])
          #             + SQR(prim[BY,k,j,i])
          #             + SQR(prim[BZ,k,j,i]) + SMALL_NUM)
          #
          #     ad = ad * FABS(prim[BX,k,j,i] * dTdx + prim[BY,k,j,i] * dTdy
          #                  + prim[BZ,k,j,i] * dTdz) * b2i*dTdsi
          #
          #     ad_dl2 = FABS(prim[BX,k,j,i]) * ad*SQR(dxfi)
          #     IF D2D: ad_dl2 = ad_dl2 + FABS(prim[BY,k,j,i]) * ad*SQR(dyfi)
          #     IF D3D: ad_dl2 = ad_dl2 + FABS(prim[BZ,k,j,i]) * ad*SQR(dzfi)
          #
          #   ELSE: pass


  for i in range(OMP_NT):
    if ad_dl2_max_loc[i] > ad_dl2_max: ad_dl2_max = ad_dl2_max_loc[i]

  IF MPI:
    var[0] = ad_dl2_max
    mpi.COMM_WORLD.Allreduce(var, var_max, op=mpi.MAX)
    ad_dl2_max = var_max[0]

  return diff.cour_diff / ad_dl2_max
