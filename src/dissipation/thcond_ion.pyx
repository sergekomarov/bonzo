# -*- coding: utf-8 -*-

IF MPI:
  from mpi4py import MPI as mpi
  from mpi4py cimport MPI as mpi

import numpy as np
cimport numpy as np
from cython.parallel import prange, parallel, threadid

from libc.math cimport isnan
from libc.stdio cimport printf

from bnz.util cimport print_root
cimport util_diffuse as utdiff


cdef void diffuse(BnzGrid grid, BnzIntegr integr, real dt):
  # only need integr to apply BCs

  # Evolve ion temperatures by super-time-stepping.

  cdef:
    GridCoord *gc = grid.coord
    real4d prim = grid.data.prim
    real4d bfld = grid.data.bfld
    BnzDiffusion diff = integr.diff
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
    # effective ion-ion collision frequency
    real3d nuii_eff = scr.nuii_eff
    # scratch STS-arrays
    real3d pres0 =scr.pres0,  pres_perp0 =scr.presi_perp0
    real3d presm1=scr.presm1, pres_perpm1=scr.presi_perpm1
    real3d dtemp0=scr.dtemp0, dtemp_perp0=scr.dtempi_perp0
    real3d dtemp =scr.dtemp,  dtemp_perp =scr.dtempi_perp
    real2d sts_coeff

  # primitive vars have already been calculated for all diffusion processes

  # apply BC to all variables (including face-centered magnetic field)
  grid.apply_grid_bc(integr, np.arange(NVAR))

  # calculate diffusive time step
  dt_diff = get_dt(prim, gc, diff)

  # calculate number of STS iterations
  s = diff.get_nsts(dt_hyp, dt_diff)
  print_root("\nion thermal conduction, N_STS=%d ... ", s)

  # calculate STS coefficients
  sts_coeff = diff.get_sts_coeff(s)

  # do STS iterations
  for n in range(1,s+1):

    # apply diffusion operator to temperature
    apply_diff_oper(dtemp,dtemp_perp, prim,bfld, gc, diff)

    # save initial T and MT
    if n==1:
      for k in range(kl,ku+1):
        for j in range(jl,ju+1):
          for i in range(il,iu+1):

            pres0[k,j,i] = prim[PR,k,j,i]
            pres_perp0[k,j,i] = prim[PPD,k,j,i]

            dtemp0[k,j,i]= dtemp[k,j,i]
            dtemp_perp0[k,j,i] = dtemp_perp[k,j,i]

    # do STS iteration
    for k in range(kl,ku+1):
      for j in range(jl,ju+1):
        for i in range(il,iu+1):

          a = prim[PR,k,j,i]
          prim[PR,k,j,i] =         ( sts_coeff[n,MU]  * prim[PR,k,j,i] +
                                     sts_coeff[n,NU]   * presm1[k,j,i]
           + (1. - sts_coeff[n,MU] - sts_coeff[n,NU])  *  pres0[k,j,i]
                         + dt_hyp * (sts_coeff[n,MUT]  *  dtemp[k,j,i]
                                   + sts_coeff[n,GAMT] * dtemp0[k,j,i]) )
          presm1[k,j,i] = a

          a = prim[PPD,k,j,i]
          prim[PPD,k,j,i] =        ( sts_coeff[n,MU]   *    prim[PPD,k,j,i] +
                                     sts_coeff[n,NU]   * pres_perpm1[k,j,i]
           + (1. - sts_coeff[n,MU] - sts_coeff[n,NU])  *  pres_perp0[k,j,i]
                         + dt_hyp * (sts_coeff[n,MUT]  *  dtemp_perp[k,j,i]
                                   + sts_coeff[n,GAMT] * dtemp_perp0[k,j,i]) )
          pres_perpm1[k,j,i] = a

    # need 1 layer of ghost cells to do 1 iteration
    if n%gc.ng==0 and n!=s:
      # only apply BC to ion pressures
      grid.apply_grid_bc(integr, np.asarray([PR,PPD]))


# ------------------------------------------------------------------------------

cdef void apply_diff_oper(real3d dtemp, real3d dtemp_perp,
                          real4d prim, real4d bfld,
                          GridCoord *gc, BnzDiffusion diff):

  # Apply ion thermal conduction matrix operator to ion temperatures.

  cdef:
    int i,j,k
    real rhoi
    real Bxh,Byh,Bzh, Bhi, bxh,byh,bzh
    real dTdx_pl, dTdy_pl, dTdz_pl, kappah_pl
    real dTdx_pd, dTdy_pd, dTdz_pd, kappah_pd
    real dBdx, dBdy, dBdz, kappah_m
    real bbgradTpl, bbgradTpd, bbgradB
    real DT, vmag, vmag_tpd
    real rho, cpl,mom, qsat, qsat_pd
    real TL,TR, TLpd,TRpd

  cdef int il,iu, jl,ju, kl,ku

  il, iu = 1, gc.Ntot[0]-2
  IF D2D: jl, ju = 1, gc.Ntot[1]-2
  ELSE: jl, ju = 0, 0
  IF D3D: kl, ku = 1, gc.Ntot[2]-2
  ELSE: kl, ku = 0, 0

  cdef:
    real one3rd=1./3
    real gamm1 = diff.gam-1.
    real a1 = diff.kl * SQRT(0.5*B_PI)
    real a2 = 0.25*(3.*B_PI-8.)

  cdef:
    DiffScratch scr = diff.scratch
    real3d fx = scr.fx_diff
    real3d fy = scr.fy_diff
    real3d fz = scr.fz_diff
    real3d fpdx = scr.fx_diff2
    real3d fpdy = scr.fy_diff2
    real3d fpdz = scr.fz_diff2

    real3d kappa_pl = scr.kappa_par
    real3d kappa_pd = scr.kappa_perp
    real3d kappa_m  = scr.kappa_mag

    real3d Tpl = scr.tempi_par
    real3d Tpd = scr.tempi_perp
    real3d Babs = scr.babs


  for k in range(gc.Ntot[2]):
    for j in range(gc.Ntot[1]):
      for i in range(gc.Ntot[0]):

        rhoi = 1./prim[RHO,k,j,i]

        Tpd[k,j,i] = rhoi * prim[PPD,k,j,i]
        Tpl[k,j,i] = rhoi * (3.*prim[PR,k,j,i] - 2.*prim[PPD,k,j,i])

        Babs[k,j,i] = SQRT(SQR(prim[BX,k,j,i]) + SQR(prim[BY,k,j,i]) + SQR(prim[BZ,k,j,i]))

        kappa_pl[k,j,i]  = prim[RHO,k,j,i] * ( 2. * Tpl[k,j,i]
                     / ( a1 * SQRT(Tpl[k,j,i])
                       + a2 * nuii_eff[k,j,i] ) )
        kappa_m[k,j,i] = prim[RHO,k,j,i] / (a1 * SQRT(Tpl[k,j,i]) + nuii_eff[k,j,i])
        kappa_pd[k,j,i] = Tpl[k,j,i] * kappa_m[k,j,i]


  for k in range(kl, ku+1):#, nogil=True, num_threads=OMP_NT, schedule='dynamic'):
    for j in range(jl, ju+1):
      for i in range(il, iu+2):

        Bxh = bfld[0,k,j,i]
        Byh = 0.5*(prim[BY,k,j,i-1] + prim[BY,k,j,i])
        Bzh = 0.5*(prim[BZ,k,j,i-1] + prim[BZ,k,j,i])
        Bhi = 1. / SQRT(Bxh*Bxh + Byh*Byh + Bzh*Bzh + SMALL_NUM)
        bxh,byh,bzh = Bxh*Bhi, Byh*Bhi, Bzh*Bhi

        #------------------------------------------------------------------

        kappah_pl = ( 2. * kappa_pl[k,j,i-1] * kappa_pl[k,j,i]
                    / (kappa_pl[k,j,i-1] + kappa_pl[k,j,i] + SMALL_NUM) )

        dTdx_pl,dTdy_pl,dTdz_pl = 0.,0.,0.

        dTdx_pl = gc.dlv_inv[0][i] * (Tpl[k,j,i] - Tpl[k,j,i-1])

        IF D2D: dTdy_pl = utdiff.tr_grad_lim_yx(Tpl, i,j,k, gc)
        IF D3D: dTdz_pl = utdiff.tr_grad_lim_zx(Tpl, i,j,k, gc)

        bbgradTpl = bxh * (bxh * dTdx_pl + byh * dTdy_pl + bzh * dTdz_pl)

        #------------------------------------------------------------------

        kappah_pd = ( 2. * kappa_pd[k,j,i-1] * kappa_pd[k,j,i]
                    / (kappa_pd[k,j,i-1] + kappa_pd[k,j,i] + SMALL_NUM) )

        dTdx_pd,dTdy_pd,dTdz_pd = 0.,0.,0.

        dTdx_pd = gc.dlv_inv[0][i] * (Tpd[k,j,i] - Tpd[k,j,i-1])

        IF D2D: dTdy_pd = utdiff.tr_grad_lim_yx(Tpd, i,j,k, gc)
        IF D3D: dTdz_pd = utdiff.tr_grad_lim_zx(Tpd, i,j,k, gc)

        bbgradTpd = bxh * (bxh * dTdx_pd + byh * dTdy_pd + bzh * dTdz_pd)

        #------------------------------------------------------------------

        kappah_m = ( 2. * kappa_m[k,j,i-1] * kappa_m[k,j,i]
                    / (kappa_m[k,j,i-1] + kappa_m[k,j,i] + SMALL_NUM) )

        dBdx,dBdy,dBdz = 0.,0.,0.

        dBdx = gc.dlv_inv[0][i] * (Babs[k,j,i] - Babs[k,j,i-1])

        IF D2D: dBdy = utdiff.tr_grad_lim_yx(Babs, i,j,k, gc)
        IF D3D: dBdz = utdiff.tr_grad_lim_zx(Babs, i,j,k, gc)

        bbgradB = bxh * (bxh * dBdx + byh * dBdy + bzh * dBdz)

        #------------------------------------------------------------------

        TR = one3rd * (2.*Tpd[k,j,i]   + Tpl[k,j,i])
        TL = one3rd * (2.*Tpd[k,j,i-1] + Tpl[k,j,i-1])

        TRpd = Tpd[k,j,i]
        TLpd = Tpd[k,j,i-1]

        #-----------------------------------------------------------------------
        # take into account that the heat flux caused by magnetic field gradient
        # is advective relative to perpendicular temperature

        DT = 1.5*(TL - TLpd + TR - TRpd)
        vmag = bbgradB * Bhi * kappah_m * DT
        vmag_tpd = vmag * TLpd if vmag > 0. else vmag * TRpd

        # heat flux of total ion thermal energy
        fx[k,j,i] = - (kappah_pd * bbgradTpd + 0.5 * kappah_pl * bbgradTpl) + vmag_tpd

        # heat flux of magnetic moment (p_perp/B)
        fpdx[k,j,i] = (- kappah_pd * bbgradTpd + vmag_tpd) * Bhi

        #------------------------------------------------------------------
        # saturation

        if diff.sat_hfi:

          cpl = SQRT(1.5*(TL + TR) - (TLpd + TRpd))
          rho = 0.5*(prim[RHO,k,j,i-1] + prim[RHO,k,j,i])
          mom = 0.3 * FABS(bxh) * rho * cpl

          qsat    = TL*mom       if fx[k,j,i]  >0. else TR*mom
          qsat_pd = TLpd*Bhi*mom if fpdx[k,j,i]>0. else TRpd*Bhi*mom

          fx[k,j,i]   =   fx[k,j,i] * qsat    / (FABS(fx[k,j,i])   + qsat    + SMALL_NUM)
          fpdx[k,j,i] = fpdx[k,j,i] * qsat_pd / (FABS(fpdx[k,j,i]) + qsat_pd + SMALL_NUM)


  IF D2D:

    for k in range(kl, ku+1):#, nogil=True, num_threads=OMP_NT, schedule='dynamic'):
      for j in range(jl, ju+2):
        for i in range(il, iu+1):

          Byh = bfld[1,k,j,i]
          Bxh = 0.5*(prim[BX,k,j-1,i] + prim[BX,k,j,i])
          Bzh = 0.5*(prim[BZ,k,j-1,i] + prim[BZ,k,j,i])
          Bhi = 1. / SQRT(Bxh*Bxh + Byh*Byh + Bzh*Bzh + SMALL_NUM)
          bxh,byh,bzh = Bxh*Bhi, Byh*Bhi, Bzh*Bhi

          #------------------------------------------------------------------

          kappah_pl = ( 2. * kappa_pl[k,j-1,i] * kappa_pl[k,j,i]
                      / (kappa_pl[k,j-1,i] + kappa_pl[k,j,i] + SMALL_NUM) )

          dTdx_pl,dTdy_pl,dTdz_pl = 0.,0.,0.

          dTdy_pl = gc.dlv_inv[1][j] * (Tpl[k,j,i] - Tpl[k,j-1,i])

          IF D2D: dTdx_pl = utdiff.tr_grad_lim_xy(Tpl, i,j,k, gc)
          IF D3D: dTdz_pl = utdiff.tr_grad_lim_zy(Tpl, i,j,k, gc)

          bbgradTpl = byh * (bxh * dTdx_pl + byh * dTdy_pl + bzh * dTdz_pl)

          #------------------------------------------------------------------

          kappah_pd = ( 2. * kappa_pd[k,j-1,i] * kappa_pd[k,j,i]
                      / (kappa_pd[k,j-1,i] + kappa_pd[k,j,i] + SMALL_NUM) )

          dTdx_pd,dTdy_pd,dTdz_pd = 0.,0.,0.

          dTdy_pd = gc.dlv_inv[1][j] * (Tpd[k,j,i]-Tpd[k,j-1,i])

          IF D2D: dTdx_pd = utdiff.tr_grad_lim_xy(Tpd, i,j,k, gc)
          IF D3D: dTdz_pd = utdiff.tr_grad_lim_zy(Tpd, i,j,k, gc)

          bbgradTpd = byh * (bxh * dTdx_pd + byh * dTdy_pd + bzh * dTdz_pd)

          #------------------------------------------------------------------

          kappah_m = ( 2. * kappa_m[k,j-1,i] * kappa_m[k,j,i]
                      / (kappa_m[k,j-1,i] + kappa_m[k,j,i] + SMALL_NUM) )

          dBdx,dBdy,dBdz = 0.,0.,0.

          dBdy = gc.dlv_inv[1][j] * (Babs[k,j,i]-Babs[k,j-1,i])

          IF D2D: dBdx = utdiff.tr_grad_lim_xy(Babs, i,j,k, gc)
          IF D3D: dBdz = utdiff.tr_grad_lim_zy(Babs, i,j,k, gc)

          bbgradB = byh * (bxh * dBdx + byh * dBdy + bzh * dBdz)

          #------------------------------------------------------------------

          TR = one3rd * (2.*Tpd[k,j,i]   + Tpl[k,j,i])
          TL = one3rd * (2.*Tpd[k,j-1,i] + Tpl[k,j-1,i])

          TRpd = Tpd[k,j,i]
          TLpd = Tpd[k,j-1,i]

          #-----------------------------------------------------------------------
          # take into account that the heat flux caused by magnetic field gradient
          # is advective relative to perpendicular temperature

          DT = 1.5*(TL - TLpd + TR - TRpd)
          vmag = bbgradB * Bhi * kappah_m * DT
          vmag_tpd = vmag * TLpd if vmag>0. else vmag * TRpd

          # heat flux of total ion thermal energy
          fy[k,j,i] = - (kappah_pd * bbgradTpd + 0.5 * kappah_pl * bbgradTpl) + vmag_tpd

          # heat flux of magnetic moment (p_perp/B)
          fpdy[k,j,i] = (- kappah_pd * bbgradTpd + vmag_tpd) * Bhi

          #-----------------------------------------------------------------------
          # saturation

          if diff.sat_hfi:

            cpl = SQRT(1.5*(TL + TR) - (TLpd + TRpd))
            rho = 0.5*(prim[RHO,k,j-1,i] + prim[RHO,k,j,i])
            mom = 0.3 * FABS(byh) * rho * cpl

            qsat    = TL*mom       if   fy[k,j,i]>0. else TR*mom
            qsat_pd = TLpd*mom*Bhi if fpdy[k,j,i]>0. else TRpd*mom*Bhi

            fy[k,j,i]   =   fy[k,j,i] * qsat    / (FABS(fy[k,j,i])   + qsat    + SMALL_NUM)
            fpdy[k,j,i] = fpdy[k,j,i] * qsat_pd / (FABS(fpdy[k,j,i]) + qsat_pd + SMALL_NUM)


  IF D3D:

    for k in range(kl, ku+2):#, nogil=True, num_threads=OMP_NT, schedule='dynamic'):
      for j in range(jl, ju+1):
        for i in range(il, iu+1):

          Bzh = bfld[2,k,j,i]
          Bxh = 0.5*(prim[BX,k-1,j,i] + prim[BX,k,j,i])
          Byh = 0.5*(prim[BY,k-1,j,i] + prim[BY,k,j,i])
          Bhi = 1. / SQRT(Bxh*Bxh + Byh*Byh + Bzh*Bzh + SMALL_NUM)
          bxh,byh,bzh = Bxh*Bhi, Byh*Bhi, Bzh*Bhi

          #------------------------------------------------------------------

          kappah_pl = ( 2. * kappa_pl[k-1,j,i] * kappa_pl[k,j,i]
                      / (kappa_pl[k-1,j,i] + kappa_pl[k,j,i] + SMALL_NUM) )

          dTdx_pl,dTdy_pl,dTdz_pl = 0.,0.,0.

          dTdz_pl = gc.dlv_inv[2][k] * (Tpl[k,j,i] - Tpl[k-1,j,i])

          IF D2D: dTdx_pl = utdiff.tr_grad_lim_xz(Tpl, i,j,k, gc)
          IF D3D: dTdy_pl = utdiff.tr_grad_lim_yz(Tpl, i,j,k, gc)

          bbgradTpl = bzh * (bxh * dTdx_pl + byh * dTdy_pl + bzh * dTdz_pl)

          #------------------------------------------------------------------

          kappah_pd = ( 2. * kappa_pd[k-1,j,i] * kappa_pd[k,j,i]
                      / (kappa_pd[k-1,j,i] + kappa_pd[k,j,i] + SMALL_NUM) )

          dTdx_pd,dTdy_pd,dTdz_pd = 0.,0.,0.

          dTdz_pd = gc.dlv_inv[2][k] * (Tpd[k,j,i]-Tpd[k-1,j,i])

          IF D2D: dTdx_pd = utdiff.tr_grad_lim_xz(Tpd, i,j,k, gc)
          IF D3D: dTdy_pd = utdiff.tr_grad_lim_yz(Tpd, i,j,k, gc)

          bbgradTpd = bzh * (bxh * dTdx_pd + byh * dTdy_pd + bzh * dTdz_pd)

          #------------------------------------------------------------------

          kappah_m = ( 2. * kappa_m[k-1,j,i] * kappa_m[k,j,i]
                      / (kappa_m[k-1,j,i] + kappa_m[k,j,i] + SMALL_NUM) )

          dBdx,dBdy,dBdz = 0.,0.,0.

          dBdz = gc.dlv_inv[2][k] * (Babs[k,j,i]-Babs[k-1,j,i])

          IF D2D: dBdx = utdiff.tr_grad_lim_xz(Babs, i,j,k, gc)
          IF D3D: dBdy = utdiff.tr_grad_lim_yz(Babs, i,j,k, gc)

          bbgradB = bzh * (bxh * dBdx + byh * dBdy + bzh * dBdz)

          #------------------------------------------------------------------

          TR = one3rd * (2.*Tpd[k,j,i]   + Tpl[k,j,i])
          TL = one3rd * (2.*Tpd[k-1,j,i] + Tpl[k-1,j,i])

          TRpd = Tpd[k,j,i]
          TLpd = Tpd[k-1,j,i]

          #-----------------------------------------------------------------------
          # take into account that the heat flux caused by magnetic field gradient
          # is advective relative to perpendicular temperature

          DT = 1.5*(TL - TLpd + TR - TRpd)
          vmag = bbgradB * Bhi * kappah_m * DT
          vmag_tpd = vmag * TLpd if vmag>0. else vmag * TRpd

          # heat flux of total ion thermal energy
          fz[k,j,i] = - (kappah_pd * bbgradTpd + 0.5 * kappah_pl * bbgradTpl) + vmag_tpd

          # heat flux of magnetic moment (p_perp/B)
          fpdz[k,j,i] = (- kappah_pd * bbgradTpd + vmag_tpd) * Bhi

          #------------------------------------------------------------------
          # saturation

          if diff.sat_hfi:

            cpl = SQRT(1.5*(TL + TR) - (TLpd + TRpd))
            rho = 0.5*(prim[RHO,k-1,j,i] + prim[RHO,k,j,i])
            mom = 0.3 * FABS(bzh) * rho * cpl

            qsat    = TL*mom       if   fz[k,j,i]>0. else TR*mom
            qsat_pd = TLpd*mom*Bhi if fpdz[k,j,i]>0. else TRpd*mom*Bhi

            fz[k,j,i]   =   fz[k,j,i] * qsat    / (FABS(fz[k,j,i])   + qsat    + SMALL_NUM)
            fpdz[k,j,i] = fpdz[k,j,i] * qsat_pd / (FABS(fpdz[k,j,i]) + qsat_pd + SMALL_NUM)

  cdef real gdx,gdy,gdz

  for k in range(kl, ku+1):
    gdz = gamm1 * gc.dlf_inv[2][k]

    for j in range(jl, ju+1):
      gdy = gamm1 * gc.dlf_inv[1][j]

      for i in range(il, iu+1):
        gdx = gamm1 * gc.dlf_inv[0][i]

        dtemp[k,j,i]      = - (  fx[k,j,i+1] -   fx[k,j,i]) * gdx
        dtemp_perp[k,j,i] = - (fpdx[k,j,i+1] - fpdx[k,j,i]) * gdx
        IF D2D:
          dtemp[k,j,i]      -= (  fy[k,j+1,i] -   fy[k,j,i]) * gdy
          dtemp_perp[k,j,i] -= (fpdy[k,j+1,i] - fpdy[k,j,i]) * gdy
        IF D3D:
          dtemp[k,j,i]      -= (  fz[k+1,j,i] -   fz[k,j,i]) * gdz
          dtemp_perp[k,j,i] -= (fpdz[k+1,j,i] - fpdz[k,j,i]) * gdz


# -------------------------------------------------------------------------------

# Calculate diffusive time-step.

cdef real get_dt_tci(real4d prim, GridCoord *gc, BnzDiffusion diff):

  cdef:
    int k,j,i
    int id
    # real dTx,dTy,dTz,dTi, B2i
    real _temp_par
    real ad, ad_dl2, ad_dl2_max = 0.

  cdef:
    real a1 = diff.kl * SQRT(0.5*B_PI)
    real a2 = 0.25*(3.*B_PI-8.)

  cdef:
    # real3d Tpl = diff.scr.tempi_par
    real ad_dl2_max_loc[OMP_NT]

    real *dxfi = gc.dlf_inv[0]
    real *dyfi = gc.dlf_inv[1]
    real *dzfi = gc.dlf_inv[2]

  IF MPI:
    cdef:
      double[::1] var     = np.empty(1, dtype='f8')
      double[::1] var_max = np.empty(1, dtype='f8')

  with nogil, parallel(num_threads=OMP_NT):

    # for k in prange(gc.Ntot[2], schedule='dynamic'):
    #   for j in range(gc.Ntot[1]):
    #     for i in range(gc.Ntot[0]):
    #
    #       Tpl[k,j,i] = (3.*prim[PR,k,j,i] - 2.*prim[PPD,k,j,i]) / prim[RHO,k,j,i]

    id = threadid()

    for k in prange(gc.k1, gc.k2+1, schedule='dynamic'):
      # dTx,dTy,dTz,dTi = 0,0,0,0
      for j in range(gc.j1, gc.j2+1):
        for i in range(gc.i1, gc.i2+1):

          _temp_par = (3.*prim[PR,k,j,i] - 2.*prim[PPD,k,j,i]) / prim[RHO,k,j,i]

          ad = 2. * _temp_par / (a1 * SQRT(_temp_par) + a2 * diff.nuii_eff[k,j,i])

          # dTx = Tpl[k,j,i+1] - Tpl[k,j,i-1]
          # IF D2D: dTy = Tpl[k,j+1,i] - Tpl[k,j-1,i]
          # IF D3D: dTz = Tpl[k+1,j,i] - Tpl[k-1,j,i]
          # dTi = 1. / SQRT(SQR(dTx) + SQR(dTy) + SQR(dTz) + SMALL_NUM)
          #
          # B2i = 1. / (SQR(prim[BX,k,j,i])
                    # + SQR(prim[BY,k,j,i])
                    # + SQR(prim[BZ,k,j,i]) + SMALL_NUM)
          #
          # ad = ad * FABS(prim[BX,k,j,i] * dTx + prim[BY,k,j,i] * dTy
          #              + prim[BZ,k,j,i] * dTz) * B2i*dTi

          ad_dl2 = SQR(dxfi[i]) * ad #* FABS(prim[BX,k,j,i])
          IF D2D: ad_dl2 = ad_dl2 + SQR(dyfi[j]) * ad #* FABS(prim[BY,k,j,i])
          IF D3D: ad_dl2 = ad_dl2 + SQR(dzfi[k]) * ad #* FABS(prim[BZ,k,j,i])

          if ad_dl2 > ad_dl2_max_loc[id]: ad_dl2_max_loc[id] = ad_dl2


  for i in range(OMP_NT):
    if ad_dl2_max_loc[i] > ad_dl2_max: ad_dl2_max = ad_dl2_max_loc[i]

  IF MPI:
    var[0] = ad_dl2_max
    mpi.COMM_WORLD.Allreduce(var, var_max, op=mpi.MAX)
    ad_dl2_max = var_max[0]

  return diff.cour_diff / ad_dl2_max
