# -*- coding: utf-8 -*-

import numpy as np
cimport numpy as np

from libc.stdlib cimport malloc, calloc, free

from bnz.utils cimport print_root
from bnz.io.read_config import read_param


cimport thcond_elec as tce
IF IONTC:
  cimport thcond_ion as tci
# cimport visc
# cimport resist
# cimport visc4
# cimport resist4

IF SPREC:
  np_real = np.float32
ELSE:
  np_real = np.float64


cdef class BnzDiffusion:

  # Allocate scratch arrays for STS integration of diffusive terms.

  def __cinit__(self, GridCoord *gc, real gam, str usr_dir):

    cdef int i,j,k

    # parabolic Courant number
    self.cour_diff = read_param("computation", "cour_diff", 'f', usr_dir)

    # Electron thermal conduction.

    self.kappa0 = read_param("physics", "kappa0", 'f',usr_dir)

    thcond_type = read_param("physics", "elec_thcond_type", 's',usr_dir)
    if thcond_type=='iso': self.thcond_type = TC_ISO
    elif thcond_type=='aniso': self.thcond_type = TC_ANISO
    else: self.thcond_type = TC_ANISO
    IF not MFIELD: self.thcond_type = TC_ISO

    self.sat_hfe = read_param("physics", "sat_hfe", 'i',usr_dir)

    # Ion thermal conduction.

    IF IONTC:
      self.nuiic0 = read_param("physics", "nuiic0", 'f',usr_dir)
      self.kl = read_param("physics", "kL", 'f',usr_dir)
    ELSE:
      self.nuiic0=0.
      self.kl=0.

    self.sat_hfi = read_param("physics", "sat_hfi", 'i',usr_dir)

    # (Hyper-)viscosity and resistivity.

    self.mu0 = read_param("physics", "mu", 'f',usr_dir)
    self.eta0 = read_param("physics", "eta", 'f',usr_dir)
    self.mu4 = read_param("physics", "mu4", 'f',usr_dir)
    self.eta4 = read_param("physics", "eta4", 'f',usr_dir)

    # gas gamma
    self.gam = gam

    # Allocate arrays.

    sh1 = (gc.Ntot[2], gc.Ntot[1], gc.Ntot[0])
    sh3 = (3, gc.Ntot[2], gc.Ntot[1], gc.Ntot[0])

    # effective ion-ion collision rate
    IF IONTC: self.nuii_eff = self.nuiic0 * np.ones(sh1, dtype=np_real)

    # Scratch arrays.

    self.scratch = DiffScratch()

    cdef DiffScratch scr = self.scratch

    if self.kappa0 != 0.:

      scr.temp = np.zeros(sh1, dtype=np_real)
      scr.pres0 = np.zeros(sh1, dtype=np_real)
      scr.presm1 = np.zeros(sh1, dtype=np_real)
      scr.dtemp = np.zeros(sh1, dtype=np_real)
      scr.dtemp0 = np.zeros(sh1, dtype=np_real)

      scr.kappa_par  = np.zeros(sh1, dtype=np_real)
      scr.kappa_mag = np.zeros(sh1, dtype=np_real)

      # for k in range(gc.Ntot[2]):
      #   for j in range(gc.Ntot[1]):
      #     for i in range(gc.Ntot[0]):
      #       if sqrt((gc.lv[0][i]-0.5)**2 + (gc.lv[1][j]-0.5)**2 + (gc.lv[2][k]-0.5)**2) < 0.15:
      #         scr.kappa_mag[k,j,i] = -1.

      scr.fx_diff = np.zeros(sh1, dtype=np_real)
      scr.fy_diff = np.zeros(sh1, dtype=np_real)
      scr.fz_diff = np.zeros(sh1, dtype=np_real)

    IF IONTC:

      if self.nuiic0 != 0. or self.kl != 0.:

        scr.tempi   = np.zeros(sh1, dtype=np_real)
        scr.tempi_par = np.zeros(sh1, dtype=np_real)
        scr.tempi_perp = np.zeros(sh1, dtype=np_real)
        scr.babs = np.zeros(sh1, dtype=np_real)

        # mean ion temperature
        scr.temp0 = np.zeros(sh1, dtype=np_real) # T0
        scr.tempm1 = np.zeros(sh1, dtype=np_real) # Tm1
        scr.dtemp0 = np.zeros(sh1, dtype=np_real) # MT0
        scr.dtemp = np.zeros(sh1, dtype=np_real) # MT

        # perpendicular ion temperature
        scr.tempi_perp0 = np.zeros(sh1, dtype=np_real) # T0
        scr.tempi_perpm1 = np.zeros(sh1, dtype=np_real) # Tm1
        scr.dtempi_perp = np.zeros(sh1, dtype=np_real) # MT
        scr.dtempi_perp0 = np.zeros(sh1, dtype=np_real) # MT0

        scr.kappa_par  = np.zeros(sh1, dtype=np_real)
        scr.kappa_perp  = np.zeros(sh1, dtype=np_real)
        scr.kappa_mag = np.zeros(sh1, dtype=np_real)

        scr.fx_diff = np.zeros(sh1, dtype=np_real)
        scr.fy_diff = np.zeros(sh1, dtype=np_real)
        scr.fz_diff = np.zeros(sh1, dtype=np_real)

        scr.fx_diff2 = np.zeros(sh1, dtype=np_real)
        scr.fy_diff2 = np.zeros(sh1, dtype=np_real)
        scr.fz_diff2 = np.zeros(sh1, dtype=np_real)

    if self.mu0 != 0. or self.mu4 != 0.:

      scr.div = np.zeros(sh1, dtype=np_real) #div(V)

    if self.eta0 != 0. or self.eta4 != 0.:

      scr.bfld0 = np.zeros(sh3, dtype=np_real)

    if self.mu0 != 0. or self.eta0 != 0. or self.mu4 != 0. or self.eta4 != 0.:

      scr.vel = np.zeros(sh3, dtype=np_real)
      scr.vel0 = np.zeros(sh3, dtype=np_real)
      scr.velm1 = np.zeros(sh3, dtype=np_real) # velm1, bfm1,             , electric field
      scr.dvel0 = np.zeros(sh3, dtype=np_real) # dvel0, dbf0, visc. flux,   Pointing flux
      scr.dvel = np.zeros(sh3, dtype=np_real) #  dvel,  dbf,  \nabla^2 vel, current


  cdef void diffuse(self, BnzGrid grid, BnzIntegr integr, real dt):

    # Integrate diffusive terms with super-time-stepping
    # using Lagrange polynomials.

    IF DIAGNOSE:
      cdef timeval tstart, tstop
      print_root(rank, "diffusion... ")
      gettimeofday(&tstart, NULL)

    # electron thermal conduction
    if self.kappa0 != 0.:
      tce.diffuse(grid,integr,dt)

    # ion thermal conduction
    IF IONTC:
      if self.nuiic0 != 0. or self.kL != 0.:
        tci.diffuse(grid,integr,dt)

    # resistivity
    if self.eta != 0.:
      res.diffuse(grid,integr,dt)

    # hyperresistivity
    if self.eta4 != 0.:
      res4.diffuse(grid,integr,dt)

    # viscosity
    if self.mu != 0.:
      visc.diffuse(grid,integr,dt)

    # hyperviscosity
    if self.mu4 != 0.:
      visc4.diffuse(grid,integr,dt)

    IF DIAGNOSE:
      gettimeofday(&tstop, NULL)
      print_root(rank, "%.1f ms\n", timediff(tstart,tstop))


  cdef int get_nsts(self, real dt_hyp, real dt_diff):

    # Get the number of STS stages.

    if dt_diff>=dt_hyp:
      return 1
    else:
      return <int>(0.5 * (SQRT(9. + 16. * dt_hyp/dt_diff) - 1.)) + 1


  cdef real2d get_sts_coeff(self, int s):

    # Set STS coefficients using Lagrange polynomials.

    cdef int j
    cdef real one3rd = 1./3
    cdef real w1

    cdef:
      real1d b = np.zeros(s+1, dtype=np_real)
      real2d coeff = np.zeros((s+1,4), dtype=np_real)

    if s != 1:
      w1 = 4./(s**2 + s - 2)
    else:
      coeff[1,MUT]=1.
      return coeff

    b[0]=one3rd
    b[1]=one3rd
    coeff[1,MU]=0.
    coeff[1,NU]=0.
    coeff[1,MUT]=w1 * b[1]
    coeff[1,GAMT]=0.

    for j in range(2,s+1):

      b[j] = <real>(j**2 + j - 2) / (2*j * (j+1))
      coeff[j,MU] = <real>(2*j-1)/j * b[j]/b[j-1]
      coeff[j,NU] = <real>(1.-j)/j * b[j]/b[j-2]
      coeff[j,MUT] = w1 * coeff[j,MU]
      coeff[j,GAMT] = (b[j-1]-1) * coeff[j,MUT]

    return coeff
