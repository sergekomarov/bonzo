# -*- coding: utf-8 -*-

from bnz.defs cimport *
from bnz.coordinates.grid cimport BnzGrid


# time integrators
ctypedef enum ThcondType:
  TC_ISO
  TC_ANISO

cdef class DiffScratch:

  cdef:
    # arrays used by electron conduction
    real3d pres0
    real3d presm1
    real3d dtemp
    real3d dtemp0

    real3d temp
    real3d kappa_par
    real3d kappa_mag
    real3d fx_diff
    real3d fy_diff
    real3d fz_diff

    # ion conduction
    real3d presi_perp0
    real3d presi_perpm1
    real3d dtempi_perp
    real3d dtempi_perp0

    real3d tempi_par
    real3d tempi_perp
    real3d babs
    real3d kappa_perp
    real3d fx_diff2
    real3d fy_diff2
    real3d fz_diff2

    # viscosity
    real3d div
    real4d vel
    real4d vel0
    real4d velm1
    real4d dvel
    real4d dvel0

    #resistivity
    real4d bfld0

cdef class BnzIntegr

cdef class BnzDiffusion:

  cdef:
    # reference to parent integrator instance
    BnzIntegr integr

    # parabolic Courant number
    real cour_diff

    real kappa0
    ThcondType thcond_type
    int sat_hfe

    real nuiic0
    real kl
    int sat_hfi

    real mu0
    real mu4
    real eta0
    real eta4

    real gam

    real3d nuii_eff

  cdef DiffScratch scratch

  cdef void diffuse(self, BnzGrid,real)
  cdef void collide(self, real4d, int*,real)

  cdef void get_nsts(self, real,real)
  cdef real2d get_sts_coeff(self,int)
