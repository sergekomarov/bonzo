# -*- coding: utf-8 -*-

import numpy as np
cimport numpy as np

from cython.parallel import prange, parallel, threadid

from bnz.utils cimport calloc_3d_array, calloc_4d_array, free_3d_array
from bnz.read_config import read_param
cimport bnz.problem
IF MHDPIC:
  from bnz.utils_particle cimport getweight1, getweight2


# =======================================

# Integration-related data and functions.

cdef class BnzIntegr:

  def __cinit__(self, bytes usr_dir):

    self.rsolver_func = NULL
    self.reconstr_func = NULL
    # self.limiter_func=NULL
    IF MHDPIC:
      self.weight_func = NULL

    # Set integrator properties from config file.

    # Courant number
    self.cour = read_param("computation", "cour", 'f', usr_dir)

    # time integration

    tintegr = read_param("computation", "tintegr", 's',usr_dir)
    if tintegr == 'vl': self.tintegr = TINT_VL
    elif tintegr == 'rk3': self.tintegr = TINT_RK3
    else: self.tintegr = TINT_VL

    # Riemann solver

    rsolver = read_param("computation", "rsolver", 's', usr_dir)
    if rsolver == 'hll': self.rsolver = RS_HLL
    elif rsolver == 'hllc': self.rsolver = RS_HLLC
    elif rsolver == 'hlld': self.rsolver = RS_HLLD
    elif rsolver == 'hllt': self.rsolver = RS_HLLT
    elif rsolver == 'hlla': self.rsolver = RS_HLLA
    else: self.rsolver = RS_HLL

    # spatial reconstruction

    reconstr = read_param("computation", "reconstr", 's',usr_dir)
    if reconstr == 'const': self.reconstr = RCN_CONST
    elif reconstr == 'linear': self.reconstr = RCN_LINEAR
    elif reconstr == 'parab': self.reconstr = RCN_PARAB
    elif reconstr == 'weno': self.reconstr = RCN_WENO
    else: self.reconstr = RCN_LINEAR

    # limiter function

    # limiter = read_param("computation", "limiter", 's', usr_dir)
    # if limiter == 'mm': self.limiter = LIM_MM
    # elif limiter == 'mc': self.limiter = LIM_MC
    # elif limiter == 'vl': self.limiter = LIM_VL
    # else: self.limiter = LIM_VL

    # liiting in characteristic variables
    self.charact_proj = read_param("computation", "charact_proj", 'i',usr_dir)
    IF CGL or TWOTEMP: self.charact_proj = 0

    # apply pressure floor
    self.pressure_floor = read_param("computation", "pressure_floor", 'f',usr_dir)

    # Set function pointers.

    if self.rsolver == RS_HLL:
      self.rsolver_func = &hll_flux

    elif self.rsolver == RS_HLLT:
      self.rsolver_func = &hllt_flux

    IF not MFIELD:
      if self.rsolver == RS_HLLC:
        self.rsolver_func = &hllc_flux
    ELSE:
      if self.rsolver == RS_HLLD:
        self.rsolver_func = &hlld_flux

    IF CGL:
      if self.rsolver == RS_HLLA:
        self.rsolver_func = &hlla_flux

    if self.reconstr == RCN_CONST:
      self.reconstr_func = &reconstr_const
      self.reconstr_order=0

    elif self.reconstr == RCN_LINEAR:
      self.reconstr_func = &reconstr_linear
      self.reconstr_order=1

    elif self.reconstr == RCN_PARAB:
      self.reconstr_func = &reconstr_parab
      self.reconstr_order=2

    elif self.reconstr == RCN_WENO:
      self.reconstr_func = &reconstr_weno
      self.reconstr_order=1

    # Diffusion parameters.

    # parabolic Courant number
    self.cour_diff = read_param("computation", "cour_diff", 'f', usr_dir)

    # super-time-stepping (1=on; 0=off)
    self.sts = read_param("computation", "sts", 'i', usr_dir)
    IF MPI: self.sts=1  # implicit solver is not compatible with MPI


    IF MHDPIC:

      # order of interpolation
      self.Ninterp = read_param("computation", "Ninterp", 'i',usr_dir)

      # Set function pointers for particle interpolation kernels.
      if self.Ninterp==1:
        self.weight_func = &getweight1
      elif self.Ninterp==2:
        self.weight_func = &getweight2


    # Set physical parameters and functions.

    # MHD physical parameters

    # adiabatic index
    self.gam = read_param("physics", "gam", 'f',usr_dir)
    IF CGL: self.gam = 5./3

    IF MHDPIC:
      # effective speed of light
      self.sol    = read_param("physics", "sol",    'f', usr_dir)
      # charge-to-mass ratio of CRs relative to thermal ions
      self.q_mc   = read_param("physics", "q_mc",   'f', usr_dir)
      # CR density
      self.rho_cr = read_param("physics", "rho_cr", 'f', usr_dir)

    # gravitation
    # self.g0 = read_param("physics", "g0", 'f',usr_dir)

    # diffusion

    # self.kappa0 = read_param("physics", "kappa0", 'f',usr_dir)
    #
    # thcond_type = read_param("physics", "elec_thcond_type", 's',usr_dir)
    # if thcond_type=='iso': self.thcond_type = TC_ISO
    # elif thcond_type=='aniso': self.thcond_type = TC_ANISO
    # else: self.thcond_type = TC_ANISO
    # IF not MFIELD: self.thcond_type = TC_ISO

    # self.sat_hfe = read_param("physics", "sat_hfe", 'i',usr_dir)

    # IF IONTC:
    #   self.nuiic0 = read_param("physics", "nuiic0", 'f',usr_dir)
    #   self.kL = read_param("physics", "kL", 'f',usr_dir)
    # ELSE:
    #   self.nuiic0=0.
    #   self.kL=0.

    # self.sat_hfi = read_param("physics", "sat_hfi", 'i',usr_dir)

    # self.mu = read_param("physics", "mu", 'f',usr_dir)
    # self.eta = read_param("physics", "eta", 'f',usr_dir)
    # self.mu4 = read_param("physics", "mu4", 'f',usr_dir)
    # self.eta4 = read_param("physics", "eta4", 'f',usr_dir)

    # solenoidal driving

    # self.f = read_param("physics", "f", 'f',usr_dir)
    # self.tau = read_param("physics", "tau", 'f',usr_dir)
    # self.Nmod = read_param("physics", "Nmod", 'i',usr_dir)

    # self.grav_pot_func = NULL


# =================================================================

cdef void init_data(self, GridCoord *gc):

  cdef:
    IntegrData dat = self.data
    IntegrScratch scr = self.scratch
    ints Nx=gc.Ntot[0], Ny=gc.Ntot[1], Nz=gc.Ntot[2]

  # allocate arrays used by MHD integrator

  sh_u = (NMODES,Nz,Ny,Nx)
  sh_3 = (3,Nz,Ny,Nx)

  # predictor-step arrays of cell-centered conserved variables
  dat.cons_s =  np.zeros(sh_u, dtype=np_real)
  if self.tintegr==TINT_RK3:
    dat.cons_ss =  np.zeros(sh_u, dtype=np_real)

  # Godunov fluxes
  dat.flux_x = np.zeros(sh_u, dtype=np_real)
  dat.flux_y = np.zeros(sh_u, dtype=np_real)
  dat.flux_z = np.zeros(sh_u, dtype=np_real)

  IF MFIELD:

    # predictor-step arrays of face-centered magnetic field
    dat.bf_s =  np.zeros(sh_3, dtype=np_real)
    if self.tintegr==TINT_RK3:
      dat.bf_ss =  np.zeros(sh_3, dtype=np_real)

    # edge- and cell-centered electric field
    dat.ee = np.zeros(sh_3, dtype=np_real)
    dat.ec = np.zeros(sh_3, dtype=np_real)

  ELSE:

    dat.bf_s = None
    if self.tintegr==TINT_RK3:
      dat.bf_ss = None
    dat.ee  = None
    dat.ec = None

  IF MPI and MHDPIC:
    # temporary array for particle deposit exchange at boundaries
    dat.fcoup_tmp = np.zeros((4,Nz,Ny,Nx), dtype=np_real)

  # allocate scratch arrays used in reconstruction and Godunov flux calculation

  scr.w_rcn = <real****>calloc_4d_array(OMP_NT, 8, NMODES, Nx+4)
  scr.wl    =  <real***>calloc_3d_array(OMP_NT,    NMODES, Nx+4)
  scr.wl_   =  <real***>calloc_3d_array(OMP_NT,    NMODES, Nx+4)
  scr.wr    =  <real***>calloc_3d_array(OMP_NT,    NMODES, Nx+4)

  # set pointers to user-defined physics functions (e.g. gravitational potential)
  problem.set_phys_ptrs_user(phys)

  # # initialize solenoidal driving force
  # if phys.f != 0.: turb_driv.init_turb_driv(grid, phys)
  #
  # # set static gravitational potential
  # if phys.g0 != 0: gravity.init_gravity(grid, phys)
  #
  # # initialize diffusion integrator
  # if (phys.mu != 0 or phys.eta !=0 or phys.mu4 != 0 or phys.eta4 != 0
  #   or phys.kappa0 != 0 or phys.kL != 0 or phys.nuiic0 != 0):
  #
  #   diffuse.init_diffuse(grid, phys, method.sts)


  def __dealloc__(self):

    cdef ints n
    for n in range(OMP_NT):
      free_3d_array(self.scratch.w_rcn[n])
    free_3d_array(self.scratch.wl)
    free_3d_array(self.scratch.wl_)
    free_3d_array(self.scratch.wr)

    self.rsolver_func = NULL
    self.reconstr_func = NULL
    # self.limiter_func=NULL
    # self.grav_pot_func = NULL
    IF MHDPIC:
      self.weight_func = NULL
