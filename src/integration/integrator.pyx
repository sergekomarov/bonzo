# -*- coding: utf-8 -*-

import numpy as np
cimport numpy as np

from bnz.util cimport calloc_3d_array, calloc_4d_array, free_3d_array, print_root
from bnz.io.read_config import read_param
from bnz.problem.problem cimport set_phys_funcs_user

cimport bnz.mhd.eos as eos
cimport ct
cimport godunov
cimport new_dt
cimport bnz.diagnostics as diagnostics
cimport bnz.srcterms.gravity as gravity
cimport bnz.srcterms.turb_driv as turb_driv
cimport bnz.coordinates.coord as coord
cimport bnz.dissipation.collisions as collisions


cdef class BnzIntegr:

  def __cinit__(self, GridCoord *gc, str usr_dir):

    init_method(self, usr_dir)
    init_physics(self, gc, usr_dir)
    init_data(self, gc)

    # initialize random-number generator
    # IF MPI:
    #   srand(dom.blocks.comm.Get_rank())
    # ELSE:
    #   np.random.seed()
    #   srand(np.random.randint(RAND_MAX))

  def __dealloc__(self):

    cdef int n
    for n in range(OMP_NT):
      free_3d_array(self.scratch.w_rcn[n])
    free_3d_array(self.scratch.wl)
    free_3d_array(self.scratch.wl_)
    free_3d_array(self.scratch.wr)

    self.rsolver_func = NULL
    self.reconstr_func = NULL


  # Integration jobs:

  cdef void integrate_hydro(self, real4d cons1,
                            real4d cons0, real4d prim0, real4d bf0,
                            GridCoord *gc, int order, real dt):

    # Integrate hydro variables by dt from U0=U(t0) to U1=U(t0+dt).

    cdef:
      timeval tstart, tstop
      IntegrData idat = self.data
      int lims[6]

    lims = &(get_integr_lims(order, gc.Ntot)[0])

    IF DIAGNOSE:
      print_root("%i-order Godunov fluxes... ", order+1)
      gettimeofday(&tstart, NULL)

    # get Goduniv fluxes flx_x,flx_y,flx_z from prim0
    godunov.godunov_fluxes(idat.flx_x, idat.flx_y, idat.flx_z,
                           prim0, bf0, gc, lims, self, order)

    IF DIAGNOSE:
      gettimeofday(&tstop, NULL)
      print_root("%.1f ms\n", timediff(tstart,tstop))
      print_root("advance cell-centered hydro variables by a half-step...\n")

    godunov.advance_hydro(cons1, cons0, idat.flx_x, idat.flx_y, idat.flx_z,
                          gc, lims, dt)
                          # advances to us from u using fluxes

  # -----------------------------------------------------------------------------

  IF MFIELD:

    cdef void integrate_field(self, real4d cons1, real4d bf1,
                              real4d prim0, real4d bf0,
                              GridCoord *gc, int order, real dt):

      # Integrate magnetic field by dt from B0=B(t0) to B1=B(t0+dt).

      cdef:
        timeval tstart, tstop
        IntegrData idat = self.data
        int lims[6]

      lims = &(get_integr_lims(order, gc.Ntot)[0])

      IF DIAGNOSE: print_root("E-field at cell centers... \n")

      # use Ohm's law to get cell-centered E-field from primitive variables
      ct.ec_from_prim(idat.ec, prim0, lims)

      IF DIAGNOSE:
        print_root("interpolate E-field to cell edges... ")
        gettimeofday(&tstart, NULL)

      # interpolate edge-centered E-field (ee) from cell-centered (ec) and fluxes
      # ct.interp_ee2(idat.ee, idat.ec, idat.flx_x, idat.flx_y, idat.flx_z, lims,
      #               prim0[RHO,...], self.dt)
      ct.interp_ee1(idat.ee, idat.ec, idat.flx_x, idat.flx_y, idat.flx_z, lims)

      IF DIAGNOSE:
        gettimeofday(&tstop, NULL)
        print_root("%.1f ms\n", timediff(tstart,tstop))
        print_root("advance B-field via CT... ")
        gettimeofday(&tstart, NULL)

      # advance B-field using edge-centered E-field via constrained transport
      ct.advance_b(bf1, bf0, idat.ee, gc, lims, dt)

      IF DIAGNOSE:
        gettimeofday(&tstop, NULL)
        print_root("%.1f ms\n", timediff(tstart,tstop))
        print_root("interpolate B-field from cell faces to centers...\n")

      ct.interp_bc(cons1, bf1, gc, lims)

  # ------------------------------------------------------------------------

  # Add source terms.

  cdef void add_source_terms(self, real4d u1, real4d bf1,
                             real4d w0, real4d bf0,
                             GridCoord *gc, int order, real dt):

    cdef:
      timeval tstart, tstop
      IntegrData idat = self.data
      int lims[6]

    lims = &(get_integr_lims(order, gc.Ntot)[0])

    coord.add_geom_src_terms(u1, w0, idat.flx_x, idat.flx_y, idat.flx_z,
                             gc, lims, dt)

    if self.gravity.g0 != 0.:

      IF DIAGNOSE:
        print_root("gravity (predictor) ... ")
        gettimeofday(&tstart, NULL)

      self.gravity.apply_gravity(u1, w0, idat.flx_x, idat.flx_y, idat.flx_z,
                                 gc, lims, dt)

      IF DIAGNOSE:
        gettimeofday(&tstop, NULL)
        print_root("%.1f ms\n", timediff(tstart,tstop))

    #--------------------------------------------------

    if self.turb_driv.f0 != 0.:

      IF DIAGNOSE:
        print_root("turbulence driving... ")
        gettimeofday(&tstart, NULL)

      self.turb_driv.apply_driv_force(u1, w0, lims, dt)

      IF DIAGNOSE:
        gettimeofday(&tstop, NULL)
        print_root("%.1f ms\n", timediff(tstart,tstop))


  # ----------------------------------------------------------------------------

  cdef void update_physics(self, GridCoord *gc, int *lims, real dt):

    # Evolve quantities not directly related to the main hydro array
    # (e.g. the driving force).

    cdef timeval tstart, tstop

    if self.turb_driv.f0 != 0.:

      IF DIAGNOSE:
        print_root("advance driving force... ")
        gettimeofday(&tstart, NULL)

      # advance drifing force from t_n-1/2 to t_n+1/2
      self.turb_driv.advance_driv_force(gc, lims, dt)

      IF DIAGNOSE:
        gettimeofday(&tstop, NULL)
        print_root("%.1f ms\n", timediff(tstart,tstop))

  # ----------------------------------------------------------------------------

  cdef void new_dt(self, real4d prim, GridCoord *gc):

    self.dt = new_dt.new_dt(prim, gc, self)


# ================================================================

cdef void init_method(BnzIntegr integr, str usr_dir):

  integr.rsolver_func = NULL
  integr.reconstr_func = NULL
  # integr.limiter_func=NULL
  # IF MHDPIC:
  #   integr.weight_func = NULL

  # Set integrator properties from config file.

  # Courant number
  integr.cour = read_param("computation", "cour", 'f', usr_dir)

  # time integration

  tintegr = read_param("computation", "tintegr", 's',usr_dir)
  if tintegr == 'vl': integr.tintegr = TINT_VL
  elif tintegr == 'rk3': integr.tintegr = TINT_RK3
  else: integr.tintegr = TINT_VL

  # Riemann solver

  rsolver = read_param("computation", "rsolver", 's', usr_dir)
  if rsolver == 'hll': integr.rsolver = RS_HLL
  elif rsolver == 'hllc': integr.rsolver = RS_HLLC
  elif rsolver == 'hlld': integr.rsolver = RS_HLLD
  elif rsolver == 'hllt': integr.rsolver = RS_HLLT
  elif rsolver == 'hlla': integr.rsolver = RS_HLLA
  else: integr.rsolver = RS_HLL

  # spatial reconstruction

  reconstr = read_param("computation", "reconstr", 's',usr_dir)
  if reconstr == 'const': integr.reconstr = RCN_CONST
  elif reconstr == 'linear': integr.reconstr = RCN_LINEAR
  elif reconstr == 'parab': integr.reconstr = RCN_PARAB
  elif reconstr == 'weno': integr.reconstr = RCN_WENO
  else: integr.reconstr = RCN_LINEAR

  # liiting in characteristic variables
  integr.charact_proj = read_param("computation", "charact_proj", 'i',usr_dir)
  IF CGL or TWOTEMP: integr.charact_proj = 0

  # apply pressure floor
  # integr.pressure_floor = read_param("computation", "pressure_floor", 'f',usr_dir)

  # Set function pointers.

  if integr.rsolver == RS_HLL:
    integr.rsolver_func = &hll_flux

  elif integr.rsolver == RS_HLLT:
    integr.rsolver_func = &hllt_flux

  IF not MFIELD:
    if integr.rsolver == RS_HLLC:
      integr.rsolver_func = &hllc_flux
  ELSE:
    if integr.rsolver == RS_HLLD:
      integr.rsolver_func = &hlld_flux

  IF CGL:
    if integr.rsolver == RS_HLLA:
      integr.rsolver_func = &hlla_flux

  if integr.reconstr == RCN_CONST:
    integr.reconstr_func = &reconstr_const
    integr.reconstr_order=0

  elif integr.reconstr == RCN_LINEAR:
    integr.reconstr_func = &reconstr_linear
    integr.reconstr_order=1

  elif integr.reconstr == RCN_PARAB:
    integr.reconstr_func = &reconstr_parab
    integr.reconstr_order=2

  elif integr.reconstr == RCN_WENO:
    integr.reconstr_func = &reconstr_weno
    integr.reconstr_order=1

  integr.time=0.
  integr.step=0
  # max time of the simulation
  integr.tmax = read_param("computation", "tmax", 'f',usr_dir)
  # timestep if fixed
  IF FIXDT: integr.dt = read_param("computation", "dt", 'f',usr_dir)


# -------------------------------------------------------------------

cdef void init_physics(BnzIntegr integr, GridCoord *gc, str usr_dir):

  # adiabatic index
  integr.gam = read_param("physics", "gam", 'f',usr_dir)
  IF CGL: integr.gam = 5./3

  IF MHDPIC:
    # effective speed of light
    integr.sol    = read_param("physics", "sol",    'f', usr_dir)
    # charge-to-mass ratio of CRs relative to thermal ions
    integr.q_mc   = read_param("physics", "q_mc",   'f', usr_dir)
    # CR density
    integr.rho_cr = read_param("physics", "rho_cr", 'f', usr_dir)

  # (these could be compiled conditionaly instead)
  self.gravity   = BnzGravity(gc, usr_dir)
  self.turb_driv = BnzTurbDriv(gc, usr_dir)
  self.diffusion = BnzDiffusion(integr, gc, usr_dir)

  # set pointers to user-defined physics functions (e.g. gravitational potential)
  set_phys_funcs_user(integr)

  # calculate static gravitational potential
  self.gravity.post_user_init(gc)


# --------------------------------------------------------------------------

cdef void init_data(BnzIntegr integr, GridCoord *gc):

  integr.data = IntegrData()
  cdef:
    IntegrData dat = integr.data
    IntegrScratch scr = integr.scratch
    int nx=gc.Ntot[0], ny=gc.Ntot[1], nz=gc.Ntot[2]

  # Allocate arrays used by the MHD integrator.

  sh_u = (NMODE,nz,ny,nx)
  sh_3 = (3,nz,ny,nx)

  # predictor-step arrays of cell-centered conserved variables
  dat.cons_s =  np.zeros(sh_u, dtype=np_real)
  if integr.tintegr==TINT_RK3:
    dat.cons_ss =  np.zeros(sh_u, dtype=np_real)

  # Godunov fluxes
  dat.flx_x = np.zeros(sh_u, dtype=np_real)
  dat.flx_y = np.zeros(sh_u, dtype=np_real)
  dat.flx_z = np.zeros(sh_u, dtype=np_real)

  IF MFIELD:

    # predictor-step arrays of face-centered magnetic field
    dat.bfld_s =  np.zeros(sh_3, dtype=np_real)
    if integr.tintegr==TINT_RK3:
      dat.bfld_ss =  np.zeros(sh_3, dtype=np_real)

    # edge- and cell-centered electric field
    dat.eflde = np.zeros(sh_3, dtype=np_real)
    dat.efldc = np.zeros(sh_3, dtype=np_real)

  ELSE:

    dat.bfld_s = None
    if integr.tintegr==TINT_RK3:
      dat.bfld_ss = None
    dat.eflde  = None
    dat.efldc = None

  # allocate scratch arrays used in reconstruction and Godunov flux calculation
  scr.w_rcn = <real****>calloc_4d_array(OMP_NT, 8, NMODE, nx+4)
  scr.wl    =  <real***>calloc_3d_array(OMP_NT,    NMODE, nx+4)
  scr.wl_   =  <real***>calloc_3d_array(OMP_NT,    NMODE, nx+4)
  scr.wr    =  <real***>calloc_3d_array(OMP_NT,    NMODE, nx+4)


# -------------------------------------------------------------

cdef int1d get_integr_lims(int order, int *Ntot):

  cdef int1d lims = np.asarray([order+1, Ntot[0]-2-order,
                                order+1, Ntot[1]-2-order,
                                order+1, Ntot[2]-2-order], dtype=int)
  IF not D2D: lims[2],lims[3] = 0,0
  IF not D3D: lims[4],lims[5] = 0,0

  return lims
