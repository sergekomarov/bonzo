# -*- coding: utf-8 -*-

import numpy as np
cimport numpy as np

from bnz.utils cimport calloc_3d_array, calloc_4d_array, free_3d_array
from bnz.io.read_config import read_param
from bnz.problem.problem cimport set_phys_funcs_user

cimport ct
cimport godunov
cimport diagnostics
cimport new_dt
cimport eos_cy
cimport bnz.srcterms.gravity as gravity
cimport bnz.srcterms.turb_driv_cy as turb_driv


cdef class BnzIntegr:

  def __cinit__(self, GridCoord *gc, bytes usr_dir):

    init_method(self, gc, usr_dir)
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

    print_root("%i-order Godunov fluxes... ", order+1)
    gettimeofday(&tstart, NULL)

    # get Goduniv fluxes flx_x,flx_y,flx_z from prim0
    godunov.godunov_fluxes(idat.flx_x, idat.flx_y, idat.flx_z,
                           prim0, bf0, gc, lims, self, order)

    gettimeofday(&tstop, NULL)
    print_root(rank, "%.1f ms\n", timediff(tstart,tstop))


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

      print_root("E-field at cell centers... \n")
      # use Ohm's law to get cell-centered E-field from primitive variables
      ct.ec_from_prim(idat.ec, prim0, lims)

      print_root("interpolate E-field to cell edges... ")
      gettimeofday(&tstart, NULL)

      # interpolate edge-centered E-field (ee) from cell-centered (ec) and fluxes
      ct.interp_ee2(idat.ee, idat.ec, idat.flx_x, idat.flx_y, idat.flx_z, lims,
                    prim0[RHO,...], self.dt)

      gettimeofday(&tstop, NULL)
      print_root(rank, "%.1f ms\n", timediff(tstart,tstop))

      print_root("advance B-field via CT... ")
      gettimeofday(&tstart, NULL)

      # advance B-field using edge-centered E-field via constrained transport
      ct.advance_b(bf1, bf0, idat.ee, gc, lims, dt)

      gettimeofday(&tstop, NULL)
      print_root(rank, "%.1f ms\n", timediff(tstart,tstop))

      print_root("interpolate B-field from cell faces to centers...\n")
      ct.interp_bc(cons1, bf1, gc, lims)

  # ------------------------------------------------------------------------

  # Add source terms.

  cdef void add_source_terms(self, real4d u1, real4d bf1,
                             real4d u0, real4d bf0,
                             GridCoord *gc, int order, real dt):

    cdef:
      timeval tstart, tstop
      IntegrData idat = self.data
      int lims[6]

    lims = &(get_integr_lims(order, gc.Ntot)[0])

    if self.gravity.g0 != 0.:

      print_root("gravity (predictor) ... ")
      gettimeofday(&tstart, NULL)

      # predictor: update cons1 using cons0 and fluxes(cons0)
      self.gravity.apply_gravity(u1, u0, idat.flx_x, idat.flx_y, idat.flx_z,
                                 gc, lims, dt)

      gettimeofday(&tstop, NULL)
      print_root("%.1f ms\n", timediff(tstart,tstop))

    #--------------------

    if self.turb_driv.f0 != 0.:

      print_root("turbulence driving... ")
      gettimeofday(&tstart, NULL)

      self.turb_driv.apply_turb_driv(u1, u0, lims, dt)

      gettimeofday(&tstop, NULL)
      print_root("%.1f ms\n", timediff(tstart,tstop))

    #--------------------

    # IF CGL or TWOTEMP:

      # print_root("collisions and effects of instabilities... ")
      # gettimeofday(&tstart, NULL)
      #
      # # apply implicit collision operator to pressures
      # collisions.collide(u1, gd.nuii_eff, lims, dt)
      #                 # updates U1 and gd.nuii_eff
      #
      # gettimeofday(&tstop, NULL)
      # print_root("%.1f ms\n", timediff(tstart,tstop))


  # ----------------------------------------------------------------------------

  cdef void update_phys(self, GridCoord *gc, int *lims, real dt):

    # Evolve quantities not directly related to the main hydro array
    # (e.g. the driving force).

    cdef timeval tstart, tstop

    if self.turb_driv.f0 != 0.:

      print_root("advance driving force... ")
      gettimeofday(&tstart, NULL)

      # advance drifing force from t_n-1/2 to t_n+1/2
      self.turb_driv.advance_driv_force(gc, lims, dt)

      gettimeofday(&tstop, NULL)
      print_root(rank, "%.1f ms\n", timediff(tstart,tstop))



  # ----------------------------------------------------------------------------

  cdef void new_dt_diag(self, real4d prim, GridCoord *gc):

    cdef timeval tstart, tstop

    print_root("set new dt...\n")
    gettimeofday(&tstart, NULL)

    self.dt = new_dt.new_dt(prim, gc, self)

    gettimeofday(&tstop, NULL)
    print_root("dt = %f, done in %.1f ms\n\n", self.dt, timediff(tstart,tstop))

  # ----------------------------------------------------------------------------

  cdef void cons2prim_diag(self, real4d prim, real4d cons, int *lims):

    cdef timeval tstart, tstop

    print_root("convert to primitive variables... ")
    gettimeofday(&tstart, NULL)

    eos_cy.cons2prim_3(prim, cons, lims, self.gam)

    gettimeofday(&tstop, NULL)
    print_root("%.1f ms\n", timediff(tstart,tstop))

  # ----------------------------------------------------------------------------

  cdef void prim2cons_diag(self, real4d cons, real4d prim, int *lims):

    cdef timeval tstart, tstop

    print_root("convert to conserved variables... ")
    gettimeofday(&tstart, NULL)

    eos_cy.prim2cons_3(cons, prim, lims, self.gam)

    gettimeofday(&tstop, NULL)
    print_root("%.1f ms\n", timediff(tstart,tstop))

  # ----------------------------------------------------------------------------

  cdef void diffuse_diag(self, real4d prim, real4d bf, BnzSim sim):

    cdef timeval tstart, tstop


    print_root(rank, "diffusion... ")
    gettimeofday(&tstart, NULL)

    # in primitive variables
    diffuse.diffuse(prim, bf, sim)

    gettimeofday(&tstop, NULL)
    print_root(rank, "%.1f ms\n", timediff(tstart,tstop))

  # ----------------------------------------------------------------------------

  cdef void print_nrg_diag(self, BnzGrid grid):

    cdef timeval tstart, tstop

    gettimeofday(&tstart, NULL)

    diagnostics.print_nrg(grid, self)

    gettimeofday(&tstop, NULL)
    print_root("mean energy densities calculated in %.1f ms\n",
               timediff(tstart,tstop))



  # ----------------------------------------------------------------------------

  # cdef void apply_bc_grid_diag(BnzSim sim, int[::1] bvars):
  #
  #   cdef:
  #     timeval tstart, tstop
  #     int rank=0
  #
  #   IF MPI: rank = mpi.COMM_WORLD.Get_rank()
  #
  #   print_root(rank, "apply MHD BC... ")
  #   gettimeofday(&tstart, NULL)
  #
  #   # in primitive variables
  #   bc_grid.apply_bc_grid(sim, bvars)
  #
  #   gettimeofday(&tstop, NULL)
  #   print_root(rank, "%.1f ms\n", timediff(tstart,tstop))






# ================================================================

cdef void init_method(BnzIntegr itg, GridCoord *gc, bytes usr_dir):

  itg.rsolver_func = NULL
  itg.reconstr_func = NULL
  # itg.limiter_func=NULL
  # IF MHDPIC:
  #   itg.weight_func = NULL

  # Set integrator properties from config file.

  # Courant number
  itg.cour = read_param("computation", "cour", 'f', usr_dir)

  # time integration

  tintegr = read_param("computation", "tintegr", 's',usr_dir)
  if tintegr == 'vl': itg.tintegr = TINT_VL
  elif tintegr == 'rk3': itg.tintegr = TINT_RK3
  else: itg.tintegr = TINT_VL

  # Riemann solver

  rsolver = read_param("computation", "rsolver", 's', usr_dir)
  if rsolver == 'hll': itg.rsolver = RS_HLL
  elif rsolver == 'hllc': itg.rsolver = RS_HLLC
  elif rsolver == 'hlld': itg.rsolver = RS_HLLD
  elif rsolver == 'hllt': itg.rsolver = RS_HLLT
  elif rsolver == 'hlla': itg.rsolver = RS_HLLA
  else: itg.rsolver = RS_HLL

  # spatial reconstruction

  reconstr = read_param("computation", "reconstr", 's',usr_dir)
  if reconstr == 'const': itg.reconstr = RCN_CONST
  elif reconstr == 'linear': itg.reconstr = RCN_LINEAR
  elif reconstr == 'parab': itg.reconstr = RCN_PARAB
  elif reconstr == 'weno': itg.reconstr = RCN_WENO
  else: itg.reconstr = RCN_LINEAR

  # limiter function

  # limiter = read_param("computation", "limiter", 's', usr_dir)
  # if limiter == 'mm': itg.limiter = LIM_MM
  # elif limiter == 'mc': itg.limiter = LIM_MC
  # elif limiter == 'vl': itg.limiter = LIM_VL
  # else: itg.limiter = LIM_VL

  # liiting in characteristic variables
  itg.charact_proj = read_param("computation", "charact_proj", 'i',usr_dir)
  IF CGL or TWOTEMP: itg.charact_proj = 0

  # apply pressure floor
  # itg.pressure_floor = read_param("computation", "pressure_floor", 'f',usr_dir)

  # Set function pointers.

  if itg.rsolver == RS_HLL:
    itg.rsolver_func = &hll_flux

  elif itg.rsolver == RS_HLLT:
    itg.rsolver_func = &hllt_flux

  IF not MFIELD:
    if itg.rsolver == RS_HLLC:
      itg.rsolver_func = &hllc_flux
  ELSE:
    if itg.rsolver == RS_HLLD:
      itg.rsolver_func = &hlld_flux

  IF CGL:
    if itg.rsolver == RS_HLLA:
      itg.rsolver_func = &hlla_flux

  if itg.reconstr == RCN_CONST:
    itg.reconstr_func = &reconstr_const
    itg.reconstr_order=0

  elif itg.reconstr == RCN_LINEAR:
    itg.reconstr_func = &reconstr_linear
    itg.reconstr_order=1

  elif itg.reconstr == RCN_PARAB:
    itg.reconstr_func = &reconstr_parab
    itg.reconstr_order=2

  elif itg.reconstr == RCN_WENO:
    itg.reconstr_func = &reconstr_weno
    itg.reconstr_order=1

  # IF MHDPIC:

    # order of interpolation
    # itg.Ninterp = read_param("computation", "Ninterp", 'i',usr_dir)

    # Set function pointers for particle interpolation kernels.
    # if itg.Ninterp==1:
    #   itg.weight_func = &getweight1
    # elif itg.Ninterp==2:
    #   itg.weight_func = &getweight2

  itg.time=0.
  itg.nstep=0
  # max time of the simulation
  itg.tmax = read_param("computation", "tmax", 'f',usr_dir)
  # timestep if fixed
  IF FIXDT: itg.dt = read_param("computation", "dt", 'f',usr_dir)


# -------------------------------------------------------------------

cdef void init_physics(BnzIntegr itg, GridCoord *gc, bytes usr_dir):

  # adiabatic index
  itg.gam = read_param("physics", "gam", 'f',usr_dir)
  IF CGL: itg.gam = 5./3

  IF MHDPIC:
    # effective speed of light
    itg.sol    = read_param("physics", "sol",    'f', usr_dir)
    # charge-to-mass ratio of CRs relative to thermal ions
    itg.q_mc   = read_param("physics", "q_mc",   'f', usr_dir)
    # CR density
    itg.rho_cr = read_param("physics", "rho_cr", 'f', usr_dir)

  # set pointers to user-defined physics functions (e.g. gravitational potential)
  set_phys_ptrs_user(itg)

  # (these could be compilated conditionaly instead:)
  self.gravity = BnzGravity(gc, grav_pot_func)
  self.turb_driv = BnzTurbDriv(gc)
  self.diffusion = BnzDiffusion(gc)


# --------------------------------------------------------------------------

cdef void init_data(BnzIntegr itg, GridCoord *gc):

  itg.data = IntegrData()
  cdef:
    IntegrData dat = itg.data
    IntegrScratch scr = itg.scratch
    int nx=gc.Ntot[0], ny=gc.Ntot[1], nz=gc.Ntot[2]

  # Allocate arrays used by the MHD integrator.

  sh_u = (NMODE,nz,ny,nx)
  sh_3 = (3,nz,ny,nx)

  # predictor-step arrays of cell-centered conserved variables
  dat.cons_s =  np.zeros(sh_u, dtype=np_real)
  if itg.tintegr==TINT_RK3:
    dat.cons_ss =  np.zeros(sh_u, dtype=np_real)

  # Godunov fluxes
  dat.flux_x = np.zeros(sh_u, dtype=np_real)
  dat.flux_y = np.zeros(sh_u, dtype=np_real)
  dat.flux_z = np.zeros(sh_u, dtype=np_real)

  IF MFIELD:

    # predictor-step arrays of face-centered magnetic field
    dat.bf_s =  np.zeros(sh_3, dtype=np_real)
    if itg.tintegr==TINT_RK3:
      dat.bf_ss =  np.zeros(sh_3, dtype=np_real)

    # edge- and cell-centered electric field
    dat.ee = np.zeros(sh_3, dtype=np_real)
    dat.ec = np.zeros(sh_3, dtype=np_real)

  ELSE:

    dat.bf_s = None
    if itg.tintegr==TINT_RK3:
      dat.bf_ss = None
    dat.ee  = None
    dat.ec = None

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
