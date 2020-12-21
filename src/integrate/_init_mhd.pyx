# -*- coding: utf-8 -*-

IF MPI:
  from mpi4py import MPI as mpi
  from mpi4py cimport MPI as mpi

import numpy as np
cimport numpy as np
import sys, os

from libc.math cimport M_PI, sqrt, fmax
from libc.stdlib cimport rand, RAND_MAX, srand, malloc, calloc, free


from bnz.utils cimport maxi,mini, print_root

cimport eos_cy
cimport ct

cimport bnz.read_config as read_config

cimport bnz.output as output
cimport bnz.restart as restart

cimport bnz.bc.bc_grid as bc_grid
IF MPI:
  cimport bnz.init_mpi_blocks as init_mpi_blocks
cimport bnz.problem.problem as problem

cimport bnz.gravity as gravity
cimport bnz.turb_driv as turb_driv
cimport bnz.dissipation.diffuse as diffuse

from bnz.coord cimport lind2gcrd_x, lind2gcrd_y, lind2gcrd_z

IF MHDPIC:
  cimport bnz.bc.bc_prt as bc_prt
  cimport bnz.utils_particle as utils_particle



IF SPREC:
  np_real = np.float32
ELSE:
  np_real = np.float64


# ====================================================

# Initialize everything, prepare for integration.

cdef void init(BnzSim sim, bytes usr_dir):

  cdef:

    BnzGrid grid = sim.grid
    GridParams gp = grid.params
    GridData gd = grid.data

    BnzMethod method = sim.method
    BnzBC bc = sim.bc
    BnzPhysics phys = sim.phys
    BnzOutput out = sim.output

  IF MHDPIC:
    cdef BnzParticles prts = sim.prts

  cdef ints rank = 0
  IF MPI:
    rank = mpi.COMM_WORLD.Get_rank()


  # Try to obtain parameters from the configuration file.
  print_root(rank, 'read the input file...\n')
  sim.output.usr_dir = usr_dir
  read_config.set_params(sim)

  print_root(rank, 'initialize grids...\n')
  init_grids(grid, phys, method)

  IF MHDPIC:
    print_root(rank, 'initialize particles...\n')
    init_particles(prts, phys, method, gp.Nact)

  if not out.restart:
    print_root(rank, 'set the problem...\n')
    problem.set_problem(sim)
    sim.t = 0.
    sim.step = 0
  else:
    print_root(rank, 'restart...\n')
    restart.set_restart(sim)

  # print 'rank =', rank, '|', dom.blocks.x1nbr_id, dom.blocks.x2nbr_id, dom.blocks.y1nbr_id, dom.blocks.y2nbr_id

  print_root(rank, 'set grid boundary condition pointers...\n')
  bc_grid.set_bc_grid_ptrs(bc, gp)
  # if rank==0: np.save('blocks2.npy', np.asarray(dom.blocks.ids))

  # print 'rank =', rank, '|', dom.blocks.x1nbr_id, dom.blocks.x2nbr_id, dom.blocks.y1nbr_id, dom.blocks.y2nbr_id

  IF MHDPIC:
    print_root(rank, 'set particle boundary condition pointers...\n')
    bc_prt.set_bc_prt_ptrs(bc, gp)

  IF MPI:
    print_root(rank, 'allocate BC MPI buffers...\n')
    init_mpi_blocks.init_bc_buffers(sim)


  cdef:
    ints lims_act[6]
    ints lims_tot[6]
  lims_act[:] = [gp.i1,gp.i2, gp.j1,gp.j2, gp.k1,gp.k2]
  lims_tot[:] = [0,gp.Ntot[0]-1, 0,gp.Ntot[1]-1, 0,gp.Ntot[2]-1]


  print_root(rank, 'apply grid BC...\n')
  IF MFIELD:
    # apply to all cell-centered hydro variables and face-centered magnetic field
    bc_grid.apply_bc_grid(sim, np.arange(NVARS))
    # get cell-centered m. field from face-centered
    ct.interp_b_field(gd.W, gd.B, lims_act)
    # apply BC to cell-centered field (need only the rightmost layer)
    bc_grid.apply_bc_grid(sim, np.array([BXC,BYC,BZC]))
  ELSE:
    # apply to all cell-centered hydro variables and face-centered magnetic field
    bc_grid.apply_bc_grid(sim, np.arange(NVARS))


  print_root(rank, 'convert to conserved variables...\n')
  eos_cy.prim2cons_3(gd.U, gd.W, lims_tot, phys.gam)

  print_root(rank, 'initialize output...\n')
  output.init_output(sim)

  print_root(rank, 'initialize integrator...\n')
  init_integrate(grid, phys, method)

  if not out.restart:
    print_root(rank, 'write initial conditions...\n')
    output.write_output(sim)



#=============================================================

# Initialize grids and MHD data structures.

cdef void init_grids(BnzGrid grid, BnzPhysics phys, BnzMethod method):

  cdef:
    GridParams gp = grid.params
    GridData gd = grid.data

  # cell size
  gp.dl[0] = gp.L[0]/gp.Nact[0]

  IF not D2D:
    gp.Nact[1] = 1
    gp.L[1] = gp.dl[0]
  IF not D3D:
    gp.Nact[2] = 1
    gp.L[2] = gp.dl[0]

  gp.dl[1] = gp.L[1]/gp.Nact[1]
  gp.dl[2] = gp.L[2]/gp.Nact[2]

  gp.dli[0] = 1./gp.dl[0]
  gp.dli[1] = 1./gp.dl[1]
  gp.dli[2] = 1./gp.dl[2]


  # set number of ghost cells

  if method.tintegr==TINT_VL:

    if method.reconstr==RCN_CONST: gp.ng=2
    elif method.reconstr==RCN_LINEAR or method.reconstr==RCN_WENO: gp.ng=3
    elif method.reconstr==RCN_PARAB: gp.ng=4

  if method.tintegr==TINT_RK3:
    gp.ng=9


  # decompose domain into MPI blocks
  IF MPI: init_mpi_blocks.init_blocks(gp)
  # after this gp.Ntot/act and gp.L are LOCAL

  # box size including ghost cells

  gp.Ntot[0] = gp.Nact[0] + 2*gp.ng
  IF D2D: gp.Ntot[1] = gp.Nact[1] + 2*gp.ng
  ELSE: gp.Ntot[1] = 1
  IF D3D: gp.Ntot[2] = gp.Nact[2] + 2*gp.ng
  ELSE: gp.Ntot[2] = 1

  IF not MPI:
  # global size is same as local without domain decomposition
    for k in range(3):
      gp.Nact_glob[k] = gp.Nact[k]
      gp.Ntot_glob[k] = gp.Ntot[k]
      gp.Lglob[k] = gp.L[k]


  gp.i1, gp.i2 = gp.ng, gp.Nact[0] + gp.ng - 1

  IF D2D: gp.j1, gp.j2 = gp.ng, gp.Nact[1] + gp.ng - 1
  ELSE: gp.j1, gp.j2 = 0,0

  IF D3D: gp.k1, gp.k2 = gp.ng, gp.Nact[2] + gp.ng - 1
  ELSE: gp.k1, gp.k2 = 0,0


  # initialize locations of cell faces
  gp.xf = np.zeros(gp.Ntot[0]+1, dtype=np_real)
  gp.yf = np.zeros(gp.Ntot[1]+1, dtype=np_real)
  gp.zf = np.zeros(gp.Ntot[2]+1, dtype=np_real)


  # cell-centered conserved variables
  gd.U = np.zeros((NWAVES, gp.Ntot[2], gp.Ntot[1], gp.Ntot[0]), dtype=np_real)

  # cell-centered primitive variables
  gd.W = np.zeros((NWAVES, gp.Ntot[2], gp.Ntot[1], gp.Ntot[0]), dtype=np_real)

  IF MFIELD:
    # face-centered magnetic field
    gd.B = np.zeros((3, gp.Ntot[2], gp.Ntot[1], gp.Ntot[0]), dtype=np_real)
  ELSE:
    # want to be able to use B as a function input without function overloading
    gd.B = None

  IF MHDPIC:
    # array to store particle feedback force
    gd.CoupF = np.zeros((4, gp.Ntot[2], gp.Ntot[1], gp.Ntot[0]), dtype=np_real)


  # initialize random number generator with different seeds
  IF MPI:
    srand(gp.rank)
  ELSE:
    np.random.seed()
    srand(np.random.randint(RAND_MAX))



# ==============================================================================

# Initialize particle data structures and set kernel function pointers.
# MHD PIC

IF MHDPIC:

  cdef void init_particles(BnzParticles prts, BnzPhysics phys,
                           BnzMethod method, ints Nact[3]):

    # set total number of particles

    IF D2D and D3D:
      prts.ppc = (<ints>(prts.ppc**(1./3)))**3
      prts.Np = prts.ppc * Nact[0] * Nact[1] * Nact[2]

    ELIF D2D:
      prts.ppc = (<ints>sqrt(prts.ppc))**2
      prts.Np = prts.ppc * Nact[0] * Nact[1]

    ELSE:
      prts.Np = prts.ppc * Nact[0]

    prts.Nprop = 10

    # allocate particle array

    prts.Npmax = prts.Np      # only when there is no injection of particles !!!
    IF MPI:
      prts.Npmax = <ints>(1.3*prts.Npmax)

    cdef ParticleData *pd = &(prts.data)

    pd.x = <real *>calloc(prts.Npmax, sizeof(real))
    pd.y = <real *>calloc(prts.Npmax, sizeof(real))
    pd.z = <real *>calloc(prts.Npmax, sizeof(real))

    pd.u = <real *>calloc(prts.Npmax, sizeof(real))
    pd.v = <real *>calloc(prts.Npmax, sizeof(real))
    pd.w = <real *>calloc(prts.Npmax, sizeof(real))
    pd.g = <real *>calloc(prts.Npmax, sizeof(real))

    pd.m = <real *>calloc(prts.Npmax, sizeof(real))
    pd.spc = <ints *>calloc(prts.Npmax, sizeof(ints))
    pd.id = <ints *>calloc(prts.Npmax, sizeof(ints))

    # allocate array that stores properties of different particle species
    prts.Ns=1    # one CR specie by default (can be overloaded in the user file)
    prts.spc_props = <SpcProps *>calloc(prts.Ns, sizeof(SpcProps))

    prts.spc_props[0].qm = phys.qomc
    prts.spc_props[0].Np = prts.Np

    # set function pointers for interpolation kernels
    if method.Ninterp==1:
      method.weight_func = utils_particle.getweight1
    elif method.Ninterp==2:
      method.weight_func = utils_particle.getweight2



# ====================================================

# Initialize integrator arrays, set function pointers.

cdef void init_integrate(BnzGrid grid, BnzPhysics phys, BnzMethod method):

  cdef:
    GridParams gp = grid.params
    GridData gd = grid.data
    GridScratch scr = grid.scr
    ints i,j,k
    ints Nx=gp.Ntot[0], Ny=gp.Ntot[1], Nz=gp.Ntot[2]
    real x=0,y=0,z=0


  # allocate arrays used by MHD integrator

  sh_u = (NWAVES,Nz,Ny,Nx)
  sh_3 = (3,Nz,Ny,Nx)

  # predictor-step arrays of cell-centered conserved variables
  gd.Us =  np.zeros(sh_u, dtype=np_real)
  if method.tintegr==TINT_RK3:
    gd.Uss =  np.zeros(sh_u, dtype=np_real)

  # Godunov fluxes
  gd.Fx =  np.zeros(sh_u, dtype=np_real)
  gd.Fy =  np.zeros(sh_u, dtype=np_real)
  gd.Fz =  np.zeros(sh_u, dtype=np_real)

  IF MFIELD:

    # predictor-step array of face-centered magnetic field
    gd.Bs =  np.zeros(sh_3, dtype=np_real)
    if method.tintegr==TINT_RK3:
      gd.Bss =  np.zeros(sh_3, dtype=np_real)

    # edge- and cell-centered electric field
    gd.E =  np.zeros(sh_3, dtype=np_real)
    gd.Ec = np.zeros(sh_3, dtype=np_real)

  ELSE:

    gd.Bs = None
    if method.tintegr==TINT_RK3:
      gd.Bss = None
    gd.E  = None
    gd.Ec = None


  # set pointers to user-defined physics functions (e.g. gravitational potential)
  problem.set_phys_ptrs_user(phys)

  # initialize solenoidal driving force
  if phys.f != 0.: turb_driv.init_turb_driv(grid, phys)

  # set static gravitational potential
  if phys.g0 != 0: gravity.init_gravity(grid, phys)


  # allocate scratch arrays

  IF MPI and MHDPIC:
    # temporary array for particle deposit exchange at boundaries
    scr.CoupF_tmp = np.zeros((4,Nz,Ny,Nx), dtype=np_real)


  # initialize diffusion integrator
  if (phys.mu != 0 or phys.eta !=0 or phys.mu4 != 0 or phys.eta4 != 0
    or phys.kappa0 != 0 or phys.kL != 0 or phys.nuiic0 != 0):

    diffuse.init_diffuse(grid, phys, method.sts)


  # set function pointers for Godunov flux calculation

  if method.rsolver == RS_HLL:
    method.rsolver_func = &HLLflux

  elif method.rsolver == RS_HLLT:
    method.rsolver_func = &HLLTflux

  IF not MFIELD:
    if method.rsolver == RS_HLLC:
      method.rsolver_func = &HLLCflux

  IF MFIELD:
    if method.rsolver == RS_HLLD:
      method.rsolver_func = &HLLDflux

  IF CGL:
    if method.rsolver == RS_HLLA:
      method.rsolver_func = &HLLAflux

  # set function pointers for spatial reconstruction

  if method.reconstr == RCN_CONST:
    method.reconstr_func = &reconstr_const
    method.reconstr_order=0

  elif method.reconstr == RCN_LINEAR:
    method.reconstr_func = &reconstr_linear
    method.reconstr_order=1

  elif method.reconstr == RCN_PARAB:
    method.reconstr_func = &reconstr_parab
    method.reconstr_order=2

  elif method.reconstr == RCN_WENO:
    method.reconstr_func = &reconstr_weno
    method.reconstr_order=1


  # initialize random-number generator
  # IF MPI:
  #   srand(dom.blocks.comm.Get_rank())
  # ELSE:
  #   np.random.seed()
  #   srand(np.random.randint(RAND_MAX))
