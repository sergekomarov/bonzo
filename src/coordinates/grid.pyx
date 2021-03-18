# -*- coding: utf-8 -*-

IF MPI:
  from mpi4py import MPI as mpi
  from mpi4py cimport MPI as mpi

import numpy as np
cimport numpy as np

import sys
from libc.stdlib cimport free, calloc

from bnz.util cimport free_2d_array, free_3d_array, print_root
from coord cimport init_coord, free_coord_data
# from interpolate cimport set_interp_coeff
cimport bnz.mhd.eos as eos

from bnz.io.read_config import read_param
from hilbertcurve import HilbertCurve

IF SPREC:
  np_real = np.float32
ELSE:
  np_real = np.float64

# ---------------------------------------------------------------------

cdef class GridData:

  def __cinit__(self, GridCoord *gc):

    sh_3 = (3, gc.Ntot[2], gc.Ntot[1], gc.Ntot[0])

    sh_u = (NMODE, gc.Ntot[2], gc.Ntot[1], gc.Ntot[0])
    sh_4 = (4,     gc.Ntot[2], gc.Ntot[1], gc.Ntot[0])

    # cell-centered conserved variables
    self.cons = np.zeros(sh_u, dtype=np_real)

    # cell-centered primitive variables
    self.prim = np.zeros(sh_u, dtype=np_real)

    IF MFIELD:
      # face-centered magnetic field
      self.bfld = np.zeros(sh_3, dtype=np_real)
    ELSE:
      # want to be able to use B as a function parameter without function overloading
      self.bfld = None

    IF MHDPIC:
      # array to store particle feedback force
      self.fcoup = np.zeros(sh_4, dtype=np_real)


# ----------------------------------------------------------------------

cdef class BnzGrid:

  def __cinit__(self, str usr_dir):

    self.usr_dir = usr_dir

    # set index-related parameters
    init_params(self.coord, usr_dir)

    # do domain decomposition
    IF MPI: domain_decomp(self.coord, usr_dir)

    # init boundary conditions
    self.bc = GridBC(self.coord, usr_dir)

    # init coordinate system
    init_coord(self.coord, usr_dir)

    # set interface interpolation coefficients
    # set_interp_coeff(self.coord)

    # init data arrays
    self.data = GridData(self.coord)

    # init particles
    IF MHDPIC: self.prts = BnzParticles(self.coord, usr_dir)

  def __dealloc__(self):

    free_coord_data(self.coord)

  cdef void cons2prim(self, int *lims, real gam):

    eos.cons2prim(self.data.prim, self.data.cons, lims, gam)

  cdef void prim2cons(self, int *lims, real gam):

    eos.prim2cons(self.data.cons, self.data.prim, lims, gam)

  cdef void apply_grid_bc(self, BnzIntegr integr, int1d bvars):
    self.bc.apply_grid_bc(self.data,self.coord, integr, bvars)

  IF MHDPIC:
    cdef void apply_prt_bc(self, BnzIntegr integr):
      self.prts.bc.apply_prt_bc(self.prts.data, self.prts.prop,
                                self.data, self.coord, integr)


# ---------------------------------------------------------------------

cdef void init_params(GridCoord *gc, str usr_dir):

  # global box size in active cells
  gc.Nact_glob[0] = read_param("computation","Nx",'i',usr_dir)
  gc.Nact_glob[1] = read_param("computation","Ny",'i',usr_dir)
  gc.Nact_glob[2] = read_param("computation","Nz",'i',usr_dir)

  IF not D2D:
    if gc.Nact_glob[1] != 1:
      print_root('Error: cannot set Ny>1 without second dimension.')
      sys.exit()

  IF not D3D:
    if gc.Nact_glob[2] != 1:
      print_root('Error: cannot set Nz>1 without third dimension.')
      sys.exit()

  # Set the number of ghost cells.

  tintegr  = read_param("computation", "tintegr", 's',usr_dir)
  reconstr = read_param("computation", "reconstr", 's',usr_dir)

  if reconstr=='const':
    gc.ng=2
    gc.interp_order=1
  elif reconstr=='linear':
    gc.ng=3
    gc.interp_order=2
  # elif reconstr=='weno':
  #   gc.ng=3
  #   gc.interp_order=3
  # elif reconstr=='parab':
  #   gc.ng=4
  #   gc.interp_order=4
  #
  # if tintegr=='rk3':
  #   gc.ng=9

  # global box size including ghost cells
  gc.Ntot_glob[0] = gc.Nact_glob[0] + 2*gc.ng
  IF D2D: gc.Ntot_glob[1] = gc.Nact_glob[1] + 2*gc.ng
  ELSE:   gc.Ntot_glob[1] = 1
  IF D3D: gc.Ntot_glob[2] = gc.Nact_glob[2] + 2*gc.ng
  ELSE:   gc.Ntot_glob[2] = 1

  # Set min/max indices of active cells assuming no MPI.
  # (will be reset when domain decomposition is performed)

  gc.i1, gc.i2 = gc.ng, gc.Nact_glob[0] + gc.ng - 1

  IF D2D: gc.j1, gc.j2 = gc.ng, gc.Nact_glob[1] + gc.ng - 1
  ELSE: gc.j1, gc.j2 = 0,0

  IF D3D: gc.k1, gc.k2 = gc.ng, gc.Nact_glob[2] + gc.ng - 1
  ELSE: gc.k1, gc.k2 = 0,0

  # Set the same local size as global for now.
  # (will be reset when domain decomposition is used)

  for n in range(3):
    gc.Nact[n] = gc.Nact_glob[n]
    gc.Ntot[n] = gc.Ntot_glob[n]

  # Choose coordinate geometry and set default grid limits.

  geom = read_param("physics","geometry",'s',usr_dir)
  cdef:
    real *lmin = gc.lmin
    real *lmax = gc.lmax

  if geom=='car':
    gc.geom=CG_CAR
    lmin[0], lmax[0] = 0.,1.
    lmin[1], lmax[1] = 0.,1.
    lmin[2], lmax[2] = 0.,1.
  elif geom=='cyl':
    gc.geom==CG_CYL
    lmin[0], lmax[0] = 0., 1.
    lmin[1], lmax[1] = 0., 2*B_PI
    lmin[2], lmax[2] = 0., 1.
  elif geom=='sph':
    gc.geom=CG_SPH
    lmin[0], lmax[0] = 0., 1.
    lmin[1], lmax[1] = 0.5*B_PI-1e-3, 0.5*B_PI+1e-3
    lmin[2], lmax[2] = 0., 2*B_PI

  # IF not D2D:
  #   if gc.geom=CG_SPH:
  #     print_root('Error: please choose cylindrical coordinates instead when '+
  #                'not using the theta (meridional) axis.')
  #     sys.exit()

  # Set global coordinate limits.

  lmin[0] = read_param("computation","xmin",'f',usr_dir)
  lmax[0] = read_param("computation","xmax",'f',usr_dir)

  IF D2D:
    lmin[1] = read_param("computation","ymin",'f',usr_dir)
    lmax[1] = read_param("computation","ymax",'f',usr_dir)

  IF D3D:
    lmin[2] = read_param("computation","zmin",'f',usr_dir)
    lmax[2] = read_param("computation","zmax",'f',usr_dir)

  # Correct boundaries as appropriate for cylindrical or spherical coordinates.

  if gc.geom==CG_CYL or gc.geom==CG_SPH:
    for n in range(3):
      if lmin[n]<0.: lmin[n]=0.

  if gc.geom==CG_CYL:
    if lmax[1]>=2*B_PI-1e-2: lmax[1] = 2*B_PI

  if gc.geom==CG_SPH:
    if lmax[1]>=B_PI-1e-2: lmax[1] = B_PI
    if lmax[2]>=2*B_PI-1e-2: lmax[2] = 2*B_PI

  # set default MPI params
  gc.rank=0
  for k in range(3):
    gc.pos[k]=0
    gc.size[k]=1
    gc.size_tot=1
  gc.ranks = NULL


#-----------------------------------------------------------

IF MPI:

  cdef void domain_decomp(GridCoord *gc, str usr_dir):

    cdef mpi.Comm comm = mpi.COMM_WORLD

    cdef:
      int i,k, size0
      int p
      long[:,::1] crds

    # read number of MPI blocks in each direction
    gc.size[0] = read_param("computation", "nblocks_x", 'i', usr_dir)
    gc.size[1] = read_param("computation", "nblocks_y", 'i', usr_dir)
    gc.size[2] = read_param("computation", "nblocks_z", 'i', usr_dir)

    gc.rank = comm.Get_rank()
    size0 = comm.Get_size()
    gc.size_tot = gc.size[0] * gc.size[1] * gc.size[2]

    # check if number of blocks is consistent with number of processes and grid size

    if gc.size_tot != size0:
      print_root("Error: Total number of MPI blocks is not equal to number of processors.\n")
      sys.exit()

    for k in range(3):
      if gc.Nact_glob[k] % gc.size[k] != 0:
        print_root("Error: Number of MPI blocks is not a multiple of grid size in %i-direction.\n", k)
        sys.exit()

    hilbert_curve = None

    # distribute MPI blocks across processors

    if gc.size[1]>1 and gc.size[2]>1:

      # check if numbers of blocks in each direction are same and equal to 2**p
      if gc.size[0]==gc.size[1] and gc.size[1]==gc.size[2] and (gc.size[0] & (gc.size[0] - 1)) == 0:

        print_root('Using 3d Hilbert-curve domain decomposition...\n')

        p=<int>np.log2(gc.size[0])

        # generate a 3D Hilbert curve
        hilbert_curve = HilbertCurve(p,3)

        crd = hilbert_curve.coordinates_from_distance(gc.rank)
        gc.pos[0] = crd[0]
        gc.pos[1] = crd[1]
        gc.pos[2] = crd[2]

        # print gc.id, crd

    elif gc.size[1]>1:

      if gc.size[0]==gc.size[1] and (gc.size[0] & (gc.size[0] - 1)) == 0:

        print_root('Using 2d Hilbert-curve domain decomposition...\n')

        p=<int>np.log2(gc.size[0])

        # generate a 2D Hilbert curve
        hilbert_curve = HilbertCurve(p,2)

        crd = hilbert_curve.coordinates_from_distance(gc.rank)
        gc.pos[0] = crd[0]
        gc.pos[1] = crd[1]
        gc.pos[2] = 0

        # print gc.id, crd

    # if not successful, use the simplest possible arrangement of blocks

    if hilbert_curve==None:

      gc.pos[2] =  gc.rank / (gc.size[0] * gc.size[1])
      gc.pos[1] = (gc.rank % (gc.size[0] * gc.size[1])) / gc.size[0]
      gc.pos[0] = (gc.rank % (gc.size[0] * gc.size[1])) % gc.size[0]

    # gather positions of all MPI blocks
    crds = np.empty((gc.size_tot, 3), dtype=np.intc)
    comm.Allgather([np.array(gc.pos), mpi.INT], [crds, mpi.INT])

    gc.ranks = <int ***>calloc_3d_array(gc.size[0], gc.size[1], gc.size[2], sizeof(int))

    for i in range(gc.size_tot):
      gc.ranks[crds[i,0], crds[i,1], crds[i,2]] = i

    # if gc.rank==0:
    #   np.save('ids.npy', np.asarray(gc.ranks))

    # for i in range(gc.size[0]):
    #   for j in range(gc.size[1]):
    #     for m in range(gc.size[2]):
    #       gc.ranks[i,j,m] = i * gc.size[1] * gc.size[2] + j * gc.size[2] + m


    # save IDs of neighboring blocks

    gc.nbr_ranks[0][0] = gc.ranks[gc.pos[0]-1, gc.pos[1],   gc.pos[2]]   if gc.pos[0] != 0 else -1
    gc.nbr_ranks[0][1] = gc.ranks[gc.pos[0]+1, gc.pos[1],   gc.pos[2]]   if gc.pos[0] != gc.size[0]-1 else -1
    gc.nbr_ranks[1][0] = gc.ranks[gc.pos[0],   gc.pos[1]-1, gc.pos[2]]   if gc.pos[1] != 0 else -1
    gc.nbr_ranks[1][1] = gc.ranks[gc.pos[0],   gc.pos[1]+1, gc.pos[2]]   if gc.pos[1] != gc.size[1]-1 else -1
    gc.nbr_ranks[2][0] = gc.ranks[gc.pos[0],   gc.pos[1],   gc.pos[2]-1] if gc.pos[2] != 0 else -1
    gc.nbr_ranks[2][1] = gc.ranks[gc.pos[0],   gc.pos[1],   gc.pos[2]+1] if gc.pos[2] != gc.size[2]-1 else -1

    # print rank, gc.x1nbr_id, gc.xid, gc.x2nbr_id
    # print rank, gc.y1nbr_id, gc.yid, gc.y2nbr_id
    # print rank, gc.z1nbr_id, gc.zid, gc.z2nbr_id

    for k in range(3): gc.Nact[k] /= gc.size[k]

    gc.Ntot[0] = gc.Nact[0] + 2*gc.ng
    IF D2D:
      gc.Ntot[1] = gc.Nact[1] + 2*gc.ng
    ELSE:
      gc.Ntot[1] = 1
    IF D3D:
      gc.Ntot[2] = gc.Nact[2] + 2*gc.ng
    ELSE:
      gc.Ntot[2] = 1

    gc.i1, gc.i2 = gc.ng, gc.Nact[0] + gc.ng - 1

    IF D2D: gc.j1, gc.j2 = gc.ng, gc.Nact[1] + gc.ng - 1
    ELSE:   gc.j1, gc.j2 = 0,0

    IF D3D: gc.k1, gc.k2 = gc.ng, gc.Nact[2] + gc.ng - 1
    ELSE:   gc.k1, gc.k2 = 0,0
