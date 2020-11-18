# -*- coding: utf-8 -*-

from mpi4py import MPI as mpi
from mpi4py cimport MPI as mpi

import numpy as np
cimport numpy as np

import sys

from libc.stdlib cimport free, calloc

from utils cimport free_2d_array, mini,maxi
from read_config import read_param
from hilbertcurve import HilbertCurve

IF SPREC:
  np_real = np.float32
ELSE:
  np_real = np.float64


#=========================================================

# Grid class, contains parameters and data of local grid.

cdef class BnzGrid:

  def __cinit__(self, bytes usr_dir):

    # self.coord = GridCoord() # structure
    self.data = GridData()
    self.scr = GridScratch()
    self.bc = GridBC()

    self.usr_dir = usr_dir

    #-----------------------------------------------------------

    cdef GridCoord *gc = &(self.coord)

    # box size in cells
    gc.Nact_glob[0] = read_param("computation","Nx",'i',usr_dir)
    gc.Nact_glob[1] = read_param("computation","Ny",'i',usr_dir)
    gc.Nact_glob[2] = read_param("computation","Nz",'i',usr_dir)

    cdef ints ndim=1
    IF D2D: ndim += 1
    IF D3D: ndim += 1

    if ndim==1:
      if gc.Nact_glob[1] != 1:
        print 'Error: cannot set Ny>1 in 1D.'
        sys.exit()

    if ndim==2:
      if gc.Nact_glob[2] != 1:
        print 'Error: cannot set Nz>1 in 2D/1D.'
        sys.exit()

    # Set the number of ghost cells.

    Nfilt = read_param("computation", "Nfilt", 'i',usr_dir)
    gc.ng = mini(4, maxi(1, Nfilt))

    gc.Ntot_glob[0] = gc.Nact_glob[0] + 2*gc.ng+1
    IF D2D: gc.Ntot_glob[1] = gc.Nact_glob[1] + 2*gc.ng+1
    ELSE:   gc.Ntot_glob[1] = 1
    IF D3D: gc.Ntot_glob[2] = gc.Nact_glob[2] + 2*gc.ng+1
    ELSE:   gc.Ntot_glob[2] = 1

    for n in range(3):
      gc.lmin[n] = 0.
      gc.lmax[n] = <real>gc.Ntot_glob[n]

    # Set min/max indices of active cells.
    # (will be reset when domain decomposition is used)

    gc.i1, gc.i2 = gc.ng, gc.Nact_glob[0] + gc.ng - 1

    IF D2D: gc.j1, gc.j2 = gc.ng, gc.Nact_glob[1] + gc.ng - 1
    ELSE: gc.j1, gc.j2 = 0,0

    IF D3D: gc.k1, gc.k2 = gc.ng, gc.Nact_glob[2] + gc.ng - 1
    ELSE: gc.k1, gc.k2 = 0,0

    # Set the same local size as global for now.
    # (will be reset when domain decomposition is used)

    for k in range(3):
      gc.Nact[k] = gc.Nact_glob[k]
      gc.Ntot[k] = gc.Ntot_glob[k]

    gc.rank=0
    for k in range(3): gc.pos[k]=0

    #-----------------------------------------------------------------------

    # Set boundary condition parameters.

    cdef ints i,k
    cdef GridBC gbc = self.bc

    for i in range(3):
      for k in range(2):
        gbc.bc_grid_funcs[i][k] = NULL

    for i in range(3):
      for k in range(2):
        gbc.bc_exch_funcs[i][k] = NULL

    # Set boundary condition flags.

    gbc.bc_flags[0][0] = read_param("physics", "bc_x1", 'i',usr_dir)
    gbc.bc_flags[0][1] = read_param("physics", "bc_x2", 'i',usr_dir)
    gbc.bc_flags[1][0] = read_param("physics", "bc_y1", 'i',usr_dir)
    gbc.bc_flags[1][1] = read_param("physics", "bc_y2", 'i',usr_dir)
    gbc.bc_flags[2][0] = read_param("physics", "bc_z1", 'i',usr_dir)
    gbc.bc_flags[2][1] = read_param("physics", "bc_z2", 'i',usr_dir)


  # =================================================================

  cdef void init(self):

    IF MPI:
      self.domain_decomp()
      self.init_bc_buffer()
    self.init_data()
    # self.init_scratch()


  # =================================================================

  IF MPI:

    cdef void domain_decomp(self):

      cdef:
        GridCoord *gc = &(self.coord)
        mpi.Comm comm = mpi.COMM_WORLD

      cdef:
        ints i,k, size0
        int p
        long[:,::1] crds

      # read number of MPI blocks in each direction
      gc.size[0] = read_param("computation", "nblocks_x", 'i', gc.usr_dir)
      gc.size[1] = read_param("computation", "nblocks_y", 'i', gc.usr_dir)
      gc.size[2] = read_param("computation", "nblocks_z", 'i', gc.usr_dir)

      gc.rank = comm.Get_rank()
      size0 = comm.Get_size()
      gc.size_tot = gc.size[0] * gc.size[1] * gc.size[2]

      # check if number of blocks is consistent with number of processes and grid size

      if gc.size_tot != size0:
        print_root("Total number of MPI blocks is not equal to number of processors!\n")
        sys.exit()

      for k in range(3):
        if gc.Nact_glob[k] % gc.size[k] != 0:
          print_root("Number of MPI blocks is not a multiple of grid size in %i-direction!\n", k)
          sys.exit()

      hilbert_curve = None

      # distribute MPI blocks across processors

      if gc.size[1]>1 and gc.size[2]>1:

        # check if numbers of blocks in each direction are same and equal to 2**p
        if gc.size[0]==gc.size[1] and gc.size[1]==gc.size[2] and (gc.size[0] & (gc.size[0] - 1)) == 0:

          print_root('Using 2d Hilbert-curve domain decomposition...\n')

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

          print_root('using 3d Hilbert-curve domain decomposition...\n')

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
      crds = np.empty((gc.size_tot, 3), dtype=np.int_)  # int_ is same as C long
      comm.Allgather([np.array(gc.pos), mpi.LONG], [crds, mpi.LONG])

      gc.ranks = <ints ***>calloc_3d_array(gc.size[0], gc.size[1], gc.size[2], sizeof(ints))

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

      gc.Ntot[0] = gc.Nact[0] + 2*gc.ng + 1
      IF D2D:
        gc.Ntot[1] = gc.Nact[1] + 2*gc.ng + 1
      ELSE:
        gc.Ntot[1] = 1
      IF D3D:
        gc.Ntot[2] = gc.Nact[2] + 2*gc.ng + 1
      ELSE:
        gc.Ntot[2] = 1

      gc.i1, gc.i2 = gc.ng, gc.Nact[0] + gc.ng - 1

      IF D2D: gc.j1, gc.j2 = gc.ng, gc.Nact[1] + gc.ng - 1
      ELSE:   gc.j1, gc.j2 = 0,0

      IF D3D: gc.k1, gc.k2 = gc.ng, gc.Nact[2] + gc.ng - 1
      ELSE:   gc.k1, gc.k2 = 0,0


    # ===================================================================

    cdef void init_bc_buffer(self):

      cdef GridCoord *gc = &(self.coord)

      cdef:
        ints bufsize, n, ndim=1
        ints Nxyz =  maxi(maxi(gc.Ntot[0],gc.Ntot[1]), gc.Ntot[2])

      IF D2D: ndim += 1
      IF D3D: ndim += 1

      n = 9 * gc.ng
      if n==3:
        bufsize = (Nxyz+1)**2 * n
      elif ndim==2:
        bufsize = (Nxyz+1) * n
      else:
        bufsize = n

      self.bc.sendbuf = np.zeros((2,bufsize), dtype=np_real)
      self.bc.recvbuf = np.zeros((2,bufsize), dtype=np_real)
      self.bc.recvbuf_size = bufsize
      self.bc.sendbuf_size = bufsize

  # end of IF MPI


  # =================================================================

  cdef void init_data(self):

    cdef:
      GridCoord  *gc  = &(self.coord)
      GridData    gd  =   self.data
      GridScratch scr =   self.scr

    sh_3 = (3, gc.Ntot[2], gc.Ntot[1], gc.Ntot[0])

    gd.ee = np.zeros(sh_3, dtype=np_real)
    gd.bf = np.zeros(sh_3, dtype=np_real)
    gd.ce = np.zeros(sh_3, dtype=np_real)

    IF MPI:
      scr.ce_tmp = np.zeros(sh_3, dtype=np_real)


  # =============================================================

  def __dealloc__(self):

    cdef:
      GridCoord *gc = &(self.coord)
      GridBC    gbc =   self.bc

    IF MPI: free_3d_array(gc.ranks)

    # Free BC pointers.

    cdef ints i,k

    for i in range(3):
      for k in range(2):
        gbc.bc_grid_funcs[i][k] = NULL

    for i in range(3):
      for k in range(2):
        gbc.bc_exch_funcs[i][k] = NULL
