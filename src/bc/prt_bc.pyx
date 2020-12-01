# -*- coding: utf-8 -*-

IF MPI:
  from mpi4py import MPI as mpi
  from mpi4py cimport MPI as mpi

import numpy as np
cimport numpy as np
from cython.parallel import prange, parallel

from libc.math cimport sqrt,floor,ceil,log,exp,sin,cos,pow,fabs,fmin,fmax

from bnz.utils cimport print_root
from bnz.io.read_config import read_param
from bnz.problem.problem cimport set_prt_bc_user
from prt_bc_funcs cimport *

IF SPREC:
  np_real = np.float32
ELSE:
  np_real = np.float64

IF MPI:
  IF SPREC:
    mpi_real = mpi.FLOAT
  ELSE:
    mpi_real = mpi.DOUBLE


cdef class GridBc:

  def __cinit__(self, ParticleProp *pp, GridCoord *gc, bytes usr_dir):

    cdef ints i,k

    for i in range(3):
      for k in range(2):
        self.prt_bc_funcs[i][k] = NULL

    # Read BC flags from config file.

    self.bc_flags[0][0] = read_param("physics", "bc_x1", 'i',usr_dir)
    self.bc_flags[0][1] = read_param("physics", "bc_x2", 'i',usr_dir)
    self.bc_flags[1][0] = read_param("physics", "bc_y1", 'i',usr_dir)
    self.bc_flags[1][1] = read_param("physics", "bc_y2", 'i',usr_dir)
    self.bc_flags[2][0] = read_param("physics", "bc_z1", 'i',usr_dir)
    self.bc_flags[2][1] = read_param("physics", "bc_z2", 'i',usr_dir)

    # Set particle BC function pointers.

    cdef:
      ints sx=gc.size[0], sy=gc.size[1], sz=gc.size[2]
      ints px=gc.pos[0], py=gc.pos[1], pz=gc.pos[2]

    # first set built-in BCs

    if self.bc_flag[0][0]==0:
      self.prt_bc_funcs[0][0] = x1_prt_bc_periodic

      # IF MPI:
      #   if px==0 and sx>1:
          # gc.nbr_ranks[0][0] = gc.ranks[sx-1, py, pz]

    if self.bc_flag[0][0]==1:
      self.prt_bc_funcs[0][0] = x1_prt_bc_outflow

    if self.bc_flag[0][0]==2:
      self.prt_bc_funcs[0][0] = x1_prt_bc_reflect

    #----------------------------------------------------------

    if self.bc_flag[0][1]==0:
      self.prt_bc_funcs[0][1] = x2_prt_bc_periodic

      # IF MPI:
      #   if px==sx-1 and sx>1:
      #     gc.nbr_ranks[0][1] = gc.ranks[0, py, pz]

    if self.bc_flag[0][1]==1:
      self.prt_bc_funcs[0][1] = x2_prt_bc_outflow

    if self.bc_flag[0][1]==2:
      self.prt_bc_funcs[0][1] = x2_prt_bc_reflect

    #----------------------------------------------------------

    if self.bc_flag[1][0]==0:
      self.prt_bc_funcs[1][0] = y1_prt_bc_periodic

      # IF MPI:
      #   if py==0 and sy>1:
      #     gc.nbr_ranks[1][0] = gc.ranks[px, sy-1, pz]

    if self.bc_flag[1][0]==1:
      self.prt_bc_funcs[1][0] = y1_prt_bc_outflow

    if self.bc_flag[1][0]==2:
      self.prt_bc_funcs[1][0] = y1_prt_bc_reflect

    #----------------------------------------------------------

    if self.bc_flag[1][1]==0:
      self.prt_bc_funcs[1][1] = y2_prt_bc_periodic

      # IF MPI:
      #   if py==sy-1 and sy>1:
      #     gc.nbr_ranks[1][1] = gc.ranks[px, 0, pz]

    if self.bc_flag[1][1]==1:
      self.prt_bc_funcs[1][1] = y2_prt_bc_outflow

    if self.bc_flag[1][1]==2:
      self.prt_bc_funcs[1][1] = y2_prt_bc_reflect

    #----------------------------------------------------------

    if self.bc_flag[2][0]==0:
      self.prt_bc_funcs[2][0] = z1_prt_bc_periodic

      # IF MPI:
      #   if pz==0 and sz>1:
      #     gc.nbr_ranks[2][0] = gc.ranks[px, py, sz-1]

    if self.bc_flag[2][0]==1:
      self.prt_bc_funcs[2][0] = z1_prt_bc_outflow

    if self.bc_flag[2][0]==2:
      self.prt_bc_funcs[2][0] = z1_prt_bc_reflect

    #----------------------------------------------------------

    if self.bc_flag[2][1]==0:
      self.prt_bc_funcs[2][1] = z2_prt_bc_periodic

      # IF MPI:
      #   if pz==sz-1 and sz>1:
      #     gc.nbr_ranks[2][1] = gc.ranks[px, py, 0]

    if self.bc_flag[2][1]==1:
      self.prt_bc_funcs[2][1] = z2_prt_bc_outflow

    if self.bc_flag[2][1]==2:
      self.prt_bc_funcs[2][1] = z2_prt_bc_reflect

    # then set remaining user-defined BCs
    set_prt_bc_user(self)

    # Check that BCs in all directions have been set.

    cdef ints i,k

    for i in range(3):
      for k in range(2):
        if self.prt_bc_funcs[i][k] == NULL:
          if k==0:
            print_root('\nWARNING: Left particle boundary condition in %i direction is not set.\n')
          else:
            print_root('\nWARNING: Right particle boundary condition in %i direction is not set.\n')

    # Allocate MPI buffers.

    IF MPI:

      cdef:
        ints bufsize
        ints Nxyz =  maxi(maxi(gc.Nact[0],gc.Nact[1]), gc.Nact[2])

      IF not PIC:

        IF D2D and D3D:
          bufsize = 5*pp.Nprop * pp.ppc * Nxyz**2
        ELIF D2D:
          bufsize = 5*pp.Nprop * pp.ppc * Nxyz
        ELSE:
          bufsize = 5*pp.Nprop * pp.ppc

      ELSE:

        IF D2D and D3D:
          bufsize = 5*pp.Nprop * pp.ppc * Nxyz**2
        ELIF D2D:
          bufsize = 5*pp.Nprop * pp.ppc * Nxyz
        ELSE:
          bufsize = 5*pp.Nprop * pp.ppc

      self.sendbuf = np.zeros((2,bufsize), dtype=np_real)
      self.recvbuf = np.zeros((2,bufsize), dtype=np_real)
      self.recvbuf_size = bufsize
      self.sendbuf_size = bufsize


  def __dealloc__(self):

    cdef ints i,k

    for i in range(3):
      for k in range(2):
        self.prt_bc_funcs[i][k] = NULL



# ===========================================================

cdef void apply_prt_bc(BnzSim sim):

  cdef ParticleBc bc = sim.grid.prts.bc

  IF MPI:

    cdef:
      BnzGrid grid = sim.grid
      GridCoord gc = grid.coord
      BnzParticles prts = grid.prts

      mpi.Comm comm = MPI.COMM_WORLD

      long[::1] cnt_send=np.zeros(1,np.int_)
      long[::1] cnt_recv=np.zeros(1,np.int_)

      mpi.Request recv_req

    # ------- data exchchange in x-direction --------------

    if gc.nbr_ranks[0][0] > -1 and gc.nbr_ranks[0][1] > -1:

      # receive from left, send to right

      recv_req = comm.Irecv([cnt_recv, 1, mpi.LONG], gc.nbr_ranks[0][0], tag=1)

      cnt_send[0] = x2_pack_shift_prt(prts, bc.sendbuf, &bc.sendbuf_size, gc.lmin[0],gc.lmax[0])
      if cnt_send[0] > 0:
        comm.Send([cnt_send, 1, mpi.LONG], gc.nbr_ranks[0][1], tag=1)

      recv_req.Wait()
      if cnt_recv[0]+1 > bc.recv_bufsize:
        realloc_recvbuf(bc.recvbuf, &bc.recvbuf_size)

      if cnt_recv[0] > 0:
        recv_req = comm.Irecv([bc.recvbuf[0,:], cnt_recv[0], mpi_real], gc.nbr_ranks[0][0], tag=1)

      if cnt_send[0] > 0:
        comm.Send([bc.sendbuf[0,:], cnt_send[0], mpi_real], gc.nbr_ranks[0][1], tag=1)

      if cnt_recv[0] > 0:
        recv_req.Wait()
        unpack_prt(prts, bc.recvbuf, cnt_recv[0])

      # receive from right, send to left

      recv_req = comm.Irecv([cnt_recv, 1, mpi.LONG], gc.nbr_ranks[0][1], tag=0)

      cnt_send[0] = x1_pack_shift_prt(prts, bc.sendbuf, &bc.sendbuf_size, gc.lmin[0],gc.lmax[0])
      if cnt_send[0] > 0:
        comm.Send([cnt_send, 1, mpi.LONG], gc.nbr_ranks[0][0], tag=0)

      recv_req.Wait()
      if cnt_recv[0]+1 > bc.recv_bufsize:
        realloc_recvbuf(bc.recvbuf, &bc.recvbuf_size)

      if cnt_recv[0] > 0:
        recv_req = comm.Irecv([bc.recvbuf[0,:], cnt_recv[0], mpi_real], gc.nbr_ranks[0][1], tag=0)

      if cnt_send[0] > 0:
        comm.Send([bc.sendbuf[0,:], cnt_send[0], mpi_real], gc.nbr_ranks[0][0], tag=0)

      if cnt_recv[0] > 0:
        recv_req.Wait()
        unpack_prt(prts, bc.recvbuf, cnt_recv[0])

    #--------------------------------------------

    elif gc.nbr_ranks[0][0] == -1 and gc.nbr_ranks[0][1] > -1:

      # send to right

      cnt_send[0] = pack_shift_prt_x2(prts, bc.sendbuf, &bc.sendbuf_size, gc.lmin[0],gc.lmax[0])

      if cnt_send[0] > 0:
        comm.Send([cnt_send, 1, mpi.LONG], gc.nbr_ranks[0][1], tag=1)
        comm.Send([bc.sendbuf[0,:], cnt_send[0], mpi_real], gc.nbr_ranks[0][1], tag=1)

      bc.prt_bc_funcs[0][0](sim)

      # receive from right

      recv_req = comm.Irecv([cnt_recv, 1, mpi.LONG], gc.nbr_ranks[0][1], tag=0)
      recv_req.Wait()

      if cnt_recv[0]+1 > bc.recv_bufsize:
        realloc_recvbuf(bc.recvbuf, &bc.recvbuf_size)

      if cnt_recv[0] > 0:
        recv_req = comm.Irecv([bc.recvbuf[0,:], cnt_recv[0], mpi_real], gc.nbr_ranks[0][1], tag=0)
        recv_req.Wait()
        unpack_prt(prts, bc.recvbuf, cnt_recv[0])

    #----------------------------------------------

    elif gc.nbr_ranks[0][0] > -1 and gc.nbr_ranks[0][1] == -1:

      # receive from left

      recv_req = comm.Irecv([cnt_recv, 1, mpi.LONG], gc.nbr_ranks[0][0], tag=1)
      recv_req.Wait()

      if cnt_recv[0]+1 > bc.recv_bufsize:
        realloc_recvbuf(bc.recvbuf, &bc.recvbuf_size)

      if cnt_recv[0] > 0:
        recv_req = comm.Irecv([bc.recvbuf[0,:], cnt_recv[0], mpi_real], gc.nbr_ranks[0][0], tag=1)
        recv_req.Wait()
        unpack_prt(prts, bc.recvbuf, cnt_recv[0])

      bc.prt_bc_funcs[0][1](sim)

      # send to left

      cnt_send[0] = pack_shift_prt_x1(prts, bc.sendbuf, &bc.sendbuf_size, gc.lmin[0],gc.lmax[0])

      if cnt_send[0] > 0:
        comm.Send([cnt_send, 1, mpi.LONG], gc.nbr_ranks[0][0], tag=0)
        comm.Send([bc.sendbuf[0,:], cnt_send[0], mpi_real], gc.nbr_ranks[0][0], tag=0)

    else:

      bc.prt_bc_funcs[0][0](sim)
      bc.prt_bc_funcs[0][1](sim)


    # ------- data exchange in y-direction --------------

    IF D2D:

      if gc.nbr_ranks[1][0] > -1 and gc.nbr_ranks[1][1] > -1:

        # receive from left, send to right

        recv_req = comm.Irecv([cnt_recv, 1, mpi.LONG], gc.nbr_ranks[1][0], tag=1)

        cnt_send[0] = pack_shift_prt_y2(prts, bc.sendbuf, &bc.sendbuf_size, gc.lmin[1],gc.lmax[1])
        if cnt_send[0] > 0:
          comm.Send([cnt_send, 1, mpi.LONG], gc.nbr_ranks[1][1], tag=1)

        recv_req.Wait()
        if cnt_recv[0]+1 > bc.recv_bufsize:
          realloc_recvbuf(bc.recvbuf, &bc.recvbuf_size)

        if cnt_recv[0] > 0:
          recv_req = comm.Irecv([bc.recvbuf[0,:], cnt_recv[0], mpi_real], gc.nbr_ranks[1][0], tag=1)

        if cnt_send[0] > 0:
          comm.Send([bc.sendbuf[0,:], cnt_send[0], mpi_real], gc.nbr_ranks[1][1], tag=1)

        if cnt_recv[0] > 0:
          recv_req.Wait()
          unpack_prt(prts, bc.recvbuf, cnt_recv[0])

        # receive from right, send to left

        recv_req = comm.Irecv([cnt_recv, 1, mpi.LONG], gc.nbr_ranks[1][1], tag=0)

        cnt_send[0] = pack_shift_prt_y1(prts, bc.sendbuf, &bc.sendbuf_size, gc.lmin[1],gc.lmax[1])
        if cnt_send[0] > 0:
          comm.Send([cnt_send, 1, mpi.LONG], gc.nbr_ranks[1][0], tag=0)

        recv_req.Wait()
        if cnt_recv[0]+1 > bc.recv_bufsize:
          realloc_recvbuf(bc.recvbuf, &bc.recvbuf_size)

        if cnt_recv[0] > 0:
          recv_req = comm.Irecv([bc.recvbuf[0,:], cnt_recv[0], mpi_real], gc.nbr_ranks[1][1], tag=0)

        if cnt_send[0] > 0:
          comm.Send([bc.sendbuf[0,:], cnt_send[0], mpi_real], gc.nbr_ranks[1][0], tag=0)

        if cnt_recv[0] > 0:
          recv_req.Wait()
          unpack_prt(prts, bc.recvbuf, cnt_recv[0])

      #--------------------------------------------

      elif gc.nbr_ranks[1][0] == -1 and gc.nbr_ranks[1][1] > -1:

        # send to right

        cnt_send[0] = pack_shift_prt_y2(prts, bc.sendbuf, &bc.sendbuf_size, gc.lmin[1],gc.lmax[1])

        if cnt_send[0] > 0:
          comm.Send([cnt_send, 1, mpi.LONG], gc.nbr_ranks[1][1], tag=1)
          comm.Send([bc.sendbuf[0,:], cnt_send[0], mpi_real], gc.nbr_ranks[1][1], tag=1)

        bc.prt_bc_funcs[1][0](sim)

        # receive from right

        recv_req = comm.Irecv([cnt_recv, 1, mpi.LONG], gc.nbr_ranks[1][1], tag=0)
        recv_req.Wait()

        if cnt_recv[0]+1 > bc.recv_bufsize:
          realloc_recvbuf(bc.recvbuf, &bc.recvbuf_size)

        if cnt_recv[0] > 0:
          recv_req = comm.Irecv([bc.recvbuf[0,:], cnt_recv[0], mpi_real], gc.nbr_ranks[1][1], tag=0)
          recv_req.Wait()
          unpack_prt(prts, bc.recvbuf, cnt_recv[0])

      #----------------------------------------------

      elif gc.nbr_ranks[1][0] > -1 and gc.nbr_ranks[1][1] == -1:

        # receive from left

        recv_req = comm.Irecv([cnt_recv, 1, mpi.LONG], gc.nbr_ranks[1][0], tag=1)
        recv_req.Wait()

        if cnt_recv[0]+1 > bc.recv_bufsize:
          realloc_recvbuf(bc.recvbuf, &bc.recvbuf_size)

        if cnt_recv[0] > 0:
          recv_req = comm.Irecv([bc.recvbuf[0,:], cnt_recv[0], mpi_real], gc.nbr_ranks[1][0], tag=1)
          recv_req.Wait()
          unpack_prt(prts, bc.recvbuf, cnt_recv[0])

        bc.prt_bc_funcs[1][1](sim)

        # send to left

        cnt_send[0] = pack_shift_prt_y1(prts, bc.sendbuf, &bc.sendbuf_size, gc.lmin[1],gc.lmax[1])

        if cnt_send[0] > 0:
          comm.Send([cnt_send, 1, mpi.LONG], gc.nbr_ranks[1][0], tag=0)
          comm.Send([bc.sendbuf[0,:], cnt_send[0], mpi_real], gc.nbr_ranks[1][0], tag=0)

      else:

        bc.prt_bc_funcs[1][0](sim)
        bc.prt_bc_funcs[1][1](sim)


    # ------- data exchange in z-direction --------------

    IF D3D:

      if gc.nbr_ranks[2][0] > -1 and gc.nbr_ranks[2][1] > -1:

        # receive from left, send to right

        recv_req = comm.Irecv([cnt_recv, 1, mpi.LONG], gc.nbr_ranks[2][0], tag=1)

        cnt_send[0] = pack_shift_prt_z2(prts, bc.sendbuf, &bc.sendbuf_size, gc.lmin[2],gc.lmax[2])
        if cnt_send[0] > 0:
          comm.Send([cnt_send, 1, mpi.LONG], gc.nbr_ranks[2][1], tag=1)

        recv_req.Wait()
        if cnt_recv[0]+1 > bc.recv_bufsize:
          realloc_recvbuf(bc.recvbuf, &bc.recvbuf_size)

        if cnt_recv[0] > 0:
          recv_req = comm.Irecv([bc.recvbuf[0,:], cnt_recv[0], mpi_real], gc.nbr_ranks[2][0], tag=1)

        if cnt_send[0] > 0:
          comm.Send([bc.sendbuf[0,:], cnt_send[0], mpi_real], gc.nbr_ranks[2][1], tag=1)

        if cnt_recv[0] > 0:
          recv_req.Wait()
          unpack_prt(prts, bc.recvbuf, cnt_recv[0])

        # receive from right, send to left

        recv_req = comm.Irecv([cnt_recv, 1, mpi.LONG], gc.nbr_ranks[2][1], tag=0)

        cnt_send[0] = pack_shift_prt_z1(prts, bc.sendbuf, &bc.sendbuf_size, gc.lmin[2],gc.lmax[2])
        if cnt_send[0] > 0:
          comm.Send([cnt_send, 1, mpi.LONG], gc.nbr_ranks[2][0], tag=0)

        recv_req.Wait()
        if cnt_recv[0]+1 > bc.recv_bufsize:
          realloc_recvbuf(bc.recvbuf, &bc.recvbuf_size)

        if cnt_recv[0] > 0:
          recv_req = comm.Irecv([bc.recvbuf[0,:], cnt_recv[0], mpi_real], gc.nbr_ranks[2][1], tag=0)

        if cnt_send[0] > 0:
          comm.Send([bc.sendbuf[0,:], cnt_send[0], mpi_real], gc.nbr_ranks[2][0], tag=0)

        if recv_req[0] > 0:
          recv_req.Wait()
          unpack_prt(prts, bc.recvbuf, cnt_recv[0])

      #--------------------------------------------

      elif gc.nbr_ranks[2][0] == -1 and gc.nbr_ranks[2][1] > -1:

        # send to right

        cnt_send[0] = pack_shift_prt_z2(prts, bc.sendbuf, &bc.sendbuf_size, gc.lmin[2],gc.lmax[2])

        if cnt_send[0] > 0:
          comm.Send([cnt_send, 1, mpi.LONG], gc.nbr_ranks[2][1], tag=1)
          comm.Send([bc.sendbuf[0,:], cnt_send[0], mpi_real], gc.nbr_ranks[2][1], tag=1)

        bc.prt_bc_funcs[2][0](sim)

        # receive from right

        recv_req = comm.Irecv([cnt_recv, 1, mpi.LONG], gc.nbr_ranks[2][1], tag=0)
        recv_req.Wait()

        if cnt_recv[0]+1 > bc.recv_bufsize:
          realloc_recvbuf(bc.recvbuf, &bc.recvbuf_size)

        if cnt_recv[0] > 0:
          recv_req = comm.Irecv([bc.recvbuf[0,:], cnt_recv[0], mpi_real], gc.nbr_ranks[2][1], tag=0)
          recv_req.Wait()
          unpack_prt(prts, bc.recvbuf, cnt_recv[0])

      #----------------------------------------------

      elif gc.nbr_ranks[2][0] > -1 and gc.nbr_ranks[2][1] == -1:

        # receive from left

        recv_req = comm.Irecv([cnt_recv, 1, mpi.LONG], gc.nbr_ranks[2][0], tag=1)
        recv_req.Wait()

        if cnt_recv[0]+1 > bc.recv_bufsize:
          realloc_recvbuf(bc.recvbuf, &bc.recvbuf_size)

        if cnt_recv[0] > 0:
          recv_req = comm.Irecv([bc.recvbuf[0,:], cnt_recv[0], mpi_real], gc.nbr_ranks[2][0], tag=1)
          recv_req.Wait()
          unpack_prt(prts, bc.recvbuf, cnt_recv[0])

        bc.prt_bc_funcs[2][1](sim)

        # send to left

        cnt_send[0] = pack_shift_prt_z1(prts, bc.sendbuf, &bc.sendbuf_size, gc.lmin[2],gc.lmax[2])

        if cnt_send[0] > 0:
          comm.Send([cnt_send, 1, mpi.LONG], gc.nbr_ranks[2][0], tag=0)
          comm.Send([bc.sendbuf[0,:], cnt_send[0], mpi_real], gc.nbr_ranks[2][0], tag=0)

      else:

        bc.prt_bc_funcs[2][0](sim)
        bc.prt_bc_funcs[2][1](sim)

  ELSE:

    bc.prt_bc_funcs[0][0](sim)
    bc.prt_bc_funcs[0][1](sim)
    IF D2D:
      bc.prt_bc_funcs[1][0](sim)
      bc.prt_bc_funcs[1][1](sim)
    IF D3D:
      bc.prt_bc_funcs[2][0](sim)
      bc.prt_bc_funcs[2][1](sim)
