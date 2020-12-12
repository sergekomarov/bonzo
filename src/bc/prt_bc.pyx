# -*- coding: utf-8 -*-

IF MPI:
  from mpi4py import MPI as mpi
  from mpi4py cimport MPI as mpi

import numpy as np
cimport numpy as np

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


cdef class PrtBc:

  def __cinit__(self, ParticleProp *pp, GridCoord *gc, bytes usr_dir):

    cdef int i,k

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
      int sx=gc.size[0], sy=gc.size[1], sz=gc.size[2]
      int px=gc.pos[0], py=gc.pos[1], pz=gc.pos[2]

    # first set built-in BCs

    if self.bc_flag[0][0]==0:
      self.prt_bc_funcs[0][0] = x1_prt_bc_periodic

    if self.bc_flag[0][0]==1:
      self.prt_bc_funcs[0][0] = x1_prt_bc_outflow

    if self.bc_flag[0][0]==2:
      self.prt_bc_funcs[0][0] = x1_prt_bc_reflect

    #----------------------------------------------------------

    if self.bc_flag[0][1]==0:
      self.prt_bc_funcs[0][1] = x2_prt_bc_periodic

    if self.bc_flag[0][1]==1:
      self.prt_bc_funcs[0][1] = x2_prt_bc_outflow

    if self.bc_flag[0][1]==2:
      self.prt_bc_funcs[0][1] = x2_prt_bc_reflect

    #----------------------------------------------------------

    if self.bc_flag[1][0]==0:
      self.prt_bc_funcs[1][0] = y1_prt_bc_periodic

    if self.bc_flag[1][0]==1:
      self.prt_bc_funcs[1][0] = y1_prt_bc_outflow

    if self.bc_flag[1][0]==2:
      self.prt_bc_funcs[1][0] = y1_prt_bc_reflect

    #----------------------------------------------------------

    if self.bc_flag[1][1]==0:
      self.prt_bc_funcs[1][1] = y2_prt_bc_periodic

    if self.bc_flag[1][1]==1:
      self.prt_bc_funcs[1][1] = y2_prt_bc_outflow

    if self.bc_flag[1][1]==2:
      self.prt_bc_funcs[1][1] = y2_prt_bc_reflect

    #----------------------------------------------------------

    if self.bc_flag[2][0]==0:
      self.prt_bc_funcs[2][0] = z1_prt_bc_periodic

    if self.bc_flag[2][0]==1:
      self.prt_bc_funcs[2][0] = z1_prt_bc_outflow

    if self.bc_flag[2][0]==2:
      self.prt_bc_funcs[2][0] = z1_prt_bc_reflect

    #----------------------------------------------------------

    if self.bc_flag[2][1]==0:
      self.prt_bc_funcs[2][1] = z2_prt_bc_periodic

    if self.bc_flag[2][1]==1:
      self.prt_bc_funcs[2][1] = z2_prt_bc_outflow

    if self.bc_flag[2][1]==2:
      self.prt_bc_funcs[2][1] = z2_prt_bc_reflect

    # then set remaining user-defined BCs
    set_prt_bc_user(self)

    # Check that BCs in all directions have been set.

    cdef int i,k

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
        long bufsize
        long Nxyz =  IMAX(IMAX(gc.Nact[0],gc.Nact[1]), gc.Nact[2])

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

    cdef int i,k

    for i in range(3):
      for k in range(2):
        self.prt_bc_funcs[i][k] = NULL



  # --------------------------------------------------------------------

  cdef void apply_prt_bc(self, PrtData *pd, PrtProp *pp,
                         GridData gd, GridCoord *gc, BnzIntegr integr):

    IF MPI:

      cdef:

        mpi.Comm comm = MPI.COMM_WORLD
        mpi.Request recv_req

        long[::1] cnt_send=np.zeros(1,np.int_)
        long[::1] cnt_recv=np.zeros(1,np.int_)

      # ------- data exchchange in x-direction --------------

      if gc.nbr_ranks[0][0] > -1 and gc.nbr_ranks[0][1] > -1:

        # receive from left, send to right

        recv_req = comm.Irecv([cnt_recv, 1, mpi.LONG], gc.nbr_ranks[0][0], tag=1)

        cnt_send[0] = x2_pack_shift_prt(pd,pp, self.sendbuf, &self.sendbuf_size, gc.lmin[0],gc.lmax[0])
        if cnt_send[0] > 0:
          comm.Send([cnt_send, 1, mpi.LONG], gc.nbr_ranks[0][1], tag=1)

        recv_req.Wait()
        if cnt_recv[0]+1 > self.recvbuf_size:
          realloc_recvbuf(self.recvbuf, &self.recvbuf_size)

        if cnt_recv[0] > 0:
          recv_req = comm.Irecv([self.recvbuf[0,:], cnt_recv[0], mpi_real], gc.nbr_ranks[0][0], tag=1)

        if cnt_send[0] > 0:
          comm.Send([self.sendbuf[0,:], cnt_send[0], mpi_real], gc.nbr_ranks[0][1], tag=1)

        if cnt_recv[0] > 0:
          recv_req.Wait()
          unpack_prt(pd,pp, self.recvbuf, cnt_recv[0])

        # receive from right, send to left

        recv_req = comm.Irecv([cnt_recv, 1, mpi.LONG], gc.nbr_ranks[0][1], tag=0)

        cnt_send[0] = x1_pack_shift_prt(pd,pp, self.sendbuf, &self.sendbuf_size, gc.lmin[0],gc.lmax[0])
        if cnt_send[0] > 0:
          comm.Send([cnt_send, 1, mpi.LONG], gc.nbr_ranks[0][0], tag=0)

        recv_req.Wait()
        if cnt_recv[0]+1 > self.recvbuf_size:
          realloc_recvbuf(self.recvbuf, &self.recvbuf_size)

        if cnt_recv[0] > 0:
          recv_req = comm.Irecv([self.recvbuf[0,:], cnt_recv[0], mpi_real], gc.nbr_ranks[0][1], tag=0)

        if cnt_send[0] > 0:
          comm.Send([self.sendbuf[0,:], cnt_send[0], mpi_real], gc.nbr_ranks[0][0], tag=0)

        if cnt_recv[0] > 0:
          recv_req.Wait()
          unpack_prt(pd,pp, self.recvbuf, cnt_recv[0])

      #--------------------------------------------

      elif gc.nbr_ranks[0][0] == -1 and gc.nbr_ranks[0][1] > -1:

        # send to right

        cnt_send[0] = pack_shift_prt_x2(pd,pp, self.sendbuf, &self.sendbuf_size, gc.lmin[0],gc.lmax[0])

        if cnt_send[0] > 0:
          comm.Send([cnt_send, 1, mpi.LONG], gc.nbr_ranks[0][1], tag=1)
          comm.Send([self.sendbuf[0,:], cnt_send[0], mpi_real], gc.nbr_ranks[0][1], tag=1)

        self.prt_bc_funcs[0][0](pd,pp, gd,gc, integr)

        # receive from right

        recv_req = comm.Irecv([cnt_recv, 1, mpi.LONG], gc.nbr_ranks[0][1], tag=0)
        recv_req.Wait()

        if cnt_recv[0]+1 > self.recvbuf_size:
          realloc_recvbuf(self.recvbuf, &self.recvbuf_size)

        if cnt_recv[0] > 0:
          recv_req = comm.Irecv([self.recvbuf[0,:], cnt_recv[0], mpi_real], gc.nbr_ranks[0][1], tag=0)
          recv_req.Wait()
          unpack_prt(pd,pp, self.recvbuf, cnt_recv[0])

      #----------------------------------------------

      elif gc.nbr_ranks[0][0] > -1 and gc.nbr_ranks[0][1] == -1:

        # receive from left

        recv_req = comm.Irecv([cnt_recv, 1, mpi.LONG], gc.nbr_ranks[0][0], tag=1)
        recv_req.Wait()

        if cnt_recv[0]+1 > self.recvbuf_size:
          realloc_recvbuf(self.recvbuf, &self.recvbuf_size)

        if cnt_recv[0] > 0:
          recv_req = comm.Irecv([self.recvbuf[0,:], cnt_recv[0], mpi_real], gc.nbr_ranks[0][0], tag=1)
          recv_req.Wait()
          unpack_prt(pd,pp, self.recvbuf, cnt_recv[0])

        self.prt_bc_funcs[0][1](pd,pp, gd,gc, integr)

        # send to left

        cnt_send[0] = pack_shift_prt_x1(pd,pp, self.sendbuf, &self.sendbuf_size, gc.lmin[0],gc.lmax[0])

        if cnt_send[0] > 0:
          comm.Send([cnt_send, 1, mpi.LONG], gc.nbr_ranks[0][0], tag=0)
          comm.Send([self.sendbuf[0,:], cnt_send[0], mpi_real], gc.nbr_ranks[0][0], tag=0)

      else:

        self.prt_bc_funcs[0][0](pd,pp, gd,gc, integr)
        self.prt_bc_funcs[0][1](pd,pp, gd,gc, integr)


      # ------- data exchange in y-direction --------------

      IF D2D:

        if gc.nbr_ranks[1][0] > -1 and gc.nbr_ranks[1][1] > -1:

          # receive from left, send to right

          recv_req = comm.Irecv([cnt_recv, 1, mpi.LONG], gc.nbr_ranks[1][0], tag=1)

          cnt_send[0] = pack_shift_prt_y2(pd,pp, self.sendbuf, &self.sendbuf_size, gc.lmin[1],gc.lmax[1])
          if cnt_send[0] > 0:
            comm.Send([cnt_send, 1, mpi.LONG], gc.nbr_ranks[1][1], tag=1)

          recv_req.Wait()
          if cnt_recv[0]+1 > self.recvbuf_size:
            realloc_recvbuf(self.recvbuf, &self.recvbuf_size)

          if cnt_recv[0] > 0:
            recv_req = comm.Irecv([self.recvbuf[0,:], cnt_recv[0], mpi_real], gc.nbr_ranks[1][0], tag=1)

          if cnt_send[0] > 0:
            comm.Send([self.sendbuf[0,:], cnt_send[0], mpi_real], gc.nbr_ranks[1][1], tag=1)

          if cnt_recv[0] > 0:
            recv_req.Wait()
            unpack_prt(pd,pp, self.recvbuf, cnt_recv[0])

          # receive from right, send to left

          recv_req = comm.Irecv([cnt_recv, 1, mpi.LONG], gc.nbr_ranks[1][1], tag=0)

          cnt_send[0] = pack_shift_prt_y1(pd,pp, self.sendbuf, &self.sendbuf_size, gc.lmin[1],gc.lmax[1])
          if cnt_send[0] > 0:
            comm.Send([cnt_send, 1, mpi.LONG], gc.nbr_ranks[1][0], tag=0)

          recv_req.Wait()
          if cnt_recv[0]+1 > self.recvbuf_size:
            realloc_recvbuf(self.recvbuf, &self.recvbuf_size)

          if cnt_recv[0] > 0:
            recv_req = comm.Irecv([self.recvbuf[0,:], cnt_recv[0], mpi_real], gc.nbr_ranks[1][1], tag=0)

          if cnt_send[0] > 0:
            comm.Send([self.sendbuf[0,:], cnt_send[0], mpi_real], gc.nbr_ranks[1][0], tag=0)

          if cnt_recv[0] > 0:
            recv_req.Wait()
            unpack_prt(pd,pp, self.recvbuf, cnt_recv[0])

        #--------------------------------------------

        elif gc.nbr_ranks[1][0] == -1 and gc.nbr_ranks[1][1] > -1:

          # send to right

          cnt_send[0] = pack_shift_prt_y2(pd,pp, self.sendbuf, &self.sendbuf_size, gc.lmin[1],gc.lmax[1])

          if cnt_send[0] > 0:
            comm.Send([cnt_send, 1, mpi.LONG], gc.nbr_ranks[1][1], tag=1)
            comm.Send([self.sendbuf[0,:], cnt_send[0], mpi_real], gc.nbr_ranks[1][1], tag=1)

          self.prt_bc_funcs[1][0](pd,pp, gd,gc, integr)

          # receive from right

          recv_req = comm.Irecv([cnt_recv, 1, mpi.LONG], gc.nbr_ranks[1][1], tag=0)
          recv_req.Wait()

          if cnt_recv[0]+1 > self.recvbuf_size:
            realloc_recvbuf(self.recvbuf, &self.recvbuf_size)

          if cnt_recv[0] > 0:
            recv_req = comm.Irecv([self.recvbuf[0,:], cnt_recv[0], mpi_real], gc.nbr_ranks[1][1], tag=0)
            recv_req.Wait()
            unpack_prt(pd,pp, self.recvbuf, cnt_recv[0])

        #----------------------------------------------

        elif gc.nbr_ranks[1][0] > -1 and gc.nbr_ranks[1][1] == -1:

          # receive from left

          recv_req = comm.Irecv([cnt_recv, 1, mpi.LONG], gc.nbr_ranks[1][0], tag=1)
          recv_req.Wait()

          if cnt_recv[0]+1 > self.recvbuf_size:
            realloc_recvbuf(self.recvbuf, &self.recvbuf_size)

          if cnt_recv[0] > 0:
            recv_req = comm.Irecv([self.recvbuf[0,:], cnt_recv[0], mpi_real], gc.nbr_ranks[1][0], tag=1)
            recv_req.Wait()
            unpack_prt(pd,pp, self.recvbuf, cnt_recv[0])

          self.prt_bc_funcs[1][1](pd,pp, gd,gc, integr)

          # send to left

          cnt_send[0] = pack_shift_prt_y1(pd,pp, self.sendbuf, &self.sendbuf_size, gc.lmin[1],gc.lmax[1])

          if cnt_send[0] > 0:
            comm.Send([cnt_send, 1, mpi.LONG], gc.nbr_ranks[1][0], tag=0)
            comm.Send([self.sendbuf[0,:], cnt_send[0], mpi_real], gc.nbr_ranks[1][0], tag=0)

        else:

          self.prt_bc_funcs[1][0](pd,pp, gd,gc, integr)
          self.prt_bc_funcs[1][1](pd,pp, gd,gc, integr)


      # ------- data exchange in z-direction --------------

      IF D3D:

        if gc.nbr_ranks[2][0] > -1 and gc.nbr_ranks[2][1] > -1:

          # receive from left, send to right

          recv_req = comm.Irecv([cnt_recv, 1, mpi.LONG], gc.nbr_ranks[2][0], tag=1)

          cnt_send[0] = pack_shift_prt_z2(pd,pp, self.sendbuf, &self.sendbuf_size, gc.lmin[2],gc.lmax[2])
          if cnt_send[0] > 0:
            comm.Send([cnt_send, 1, mpi.LONG], gc.nbr_ranks[2][1], tag=1)

          recv_req.Wait()
          if cnt_recv[0]+1 > self.recvbuf_size:
            realloc_recvbuf(self.recvbuf, &self.recvbuf_size)

          if cnt_recv[0] > 0:
            recv_req = comm.Irecv([self.recvbuf[0,:], cnt_recv[0], mpi_real], gc.nbr_ranks[2][0], tag=1)

          if cnt_send[0] > 0:
            comm.Send([self.sendbuf[0,:], cnt_send[0], mpi_real], gc.nbr_ranks[2][1], tag=1)

          if cnt_recv[0] > 0:
            recv_req.Wait()
            unpack_prt(pd,pp, self.recvbuf, cnt_recv[0])

          # receive from right, send to left

          recv_req = comm.Irecv([cnt_recv, 1, mpi.LONG], gc.nbr_ranks[2][1], tag=0)

          cnt_send[0] = pack_shift_prt_z1(pd,pp, self.sendbuf, &self.sendbuf_size, gc.lmin[2],gc.lmax[2])
          if cnt_send[0] > 0:
            comm.Send([cnt_send, 1, mpi.LONG], gc.nbr_ranks[2][0], tag=0)

          recv_req.Wait()
          if cnt_recv[0]+1 > self.recvbuf_size:
            realloc_recvbuf(self.recvbuf, &self.recvbuf_size)

          if cnt_recv[0] > 0:
            recv_req = comm.Irecv([self.recvbuf[0,:], cnt_recv[0], mpi_real], gc.nbr_ranks[2][1], tag=0)

          if cnt_send[0] > 0:
            comm.Send([self.sendbuf[0,:], cnt_send[0], mpi_real], gc.nbr_ranks[2][0], tag=0)

          if recv_req[0] > 0:
            recv_req.Wait()
            unpack_prt(pd,pp, self.recvbuf, cnt_recv[0])

        #--------------------------------------------

        elif gc.nbr_ranks[2][0] == -1 and gc.nbr_ranks[2][1] > -1:

          # send to right

          cnt_send[0] = pack_shift_prt_z2(pd,pp, self.sendbuf, &self.sendbuf_size, gc.lmin[2],gc.lmax[2])

          if cnt_send[0] > 0:
            comm.Send([cnt_send, 1, mpi.LONG], gc.nbr_ranks[2][1], tag=1)
            comm.Send([self.sendbuf[0,:], cnt_send[0], mpi_real], gc.nbr_ranks[2][1], tag=1)

          self.prt_bc_funcs[2][0](pd,pp, gd,gc, integr)

          # receive from right

          recv_req = comm.Irecv([cnt_recv, 1, mpi.LONG], gc.nbr_ranks[2][1], tag=0)
          recv_req.Wait()

          if cnt_recv[0]+1 > self.recvbuf_size:
            realloc_recvbuf(self.recvbuf, &self.recvbuf_size)

          if cnt_recv[0] > 0:
            recv_req = comm.Irecv([self.recvbuf[0,:], cnt_recv[0], mpi_real], gc.nbr_ranks[2][1], tag=0)
            recv_req.Wait()
            unpack_prt(pd,pp, self.recvbuf, cnt_recv[0])

        #----------------------------------------------

        elif gc.nbr_ranks[2][0] > -1 and gc.nbr_ranks[2][1] == -1:

          # receive from left

          recv_req = comm.Irecv([cnt_recv, 1, mpi.LONG], gc.nbr_ranks[2][0], tag=1)
          recv_req.Wait()

          if cnt_recv[0]+1 > self.recvbuf_size:
            realloc_recvbuf(self.recvbuf, &self.recvbuf_size)

          if cnt_recv[0] > 0:
            recv_req = comm.Irecv([self.recvbuf[0,:], cnt_recv[0], mpi_real], gc.nbr_ranks[2][0], tag=1)
            recv_req.Wait()
            unpack_prt(pd,pp, self.recvbuf, cnt_recv[0])

          self.prt_bc_funcs[2][1](pd,pp, gd,gc, integr)

          # send to left

          cnt_send[0] = pack_shift_prt_z1(pd,pp, self.sendbuf, &self.sendbuf_size, gc.lmin[2],gc.lmax[2])

          if cnt_send[0] > 0:
            comm.Send([cnt_send, 1, mpi.LONG], gc.nbr_ranks[2][0], tag=0)
            comm.Send([self.sendbuf[0,:], cnt_send[0], mpi_real], gc.nbr_ranks[2][0], tag=0)

        else:

          self.prt_bc_funcs[2][0](pd,pp, gd,gc, integr)
          self.prt_bc_funcs[2][1](pd,pp, gd,gc, integr)

    ELSE:

      self.prt_bc_funcs[0][0](pd,pp, gd,gc, integr)
      self.prt_bc_funcs[0][1](pd,pp, gd,gc, integr)
      IF D2D:
        self.prt_bc_funcs[1][0](pd,pp, gd,gc, integr)
        self.prt_bc_funcs[1][1](pd,pp, gd,gc, integr)
      IF D3D:
        self.prt_bc_funcs[2][0](pd,pp, gd,gc, integr)
        self.prt_bc_funcs[2][1](pd,pp, gd,gc, integr)
