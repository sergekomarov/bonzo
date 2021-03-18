# -*- coding: utf-8 -*-

IF MPI:
  from mpi4py import MPI as mpi
  from mpi4py cimport MPI as mpi

import numpy as np
cimport numpy as np

from bnz.io.read_config import read_param
from bnz.util cimport print_root
from bnz.problem.problem cimport set_grid_bc_user
from grid_bc_funcs cimport *
IF MHDPIC:
  from exch_bc_funcs cimport *

IF SPREC:
  np_real = np.float32
ELSE:
  np_real = np.float64
IF MPI:
  IF SPREC:
    mpi_real = mpi.FLOAT
  ELSE:
    mpi_real = mpi.DOUBLE


cdef class GridBC:

  def __cinit__(self, GridCoord *gc, str usr_dir):

    cdef int i,k

    for i in range(3):
      for k in range(2):
        self.grid_bc_funcs[i][k] = NULL

    IF MHDPIC:
      for i in range(3):
        for k in range(2):
          self.exch_bc_funcs[i][k] = NULL

    # Set boundary condition flags.

    self.bc_flags[0][0] = read_param("physics", "bc_x1", 'i',usr_dir)
    self.bc_flags[0][1] = read_param("physics", "bc_x2", 'i',usr_dir)
    self.bc_flags[1][0] = read_param("physics", "bc_y1", 'i',usr_dir)
    self.bc_flags[1][1] = read_param("physics", "bc_y2", 'i',usr_dir)
    self.bc_flags[2][0] = read_param("physics", "bc_z1", 'i',usr_dir)
    self.bc_flags[2][1] = read_param("physics", "bc_z2", 'i',usr_dir)

    # Reset flags as appropriate for curvilinear coordinates.

    # special reflective at r=0
    if gc.geom==CG_SPH or gc.geom==CG_CYL:
      if gc.lmin[0]==0.: self.bc_flags[0][0] = 2

    IF D2D:
      if gc.geom==CG_CYL:
        # periodic at phi=0, phi=pi
        if gc.lmin[1]==0. and gc.lmax[1]==2*B_PI:
          self.bc_flags[1][0] = 0
          self.bc_flags[1][1] = 0
      if gc.geom==CG_SPH:
        # special reflective at theta=0 or theta=pi
        if gc.lmin[1]==0.:   self.bc_flags[1][0] = 2
        if gc.lmax[1]==B_PI: self.bc_flags[1][1] = 2
    IF D3D:
      if gc.geom==CG_SPH:
        # periodic at phi=0, phi=2pi
        if gc.lmin[2]==0. and gc.lmax[2]==2*B_PI:
          self.bc_flags[2][0] = 0
          self.bc_flags[2][1] = 0

    # Set BC function pointers.

    cdef:
      int sx=gc.size[0], sy=gc.size[1], sz=gc.size[2]
      int px=gc.pos[0], py=gc.pos[1], pz=gc.pos[2]
      int ph_pos, th_pos

    # left x boundary

    if self.bc_flags[0][0]==0:
      self.grid_bc_funcs[0][0] = x1_grid_bc_periodic
      IF MHDPIC:
        self.exch_bc_funcs[0][0] = x1_exch_bc_periodic

      IF MPI:
        if px==0 and sx>1:
          gc.nbr_ranks[0][0] = gc.ranks[sx-1][py][pz]

    if self.bc_flags[0][0]==1:
      self.grid_bc_funcs[0][0] = x1_grid_bc_outflow
      IF MHDPIC:
        self.exch_bc_funcs[0][0] = x1_exch_bc_outflow

    if self.bc_flags[0][0]==2:
      self.grid_bc_funcs[0][0] = x1_grid_bc_reflect
      IF MHDPIC:
        self.exch_bc_funcs[0][0] = x1_exch_bc_reflect

    # right x boundary

    if self.bc_flags[0][1]==0:
      self.grid_bc_funcs[0][1] = x2_grid_bc_periodic
      IF MHDPIC:
        self.exch_bc_funcs[0][1] = x2_exch_bc_periodic

      IF MPI:
        if px==sx-1 and sx>1:
          gc.nbr_ranks[0][1] = gc.ranks[0][py][pz]

    if self.bc_flags[0][1]==1:
      self.grid_bc_funcs[0][1] = x2_grid_bc_outflow
      IF MHDPIC:
        self.exch_bc_funcs[0][1] = x2_exch_bc_outflow

    if self.bc_flags[0][1]==2:
      self.grid_bc_funcs[0][1] = x2_grid_bc_reflect
      IF MHDPIC:
        self.exch_bc_funcs[0][1] = x2_exch_bc_reflect

    # left y boundary

    if self.bc_flags[1][0]==0:
      self.grid_bc_funcs[1][0] = y1_grid_bc_periodic
      IF MHDPIC:
        self.exch_bc_funcs[1][0] = y1_exch_bc_periodic

      IF MPI:
        if py==0 and sy>1:
          gc.nbr_ranks[1][0] = gc.ranks[px][sy-1][pz]

    if self.bc_flags[1][0]==1:
      self.grid_bc_funcs[1][0] = y1_grid_bc_outflow
      IF MHDPIC:
        self.exch_bc_funcs[1][0] = y1_exch_bc_outflow

    if self.bc_flags[1][0]==2:
      self.grid_bc_funcs[1][0] = y1_grid_bc_reflect
      IF MHDPIC:
        self.exch_bc_funcs[1][0] = y1_exch_bc_reflect

    # right y boundary

    if self.bc_flags[1][1]==0:
      self.grid_bc_funcs[1][1] = y2_grid_bc_periodic
      IF MHDPIC:
        self.exch_bc_funcs[1][1] = y2_exch_bc_periodic

      IF MPI:
        if py==sy-1 and sy>1:
          gc.nbr_ranks[1][1] = gc.ranks[px][0][pz]

    if self.bc_flags[1][1]==1:
      self.grid_bc_funcs[1][1] = y2_grid_bc_outflow
      IF MHDPIC:
        self.exch_bc_funcs[1][1] = y2_exch_bc_outflow

    if self.bc_flags[1][1]==2:
      self.grid_bc_funcs[1][1] = y2_grid_bc_reflect
      IF MHDPIC:
        self.exch_bc_funcs[1][1] = y2_exch_bc_reflect

    # left z boundary

    if self.bc_flags[2][0]==0:
      self.grid_bc_funcs[2][0] = z1_grid_bc_periodic
      IF MHDPIC:
        self.exch_bc_funcs[2][0] = z1_exch_bc_periodic

      IF MPI:
        if pz==0 and sz>1:
          gc.nbr_ranks[2][0] = gc.ranks[px][py][sz-1]

    if self.bc_flags[2][0]==1:
      self.grid_bc_funcs[2][0] = z1_grid_bc_outflow
      IF MHDPIC:
        self.exch_bc_funcs[2][0] = z1_exch_bc_outflow

    if self.bc_flags[2][0]==2:
      self.grid_bc_funcs[2][0] = z1_grid_bc_reflect
      IF MHDPIC:
        self.exch_bc_funcs[2][0] = z1_exch_bc_reflect

    # right z boundary

    if self.bc_flags[2][1]==0:
      self.grid_bc_funcs[2][1] = z2_grid_bc_periodic
      IF MHDPIC:
        self.exch_bc_funcs[2][1] = z2_exch_bc_periodic

      IF MPI:
        if pz==sz-1 and sz>1:
          gc.nbr_ranks[2][1] = gc.ranks[px][py][0]

    if self.bc_flags[2][1]==1:
      self.grid_bc_funcs[2][1] = z2_grid_bc_outflow
      IF MHDPIC:
        self.exch_bc_funcs[2][1] = z2_exch_bc_outflow

    if self.bc_flags[2][1]==2:
      self.grid_bc_funcs[2][1] = z2_grid_bc_reflect
      IF MHDPIC:
        self.exch_bc_funcs[2][1] = z2_exch_bc_reflect

    # ---------------------------------------------------------------

    # special reflective BC for cylindrical and spherical coordinates

    if gc.geom==CG_CYL:

      if gc.lmin[0]==0.:

        self.grid_bc_funcs[0][0] = r1_grid_bc_cyl
        IF MHDPIC:
          self.exch_bc_funcs[0][0] = r1_exch_bc_cyl

        IF MPI:
          if px==0 and sy>1:
            # translation phi->phi+pi
            ph_pos = (py + sy/2) % sy
            gc.nbr_ranks[0][0] = gc.ranks[0][ph_pos][pz]

    if gc.geom==CG_SPH:

      if gc.lmin[0]==0.:

        self.grid_bc_funcs[0][0] = r1_grid_bc_sph
        IF MHDPIC:
          self.exch_bc_funcs[0][0] = r1_exch_bc_sph

        IF MPI:
          if px==0 and (sy>1 or sz>1):
            # translation phi->phi+pi, theta->pi-theta
            th_pos = sy-py-1
            ph_pos = (pz + sz/2) % sz
            gc.nbr_ranks[0][0] = gc.ranks[0][th_pos][ph_pos]

      if gc.lmin[1]==0.:
        self.grid_bc_funcs[1][0] = th1_grid_bc_sph
        IF MHDPIC:
          self.exch_bc_funcs[1][0] = th1_exch_bc_sph

        IF MPI:
          if py==0 and sz>1:
            # translation phi->phi+pi
            ph_pos = (pz + sz/2) % sz
            gc.nbr_ranks[1][0] = gc.ranks[px][0][ph_pos]

      if gc.lmax[1]==B_PI:
        self.grid_bc_funcs[1][1] = th2_grid_bc_sph
        IF MHDPIC:
          self.exch_bc_funcs[1][1] = th2_exch_bc_sph

        IF MPI:
          if py==sy-1 and sz>1:
            # translation phi->phi+pi
            ph_pos = (pz + sz/2) % sz
            gc.nbr_ranks[1][1] = gc.ranks[px][sy-1][ph_pos]


    # --------------------------------------------------------------------------

    # Set remaining user-defined BCs.

    set_grid_bc_user(self)

    # Check if all BCs have been set.

    cdef int i,k

    for i in range(3):
      for k in range(2):

        if self.grid_bc_funcs[i][k] == NULL:
          if k==0:
            print_root('\nWARNING: Left boundary condition in %i direction is not set.\n')
          else:
            print_root('\nWARNING: Right boundary condition in %i direction is not set.\n')

        IF MHDPIC:
          if self.exch_bc_funcs[i][k] == NULL:
            if k==0:
              print_root('\nWARNING: Left exchange boundary condition in %i direction is not set.\n')
            else:
              print_root('\nWARNING: Right exchange boundary condition in %i direction is not set.\n')

    # --------------------------------------------------------------------------

    IF MPI:

      # Allocate BC buffers for MPI.

      cdef:
        int n, ndim=1
        long bufsize
        long Nxyz =  IMAX(IMAX(gc.Ntot[0],gc.Ntot[1]), gc.Ntot[2])

      IF D2D: ndim += 1
      IF D3D: ndim += 1

      IF MHDPIC:   n = (NMODE+7) * gc.ng
      ELIF MFIELD: n = (NMODE+3) * gc.ng
      ELSE:        n =  NMODE    * gc.ng

      if ndim=3:
        bufsize = (Nxyz+1)**2 * n
      elif ndim==2:
        bufsize = (Nxyz+1) * n
      else:
        bufsize = n

      self.sendbuf = np.zeros((2,bufsize), dtype=np_real)
      self.recvbuf = np.zeros((2,bufsize), dtype=np_real)
      self.recvbuf_size = bufsize
      self.sendbuf_size = bufsize


  # -------------------------------------------------------------------------

  cdef void apply_grid_bc(self, GridData gd, GridCoord *gc,
                          BnzIntegr integr, int1d bvars):

    cdef:
      PackFunc pack_func = pack_grid_all
      UnpackFunc unpack_func = unpack_grid_all

    IF DIAGNOSE:
      cdef timeval tstart, tstop
      print_root("apply grid BC... ")
      gettimeofday(&tstart, NULL)

    apply_bc(self, gd,gc, integr, bvars, self.grid_bc_funcs, pack_func, unpack_func)

    IF DIAGNOSE:
      gettimeofday(&tstop, NULL)
      print_root("%.1f ms\n", timediff(tstart,tstop))

  IF MHDPIC:

    cdef void apply_exch_bc(self, GridData gd, GridCoord *gc,
                            BnzIntegr integr, int1d bvars):

      cdef:
        PackFunc pack_func = pack_exch_all
        UnpackFunc unpack_func = unpack_exch_all

      IF DIAGNOSE:
        cdef timeval tstart, tstop
        print_root("apply exchange BC... ")
        gettimeofday(&tstart, NULL)

      apply_bc(self, gd,gc, integr, bvars, self.exch_bc_funcs, pack_func, unpack_func)

      IF DIAGNOSE:
        gettimeofday(&tstop, NULL)
        print_root("%.1f ms\n", timediff(tstart,tstop))

  #--------------------------------------------------------------------------

  def __dealloc__(self):

    cdef int i,k

    for i in range(3):
      for k in range(2):
        self.grid_bc_funcs[i][k] = NULL

    IF MHDPIC:
      for i in range(3):
        for k in range(2):
          self.exch_bc_funcs[i][k] = NULL


# --------------------------------------------------------------------------

cdef void apply_bc(GridBC gbc, GridData gd, GridCoord *gc, BnzIntegr integr,
                   int1d bvars, GridBCFunc bc_funcs[3][2],
                   PackFunc pack_func, UnpackFunc unpack_func):

  IF MPI:

    cdef int nx=gc.Ntot[0], ny=gc.Ntot[1], nz=gc.Ntot[2], ng=gc.ng

    if bvars is None:
      # if bvars not specified, apply BC to all variables
      bvars = np.arange(NVAR, dype=np.intp)

    cdef int nbvar = bvars.size

    cdef:

      mpi.Comm comm = mpi.COMM_WORLD
      int done
      mpi.Request send_req1, send_req2, recv_req1, recv_req2

      int cnt1,cnt2

      real2d sendbuf = gbc.sendbuf
      real2d recvbuf = gbc.recvbuf
      int **nbr_ranks = gc.nbr_ranks

    cdef:
      int rtagl, rtagr
      int stagl, stagr

    # CHECK THIS!
    # Do MPI exchange from z- to x-direction in order to use the same procedure
    # to set ghost cells normally as well as to exchange particle deposits
    # that must be done in reverse (z to x) for setting corners correctly.

    # ------- data exchange in z-direction --------------

    IF D3D:

      rtagl,rtagr=1,0
      stagl,stagr=0,1

      cnt1 = nx * ny * ng * nbvar
      cnt2 = cnt1

      if nbr_ranks[2][0] > -1 and nbr_ranks[2][1] > -1:

        recv_req1 = comm.Irecv([recvbuf[0,:], cnt1, mpi_real], nbr_ranks[2][0], tag=rtagl)
        recv_req2 = comm.Irecv([recvbuf[1,:], cnt2, mpi_real], nbr_ranks[2][1], tag=rtagr)

        pack_func(gd,gc, bvars, sendbuf[0,:], ZAX, 0)
        send_req1 = comm.Isend([sendbuf[0,:], cnt2, mpi_real], nbr_ranks[2][0], tag=stagl)
        pack_func(gd,gc, bvars, sendbuf[1,:], ZAX, 1)
        send_req2 = comm.Isend([sendbuf[1,:], cnt1, mpi_real], nbr_ranks[2][1], tag=stagr)

        mpi.Request.Waitall([send_req1,send_req2])

        done = mpi.Request.Waitany([recv_req1,recv_req2])
        if done==0:   unpack_func(gd,gc, bvars, recvbuf[0,:], ZAX,0)
        elif done==1: unpack_func(gd,gc, bvars, recvbuf[1,:], ZAX,1)

        done = mpi.Request.Waitany([recv_req1,recv_req2])
        if done==0:   unpack_func(gd,gc, bvars, recvbuf[0,:], ZAX,0)
        elif done==1: unpack_func(gd,gc, bvars, recvbuf[1,:], ZAX,1)

      elif nbr_ranks[2][0] == -1 and nbr_ranks[2][1] > -1:

        recv_req2 = comm.Irecv([recvbuf[1,:], cnt2, mpi_real], nbr_ranks[2][1], tag=rtagr)

        pack_func(gd,gc, bvars, sendbuf[1,:], ZAX, 1)
        send_req2 = comm.Isend([sendbuf[1,:], cnt1, mpi_real], nbr_ranks[2][1], tag=stagr)

        bc_funcs[2][0](sim, bvars)

        send_req2.Wait()
        recv_req2.Wait()
        unpack_func(gd,gc, bvars, recvbuf[1,:], ZAX,1)

      elif nbr_ranks[2][0] > -1 and nbr_ranks[2][1] == -1:

        recv_req1 = comm.Irecv([recvbuf[0,:], cnt1, mpi_real], nbr_ranks[2][0], tag=rtagl)

        pack_func(gd,gc, bvars, sendbuf[0,:], ZAX, 0)
        send_req1 = comm.Isend([sendbuf[0,:], cnt2, mpi_real], nbr_ranks[2][0], tag=stagl)

        bc_funcs[2][1](sim, bvars)

        send_req1.Wait()
        recv_req1.Wait()
        unpack_func(gd,gc, bvars, recvbuf[0,:], ZAX,0)

      else:

        bc_funcs[2][0](sim, bvars)
        bc_funcs[2][1](sim, bvars)


    # ------- data exchange in y-direction --------------

    IF D2D:

      rtagl,rtagr=1,0
      stagl,stagr=0,1

      if gc.geom==CG_SPH:
        if gc.lf[1][gc.j1]==0.:
          stagl=2
          rtagl=2
        if gc.lf[1][gc.j2+1]==B_PI:
          stagr=2
          rtagr=2

      cnt1 = nx * nz * ng * nbvar
      cnt2 = cnt1

      if nbr_ranks[1][0] > -1 and nbr_ranks[1][1] > -1:

        recv_req1 = comm.Irecv([recvbuf[0,:], cnt1, mpi_real], nbr_ranks[1][0], tag=rtagl)
        recv_req2 = comm.Irecv([recvbuf[1,:], cnt2, mpi_real], nbr_ranks[1][1], tag=rtagr)

        pack_func(gd,gc, bvars, sendbuf[0,:], YAX, 0)
        send_req1 = comm.Isend([sendbuf[0,:], cnt2, mpi_real], nbr_ranks[1][0], tag=stagl)

        pack_func(gd,gc, bvars, sendbuf[1,:], YAX, 1)
        send_req2 = comm.Isend([sendbuf[1,:], cnt1, mpi_real], nbr_ranks[1][1], tag=stagr)

        mpi.Request.Waitall([send_req1,send_req2])

        done = mpi.Request.Waitany([recv_req1,recv_req2])
        if done==0:   unpack_func(gd,gc, bvars, recvbuf[0,:], YAX,0)
        elif done==1: unpack_func(gd,gc, bvars, recvbuf[1,:], YAX,1)

        done = mpi.Request.Waitany([recv_req1,recv_req2])
        if done==0:   unpack_func(gd,gc, bvars, recvbuf[0,:], YAX,0)
        elif done==1: unpack_func(gd,gc, bvars, recvbuf[1,:], YAX,1)

      elif nbr_ranks[1][0] == -1 and nbr_ranks[1][1] > -1:

        recv_req2 = comm.Irecv([recvbuf[1,:], cnt2, mpi_real], nbr_ranks[1][1], tag=rtagr)

        pack_func(gd,gc, bvars, sendbuf[1,:], YAX, 1)
        send_req2 = comm.Isend([sendbuf[1,:], cnt1, mpi_real], nbr_ranks[1][1], tag=stagr)

        bc_funcs[1][0](sim, bvars)

        send_req2.Wait()
        recv_req2.Wait()
        unpack_func(gd,gc, bvars, recvbuf[1,:], YAX,1)

      elif nbr_ranks[1][0] > -1 and nbr_ranks[1][1] == -1:

        recv_req1 = comm.Irecv([recvbuf[0,:], cnt1, mpi_real], nbr_ranks[1][0], tag=rtagl)

        pack_func(gd,gc, bvars, sendbuf[0,:], YAX, 0)
        send_req1 = comm.Isend([sendbuf[0,:], cnt2, mpi_real], nbr_ranks[1][0], tag=stagl)

        bc_funcs[1][1](sim, bvars)

        send_req1.Wait()
        recv_req1.Wait()
        unpack_func(gd,gc, bvars, recvbuf[0,:], YAX,0)

      else:

        bc_funcs[1][0](sim, bvars)
        bc_funcs[1][1](sim, bvars)


    # ------- data exchange in x-direction --------------

    rtagl,rtagr=1,0
    stagl,stagr=0,1

    if gc.geom==CG_CYL or gc.geom==CG_SPH:
      if gc.lf[0][gc.i1]==0.:
        stagl=2
        rtagl=2

    cnt1 = ny * nz * ng * nbvar
    cnt2 = cnt1

    if nbr_ranks[0][0] > -1 and nbr_ranks[0][1] > -1:

      recv_req1 = comm.Irecv([recvbuf[0,:], cnt1, mpi_real], nbr_ranks[0][0], tag=rtagl)
      recv_req2 = comm.Irecv([recvbuf[1,:], cnt2, mpi_real], nbr_ranks[0][1], tag=rtagr)

      pack_func(gd,gc, bvars, sendbuf[0,:], XAX, 0)
      send_req1 = comm.Isend([sendbuf[0,:], cnt2, mpi_real], nbr_ranks[0][0], tag=stagl)

      pack_func(gd,gc, bvars, sendbuf[1,:], XAX, 1)
      send_req2 = comm.Isend([sendbuf[1,:], cnt1, mpi_real], nbr_ranks[0][1], tag=stagr)

      mpi.Request.Waitall([send_req1,send_req2])

      done = mpi.Request.Waitany([recv_req1,recv_req2])
      if done==0:   unpack_func(gd,gc, bvars, recvbuf[0,:], XAX,0)
      elif done==1: unpack_func(gd,gc, bvars, recvbuf[1,:], XAX,1)

      done = mpi.Request.Waitany([recv_req1,recv_req2])
      if done==0:   unpack_func(gd,gc, bvars, recvbuf[0,:], XAX,0)
      elif done==1: unpack_func(gd,gc, bvars, recvbuf[1,:], XAX,1)

    elif nbr_ranks[0][0] == -1 and nbr_ranks[0][1] > -1:

      recv_req2 = comm.Irecv([recvbuf[1,:], cnt2, mpi_real], nbr_ranks[0][1], tag=rtagr)

      pack_func(gd,gc, bvars, sendbuf[1,:], XAX, 1)
      send_req2 = comm.Isend([sendbuf[1,:], cnt1, mpi_real], nbr_ranks[0][1], tag=stagr)

      bc_funcs[0][0](sim, bvars)

      send_req2.Wait()
      recv_req2.Wait()
      unpack_func(gd,gc, bvars, recvbuf[1,:], XAX,1)

    elif nbr_ranks[0][0] > -1 and nbr_ranks[0][1] == -1:

      recv_req1 = comm.Irecv([recvbuf[0,:], cnt1, mpi_real], nbr_ranks[0][0], tag=rtagl)

      pack_func(gd,gc, bvars, sendbuf[0,:], XAX, 0)
      send_req1 = comm.Isend([sendbuf[0,:], cnt2, mpi_real], nbr_ranks[0][0], tag=stagl)

      bc_funcs[0][1](sim, bvars)

      send_req1.Wait()
      recv_req1.Wait()
      unpack_func(gd,gc, bvars, recvbuf[0,:], XAX,0)

    else:

      bc_funcs[0][0](sim, bvars)
      bc_funcs[0][1](sim, bvars)

  ELSE:

    IF D3D:
      bc_funcs[2][0](sim, bvars)
      bc_funcs[2][1](sim, bvars)

    IF D2D:
      bc_funcs[1][0](sim, bvars)
      bc_funcs[1][1](sim, bvars)

    bc_funcs[0][0](sim, bvars)
    bc_funcs[0][1](sim, bvars)
