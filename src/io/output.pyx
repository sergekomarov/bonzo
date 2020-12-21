# -*- coding: utf-8 -*-
IF MPI:
  from mpi4py import MPI as mpi
  from mpi4py cimport MPI as mpi

import numpy as np
cimport numpy as np
from cython.parallel import prange, parallel, threadid
import os
import h5py

# from h5write cimport h5write

IF SPREC:
  np_real = np.float32
ELSE:
  np_real = np.float64


cdef np.ndarray[double, ndim=1] get_hst_vars_grid(BnzGrid grid, BnzIntegr integr):

  # Calculate grid history variables, e.g., mean energy densities.

  cdef:
    int i,j,k,n,m
    int id
    real bx2,by2,bz2
    real engk,engt,engm, rhoi
    int nhst_grid

  cdef:
    GridCoord *gc = grid.coord
    GridData gd = grid.data

  cdef real4d u = gd.cons

  # get number of all (global) active cells
  cdef int ncells = gc.Nact_glob[0] * gc.Nact_glob[1] * gc.Nact_glob[2]

  IF MFIELD: nhst_grid=16
  ELSE: nhst_grid=12

  cdef real2d means_loc = np.zeros((OMP_NT, nhst_grid-2), dtype=np_real)
  cdef double[::1] hst_vars = np.zeros(nhst_grid, dtype='f8')

  IF MPI:
    cdef:
      mpi.Comm comm = mpi.COMM_WORLD
      double[::1] vars     = np.empty(nhst_grid-2, dtype='f8')
      double[::1] vars_sum = np.empty(nhst_grid-2, dtype='f8')

  hst_vars[0] = integr.step
  hst_vars[1] = integr.time

  # Calculate MHD history variables.

  with nogil, parallel(num_threads=OMP_NT):

    id = threadid()

    for k in prange(gc.k1, gc.k2+1, schedule='dynamic'):

      engm=0.

      for j in range(gc.j1, gc.j2+1):
        for i in range(gc.i1, gc.i2+1):

          rhoi = 1./u[RHO,k,j,i]

          engk = 0.5*rhoi * (SQR(u[MX,k,j,i]) + SQR(u[MY,k,j,i]) + SQR(u[MZ,k,j,i]))
          IF MFIELD:
            bx2, by2, bz2 = SQR(u[BX,k,j,i]), SQR(u[BY,k,j,i]), SQR(u[BZ,k,j,i])
            engm = 0.5*(bx2 + by2 + bz2)

          engt = u[EN,k,j,i] - engk - engm

          n=0

          # gas density
          means_loc[id,n] = means_loc[id,n] + u[RHO,k,j,i]
          n = n+1
          # gas kinetic energy
          means_loc[id,n] = means_loc[id,n] + engk
          n = n+1
          # thermal energy density
          means_loc[id,n] = means_loc[id,n] + engt
          n = n+1
          IF MFIELD:
            # magnetic energy density
            means_loc[id,n] = means_loc[id,n] + engm
            n = n+1
          # total energy density
          means_loc[id,n] = means_loc[id,n] + u[EN,k,j,i]
          n = n+1
          # momenta
          means_loc[id,n] = means_loc[id,n] + u[MX,k,j,i]
          n = n+1
          means_loc[id,n] = means_loc[id,n] + u[MY,k,j,i]
          n = n+1
          means_loc[id,n] = means_loc[id,n] + u[MZ,k,j,i]
          n = n+1

          IF MFIELD:

            # mean magnetic field components
            means_loc[id,n] = means_loc[id,n] + u[BX,k,j,i]
            n = n+1
            means_loc[id,n] = means_loc[id,n] + u[BY,k,j,i]
            n = n+1
            means_loc[id,n] = means_loc[id,n] + u[BZ,k,j,i]
            n = n+1

            # mean square magnetic field components
            means_loc[id,n] = means_loc[id,n] + bx2
            n = n+1
            means_loc[id,n] = means_loc[id,n] + by2
            n = n+1
            means_loc[id,n] = means_loc[id,n] + bz2
            n = n+1

  for k in range(2,nhst_grid):
    for n in range(OMP_NT):
      hst_vars[k] += means_loc[n,k-2]

  IF MPI:
    vars = hst_vars[2:]
    comm.Allreduce(vars, vars_sum, op=mpi.SUM)
    hst_vars[2:] = vars_sum

  for k in range(2,nhst_grid):
    hst_vars[k] = hst_vars[k] / ncells

  IF MFIELD:
    # rms magnetic field components
    for k in range(nhst_grid-3,nhst_grid):
      hst_vars[k] = SQRT(FABS(hst_vars[k] - SQR(hst_vars[k-3])))

  return np.asarray(hst_vars)


# ---------------------------------------------------------------------------------

IF MHDPIC:

  cdef np.ndarray[double, ndim=1] get_hst_vars_prt(BnzGrid grid, BnzIntegr integr):

    # Calculate CR particles history variables.

    cdef:
      int k,j,i
      long n
      int id

    cdef:
      GridCoord *gc = grid.params
      BnzParticles prts = grid.prts
      ParticleProp *pp = prts.prop
      ParticleData *pd = prts.data

    # get number of all (global) cells
    cdef int ncells = gc.Nact_glob[0] * gc.Nact_glob[1] * gc.Nact_glob[2]

    cdef int nhst_prt=8

    real2d means_loc = np.zeros((OMP_NT, nhst_prt-1), dtype=np_real)
    cdef:
      double[::1] hst_vars = np.zeros(nhst_prt, dtype='f8')
    IF MPI:
      cdef:
        mpi.Comm comm = mpi.COMM_WORLD
        double[::1] vars     = np.empty(nhst_prt, dtype='f8')
        double[::1] vars_sum = np.empty(nhst_prt, dtype='f8')

    hst_vars[0] = pp.Np

    with nogil, parallel(num_threads=OMP_NT):

      id = threadid()

      for n in prange(pp.Np, schedule='dynamic'):

        # particle kinetic energy

        means_loc[id,0] = means_loc[id,0] + (pd.g[n]-1.)
        means_loc[id,1] = means_loc[id,1] + pd.u[n]
        means_loc[id,2] = means_loc[id,2] + pd.v[n]
        means_loc[id,3] = means_loc[id,3] + pd.w[n]
        means_loc[id,4] = means_loc[id,4] + SQR(pd.u[n])
        means_loc[id,5] = means_loc[id,5] + SQR(pd.v[n])
        means_loc[id,6] = means_loc[id,6] + SQR(pd.w[n])

    # add up contributions from different threads
    for k in range(1,nhst_prt):
      for n in range(OMP_NT):
        hst_vars[k] += means_loc[n,k-1]

    # add up contributions from MPI blocks
    IF MPI:
      vars[:] = hst_vars
      comm.Allreduce(vars, vars_sum, op=mpi.SUM)
      hst_vars[:] = vars_sum

    # mean particle density
    hst_vars[0] = hst_vars[0] * integr.rho_cr / (ncells*pp.ppc)
    # mean particle energy density
    hst_vars[1] = hst_vars[1] * integr.rho_cr * SQR(integr.sol) / (ncells*pp.ppc)

    # divide by the total number of particles
    for k in range(2,nhst_prt):
      hst_vars[k] = hst_vars[k]/hst_vars[0]

    # calculate rms velocities
    for k in range(nhst_prt-3,nhst_prt):
      hst_vars[k] = SQRT(FABS(hst_vars[k] - SQR(hst_vars[k-3])))

  return np.asarray(hst_vars)


# -------------------------------------------------------------------

cdef void write_history(BnzIO out, BnzGrid grid, BnzIntegr integr):

  # Write history variables.

  cdef np.ndarray[double, ndim=1] hst_vars

  cdef int rank=0
  IF MPI: rank=mpi.COMM_WORLD.Get_rank()

  hst_vars = get_hst_vars_grid(grid,integr)
  IF MHDPIC:
    hst_vars = np.append(hst_vars, get_hst_vars_prt(grid,integr))

  # add user-defined history variables
  for k in range(len(out.hst_funcs_u)):
    np.append(hst_vars, <double>out.hst_funcs_u[k](grid,integr))

  if rank==0:
    hst_path = os.path.join(out.usr_dir, 'out', 'hst.txt')
    with open(hst_path, 'a') as f:
      for var in hst_vars:
        f.write('{:<14e}'.format(var))
      f.write('\n')


# ---------------------------------------------------------------------------

cdef void write_grid(BnzIO out, BnzGrid grid, BnzIntegr integr, int rst):

  # Write the computational grid.

  cdef:
    int k,j,i,n,m
    int i1c,i2c, j1c,j2c, k1c,k2c
    int i1t,i2t, j1t,j2t, k1t,k2t
    # int grid_dim[3]
    # int chunk_dim[3]
    # int block_dim[3]
    # int offset[3]
    int write_ghost
    VarType var_type

  cdef:
    GridCoord *gc = grid.coord
    GridData gd = grid.data

  cdef:
    int i0 = gc.pos[0] * gc.Nact[0]
    int j0 = gc.pos[1] * gc.Nact[1]
    int k0 = gc.pos[2] * gc.Nact[2]

  if rst:
    write_ghost=0
  else:
    write_ghost = out.write_ghost

  if rst:
    var_type = VAR_PRIM
  else:
    var_type = out.var_type

  # Set names of variables.

  if var_type==VAR_PRIM:

    var_names = ['rho','vx','vy','vz','p','psc']

    IF TWOTEMP:
      var_names.insert(5, 'pe')
    IF CGL:
      var_names.insert(5, 'ppd')

  elif var_type==VAR_CONS:

    var_names = ['rho','mx','my','mz','en','psc']
    IF TWOTEMP:
      var_names.insert(5, 'se')
    IF CGL:
      var_names.insert(5, 'loga')

  IF MFIELD: var_names.extend(['bxc','byc','bzc', 'bxf','byf','bzf'])

  # Set starting and ending indexes of data chunks for parallel write.

  if write_ghost:

    i1c = 0 if gc.pos[0]==0 else gc.i1
    j1c = 0 if gc.pos[1]==0 else gc.j1
    k1c = 0 if gc.pos[2]==0 else gc.k1

    i2c = gc.Ntot[0] if gc.pos[0]==gc.size[0]-1 else gc.i2+1
    j2c = gc.Ntot[1] if gc.pos[1]==gc.size[1]-1 else gc.j2+1
    k2c = gc.Ntot[2] if gc.pos[2]==gc.size[2]-1 else gc.k2+1

    i1t, i2t = i1c + i0, i2c + i0
    j1t, j2t = j1c + j0, j2c + j0
    k1t, k2t = k1c + k0, k2c + k0

  else:

    i1c, i2c = gc.i1, gc.i2+1
    j1c, j2c = gc.j1, gc.j2+1
    k1c, k2c = gc.k1, gc.k2+1

    i1t, i2t = i0, i0 + gc.Nact[0]
    j1t, j2t = j0, j0 + gc.Nact[1]
    k1t, k2t = k0, k0 + gc.Nact[2]

  # full grid dimensions

  if write_ghost:
    shg = np.asarray(gc.Ntot_glob)
  else:
    shg = np.asarray(gc.Nact_glob)

  grid_dim = (shg[2],shg[1],shg[0])
  # dimensions and offsets of MPI block on full grid
  # block_dim = (k2t-k1t, j2t-j1t, i2t-i1t)
  # offset = (k1t, j1t, i1t)

  chunked=0
  chunks=None
  if not write_ghost:
    chunked=1
    chunks = (k2t-k1t, j2t-j1t, i2t-i1t)

  # Generate file name.
  if rst:
    file_name = os.path.join(out.usr_dir, 'rst', 'rst_grid.h5')
  else:
    file_name = ( os.path.join(out.usr_dir, 'out', 'grid') +
                  '_{:03d}.h5'.format(<int>(integr.time / out.grid_dt)) )

  # --------------------

  driver=None
  driver_args={}
  IF MPI:
    driver='mpio'
    driver_args['comm'] = mpi.COMM_WORLD

  f = h5py.File(file_name,'w', driver=driver, **driver_args)

  for var_name in var_names:
    f.create_dataset(var_name, grid_dim, chunks=chunks, dtype=np_real)

  sht = np.s_[k1t:k2t, j1t:j2t, i1t:i2t]
  shc = np.s_[k1c:k2c, j1c:j2c, i1c:i2c]

  if var_type==VAR_PRIM:
    for n in range(NMODE):
      f[var_names[n]][sht] = gd.prim[n][shc]
  elif var_type==VAR_CONS:
    for n in range(NMODE):
      f[var_names[n]][sht] = gd.cons[n][shc]
  IF MFIELD:
    for n in range(3):
      f[var_names[NMODE+n]][sht] = gd.bfld[n][shc]

  f.attrs['time'] = integr.time
  f.attrs['step'] = integr.step
  f.attrs['dt'] = integr.dt
  f.attrs['xf'] = np.asarray(gc.lf[0])
  f.attrs['yf'] = np.asarray(gc.lf[1])
  f.attrs['zf'] = np.asarray(gc.lf[2])
  # f.attrs['Lx'] = gc.lmax[0]-gc.lmin[0]
  # f.attrs['Ly'] = gc.lmax[1]-gc.lmin[1]
  # f.attrs['Lz'] = gc.lmax[2]-gc.lmin[2]

  f.close()

  # Create a dictionary of datasets.

  # dsets={}
  #
  # IF PIC:
  #   for n in range(3):
  #     dsets[var_names[n]]   = (gd.efld[n, k1c:k2c, j1c:j2c, i1c:i2c], 'r')
  #     dsets[var_names[n+3]] = (gd.bfld[n, k1c:k2c, j1c:j2c, i1c:i2c], 'r')
  #     dsets[var_names[n+6]] = (gd.curr[n, k1c:k2c, j1c:j2c, i1c:i2c], 'r')
  # ELSE:
  #   if var_type==VAR_PRIM:
  #     for n in range(NMODE):
  #       dsets[var_names[n]] = (gd.prim[n, k1c:k2c, j1c:j2c, i1c:i2c], 'r')
  #   elif var_type==VAR_CONS:
  #     for n in range(NMODE):
  #       dsets[var_names[n]] = (gd.cons[n, k1c:k2c, j1c:j2c, i1c:i2c], 'r')
  #   IF MFIELD:
  #     for n in range(3):
  #       dsets[var_names[NMODE+n]] = (gd.bfld[n, k1c:k2c, j1c:j2c, i1c:i2c], 'r')
  #
  # # Create a dictionary of attributes.
  #
  # attrs = {}
  # attrs['t'] = (integr.time, 'r')
  # attrs['step'] = (integr.step, 'l')
  # attrs['dt'] = (integr.dt, 'r')
  # attrs['dx'] = (gc.dlf[0][0], 'r')
  # attrs['dy'] = (gc.dlf[1][0], 'r')
  # attrs['dz'] = (gc.dlf[2][0], 'r')
  # attrs['Lx'] = (gc.lmax[0]-gc.lmin[0], 'r')
  # attrs['Ly'] = (gc.lmax[1]-gc.lmin[1], 'r')
  # attrs['Lz'] = (gc.lmax[2]-gc.lmin[2], 'r')
  #
  # h5write(file_name, dsets,attrs, 3,
  #         grid_dim, chunk_dim, block_dim, offset, chunked)


# -----------------------------------------------------------------------

cpdef void write_slice(BnzIO out, BnzGrid grid, BnzIntegr integr):

  cdef:
    int k,j,i,n, n1
    int i1c,i2c, j1c,j2c, k1c,k2c
    int i1t,i2t, j1t,j2t, k1t,k2t, islc
    # int slc_dim[3]
    # int chunk_dim[3]
    # int block_dim[3]
    # int offset[3]

  cdef:
    GridCoord *gc = grid.coord
    GridData gd = grid.data

  cdef int rank=0
  IF MPI: rank=mpi.COMM_WORLD.Get_rank()

  cdef:
    int i0 = gc.pos[0] * gc.Nact[0]
    int j0 = gc.pos[1] * gc.Nact[1]
    int k0 = gc.pos[2] * gc.Nact[2]

  if out.var_type==VAR_PRIM:

    var_names = ['rho','vx','vy','vz','p','psc']

    IF TWOTEMP:
      var_names.insert(5, 'pe')
    IF CGL:
      var_names.insert(5, 'ppd')

  elif out.var_type==VAR_CONS:

    var_names = ['rho','mx','my','mz','en','psc']
    IF TWOTEMP:
      var_names.insert(5, 'se')
    IF CGL:
      var_names.insert(5, 'loga')

  IF MFIELD: var_names.extend(['bxc','byc','bzc', 'bxf','byf','bzf'])

  # Set starting and ending indexes of data chunks for parallel write.

  # obtain the local coordinate of the slice
  # check if the slice is within the coordinate range of the current processor

  if out.slc_axis==0:
    if out.slc_pos / gc.Nact[0] == gc.pos[0]:
      islc = out.slc_pos % gc.Nact[0] + gc.i1
      i1c, i2c = islc, islc+1
      i1t, i2t = 0, 1
    else:
      i1c, i2c = 0, 0
      i1t, i2t = 0, 0

  if out.slc_axis==1:
    if out.slc_pos / gc.Nact[1] == gc.pos[1]:
      islc = out.slc_pos % gc.Nact[1] + gc.j1
      j1c, j2c = islc, islc+1
      j1t, j2t = 0, 1
    else:
      j1c, j2c = 0, 0
      j1t, j2t = 0, 0

  if out.slc_axis==2:
    if out.slc_pos / gc.Nact[2] == gc.pos[2]:
      islc = out.slc_pos % gc.Nact[2] + gc.k1
      k1c, k2c = islc, islc+1
      k1t, k2t = 0, 1
    else:
      k1c, k2c = 0, 0
      k1t, k2t = 0, 0

  if out.write_ghost:

    if out.slc_axis != 0:

      i1c = 0 if gc.pos[0]==0 else gc.i1
      i2c = gc.Ntot[0] if gc.pos[0]==gc.size[0]-1 else gc.i2+1
      i1t, i2t = i1c + i0, i2c + i0

    if out.slc_axis != 1:

      j1c = 0 if gc.pos[1]==0 else gc.j1
      j2c = gc.Ntot[1] if gc.pos[1]==gc.size[1]-1 else gc.j2+1
      j1t, j2t = j1c + j0, j2c + j0

    if out.slc_axis != 2:

      k1c = 0 if gc.pos[2]==0 else gc.k1
      k2c = gc.Ntot[2] if gc.pos[2]==gc.size[2]-1 else gc.k2+1
      k1t, k2t = k1c + k0, k2c + k0

  else:

    if out.slc_axis != 0:
      i1c, i2c = gc.i1, gc.i2+1
      i1t, i2t = i0, i0 + gc.Nact[0]
    if out.slc_axis != 1:
      j1c, j2c = gc.j1, gc.j2+1
      j1t, j2t = j0, j0 + gc.Nact[1]
    if out.slc_axis != 2:
      k1c, k2c = gc.k1, gc.k2+1
      k1t, k2t = k0, k0 + gc.Nact[2]

  # full slice dimensions

  if out.write_ghost:
    shg = np.asarray(gc.Ntot_glob)
  else:
    shg = np.asarray(gc.Nact_glob)

  if out.slc_axis==0:
    slc_dim = (shg[2], shg[1], 1)
    chunk_dim = (k2t-k1t, j2t-j1t, 1)
  elif out.slc_axis==1:
    slc_dim = (shg[2], 1, shg[0])
    chunk_dim = (k2t-k1t, 1, i2t-i1t)
  elif out.slc_axis==2:
    slc_dim = (1, shg[1], shg[0])
    chunk_dim = (1, j2t-j1t, i2t-i1t)

  # dimensions and offsets of MPI block on full grid
  # block_dim = (k2t-k1t, j2t-j1t, i2t-i1t)
  # offset = (k1t, j1t, i1t)

  chunked=0
  chunks=None
  if not out.write_ghost:
    chunked=1
    chunks=chunk_dim

  # Generate file name.
  file_name = (os.path.join(out.usr_dir, 'out', 'slc') +
                '_{:03d}.h5'.format(<int>(integr.time/out.slc_dt)) )

  # --------------------

  driver=None
  driver_args={}
  IF MPI:
    driver='mpio'
    driver_args['comm'] = mpi.COMM_WORLD

  f = h5py.File(file_name,'w', driver=driver, **driver_args)

  for var_name in var_names:
    f.create_dataset(var_name, slc_dim, chunks=chunks, dtype=np_real)

  sht = np.s_[k1t:k2t, j1t:j2t, i1t:i2t]
  shc = np.s_[k1c:k2c, j1c:j2c, i1c:i2c]

  if var_type==VAR_PRIM:
    for n in range(NMODE):
      f[var_names[n]][sht] = gd.prim[n][shc]
  elif var_type==VAR_CONS:
    for n in range(NMODE):
      f[var_names[n]][sht] = gd.cons[n][shc]
  IF MFIELD:
    for n in range(3):
      f[var_names[NMODE+n]][sht] = gd.bfld[n][shc]

  f.attrs['time'] = integr.time
  f.attrs['step'] = integr.step
  f.attrs['dt'] = integr.dt
  f.attrs['xf'] = np.asarray(gc.lf[0])
  f.attrs['yf'] = np.asarray(gc.lf[1])
  f.attrs['zf'] = np.asarray(gc.lf[2])
  # f.attrs['Lx'] = gc.lmax[0]-gc.lmin[0]
  # f.attrs['Ly'] = gc.lmax[1]-gc.lmin[1]
  # f.attrs['Lz'] = gc.lmax[2]-gc.lmin[2]

  f.close()

  # Create a dictionary of datasets.

  # dsets={}
  #
  # IF PIC:
  #   for n in range(3):
  #     dsets[var_names[n]]   = (gd.efld[n, k1c:k2c, j1c:j2c, i1c:i2c], 'r')
  #     dsets[var_names[n+3]] = (gd.bfld[n, k1c:k2c, j1c:j2c, i1c:i2c], 'r')
  #     dsets[var_names[n+6]] = (gd.curr[n, k1c:k2c, j1c:j2c, i1c:i2c], 'r')
  # ELSE:
  #   if out.var_type==VAR_PRIM:
  #     for n in range(NMODE):
  #       dsets[var_names[n]] = (gd.prim[n, k1c:k2c, j1c:j2c, i1c:i2c], 'r')
  #   elif out.var_type==VAR_CONS:
  #     for n in range(NMODE):
  #       dsets[var_names[n]] = (gd.cons[n, k1c:k2c, j1c:j2c, i1c:i2c], 'r')
  #   IF MFIELD:
  #     for n in range(3):
  #       dsets[var_names[NMODE+n]] = (gd.bfld[n, k1c:k2c, j1c:j2c, i1c:i2c], 'r')
  #
  # # Create a dictionary of attributes.
  #
  # attrs = {}
  # attrs['t'] = (integr.time, 'r')
  # attrs['step'] = (integr.step, 'l')
  # attrs['dt'] = (integr.dt, 'r')
  # attrs['dx'] = (gc.dlf[0][0], 'r')
  # attrs['dy'] = (gc.dlf[1][0], 'r')
  # attrs['dz'] = (gc.dlf[2][0], 'r')
  # attrs['Lx'] = (gc.lmax[0]-gc.lmin[0], 'r')
  # attrs['Ly'] = (gc.lmax[1]-gc.lmin[1], 'r')
  # attrs['Lz'] = (gc.lmax[2]-gc.lmin[2], 'r')
  #
  # h5write(file_name, dsets,attrs, 3
  #         slc_dim, chunk_dim, block_dim, offset, chunked)


# ----------------------------------------------------------------------------------

# Write particles.

IF MHDPIC:

  cdef void write_particles(BnzIO out, BnzGrid grid, BnzIntegr integr, int rst):

    cdef:
      GridCoord *gc = grid.coord
      BnzParticles prts = grid.prts
      ParticleData *pd = prts.data
      ParticleProp *pp = prts.prop

    cdef int rank=0
    IF MPI: rank = mpi.COMM_WORLD.Get_rank()

    # Count selected particles and save their local indices.

    cdef long n, np_sel = 0
    cdef long[::1] ind_sel = np.arange(pp.Np, dtype=np.int_)  #!!!!!!!!!!!!

    if rst or (out.prt_sel_func == NULL and out.prt_stride==1):
      np_sel = pp.Np

    elif out.prt_sel_func != NULL:

      for n in range(pp.Np):

        if out.prt_sel_func(pd, n):
          ind_sel[np_sel] = n
          np_sel += 1

    elif out.prt_stride>1:

      for n in range(pp.Np):

        if pd.id[n] % out.prt_stride == 0:
          ind_sel[np_sel] = n
          np_sel += 1

    # Calculate particle numbers on each processor.

    cdef:
      long[::1] np_sel_loc = np.array([np_sel], dtype=np.int_)
      long[::1] np_sel_proc
      long np_sel_tot

    IF MPI:
      # gather numbers of particles on different processors
      np_sel_proc = np.zeros(gc.size_tot, dtype=np.int_)
      # comm.Barrier()
      mpi.COMM_WORLD.Allgather(np_sel_loc, np_sel_proc)
      np_sel_tot = np_sel_proc.sum()
    ELSE:
      np_sel_proc = np_sel_loc
      np_sel_tot  = np_sel

    # Create file.

    if rst:
      fname = os.path.join(out.usr_dir, 'rst', 'rst_prt.h5')
    else:
      fname = ( os.path.join(out.usr_dir, 'out', 'prt')
              + '_{:03d}.h5'.format(<int>(integr.time/out.prt_dt)) )

    driver=None
    driver_args={}
    IF MPI:
      driver='mpio'
      driver_args['comm'] = mpi.COMM_WORLD

    f = h5py.File(fname,'w', driver=driver, **driver_args)

    # Create datasets.

    var_names = ['x','y','z','u','v','w','g','m','spc','id']#,'proc']
    for var_name in var_names[:8]:
      f.create_dataset(var_name, np_sel_tot, dtype=np_real)
    f.create_dataset(var_names[8], np_sel_tot, dtype=np.intc)
    f.create_dataset(var_names[9], np_sel_tot, dtype=np.int_)
    # f.create_dataset(var_names[10], Np_sel_tot, dtype='i8')

    # Set attributes.

    f.attrs['time'] = integr.time
    f.attrs['step'] = integr.step
    f.attrs['dt'] = integr.dt
    f.attrs['Np'] = np_sel_tot
    f.attrs['Ns'] = pp.Ns
    # cdef long[::1] np_spc = np.zeros(pp.Ns, dtype=np.int_)
    # for n in range(pp.Ns):
    #   np_spc[n] = pp.spc_props[n].Np
    # f.attrs['Np_spc'] = np_spc

    if rst:
      # numbers of selected particles on every processor
      f.attrs['Np_proc'] = np.asarray(np_sel_proc, dtype=np.int_)

    # global index of the first selected particle on current processor
    cdef long i, n0=np_sel_proc[:rank].sum()

    for n in range(n0,n0+np_sel):

      i = ind_sel[n]

      f['x'][n] = pd.x[i]
      f['y'][n] = pd.y[i]
      f['z'][n] = pd.z[i]

      f['u'][n] = pd.u[i]
      f['v'][n] = pd.v[i]
      f['w'][n] = pd.w[i]
      f['g'][n] = pd.g[i]
      f['m'][n] = pd.m[i]
      f['spc'][n] = pd.spc[i]
      f['id'][n] = pd.id[i]
      # f['proc'][n] = rank

    f.close()

    # cdef:
    #   real *xs
    #   real *ys
    #   real *zs
    #   real *us
    #   real *vs
    #   real *ws
    #   real *gs
    #   real *ms
    #   int *spcs
    #   long *ids
    #
    # if out.prt_stride>1 or out.prt_sel_func!=NULL:
    #
    #   # create arrays of selected particles
    #
    #   xs = <real*>calloc(np_sel, sizeof(real))
    #   ys = <real*>calloc(np_sel, sizeof(real))
    #   zs = <real*>calloc(np_sel, sizeof(real))
    #   us = <real*>calloc(np_sel, sizeof(real))
    #   vs = <real*>calloc(np_sel, sizeof(real))
    #   ws = <real*>calloc(np_sel, sizeof(real))
    #   gs = <real*>calloc(np_sel, sizeof(real))
    #   ms = <real*>calloc(np_sel, sizeof(real))
    #   spcs = <int*>calloc(np_sel, sizeof(int))
    #   ids = <long*>calloc(np_sel, sizeof(long))
    #
    #   for n in range(np_sel):
    #
    #     i = ind_sel[n]
    #     xs[n] = pd.x[i]
    #     ys[n] = pd.y[i]
    #     zs[n] = pd.z[i]
    #     us[n] = pd.u[i]
    #     vs[n] = pd.v[i]
    #     ws[n] = pd.w[i]
    #     gs[n] = pd.g[i]
    #     ms[n] = pd.m[i]
    #     spcs[n] = pd.spc[i]
    #     ids[n] = pd.id[i]
    #
    # else:
    #   xs = pd.x
    #   ys = pd.y
    #   zs = pd.z
    #   us = pd.u
    #   vs = pd.v
    #   ws = pd.w
    #   gs = pd.g
    #   ms = pd.m
    #   spcs = pd.spc
    #   ids = pd.id
    #
    # if rst:
    #   fname = os.path.join(out.usr_dir, 'rst', 'rst_prt.h5')
    # else:
    #   fname = ( os.path.join(out.usr_dir, 'out', 'prt')
    #           + '_{:03d}.h5'.format(<int>(integr.time/out.prt_dt))
    #
    # glob_dim = [np_sel_tot]
    # loc_dim = [np_sel]
    # chunk_dim = [np_sel]
    # offset = [np_sel_proc[:rank].sum()]
    # chunked=0
    #
    # dsets={}
    # dsets['x'] = (xs, 'r')
    # dsets['y'] = (ys, 'r')
    # dsets['z'] = (zs, 'r')
    # dsets['u'] = (us, 'r')
    # dsets['v'] = (vs, 'r')
    # dsets['w'] = (ws, 'r')
    # dsets['g'] = (gs, 'r')
    # dsets['m'] = (ms, 'r')
    # dsets['spc'] = (spcs, 'i')
    # dsets['id'] = (ids, 'l')
    #
    # cdef long[::1] np_spc = np.zeros(pp.Ns, dtype=np.int_)
    # for i in range(pp.Ns):
    #   np_spc[i] = pp.spc_props[i].Np
    #
    # dsets['Np_spc'] = (np_spcs, 'l')
    # dsets['Np_proc'] = (np_sel_proc, 'l')
    #
    # attrs = {}
    # attrs['t'] = (integr.time, 'r')
    # attrs['step'] = (integr.step, 'l')
    # attrs['dt'] = (integr.dt, 'r')
    # attrs['Np'] = (np_sel_tot, 'l')
    # # attrs['Np_loc'] = (np_sel, 'l')
    #
    # h5write(file_name, dsets,attrs, 1
    #         slc_dim, chunk_dim, block_dim, offset, chunked)
    #
    # if out.prt_stride>1 or out.prt_sel_func!=NULL:
    #   free(xs)
    #   free(ys)
    #   free(zs)
    #   free(us)
    #   free(vs)
    #   free(ws)
    #   free(gs)
    #   free(ms)
    #   free(spcs)
    #   free(ids)
