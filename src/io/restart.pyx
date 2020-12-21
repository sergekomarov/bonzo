# -*- coding: utf-8 -*-

IF MPI:
  from mpi4py import MPI as mpi
  from mpi4py cimport MPI as mpi

import numpy as np
cimport numpy as np
import os
from cython.parallel import prange, parallel, threadid
import h5py

# from h5restart cimport h5restart3

IF SPREC:
  np_real = np.float32
ELSE:
  np_real = np.float64


#-------------------------------------------------------------------------

cdef void set_restart_grid(BnzIO out, BnzGrid grid, BnzIntegr integr):

  cdef:
    int i1t,i2t, j1t,j2t, k1t,k2t, n
    int i1c,i2c, j1c,j2c, k1c,k2c
    int i,j,k, it,jt,kt
    # int block_dim[3]
    # int offset[3]
    # int offset_loc[3]

  cdef:
    GridCoord *gc = grid.coord
    GridData gd = grid.data

  cdef:
    int i0 = gc.pos[0] * gc.Nact[0]
    int j0 = gc.pos[1] * gc.Nact[1]
    int k0 = gc.pos[2] * gc.Nact[2]

  # Set names of variables.
  var_names = ['rho','vx','vy','vz','p','psc']
  IF MFIELD: var_names.extend(['bxc','byc','bzc','bxf','byf','bzf'])
  IF TWOTEMP:
    var_names.insert(5, 'pe')
  IF CGL:
    var_names.insert(5, 'ppd')

  # Set starting and ending indexes of data chunks for parallel write.

  i1c, j1c, k1c = gc.i1,   gc.j1,   gc.k1
  i2c, j2c, k2c = gc.i2+1, gc.j2+1, gc.k2+1

  i1t, i2t = i0, gc.Nact[0]+i0
  j1t, j2t = j0, gc.Nact[1]+j0
  k1t, k2t = k0, gc.Nact[2]+k0

  # dimensions and offsets of MPI block on full grid
  block_dim = (k2t-k1t, j2t-j1t, i2t-i1t)
  # offset    = (k1t, j1t, i1t)
  # offset_loc = [k1c, j1c, i1c]

  file_name = os.path.join(out.usr_dir, 'rst', 'rst_grid.h5')

  # ---------------------------

  driver=None
  driver_args={}
  IF MPI:
    driver='mpio'
    driver_args['comm']=mpi.COMM_WORLD

  f = h5py.File(file_name, 'r', driver=driver, **driver_args)

  sht = np.s_[k1t:k2t, j1t:j2t, i1t:i2t]
  shc = np.s_[k1c:k2c, j1c:j2c, i1c:i2c]

  for n in range(NMODE):
    gd.prim[n][shc] = f[var_names[n]][sht]
  IF MFIELD:
    for n in range(3):
      gd.bfld[n][shc] = f[var_names[NMODE+n]][sht]

  integr.time = f.attrs['time'] #17.00013
  integr.step = f.attrs['step'] #96443


  # dsets={}
  #
  # IF PIC:
  #   for n in range(3):
  #     dsets[var_names[n]]   = (gd.efld[n, k1c:k2c, j1c:j2c, i1c:i2c], 'r')
  #     dsets[var_names[n+3]] = (gd.bfld[n, k1c:k2c, j1c:j2c, i1c:i2c], 'r')
  #     dsets[var_names[n+6]] = (gd.curr[n, k1c:k2c, j1c:j2c, i1c:i2c], 'r')
  # ELSE:
  #   for n in range(NMODE):
  #     dsets[var_names[n]] = (gd.prim[n, k1c:k2c, j1c:j2c, i1c:i2c], 'r')
  #   IF MFIELD:
  #     for n in range(3):
  #       dsets[var_names[NMODE+n]] = (gd.bfld[n, k1c:k2c, j1c:j2c, i1c:i2c], 'r')
  #
  # # Create dictionary of attributes.
  #
  # attrs = {}
  # attrs['t'] = (<real>0., 'r')
  # attrs['step'] = (<long>0, 'l')
  #
  # h5restart(file_name, dsets,attrs, 3, block_dim, offset)
  #
  # integr.time = attrs['t'][0] #17.00013
  # integr.step = attrs['step'][0] #96443



#---------------------------------------------------------------------------

IF MHDPIC:

  cdef void set_restart_particles(BnzIO out, BnzGrid grid, BnzIntegr integr):

    cdef:
      GridCoord *gc = grid.coord
      BnzParticles prts = grid.prts
      ParticleProp *pp = prts.prop
      ParticleData *pd = prts.data
      int rank=0

    IF MPI: rank = mpi.COMM_WORLD.Get_rank()

    # Open HDF5 file.

    driver=None
    driver_args={}
    IF MPI:
      driver='mpio'
      driver_args['comm']=mpi.COMM_WORLD

    fname = os.path.join(out.usr_dir, 'rst', 'rst_prt.hdf5')
    f = h5py.File(fname, 'r', driver=driver, **driver_args)

    # set local numbers of particles
    pp.Np = f.attrs['Np_proc'][rank]
    # for n in range(pp.Ns):
    #   pp.spc_props[n].Np = f.attrs['Nps'][n]

    cdef long np_tot, nmin,nmax, n,n1

    # total number of particles on all processors
    np_tot = f.attrs['Np']

    # Read particles.

    nmin = f['Np_proc'][:rank].sum()
    nmax = f['Np_proc'][:rank+1].sum()

    for n in range(nmin, nmax):

      n1 = n-nmin
      pd.x[n1] = f['x'][n]
      pd.y[n1] = f['y'][n]
      pd.z[n1] = f['z'][n]
      pd.u[n1] = f['u'][n]
      pd.v[n1] = f['v'][n]
      pd.w[n1] = f['w'][n]
      pd.g[n1] = f['g'][n]
      pd.m[n1] = f['m'][n]
      pd.spc[n1] = f['spc'][n]
      pd.id[n1] = f['id'][n]

    f.close()
