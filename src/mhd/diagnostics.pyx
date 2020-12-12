# -*- coding: utf-8 -*-

IF MPI:
  from mpi4py import MPI as mpi
  from mpi4py cimport MPI as mpi

import numpy as np
cimport numpy as np
from cython.parallel import prange, parallel, threadid

from libc.stdlib cimport malloc, calloc, free
from libc.stdio cimport printf

from bnz.utils cimport print_root
from bnz.coord.coord_cy cimport get_cell_vol


cdef void print_nrg(BnzGrid grid, BnzIntegr integr):

  # in primitive variables

  cdef:
    int i,j,k, n
    int id
    real dv
    real ek,em,et,ep
    real ekm=0., etm=0., emm=0., epm=0.

  cdef:
    GridCoord *gc = grid.coord
    real4d w = grid.data.prim
  IF MHDPIC:
    cdef BnzParticles prts = grid.prts

  cdef:
    real vol = (gc.lmax[0]-gc.lmin[0]) * (gc.lmax[1]-gc.lmin[1]) * (gc.lmax[2]-gc.lmin[2])
    real gamm1i = 1./(integr.gam-1)

  cdef:
    real1d ek_loc = np.zeros(OMP_NT)
    real1d em_loc = np.zeros(OMP_NT)
    real1d et_loc = np.zeros(OMP_NT)
    IF MHDPIC:
      real1d ep_loc = np.zeros(OMP_NT)

  IF MPI:
    cdef int varlen = 2
    IF MFIELD: varlen += 1
    IF MHDPIC: varlen += 1
    cdef:
      double[::1] var     = np.empty(varlen, dtype='f8')
      double[::1] var_sum = np.empty(varlen, dtype='f8')


  with nogil, parallel(num_threads=OMP_NT):
    id = threadid()

    for k in prange(gc.k1, gc.k2+1, schedule='dynamic'):
      for j in range(gc.j1, gc.j2+1):
        for i in range(gc.i1, gc.i2+1):

          dv = get_cell_vol(gc, i,j,k)

          ek = 0.5*(SQR(w[VX,k,j,i]) + SQR(w[VY,k,j,i]) + SQR(w[VZ,k,j,i])) * w[RHO,k,j,i]
          IF MFIELD: em = 0.5*(SQR(w[BX,k,j,i]) + SQR(w[BY,k,j,i]) + SQR(w[BZ,k,j,i]))

          et = gamm1i * w[PR,k,j,i]
          IF TWOTEMP: et = et + gamm1i * e[PE,k,j,i]

          ek_loc[id] = ek_loc[id] + ek*dv
          et_loc[id] = et_loc[id] + et*dv
          IF MFIELD: em_loc[id] = em_loc[id] + em*dv

  IF MHDPIC:
    with nogil, parallel(num_threads=OMP_NT):
      id = threadid()
      for n in prange(prts.prop.Np, schedule='dynamic'):
        ep_loc[id] = ep_loc[id] + (prts.data.g[n]-1.)*dv


  for i in range(OMP_NT):

    ekm += ek_loc[i]
    etm += et_loc[i]
    IF MFIELD: emm += em_loc[i]
    IF MHDPIC: epm += ep_loc[i]

  ekm /= vol
  etm /= vol
  IF MFIELD: emm /= vol
  IF MHDPIC: epm *= integr.rho_cr * integr.sol**2 / (prts.prop.ppc * vol)

  IF MPI:

    var[0], var[1] = ekm, etm
    IF MFIELD: var[2] = emm
    IF MHDPIC: var[3] = epm
    mpi.COMM_WORLD.Allreduce(var, var_sum, op=mpi.SUM)
    ekm, etm = var_sum[0], var_sum[1]
    IF MFIELD: emm = var_sum[2]
    IF MHDPIC: epm = var_sum[3]

  print_root("\n----- mean energy densities -------\n")
  print_root("Ek = %f\n", ekm)
  print_root("Et = %f\n", etm)
  IF MFIELD:
    print_root("Em = %f\n", emm)
  IF MHDPIC:
    print_root("Ep = %f\n", epm)
    print_root("Etot = %f\n", ekm+etm+emm+epm)
  ELSE:
    print_root("Etot = %f\n", ekm+etm+emm)


  print_root("-----------------------------------\n")

  return
