# -*- coding: utf-8 -*-

IF MPI:
  from mpi4py import MPI as mpi
  from mpi4py cimport MPI as mpi

import numpy as np
cimport numpy as np

from cython.parallel import parallel, prange, threadid

from libc.stdlib cimport rand, srand

from utils cimport rand01, print_root

IF SPREC:
  np_real = np.float32
ELSE:
  np_real = np.float64


cdef class BnzTurbDriv:

  def __cinit__(self, GridCoord *gc, str usr_dir):

    self.f0 = read_param("physics", "f", 'f',usr_dir)
    self.tau = read_param("physics", "tau", 'f',usr_dir)
    self.nmod = read_param("physics", "nmod", 'i',usr_dir)

    if self.f0 != 0.:
      self.fdriv = np.zeros((3,gc.Ntot[2],gc.Ntot[1],gc.Ntot[0]), dtype=np_real)


  cdef void advance_driv_force(self, GridCoord *gc, int *lims, real dt):

    # number of wave modes excluding k<0 and k=0
    cdef int nmod, nmod2
    nmod  = self.nmod
    nmod2 = 2*nmod+1

    # Set mode amplitudes and phases.

    cdef real1d c1,c2
    c1 = np.empty(3*nmod2**3, dtype=np_real)
    c2 = np.empty(3*nmod2**3, dtype=np_real)

    cdef int rank=0
    IF MPI: rank = mpi.COMM_WORLD.Get_rank()

    cdef:
      int p,q,s,n, m
      real norm, w1,w2

    # srand?

    if rank==0:

      for p in range(nmod2):
        for q in range(nmod2):
          for s in range(nmod2):
            for n in range(3):

              if p==nmod or q==nmod or s==nmod:
                norm = 0.
              else:
                norm = 1./(SQR(p-nmod) + SQR(q-nmod) + SQR(s-nmod))

              w1 = rand01()
              w2 = rand01()
              if w1==0.: w1 = 1e-20

              m = n + 3*(s + nmod2*q + nmod2*nmod2*p)

              c1[m] = norm * SQRT(-2 * LOG(w1)) * COS(2*B_PI * w2)
              c2[m] = norm * SQRT(-2 * LOG(w1)) * SIN(2*B_PI * w2)

    IF MPI:
      mpi.COMM_WORLD.Bcast(c1, root=0)
      mpi.COMM_WORLD.Bcast(c2, root=0)

    cdef:
      int j,k
      real dt_tau, f1, kx0, ky0, kz0

    dt_tau = dt/self.tau
    f1 = SQRT(dt_tau)*self.f0

    kx0 = 2.*B_PI / (gc.lmax[0]-gc.lmin[0])
    ky0 = 2.*B_PI / (gc.lmax[1]-gc.lmin[1])
    kz0 = 2.*B_PI / (gc.lmax[2]-gc.lmin[2])

    with nogil, parallel(num_threads=OMP_NT):

      for k in prange(lims[4],lims[5]+1, schedule='dynamic'):
        z = gc.lv[2][k]

        for j in range(lims[2],lims[3]+1):
          y = gc.lv[1][j]

          advance_driv_force_i(&fdriv[0,k,j,0], &fdriv[1,k,j,0], &fdriv[2,k,j,0],
                               &c1[0], &c2[0],
                               &(gc.lv[0][0]), y, z,
                               kx0, ky0, kz0, dt_tau, f1, nmod)


  cdef void apply_driv_force(self, real4d u1, real4d u0, int *lims, real dt) nogil:

    cdef int n,k,j,i

    for n in range(3):
      for k in range(lims[4],lims[5]+1):
        for j in range(lims[2],lims[3]+1):
          for i in range(lims[0],lims[1]+1):

            u1[MX+n,k,j,i] = u1[MX+n,k,j,i] + dt * self.fdriv[n,k,j,i]

    for k in prange(lims[4],lims[5]+1, nogil=True, num_threads=OMP_NT, schedule='dynamic'):
      for j in range(lims[2],lims[3]+1):
        for i in range(lims[0],lims[1]+1):

          u1[EN,k,j,i] = u1[EN,k,j,i] + ( dt / u0[RHO,k,j,i]
                        * (self.fdriv[0,k,j,i] * u0[MX,k,j,i]
                         + self.fdriv[1,k,j,i] * u0[MY,k,j,i]
                         + self.fdriv[2,k,j,i] * u0[MZ,k,j,i]) )
