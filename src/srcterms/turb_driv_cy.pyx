# -*- coding: utf-8 -*-

IF MPI:
  from mpi4py import MPI as mpi
  from mpi4py cimport MPI as mpi

import numpy as np
cimport numpy as np

from cython.parallel import parallel, prange, threadid

from libc.stdlib cimport rand, srand

from utils cimport rand01, print_root, memview2carray_4d

IF SPREC:
  np_real = np.float32
ELSE:
  np_real = np.float64


cdef class BnzTurbDriv:

  def __cinit__(self, GridCoord *gc):

    self.f0 = read_param("physics", "f", 'f',usr_dir)
    self.tau = read_param("physics", "tau", 'f',usr_dir)
    self.Nmod = read_param("physics", "Nmod", 'i',usr_dir)

    if self.f0 != 0.:
      self.fdriv = np.zeros((3,gc.Ntot[2],gc.Ntot[1],gc.Ntot[0]), dtype=np_real)


  cdef void advance_driv_force(self, GridCoord *gc, int *lims, real dt):

    # CLEANUP
    cdef real ****fdriv_c = memview2carray_4d(self.fdriv, 3,gc.Ntot[2],gc.Ntot[1])

    advance_driv_force_c(fdriv_c, gc, lims, self.f0, self.tau, self.nmod, dt)

    # cdef:
    #   int i,j,k, n, p,q,s
    #   real x,y,z
    #   real th, cn0,sn0, cn,sn, cnx,cny,cnz, snx,sny,snz
    #   real ax,ay,az, gx,gy,gz, tx,ty,tz
    #   real dt_tau, f1, w1,w2
    #   int Nmod, Nmod2
    #
    #   double Lxi = 1./gp.Lglob[0]
    #   double Lyi = 1./gp.Lglob[1]
    #   double Lzi = 1./gp.Lglob[2]
    #
    # cdef real4d C1,C2
    #
    # cdef int rank=0
    # IF MPI: rank = mpi.COMM_WORLD.Get_rank()
    #
    #
    # # number of wave modes excluding k<0 and k=0
    #
    # Nmod  = phys.Nmod
    # Nmod2 = 2*Nmod+1
    #
    # # set mode amplitudes and phases
    #
    # C1 = np.empty((Nmod2,Nmod2,Nmod2,3), dtype=np_real)
    # C2 = np.empty((Nmod2,Nmod2,Nmod2,3), dtype=np_real)
    #
    # if rank==0:
    #
    #   for p in range(Nmod2):
    #     for q in range(Nmod2):
    #       for s in range(Nmod2):
    #         for n in range(3):
    #
    #           w1 = rand01()
    #           w2 = rand01()
    #           if w1==0.: w1 = 1e-20
    #
    #           C1[p,q,s,n] = sqrt(-2 * log(w1)) * cos(2*M_PI * w2)
    #           C2[p,q,s,n] = sqrt(-2 * log(w1)) * sin(2*M_PI * w2)
    #
    #   for n in range(3):
    #     C1[0,0,0,n]=0.
    #     C2[0,0,0,n]=0.
    #
    # IF MPI:
    #   mpi.COMM_WORLD.Bcast(C1, root=0)
    #   mpi.COMM_WORLD.Bcast(C2, root=0)
    #
    # dt_tau = dt/phys.tau
    # f1 = sqrt(dt_tau)*phys.f
    #
    # with nogil, parallel(num_threads=OMP_NT):
    #
    #   for k in prange(lims[4],lims[5]+1, schedule='dynamic'):
    #     x,y,z=0,0,0
    #     lind2gcrd_z(&z, k, gp)
    #
    #     for j in range(lims[2],lims[3]+1):
    #       lind2gcrd_y(&y, j, gp)
    #
    #       for i in range(lims[0],lims[1]+1):
    #         lind2gcrd_x(&x, i, gp)
    #
    #         tx,ty,tz = -Nmod*Lxi, -Nmod*Lyi, -Nmod*Lzi
    #
    #         th = 2*M_PI*(tx * x + ty * y + tz * z)
    #
    #         cn = cos(th)
    #         sn = sin(th)
    #
    #         cnx = cos(2*M_PI*x)
    #         cny = cos(2*M_PI*y)
    #         cnz = cos(2*M_PI*z)
    #
    #         snx = sin(2*M_PI*x)
    #         sny = sin(2*M_PI*y)
    #         snz = sin(2*M_PI*z)
    #
    #         gx,gy,gz=0,0,0
    #
    #         for p in range(Nmod2):
    #           tx = (p-Nmod)*Lxi
    #
    #           for q in range(Nmod2):
    #             ty = (q-Nmod)*Lyi
    #
    #             for s in range(Nmod2):
    #               tz = (s-Nmod)*Lzi
    #
    #               ax = - C1[p,q,s,0]*sn + C2[p,q,s,0]*cn
    #               ay = - C1[p,q,s,1]*sn + C2[p,q,s,1]*cn
    #               az = - C1[p,q,s,2]*sn + C2[p,q,s,2]*cn
    #
    #               gx = gx + ty * az - tz * ay
    #               gy = gy - tx * az - tz * ax
    #               gz = gz + tx * ay - ty * ax
    #
    #               cn0 = cn
    #               cn = cn0 * cnx - sn  * snx
    #               sn = sn  * cnx + cn0 * snx
    #
    #             cn0 = cn
    #             cn = cn0 * cny - sn  * sny
    #             sn = sn  * cny + cn0 * sny
    #
    #           cn0 = cn
    #           cn = cn0 * cnz - sn  * snz
    #           sn = sn  * cnz + cn0 * snz
    #
    #         DrivF[0,k,j,i] = (1-dt_tau) * DrivF[0,k,j,i] + f1 * gx
    #         DrivF[1,k,j,i] = (1-dt_tau) * DrivF[1,k,j,i] + f1 * gy
    #         DrivF[2,k,j,i] = (1-dt_tau) * DrivF[2,k,j,i] + f1 * gz


  cdef void apply_turb_driv(self, real4d u1, real4d u0, int *lims, real dt) nogil:

    cdef int n,k,j,i

    for n in range(3):
      for k in range(lims[4],lims[5]+1):
        for j in range(lims[2],lims[3]+1):
          for i in range(lims[0],lims[1]+1):

            u1[MX+n,k,j,i] = u1[MX+n,k,j,i] + dt * self.fdriv[n,k,j,i]

    for k in range(lims[4],lims[5]+1):
      for j in range(lims[2],lims[3]+1):
        for i in range(lims[0],lims[1]+1):

          u1[EN,k,j,i] = u1[EN,k,j,i] + ( dt / u0[RHO,k,j,i]
                        * (self.fdriv[0,k,j,i] * u0[MX,k,j,i]
                         + self.fdriv[1,k,j,i] * u0[MY,k,j,i]
                         + self.fdriv[2,k,j,i] * u0[MZ,k,j,i]) )
