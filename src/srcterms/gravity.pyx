# -*- coding: utf-8 -*-

import numpy as np
cimport numpy as np

from cython.parallel import parallel, prange, threadid

from libc.stdlib cimport rand, RAND_MAX, srand
from libc.stdlib cimport malloc, calloc, free

from utils cimport cost, rand01, print_root

IF SPREC:
  np_real = np.float32
ELSE:
  np_real = np.float64

cdef class BnzGravity:

  def __cinit__(self, GravPotFunc grav_pot_func, GridCoord *gc, bytes usr_dir):

    cdef:
      GridCoord *gc = grid.coord
      int k,j,i
      real x=0,y=0,z=0

    self.g0 = read_param("physics", "g0", 'f',usr_dir)

    if self.g0!=0.:

      # check if the user has not provided a gravitational potential function
      if grav_pot_func!=NULL:
        self.grav_pot_func=grav_pot_func
      else:
        self.grav_pot_func=&grav_pot_func_def

      self.grav_pot = np.zeros((gp.Ntot[2],gp.Ntot[1],gp.Ntot[0]), dtype=np_real)

      for k in range(gc.Ntot[2]):
        for j in range(gc.Ntot[1]):
          for i in range(gc.Ntot[0]):
            self.grav_pot[k,j,i] = self.grav_pot_func(
                                  gc.lv[0][i], gc.lv[1][j], gc.lv[2][k], self.g0)


  cdef void apply_gravity(self, real4d u1, real4d u0,
                  real4d fx0, real4d fy0, real4d fz0,
                  GridCoord *gc, int *lims, real dt) nogil:

    cdef:
      int i,j,k
      real x,y,z, phil,phic,phir
      real3d phi = self.grav_pot

    for k in range(lims[4],lims[5]+1):
      for j in range(lims[2],lims[3]+1):
        for i in range(lims[0],lims[1]+1):

          # COORDINATES

          phic = phi[k,j,i]
          phil = 0.5*(phi[k,j,i-1] + phi[k,j,i])
          phir = 0.5*(phi[k,j,i] + phi[k,j,i+1])

          u1[MX,k,j,i] = u1[MX,k,j,i] + dtdx * u0[RHO,k,j,i] * (phil - phir)
          u1[EN,k,j,i] = u1[EN,k,j,i] - dtdx * (
              fx0[RHO,k,j,i+1] * (phir-phic) - fx0[RHO,k,j,i] * (phil-phic))

          IF D2D:

            phil = 0.5*(phi[k,j-1,i] + phi[k,j,i])
            phir = 0.5*(phi[k,j,i] + phi[k,j+1,i])

            u1[MY,k,j,i] = u1[MY,k,j,i] + dtdy * u0[RHO,k,j,i] * (phil - phir)
            u1[EN,k,j,i] = u1[EN,k,j,i] - dtdy * (
                fy0[RHO,k,j+1,i] * (phir-phic) - fy0[RHO,k,j,i] * (phil-phic))

          IF D3D:

            phil = 0.5*(phi[k-1,j,i] + phi[k,j,i])
            phir = 0.5*(phi[k,j,i] + phi[k+1,j,i])

            u1[MZ,k,j,i] = u1[MZ,k,j,i] + dtdz * u0[RHO,k,j,i] * (phil - phir)
            u1[EN,k,j,i] = u1[EN,k,j,i] - dtdz * (
                fz0[RHO,k+1,j,i] * (phir-phic) - fz0[RHO,k,j,i] * (phil-phic))


cdef inline real grav_pot_func_def(real x, real y, real z, real g0) nogil:

  return g0*y
