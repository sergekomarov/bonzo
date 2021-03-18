# -*- coding: utf-8 -*-

import numpy as np
cimport numpy as np

IF SPREC:
  np_real = np.float32
ELSE:
  np_real = np.float64

cdef class BnzGravity:

  def __cinit__(self, GridCoord *gc, str usr_dir):

    self.g0[0] = read_param("physics", "gx", 'f',usr_dir)
    self.g0[1] = read_param("physics", "gy", 'f',usr_dir)
    self.g0[2] = read_param("physics", "gz", 'f',usr_dir)
    self.grav_pot_func = NULL


  cdef void post_user_init(self, GridCoord *gc):

    cdef int k,j,i

    if self.g0[0]!=0. or self.g0[1]!=0. or self.g0[2]!=0.:

      # if user potential has not been set, use constant grav. acceleration
      if self.grav_pot_func==NULL:
        self.const_g=1

      else:

        self.grav_pot = np.zeros((gc.Ntot[2],gc.Ntot[1],gc.Ntot[0]), dtype=np_real)

        for k in range(gc.Ntot[2]):
          for j in range(gc.Ntot[1]):
            for i in range(gc.Ntot[0]):
              self.grav_pot[k,j,i] = self.grav_pot_func(
                                    gc.lv[0][i], gc.lv[1][j], gc.lv[2][k], self.g0)


  cdef void apply_gravity(self, real4d u1, real4d w0,
                  real4d fx0, real4d fy0, real4d fz0,
                  GridCoord *gc, int *lims, real dt) nogil:

    if self.const_g:
      apply_gravity_const(self, u1,w0, gc,lims, dt)
    else:
      apply_gravity_phi(self, u1,w0, fx0,fy0,fz0, gc,lims, dt)


#--------------------------------------------------------------------------

cdef void apply_gravity_const(BnzGravity grav, real4d u1, real4d w0,
                              GridCoord *gc, int *lims, real dt) nogil:

  cdef int i,j,k

  for k in range(lims[4],lims[5]+1):
    for j in range(lims[2],lims[3]+1):
      for i in range(lims[0],lims[1]+1):

        for n in range(3):
          u1[MX+n,k,j,i] += u0[RHO,k,j,i] * grav.g0[n] * dt

        u1[EN,k,j,i] += dt * (u0[MX,k,j,i] * grav.g0[0] +
                              u0[MY,k,j,i] * grav.g0[1] +
                              u0[MZ,k,j,i] * grav.g0[2])


cdef void apply_gravity_phi(BnzGravity grav, real4d u1, real4d w0,
                            real4d fx0, real4d fy0, real4d fz0,
                            GridCoord *gc, int *lims, real dt) nogil:

  cdef:
    int i,j,k
    real phil,phic,phir, dtdx,dtdy,dtdz
    real3d phi = grav.grav_pot

  for k in range(lims[4],lims[5]+1):
    for j in range(lims[2],lims[3]+1):
      for i in range(lims[0],lims[1]+1):

        dtdx = dt * gc.dlf_inv[0][i]

        phic = phi[k,j,i]
        phil = 0.5*(phi[k,j,i-1] + phi[k,j,i])
        phir = 0.5*(phi[k,j,i] + phi[k,j,i+1])

        u1[MX,k,j,i] = u1[MX,k,j,i] + dtdx * w0[RHO,k,j,i] * (phil - phir)
        u1[EN,k,j,i] = u1[EN,k,j,i] - dtdx * (
            fx0[RHO,k,j,i+1] * (phir-phic) - fx0[RHO,k,j,i] * (phil-phic))

        IF D2D:

          dtdy = dt * gc.dlf_inv[1][j] * gc.syxv[i]

          phil = 0.5*(phi[k,j-1,i] + phi[k,j,i])
          phir = 0.5*(phi[k,j,i] + phi[k,j+1,i])

          u1[MY,k,j,i] = u1[MY,k,j,i] + dtdy * w0[RHO,k,j,i] * (phil - phir)
          u1[EN,k,j,i] = u1[EN,k,j,i] - dtdy * (
              fy0[RHO,k,j+1,i] * (phir-phic) - fy0[RHO,k,j,i] * (phil-phic))

        IF D3D:

          dtdz = dt * gc.dlf_inv[2][k] * gc.szxv[i] * gc.szyv[j]

          phil = 0.5*(phi[k-1,j,i] + phi[k,j,i])
          phir = 0.5*(phi[k,j,i] + phi[k+1,j,i])

          u1[MZ,k,j,i] = u1[MZ,k,j,i] + dtdz * w0[RHO,k,j,i] * (phil - phir)
          u1[EN,k,j,i] = u1[EN,k,j,i] - dtdz * (
              fz0[RHO,k+1,j,i] * (phir-phic) - fz0[RHO,k,j,i] * (phil-phic))
