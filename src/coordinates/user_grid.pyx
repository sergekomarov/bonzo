# -*- coding: utf-8 -*-

from bnz.defs cimport *
from coord cimport GridCoord

import numpy as np
cimport numpy as np


cpdef set_user_coord_x(GridCoord *gc):

  cdef:
    int i
    real dx,x0

  dx = (gc.lmax[0]-gc.lmin[0]) / gc.Nact_glob[0]
  x0 = gc.pos[0]*gc.Nact[0]*dx

  for i in range(gc.Ntot[0]+1):

    gc.lf[0][i] = x0 + (i-gc.i1)*dx


cpdef set_user_coord_y(GridCoord *gc):

  cdef:
    int j
    real dy,y0

  dy = (gc.lmax[1]-gc.lmin[1]) / gc.Nact_glob[1]
  y0 = gc.pos[1]*gc.Nact[1]*dy

  for j in range(gc.Ntot[1]+1):

    gc.lf[1][j] = y0 + (j-gc.j1)*dy


cpdef set_user_coord_z(GridCoord *gc):

  cdef:
    int k
    real dz, z0

  dz = (gc.lmax[2]-gc.lmin[2]) / gc.Nact_glob[2]
  z0 = gc.pos[2]*gc.Nact[2]*dz

  for k in range(gc.Ntot[2]+1):

    gc.lf[2][k] = z0 + (k-gc.k1)*dz
