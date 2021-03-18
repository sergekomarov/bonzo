# -*- coding: utf-8 -*-

import numpy as np
cimport numpy as np

cimport bnz.integration.vanleer as vl
IF SPREC:
  np_real = np.float32
ELSE:
  np_real = np.float64


def update_integr_params(integr_params, BnzIntegr integr):

  integr_params = {
      'time': integr.time,
      'step': integr.step,
      'dt': integr.dt,
      'tmax': integr.tmax,
      'cour': integr.cour,
      'char_proj': integr.char_proj,
      'gam': integr.gam
      }

def update_coord_params(coord_params, BnzGrid grid):

  cdef GridCoord *gc = grid.coord

  # ADD SPACINGS

  coord_params = {
    'Nact': list(gc.Nact_glob),
    'Ntot': list(gc.Ntot_glob),
    'ng': gc.ng,
    'lmin': list(gc.lmin),
    'lmax': list(gc.lmax),
    # 'lf': ptr2nparr2(gc.lf, gc.N),
    # 'lv': ptr2nparr2(gc.lv)
  }

# def ptr2nparr2(real **ptr, int ny, int nx):
#   cdef np.ndarray[real,ndim=2] arr = np.zeros((ny,nx), dtype=np_real)
#   for j in range(ny):
#     for i in range(nx)
#       arr[j,i] = ptr[j][i]
#   return arr

# def join_coord(GridCoord *gc):
#
#   for n in range(3):
#
#     mpi.concat(lfx_all, lf[0])


class Simulation:

  cdef BnzGrid grid
  cdef BnzIntegr integr
  cdef BnzIO io

  def __init__(self, usr_dir):

    self.usr_dir = usr_dir
    self.integr_params = {}
    self.coord_params = {}

    vl.init(self.grid, self.integr, self.io, usr_dir)

    update_integr_params(self.integr_params, self.integr)
    update_coord_params(self.coord_params, self.grid)

  def advance(self, tmax):
    vl.advance(self.grid, self.integr, self.io, tmax)
    update_integr_params(self.integr_params, self.integr)

  @property
  def prim(self):
    return np.asarray(grid.data.prim)

  @property
  def cons(self):
    return np.asarray(grid.data.cons)

  @property
  def bfld(self):
    return np.asarray(grid.data.bfld)
