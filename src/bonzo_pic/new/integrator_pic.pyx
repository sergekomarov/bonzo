# -*- coding: utf-8 -*-

import numpy as np
cimport numpy as np

from read_config import read_param


# =======================================

# Integration-related data and functions.

cdef class BnzIntegrator:

  def __cinit__(self, bytes usr_dir):

    # Courant number
    self.cour = read_param("computation", "cour", 'f', usr_dir)

    # Number of passes of current filter.
    self.Nfilt = read_param("computation", "Nfilt", 'i',usr_dir)



# =================================================================

cdef void init_data(self):

  cdef:
    GridCoord gc = self.coord
    GridData gd = self.data
    ints i,j,k
    ints Nx=gc.Ntot[0], Ny=gc.Ntot[1], Nz=gc.Ntot[2]

  # allocate arrays used by MHD integrator

  sh_u = (NWAVES,Nz,Ny,Nx)
  sh_3 = (3,Nz,Ny,Nx)

  # set pointers to user-defined physics functions (e.g. gravitational potential)
  problem.set_phys_ptrs_user(phys)


  def __dealloc__(self):

    return
