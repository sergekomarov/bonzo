# -*- coding: utf-8 -*-

import numpy as np
cimport numpy as np

from libc.stdlib cimport free, calloc
from bnz.io.read_config import read_param

IF SPREC:
  np_real = np.float32
ELSE:
  np_real = np.float64


cdef class BnzParticles:

  def __cinit__(self, GridCoord *gc, bytes usr_dir):

    self.usr_dir = usr_dir

    cdef:
      PrtProp *pp = self.prop
      PrtData  *pd = self.data

    # number of particles per cell
    pp.ppc = <int>read_param("computation","ppc",'i',usr_dir)

    IF D2D and D3D:
      pp.ppc = (<int>(pp.ppc**(1./3)))**3
    ELIF D2D:
      pp.ppc = (<int>sqrt(pp.ppc))**2

    # pp.c      = read_param("physics", "c",      'f', usr_dir)
    # pp.rho_cr = read_param("physics", "rho_cr", 'f', usr_dir)

    # total number of particles
    pp.Np = <long>pp.ppc * gc.Nact[0] * gc.Nact[1] * gc.Nact[2]

    # Set properties of particle species.

    # number of species
    pp.Ns=1

    pp.spc_props = <SpcProp*>calloc(pp.Ns, sizeof(SpcProp))

    # charge-to-mass ratio
    pp.spc_props[0].qm = <real>read_param("physics", "q_mc", 'f', usr_dir)
    # numbers of particles
    pp.spc_props[0].Np = pp.Np

    # number of particle properties
    pp.Nprop = 10

    # Init data structures.

    # maximum size of particle array
    pp.Npmax = pp.Np      # only when there is no injection of particles !!!
    IF MPI: pp.Npmax = <long>(1.3*pp.Npmax)

    pd.x = <real *>calloc(pp.Npmax, sizeof(real))
    pd.y = <real *>calloc(pp.Npmax, sizeof(real))
    pd.z = <real *>calloc(pp.Npmax, sizeof(real))

    pd.u = <real *>calloc(pp.Npmax, sizeof(real))
    pd.v = <real *>calloc(pp.Npmax, sizeof(real))
    pd.w = <real *>calloc(pp.Npmax, sizeof(real))
    pd.g = <real *>calloc(pp.Npmax, sizeof(real))

    pd.m = <real *>calloc(pp.Npmax, sizeof(real))
    pd.spc = <int *>calloc(pp.Npmax, sizeof(int))
    pd.id = <long *>calloc(pp.Npmax, sizeof(int))

    # Init boundary conditions.

    self.bc = PrtBc(pp, gc, usr_dir)


  def __dealloc__(self):

    # Free array of particle species.

    free(self.prop.spc_props)

    # Free particle data.

    free(self.data.x)
    free(self.data.y)
    free(self.data.z)

    free(self.data.u)
    free(self.data.v)
    free(self.data.w)
    free(self.data.g)

    free(self.data.id)
    free(self.data.spc)
    free(self.data.m)
