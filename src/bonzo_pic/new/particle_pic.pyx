# -*- coding: utf-8 -*-

import numpy as np
cimport numpy as np

from libc.stdlib cimport free, calloc
from utils cimport mini,maxi
from read_config import read_param

IF SPREC:
  np_real = np.float32
ELSE:
  np_real = np.float64


cdef class BnzParticles:

  def __cinit__(self, bytes usr_dir):

    self.bc = ParticleBC()
    # self.prop = ParticleProp()  # structure
    # self.data = ParticleData()  # structure

    cdef ParticleProp *pp = &(self.prop)

    # number of particles per cell
    pp.ppc = <ints>read_param("computation","ppc",'i',usr_dir)

    IF D2D and D3D:
      pp.ppc = (<ints>(pp.ppc**(1./3)))**3
    ELIF D2D:
      pp.ppc = (<ints>sqrt(pp.ppc))**2

    # ppc needs to be even to have equal numbers of electrons and ions
    if pp.ppc % 2 != 0: pp.ppc += 1

    pp.c = read_param("computation", "cour", 'f',usr_dir)
    pp.mime = read_param("physics",  "mime", 'f',usr_dir)
    cdef real c_ompe = read_param("physics", "c_ompe", 'f',usr_dir)
    pp.me = pp.c / (c_ompe * pp.ppc)

    # Set properties of particle species.

    pp.Ns=2    # 0: ions; 1: electrons by default
    pp.spc_props = <SpcProp*>calloc(pp.Ns, sizeof(SpcProp))

    pp.spc_props[0].qm =  1./pp.mime
    pp.spc_props[1].qm = -1.
    pp.spc_props[0].Np=  <ints>(0.5*pp.Np)
    pp.spc_props[1].Np=  <ints>(0.5*pp.Np)
    pp.Nprop = 10

    # Particle BC.

    cdef ints i,k

    for i in range(3):
      for k in range(2):
        self.bc.bc_prt_funcs[i][k] = NULL

    self.bc.bc_flags[0][0] = read_param("physics", "bc_x1", 'i',usr_dir)
    self.bc.bc_flags[0][1] = read_param("physics", "bc_x2", 'i',usr_dir)
    self.bc.bc_flags[1][0] = read_param("physics", "bc_y1", 'i',usr_dir)
    self.bc.bc_flags[1][1] = read_param("physics", "bc_y2", 'i',usr_dir)
    self.bc.bc_flags[2][0] = read_param("physics", "bc_z1", 'i',usr_dir)
    self.bc.bc_flags[2][1] = read_param("physics", "bc_z2", 'i',usr_dir)


  #=========================================================================

  cdef void init(self, ints *Nact):

    self.init_data(Nact)
    self.init_bc_buffer(Nact)


  #=========================================================================

  cdef void init_data(self, ints *Nact):

    # Call AFTER domain decomposition.

    cdef:
      ParticleProp *pp = &(self.prop)
      ParticleData *pd = &(self.data)

    pp.Np = pp.ppc * Nact[0] * Nact[1] * Nact[2]

    pp.Npmax = pp.Np      # only when there is no injection of particles !!!
    IF MPI: pp.Npmax = <ints>(1.3*pp.Npmax)

    pd.x = <real *>calloc(pp.Npmax, sizeof(real))
    pd.y = <real *>calloc(pp.Npmax, sizeof(real))
    pd.z = <real *>calloc(pp.Npmax, sizeof(real))

    pd.u = <real *>calloc(pp.Npmax, sizeof(real))
    pd.v = <real *>calloc(pp.Npmax, sizeof(real))
    pd.w = <real *>calloc(pp.Npmax, sizeof(real))
    pd.g = <real *>calloc(pp.Npmax, sizeof(real))

    pd.m = <real *>calloc(pp.Npmax, sizeof(real))
    pd.spc = <ints *>calloc(pp.Npmax, sizeof(ints))
    pd.id = <ints *>calloc(pp.Npmax, sizeof(ints))


  #====================================================================

  cdef void init_bc_buffer(self, ints *Nact):

    cdef ParticleProp *pp = &(self.prop)

    cdef:
      ints bufsize
      ints Nxyz =  maxi(maxi(Nact[0],Nact[1]), Nact[2])

    IF D2D and D3D:
      bufsize = 5*pp.Nprop * pp.ppc * Nxyz**2
    ELIF D2D:
      bufsize = 5*pp.Nprop * pp.ppc * Nxyz
    ELSE:
      bufsize = 5*pp.Nprop * pp.ppc

    self.bc.sendbuf = np.zeros((2,bufsize), dtype=np_real)
    self.bc.recvbuf = np.zeros((2,bufsize), dtype=np_real)
    self.bc.recvbuf_size = bufsize
    self.bc.sendbuf_size = bufsize


  # ==================================================================

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

    # Free BC pointers.

    cdef ints i,k

    for i in range(3):
      for k in range(2):
        self.bc.bc_prt_funcs[i][k] = NULL
