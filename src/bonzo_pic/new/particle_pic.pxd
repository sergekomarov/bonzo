# -*- coding: utf-8 -*-

from bnz.defs_cy cimport *

cdef extern from "particle.h" nogil:

  # Properties of a particle specie.

  ctypedef struct SpcProp:
    real qm     # charge-to-mass ratio
    ints Np     # number of particles

cdef extern from "particle.h" nogil:

  # General particle properties.

  ctypedef struct ParticleProp:

    ints ppc           # number of particles per cell
    ints Nprop         # number of particle properties
    ints Npmax         # length of particle array
    ints Np            # number of active particles of all species
    ints Ns            # number of species
    SpcProp *spc_props # properties of different species

    IF MHDPIC:
      real c           # effective speed of light
      real q_mc        # charge-to-mass ratio of CRs relative to thermal ions
      real rho_cr      # CR density
    IF PIC:
      real c           # speed of light <-> Courant number
      real me          # electron mass
      real mime        # ion-electron mass ratio
      # real c_ompe    # electron skin depth


  # Structure containing arrays of particle properties.

  ctypedef struct ParticleData:

    # coordinates
    real *x
    real *y
    real *z

    # four-velocities
    real *u
    real *v
    real *w

    real *g     # relativistic gamma

    real *m     # mass
    ints *spc   # specie
    ints *id    # particle ID


# Particle boundary conditions.

cdef class BnzSim

# particle BC function pointer
ctypedef void (*BCFuncPrt)(BnzSim)

cdef class ParticleBC:

  # BC flags
  cdef int bc_flags[3][2]

  # BC function pointers
  cdef BCFuncPrt bc_prt_funcs[3][2]

  # BC buffers
  IF MPI:
    cdef:
      real2d sendbuf, recvbuf
      ints recvbuf_size, sendbuf_size


# Particle class.

cdef class BnzParticles:

  cdef:
    ParticleProp *prop
    ParticleData *data
    ParticleBC bc

  cdef:
    void init(self, *ints)
    void init_data(self, *ints)
    void init_bc_buffer(self, *ints)
