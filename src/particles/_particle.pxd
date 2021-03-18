# -*- coding: utf-8 -*-

from bnz.defs cimport *
from bnz.coordinates.coord cimport GridCoord

cdef extern from "particle_c.h" nogil:

  # Properties of a particle specie.

  ctypedef struct SpcProp:
    real qm     # charge-to-mass ratio
    long Np     # number of particles

  # General particle properties.

  ctypedef struct PrtProp:

    int ppc            # number of particles per cell
    int Nprop          # number of particle properties
    long Npmax         # length of particle array
    long Np            # number of active particles of all species
    int Ns             # number of species
    SpcProp *spc_props # properties of particle species

    # IF MHDPIC:
    #   real c         # effective speed of light
    #   real q_mc      # charge-to-mass ratio of CRs relative to thermal ions
    #   real rho_cr    # CR density

  # Structure containing arrays of particle properties.

  ctypedef struct PrtData:

    # coordinates
    real *x
    real *y
    real *z

    # four-velocities
    real *u
    real *v
    real *w

    real *g    # relativistic gamma

    real *m    # mass
    int *spc   # specie
    long *id   # particle ID


# from bnz.particles.prt_bc cimport PrtBc
cdef class PrtBc

# Particle class.

cdef class BnzParticles:

  cdef:
    PrtProp *prop
    PrtData *data
    PrtBc bc

  cdef str usr_dir
