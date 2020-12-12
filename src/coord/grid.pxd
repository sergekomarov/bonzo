# -*- coding: utf-8 -*-

from bnz.defs_cy cimport *

cdef class GridData:

  cdef:

    real4d cons         # cell-centered conserved variables
    real4d prim         # cell-centered primitive variables

    real4d bf           # face-centered magnetic field
    real4d bf_init      # initial magnetic field
    # real3d phi        # static gravitational potential
    # real4d fdriv      # driving force
    # real3d nuii_eff   # effective ion collision rate

    real4d fcoup        # particle feedback array


# circular import: can use forward declarations instead?
from coord_cy cimport GridCoord
from bnz.bc.grid_bc cimport GridBc
from bnz.mhdpic.particle cimport BnzParticles

cdef class BnzGrid:

  cdef:
    GridCoord *coord       # grid coordinates
    GridData data          # grid data
    # GridScratch scratch  # scratch arrays
    GridBc bc              # boundary conditions
    BnzParticles prts      # particles

  cdef bytes usr_dir       # user directory, contains config file



# Scratch arrays used by the integrator (mainly by diffusion routines).

# cdef class GridScratch:
#
#   cdef:
#     # scratch arrays used by reconstruction routines
#     real4d scr_reconstr
#
#     # divergence of velocity field
#     real3d div
#
#     # electron and ion temperatures
#     real3d Te,Ti,Tipl,Tipd
#
#     # magnetic field strength (to calculate magnetic gradients)
#     real3d Babs
#
#     # super-time-stepping arrays (thermal conduction)
#     real3d T0, Tm1, MT0, MT
#     real3d Tipd0, Tipdm1, MTipd0, MTipd
#
#     # STS arrays of velocities / magnetic field (viscosity / resistivity)
#     real4d V0,Vm1, MV0, MV
#
#     # L/R interface values of ion temperature used for advective terms
#     # real1d TiL,TiR, TipdL,TipdR
#
#     # thermal conductivities
#     real3d kappa_pl, kappa_pd, kappa_mag
#
#     # diffusive fluxes
#     real3d Fx_diff1, Fy_diff1, Fz_diff1
#     real3d Fx_diff2, Fy_diff2, Fz_diff2
#
#   # cdef real4d fcoup_tmp   # temporary copy of the particle feedback array
#
#
