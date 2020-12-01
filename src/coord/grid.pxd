# -*- coding: utf-8 -*-

from bnz.defs_cy cimport *


cdef extern from "coord.h" nogil:

  ctypedef enum CoordScale:
    CS_UNI,
    CS_LOG,
    CS_USR

  ctypedef enum CoordGeom:
    CG_CAR,
    CG_CYL,
    CG_SPH

  ctypedef struct GridCoord:

    # Cells and indices.

    ints Nact[3]        # numbers of active cells cells on local grid
    ints Ntot[3]        # numbers of cells including ghosts
    ints Nact_glob[3]   # active cells in full domain
    ints Ntot_glob[3]   # all cells in full domain
    ints ng              # number of ghost cells
    ints i1,i2          # min and max indices of active cells on local grid
    ints j1,j2
    ints k1,k2

    # Coordinates.

    real lmin[3]        # coordinates of left border of global domain
    real lmax[3]        # coordinates of right border of global domain

    real **lf           # coordinates of cell faces
    real **lv           # cell baricenters

    # cell spacings
    real **dlf          # between cell faces
    real **dlv          # between cell centers
    real **dlf_inv
    real **dlv_inv      # inverse spacings

    CoorgGeom geom       # coordinate geometry
    CoordScale scale[3]  # scale of the coordinate axes

    # auxilary coefficients to calculate cell volumes, areas, and lengths
    real *rinv_mean
    real *d2r
    real *d3r
    real *sin_thf
    real *sin_thc
    real *dcos_thf

    # auxilary coefficients to calculate geometric source terms
    real *src_coeff1
    real **src_coeff2

    # coefficients used in parabolic reconstruction (Mignone paper)
    real **hp_ratio
    real **hm_ratio

    # interpolation coefficients
    real **cm
    real **cp

    # MPI block IDs
    ints rank            # MPI rank of the grid
    ints pos[3]          # 3D index of the grid on the current processor
    ints size[3]         # number of MPI blocks (grids) in x,y,z directions
    ints size_tot        # total number of blocks

    ints ***ranks        # 3D array of grid ranks
    ints nbr_ranks[3][2] # ranks of neighboring grids
    # nbr_ids[axis,L(0)/R(1)]



#=====================================================================

# All grid data.

cdef class GridData:

  cdef:

    real4d cons         # cell-centered conserved variables
    real4d prim         # cell-centered primitive variables

    real4d bf           # face-centered magnetic field
    # real4d ec           # cell-centered electric field
    # real4d ee           # edge-centered electric field

    # real4d flux_x, flux_y, flux_z   # Godunov fluxes

    # real4d cons_s       # predictor-step arrays of cell-centered conserved variables
    # real4d cons_ss
    #
    # real4d bf_s         # predictor-step arrays of face-centered magnetic field
    # real4d bf_ss

    real4d bf_init        # initial magnetic field
    # real3d phi          # static gravitational potential
    # real4d fdriv        # driving force
    # real3d nuii_eff     # effective ion collision rate

    cdef real4d fcoup     # particle feedback array



#=======================================================================

# Scratch arrays used by the integrator (mainly by diffusion routines).

cdef class GridScratch:

  cdef:
    # scratch arrays used by reconstruction routines
    real4d scr_reconstr

    # divergence of velocity field
    real3d div

    # electron and ion temperatures
    real3d Te,Ti,Tipl,Tipd

    # magnetic field strength (to calculate magnetic gradients)
    real3d Babs

    # super-time-stepping arrays (thermal conduction)
    real3d T0, Tm1, MT0, MT
    real3d Tipd0, Tipdm1, MTipd0, MTipd

    # STS arrays of velocities / magnetic field (viscosity / resistivity)
    real4d V0,Vm1, MV0, MV

    # L/R interface values of ion temperature used for advective terms
    # real1d TiL,TiR, TipdL,TipdR

    # thermal conductivities
    real3d kappa_pl, kappa_pd, kappa_mag

    # diffusive fluxes
    real3d Fx_diff1, Fy_diff1, Fz_diff1
    real3d Fx_diff2, Fy_diff2, Fz_diff2

  # cdef real4d fcoup_tmp   # temporary copy of the particle feedback array



# =========================================================================

cdef class GridBc

# Grid class, contains parameters and data of local grid.

cdef class BnzGrid:

  cdef:
    GridCoord coord      # grid coordinates
    GridData data        # grid data
    GridScratch scratch  # scratch arrays
    GridBc bc            # boundary conditions

  cdef BnzParticles prts # particles

  cdef bytes usr_dir     # user directory, contains config file

  cdef void init(self)
