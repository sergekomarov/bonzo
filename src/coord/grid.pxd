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
    int ng              # number of ghost cells
    ints i1,i2          # min and max indices of active cells on local grid
    ints j1,j2
    ints k1,k2

    # Coordinates.

    real lmin[3]        # coordinates of left border of global domain
    real lmax[3]        # coordinates of right border of global domain
    #real l1[3]         # coordinates of left border of local grid
    #real l2[3]         # coordinates of right border of local grid

    IF not PIC:

      real **lf           # coordinates of cell faces
      real **lv           # cell baricenters

      real ***dv          # cell volumes
      real ****da         # cell areas in all directions
      real ****ds         # lengths of cell edges

      # cell spacings
      real **dlf          # between cell faces
      real **dlv          # between cell centers
      real **dlf_inv
      real **dlv_inv      # inverse spacings

      CoorgGeom coord_geom       # coordinate geometry
      CoordScale coord_scale[3]  # scale of the coordinate axes

      # MPI block IDs
      ints rank             # MPI rank of the grid
      ints pos[3]           # 3D index of the grid on the current processor

      # Auxilary coefficients for non-cartesian coordinates.
      real *rinv_mean
      real *src_coeff1
      real **src_coeff2
      real **hp_ratio
      real **hm_ratio

    IF MPI:

      ints size[3]         # number of MPI blocks (grids) in x,y,z directions
      ints size_tot        # total number of blocks

      ints ***ranks        # 3D array of grid ranks

      ints nbr_ranks[3][2] # ranks of neighboring grids
      # nbr_ids[axis,L(0)/R(1)]


# ===========================================================================


#=====================================================================

# All grid data.

cdef class GridData:

  IF not PIC:

    cdef:

      real4d U            # cell-centered conserved variables
      real4d W            # cell-centered primitive variables

      real4d Fx, Fy, Fz   # Godunov fluxes
      real4d Us           # predictor-step arrays of cell-centered conserved variables
      real4d Uss

      real3d Phi          # static gravitational potential
      real4d DrivF        # driving force
      CosTab costab       # cosine lookup table for the turbulence module

      real4d B            # face-centered magnetic field
      real4d Ec, E        # cell-centered and edge-centered electric field
      real4d Bs           # predictor-step arrays of face-centered magnetic field
      real4d Bss

      real4d Binit        # initial magnetic field

      real3d nuii_eff     # effective ion collision rate

  IF MHDPIC:
    cdef real4d CoupF     # particle feedback array

  IF PIC:
    cdef:
      real4d E            # edge-centered electric field
      real4d B            # face-centered magnetic field
      real4d J            # edge-centered currents



#=======================================================================

# Scratch arrays used by the integrator (mainly by diffusion routines).

cdef class GridScratch:

  cdef:

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

  IF MHDPIC:
    cdef real4d CoupF_tmp   # temporary copy of the particle feedback array
  IF PIC:
    cdef real4d Jtmp        # temporary copy of the particle current array


# =========================================================================

cdef class BnzSim

# grid BC function pointer
ctypedef void (*BCFuncGrid)(BnzSim, ints[::1])

# Boundary condition class.

cdef class GridBC:

  # BC flags
  # 0 - periodic; 1 - outflow; 2 - reflective / conductive; 3 - user-defined
  cdef int bc_flags[3][2]

  # array of grid BC function pointers
  cdef BCFuncGrid bc_grid_funcs[3][2]

  IF PIC or MHDPIC:
    # exchange BC for currents / particle feedback
    cdef BCFuncGrid bc_exch_funcs[3][2]

  IF MPI:
    cdef:
      real2d sendbuf, recvbuf    # send/receive buffers for boundary conditions
      ints recvbuf_size, sendbuf_size   # buffer sizes


#====================================================================

# Grid class, contains parameters and data of local grid.

cdef class BnzGrid:

  cdef:
    GridCoord coord      # grid coordinates
    GridData data        # grid data
    GridScratch scr      # scratch arrays
    GridBC bc            # boundary conditions

  IF PIC or MHDPIC:
    cdef BnzParticles prts

  cdef bytes usr_dir     # user directory, contains config file

  cdef:
    init_data(self)
    init_bc_buffer(self)
  IF MPI:
    cdef domain_decomp(self)
