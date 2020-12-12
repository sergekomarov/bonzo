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

    int Nact[3]        # numbers of active cells cells on local grid
    int Ntot[3]        # numbers of cells including ghosts
    int Nact_glob[3]   # active cells in full domain
    int Ntot_glob[3]   # all cells in full domain
    int ng              # number of ghost cells
    int i1,i2          # min and max indices of active cells on local grid
    int j1,j2
    int k1,k2

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

    # scale factors to use e.g. to calculate gradients
    real *syxf
    real *syxv
    real *szxf
    real *szxv
    real *szyf
    real *szyv

    # temporary arrays to calculate Laplacian
    real **lapl_tmp_xy1
    real **lapl_tmp_xy2

    # MPI block IDs
    int rank            # MPI rank of the grid
    int pos[3]          # 3D index of the grid on the current processor
    int size[3]         # number of MPI blocks (grids) in x,y,z directions
    int size_tot        # total number of blocks
    int ***ranks        # 3D array of grid ranks
    int nbr_ranks[3][2] # ranks of neighboring grids
    # nbr_ids[axis,L(0)/R(1)]


from bnz.bc.grid_bc cimport GridBc

cdef void init_coord(GridCoord*, GridBc, bytes)
cdef void free_coord_data(GridCoord*)

cdef void add_geom_src_terms(real4d,real4d, real4d,real4d,real4d,
                             GridCoord*, int*, real) nogil

# cdef void add_laplacian(GridCoord*, real4d, int) nogil

cdef real get_edge_len_x(GridCoord*, int,int,int) nogil
cdef real get_edge_len_y(GridCoord*, int,int,int) nogil
cdef real get_edge_len_z(GridCoord*, int,int,int) nogil

cdef real get_centr_len_x(GridCoord*, int,int,int) nogil
cdef real get_centr_len_x(GridCoord*, int,int,int) nogil
cdef real get_centr_len_x(GridCoord*, int,int,int) nogil

cdef real get_face_area_x(GridCoord*, int,int,int) nogil
cdef real get_face_area_y(GridCoord*, int,int,int) nogil
cdef real get_face_area_z(GridCoord*, int,int,int) nogil

cdef real get_cell_vol(GridCoord*, int,int,int) nogil
