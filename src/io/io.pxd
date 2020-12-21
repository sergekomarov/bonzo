# -*- coding: utf-8 -*-

from bnz.defs cimport *
from bnz.coordinates.coord cimport GridCoord
from bnz.coordinates.grid cimport GridData, BnzGrid
from bnz.particles.particle cimport BnzParticles
from bnz.integrate.integrator cimport BnzIntegr

# primitive or conservative variables in grid output
cdef enum VarType:
  VAR_PRIM
  VAR_CONS
  VAR_PIC

# Function pointer to calculate user history variables.
ctypedef real(*HstFunc)(BnzGrid,BnzIntegr)

IF MHDPIC:
  # Pointer to user-defined particle selection function.
  ctypedef int(*PrtSelFunc)(ParticleData*,long)


cdef class BnzIO:

  cdef:

    str usr_dir

    real hst_dt     # write history every hst_dt
    real grid_dt    # the grid every grid_dt
    real slc_dt     # a slice every slc_st
    real prt_dt     # particles every prt_dt
    real rst_dt     # restart files every rst_dt

    int use_npy     # output .npy arrays instaad of HDF5 (only without MPI)

    # history
    list hst_funcs_u    # user history variable functions
    list hst_names_u    # names of user history variables
    int nhst        # total number (user+default) of active history variables

    # grids
    VarType var_type     # primitive/conservative variables in grid output

    # grid slice
    int slc_axis         # axis perpendicular to the slice
    int slc_loc          # cell index of the slice along the slc_axis

    int write_ghost      # 1 to write ghost cells

    int restart          # 1 to restart simulation

    # particles
    PrtSelFunc prt_sel_func  # particle selection function
    int prt_stride           # write every prt_stride particle


  cdef void write_output(self, BnzGrid,BnzIntegr)
  cdef void write_restart(self, BnzGrid,BnzIntegr)
  cdef void set_restart(self, BnzGrid,BnzIntegr)
