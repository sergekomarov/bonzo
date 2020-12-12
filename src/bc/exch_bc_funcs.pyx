# -*- coding: utf-8 -*-

import numpy as np
cimport numpy as np
from cython.parallel import prange, parallel

from utils_bc cimport *

IF MPI:
  IF SPREC:
    mpi_real = mpi.FLOAT
  ELSE:
    mpi_real = mpi.DOUBLE


# ====================================================================

# Periodic exchange BC for CR deposits.

cdef void x1_exch_bc_periodic(GridData gd, GridCoord *gc,  BnzIntegr integr, int1d bvars):

  cdef:
    int i1=gc.i1, i2=gc.i2, ng=gc.ng
    int n

  for n in range(4):
    copy_add_layer_x(gd.fcoup[n,...], i2-ng+1, i1-ng, ng, gc.Ntot)


cdef void x2_exch_bc_periodic(GridData gd, GridCoord *gc,  BnzIntegr integr, int1d bvars):

  cdef:
    int i1=gc.i1, i2=gc.i2, ng=gc.ng
    int n

  for n in range(4):
    copy_add_layer_x(gd.fcoup[n,...], i1, i2+1, ng, gc.Ntot)


cdef void y1_exch_bc_periodic(GridData gd, GridCoord *gc,  BnzIntegr integr, int1d bvars):

  cdef:
    int j1=gc.j1, j2=gc.j2, ng=gc.ng
    int n

  for n in range(4):
    copy_add_layer_y(gd.fcoup[n,...], j2-ng+1, j1-ng, ng, gc.Ntot)


cdef void y2_exch_bc_periodic(GridData gd, GridCoord *gc,  BnzIntegr integr, int1d bvars):

  cdef:
    int j1=gc.j1, j2=gc.j2, ng=gc.ng
    int n

  for n in range(4):
    copy_add_layer_y(gd.fcoup[n,...], j1, j2+1, ng, gc.Ntot)


cdef void z1_exch_bc_periodic(GridData gd, GridCoord *gc,  BnzIntegr integr, int1d bvars):

  cdef:
    int k1=gc.k1, k2=gc.k2, ng=gc.ng
    int n

  for n in range(4):
    copy_add_layer_z(gd.fcoup[n,...], k2-ng+1, k1-ng, ng, gc.Ntot)


cdef void z2_exch_bc_periodic(GridData gd, GridCoord *gc,  BnzIntegr integr, int1d bvars):

  cdef:
    int k1=gc.k1, k2=gc.k2, ng=gc.ng
    int n

  for n in range(4):
    copy_add_layer_z(gd.fcoup[n,...], k1, k2+1, ng, gc.Ntot)


# ==================================================================================

# Outflow exchange BC for CR deposits.
# Treated automaticaly if particles are removed 1 ghost cell away from active domain.

cdef void x1_exch_bc_outflow(GridData gd, GridCoord *gc,  BnzIntegr integr, int1d bvars):

  cdef int n
  for n in range(4):
    prolong_x(gd.fcoup[n,...], 0, gc.i1-1, gc.ng, gc.Ntot)


cdef void x2_exch_bc_outflow(GridData gd, GridCoord *gc,  BnzIntegr integr, int1d bvars):

  cdef int n
  for n in range(4):
    prolong_x(gd.fcoup[n,...], 1, gc.i2+1, gc.ng, gc.Ntot)


cdef void y1_exch_bc_outflow(GridData gd, GridCoord *gc,  BnzIntegr integr, int1d bvars):

  cdef int n
  for n in range(4):
    prolong_y(gd.fcoup[n,...], 0, gc.j1-1, gc.ng, gc.Ntot)


cdef void y2_exch_bc_outflow(GridData gd, GridCoord *gc,  BnzIntegr integr, int1d bvars):

  cdef int n
  for n in range(4):
    prolong_y(gd.fcoup[n,...], 1, gc.j2+1, gc.ng, gc.Ntot)


cdef void z1_exch_bc_outflow(GridData gd, GridCoord *gc,  BnzIntegr integr, int1d bvars):

  cdef int n
  for n in range(4):
    prolong_z(gd.fcoup[n,...], 0, gc.k1-1, gc.ng, gc.Ntot)


cdef void z2_exch_bc_outflow(GridData gd, GridCoord *gc,  BnzIntegr integr, int1d bvars):

  cdef int n
  for n in range(4):
    prolong_z(gd.fcoup[n,...], 1, gc.k2+1, gc.ng, gc.Ntot)


# ====================================================================

# Reflective exchange BC for CR deposits.

cdef void x1_exch_bc_reflect(GridData gd, GridCoord *gc,  BnzIntegr integr, int1d bvars):

  cdef:
    int i1=gc.i1, i2=gc.i2, ng=gc.ng
    int n

  # map deposits from ghost cells to active domain
  for n in range(4):
    if n==1:
      # change sign of normal current
      copy_add_reflect_layer_x(gd.fcoup[1,...], -1, i1,i1-ng, ng, gc.Ntot)
      # copy_reflect_layer_x(gd.fcoup[1,....], -1, i1-ng,i1, ng, gc.Ntot)
    else:
      # keep sign of transverse currents
      copy_add_reflect_layer_x(gd.fcoup[n,...], 1, i1,i1-ng, ng, gc.Ntot)
      # copy_reflect_layer_x(gd.fcoup[n,...], 1, i1-ng,i1, ng, gc.Ntot)


#----------------------------------------------------------------

cdef void x2_exch_bc_reflect(GridData gd, GridCoord *gc,  BnzIntegr integr, int1d bvars):

  cdef:
    int i1=gc.i1, i2=gc.i2, ng=gc.ng
    int n

  for n in range(4):
    if n==1:
      copy_add_reflect_layer_x(gd.fcoup[1,...], -1, i2-ng+1,i2+1, ng, gc.Ntot)
      # copy_reflect_layer_x(gd.fcoup[1,...], -1, i2+1,i2-ng+1, ng, gc.Ntot)
    else:
      copy_add_reflect_layer_x(gd.fcoup[n,...], 1, i2-ng+1,i2+1, ng, gc.Ntot)
      # copy_reflect_layer_x(gd.fcoup[n,...], 1, i2+1,i2-ng+1, ng, gc.Ntot)


#----------------------------------------------------------------

cdef void y1_exch_bc_reflect(GridData gd, GridCoord *gc,  BnzIntegr integr, int1d bvars):

  cdef:
    int j1=gc.j1, j2=gc.j2, ng=gc.ng
    int n

  for n in range(4):
    if n==2:
      copy_add_reflect_layer_y(gd.fcoup[2,...], -1, j1,j1-ng, ng, gc.Ntot)
      # copy_reflect_layer_y(gd.fcoup[2,....], -1, j1-ng,j1, ng, gc.Ntot)
    else:
      copy_add_reflect_layer_y(gd.fcoup[n,...], 1, j1,j1-ng, ng, gc.Ntot)
      # copy_reflect_layer_y(gd.fcoup[n,...], 1, j1-ng,j1, ng, gc.Ntot)


#----------------------------------------------------------------

cdef void y2_exch_bc_reflect(GridData gd, GridCoord *gc,  BnzIntegr integr, int1d bvars):

  cdef:
    int j1=gc.j1, j2=gc.j2, ng=gc.ng
    int n

  for n in range(4):
    if n==2:
      copy_add_reflect_layer_y(gd.fcoup[2,...], -1, j2-ng+1,j2+1, ng, gc.Ntot)
      # copy_reflect_layer_y(gd.fcoup[2,...], -1, j2+1,j2-ng+1, ng, gc.Ntot)
    else:
      copy_add_reflect_layer_y(gd.fcoup[n,...], 1, j2-ng+1,j2+1, ng, gc.Ntot)
      # copy_reflect_layer_y(gd.fcoup[n,...], 1, j2+1,j2-ng+1, ng, gc.Ntot)


#----------------------------------------------------------------

cdef void z1_exch_bc_reflect(GridData gd, GridCoord *gc,  BnzIntegr integr, int1d bvars):

  cdef:
    int k1=gc.k1, k2=gc.k2, ng=gc.ng
    int n

  for n in range(4):
    if n==3:
      copy_add_reflect_layer_z(gd.fcoup[3,...], -1, k1,k1-ng, ng, gc.Ntot)
      # copy_reflect_layer_z(gd.fcoup[3,....], -1, k1-ng,k1, ng, gc.Ntot)
    else:
      copy_add_reflect_layer_z(gd.fcoup[n,...], 1, k1,k1-ng, ng, gc.Ntot)
      # copy_reflect_layer_z(gd.fcoup[n,...], 1, k1-ng,k1, ng, gc.Ntot)


#----------------------------------------------------------------

cdef void z2_exch_bc_reflect(GridData gd, GridCoord *gc,  BnzIntegr integr, int1d bvars):

  cdef:
    int k1=gc.k1, k2=gc.k2, ng=gc.ng
    int n

  for n in range(4):
    if n==3:
      copy_add_reflect_layer_z(gd.fcoup[3,...], -1, k2-ng+1,k2+1, ng, gc.Ntot)
      # copy_reflect_layer_z(gd.fcoup[3,...], -1, k2+1,k2-ng+1, ng, gc.Ntot)
    else:
      copy_add_reflect_layer_z(gd.fcoup[n,...], 1, k2-ng+1,k2+1, ng, gc.Ntot)
      # copy_reflect_layer_z(gd.fcoup[n,...], 1, k2+1,k2-ng+1, ng, gc.Ntot)


# =========================================================================

cdef void r1_exch_bc_sph(GridData gd, GridCoord *gc,  BnzIntegr integr, int1d bvars):

  cdef int n, i1=gc.i1, ng=gc.ng

  for n in range(4):
    if n==1:
      copy_add_layer_r_sph(gd.fcoup[n,...], -1, i1, i1-ng, ng, gc.Ntot,gc.Nact)
    else:
      copy_add_layer_r_sph(gd.fcoup[n,...], 1, i1, i1-ng, ng, gc.Ntot,gc.Nact)


cdef void r1_exch_bc_cyl(GridData gd, GridCoord *gc,  BnzIntegr integr, int1d bvars):

  cdef int n, ng=gc.ng, i1=gc.i1

  for n in range(4):
    if n==1:
      copy_add_layer_r_cyl(gd.fcoup[n,...], -1, i1, i1-ng, ng, gc.Ntot, gc.Nact)
    else:
      copy_add_layer_r_cyl(gd.fcoup[n,...], 1, i1, i1-ng, ng, gc.Ntot, gc.Nact)


cdef void th1_exch_bc_sph(GridData gd, GridCoord *gc,  BnzIntegr integr, int1d bvars):

  cdef int n, ng=gc.ng, j1=gc.j1

  for n in range(4):
    if n==2:
      copy_add_layer_th_sph(gd.fcoup[n,...], -1, j1, j1-ng, ng, gc.Ntot,gc.Nact)
    else:
      copy_add_layer_th_sph(gd.fcoup[n,...], 1, j1, j1-ng, ng, gc.Ntot,gc.Nact)


cdef void th2_exch_bc_sph(GridData gd, GridCoord *gc,  BnzIntegr integr, int1d bvars):

  cdef int n, ng=gc.ng, j2=gc.j2

  for n in range(4):
    if n==2:
      copy_add_layer_th_sph(gd.fcoup[n,...], -1, j2-ng+1, j2+1, ng, gc.Ntot,gc.Nact)
    else:
      copy_add_layer_th_sph(gd.fcoup[n,...], 1, j2-ng+1, j2+1, ng, gc.Ntot,gc.Nact)



# ==================================================================================

IF MPI:

  cdef void pack_exch_all(GridData gd, GridCoord *gc,  int1d bvars, real1d buf, int ax, int side):

    cdef:
      int n
      long offset
      int nx=gc.Ntot[0], ny=gc.Ntot[1], nz=gc.Ntot[2], ng=gc.ng
      int i1=gc.i1, i2=gc.i2, j1=gc.j1, j2=gc.j2, k1=gc.k1, k2=gc.k2
      int1d lims, pack_order

    pack_order = np.ones(3, dtype=np.intp)
    sign=1
    offset=0

    # First treat special boundaries in curvilinear geometry.

    if gc.geom==CG_SPH:

      # r=0 boundary
      if ax==0 and side==0 and gc.lf[0][i1]==0.:

        # f(-r,theta,phi)=f(r,pi-theta,phi+pi) -> reflect along r and theta
        pack_order[0] = -1
        pack_order[1] = -1

        for n in range(4):

          lims = np.array([0,i1-1, 0,ny-1, 0,nz-1], dtype=np.intp)

          # reflect vector r- and theta-components
          if n==1 or n==2:
            sign=-1
          else:
            sign=1

          pack(gd.fcoup[n,...], buf, &offset, lims, pack_order, sign)

        return

      # pole theta=0
      elif ax==1 and side==0 and gc.lf[1][j1]==0.:

        # f(r,-theta,phi)=f(r,theta,phi+pi) -> reflect along theta
        pack_order[1] = -1

        for n in range(4):

          lims = np.array([0,nx-1, 0,j1-1, 0,nz-1])
          sign = -1 if n==2 else 1

          pack(gd.fcoup[n,...], buf, &offset, lims, pack_order, sign)

        return

      # pole theta=pi
      elif ax==1 and side==1 and gc.lf[1][j2+1]==B_PI:

        # f(r,pi+theta,phi)=f(r,pi-theta,phi+pi) -> reflect along theta
        pack_order[1] = -1

        for n in range(4):

          lims = np.array([0,nx-1, j2+1,j2+ng, 0,nz-1])
          sign = -1 if n==2 else 1

          pack(gd.fcoup[n,...], buf, &offset, lims, pack_order, sign)

        return

    elif bc.geom==CG_CYL:

      # r=0 boundary
      if ax==0 and side==0 and gc.lf[0][i1]==0.:

        # f(-r,phi,z)=f(r,phi+pi,z) -> reflect along r
        pack_order[0] = -1

        for n in range(4):

          lims = np.array([0,i1-1, 0,ny-1, 0,nz-1])
          sign = -1 if n==1 else 1

          pack(gd.fcoup[n,...], buf, &offset, lims, pack_order, sign)

        return

    # Now treat ordinary MPI boundaries.

    if side==0:
      if ax==0:   lims = np.array([0,i1-1, 0,ny-1, 0,nz-1])
      elif ax==1: lims = np.array([0,nx-1, 0,j1-1, 0,nz-1])
      elif ax==2: lims = np.array([0,nx-1, 0,ny-1, 0,k1-1])
    elif side==1:
      if ax==0:   lims = np.array([i2+1,i2+ng, 0,ny-1, 0,nz-1])
      elif ax==1: lims = np.array([0,nx-1, j2+1,j2+ng, 0,nz-1])
      elif ax==2: lims = np.array([0,nx-1, 0,ny-1, k2+1,k2+ng])

    for n in range(4):
      pack(gd.fcoup[n,...], buf, &offset, lims, pack_order, 1)

    return


  # ----------------------------------------------------------------------------------

  cdef void unpack_exch_all(GridData gd, GridCoord *gc,  int1d bvars, real1d buf, int ax, int side):

    cdef:
      int n
      long offset
      int nx=gc.Ntot[0], ny=gc.Ntot[1], nz=gc.Ntot[2], ng=gc.ng
      int i1=gc.i1, i2=gc.i2, j1=gc.j1, j2=gc.j2, k1=gc.k1, k2=gc.k2
      int1d lims

    offset=0

    if side==0:
      if ax==0:   lims = np.array([i1,i1+ng-1, 0,ny-1, 0,nz-1])
      elif ax==1: lims = np.array([0,nx-1, j1,j1+ng-1, 0,nz-1])
      elif ax==2: lims = np.array([0,nx-1, 0,ny-1, k1,k1+ng-1])
    elif side==1:
      if ax==0:   lims = np.array([i2-ng+1,i2, 0,ny-1, 0,nz-1])
      elif ax==1: lims = np.array([0,nx-1, j2-ng+1,j2, 0,nz-1])
      elif ax==2: lims = np.array([0,nx-1, 0,ny-1, k2-ng+1,k2])

    for n in range(4):
      unpack_add(gd.fcoup[n,...], buf, &offset, lims)

    return
