# -*- coding: utf-8 -*-

import numpy as np
cimport numpy as np
from cython.parallel import prange, parallel
from util_bc cimport *

# ------------------------------------------------------------------------------

cdef list get_bvar_fld_list(GridData gd, int1d bvars):

  # Form a list of references to 3D arrays that need to be updated.

  cdef int bvar
  flds=[]

  for bvar in bvars:
    if bvar<NMODE:
      # add cell centered fields
      flds.append(gd.prim[bvar,...])
    elif bvar<NMODE+3:
      # add face-centered magnetic field
      flds.append(gd.bfld[bvar-NMODE,...])
    else:
      # add particle feedback force
      flds.append(gd.fcoup[bvar-(NMODE+3),...])


  return flds


# ------------------------------------------------------------------------------

# Periodic MHD BC.

cdef void x1_grid_bc_periodic(GridData gd, GridCoord *gc, BnzIntegr integr, int1d bvars):

  cdef:
    int n, ng=gc.ng, i1=gc.i1, i2=gc.i2
    int nbvar=bvars.size

  flds = get_bvar_fld_list(gd, bvars)

  for n in range(nbvar):
    copy_layer_x(flds[n], i1-ng, i2-ng+1, ng, gc.Ntot)


cdef void x2_grid_bc_periodic(GridData gd, GridCoord *gc, BnzIntegr integr, int1d bvars):

  cdef:
    int n, ng=gc.ng, i1=gc.i1, i2=gc.i2
    int nbvar=bvars.size

  flds = get_bvar_fld_list(gd, bvars)

  for n in range(nbvar):
    copy_layer_x(flds[n], i2+1, i1, ng, gc.Ntot)


cdef void y1_grid_bc_periodic(GridData gd, GridCoord *gc, BnzIntegr integr, int1d bvars):

  cdef:
    int n, ng=gc.ng, j1=gc.j1, j2=gc.j2
    int nbvar=bvars.size

  flds = get_bvar_fld_list(gd, bvars)

  for n in range(nbvar):
    copy_layer_y(flds[n], j1-ng, j2-ng+1, ng, gc.Ntot)


cdef void y2_grid_bc_periodic(GridData gd, GridCoord *gc, BnzIntegr integr, int1d bvars):

  cdef:
    int n, ng=gc.ng, j1=gc.j1, j2=gc.j2
    int nbvar=bvars.size

  flds = get_bvar_fld_list(gd, bvars)

  for n in range(nbvar):
    copy_layer_y(flds[n], j2+1, j1, ng, gc.Ntot)


cdef void z1_grid_bc_periodic(GridData gd, GridCoord *gc, BnzIntegr integr, int1d bvars):

  cdef:
    int n, ng=gc.ng, k1=gc.k1, k2=gc.k2
    int nbvar=bvars.size

  flds = get_bvar_fld_list(gd, bvars)

  for n in range(nbvar):
    copy_layer_z(flds[n], k1-ng, k2-ng+1, ng, gc.Ntot)


cdef void z2_grid_bc_periodic(GridData gd, GridCoord *gc, BnzIntegr integr, int1d bvars):

  cdef:
    int n, ng=gc.ng, k1=gc.k1, k2=gc.k2
    int nbvar=bvars.size

  flds = get_bvar_fld_list(gd, bvars)

  for n in range(nbvar):
    copy_layer_z(flds[n], k2+1, k1, ng, gc.Ntot)


# ------------------------------------------------------------------------------

# Outflow BC.

cdef void x1_grid_bc_outflow(GridData gd, GridCoord *gc, BnzIntegr integr, int1d bvars):

  cdef:
    int n, ng=gc.ng, i1=gc.i1
    int nbvar=bvars.size

  flds = get_bvar_fld_list(gd, bvars)

  for n in range(nbvar):
    prolong_x(flds[n], 0, i1-1, ng, gc.Ntot)


cdef void x2_grid_bc_outflow(GridData gd, GridCoord *gc,  BnzIntegr integr, int1d bvars):

  cdef:
    int n, ng=gc.ng, i2=gc.i2
    int nbvar=bvars.size

  flds = get_bvar_fld_list(gd, bvars)

  for n in range(nbvar):

    # if bvar_types[n]==CC_FLD:
    prolong_x(flds[n], 1, i2+1, ng, gc.Ntot)

    # elif bvar_types[n]== FC_FLD or bvar_types[n]==EC_FLD:
    #   prolong_x(flds[n], 1, i2+2, ng-1, gc.Ntot)


cdef void y1_grid_bc_outflow(GridData gd, GridCoord *gc,  BnzIntegr integr, int1d bvars):

  cdef:
    int n, ng=gc.ng, j1=gc.j1
    int nbvar=bvars.size

  flds = get_bvar_fld_list(gd, bvars)

  for n in range(nbvar):
    prolong_y(flds[n], 0, j1-1, ng, gc.Ntot)


cdef void y2_grid_bc_outflow(GridData gd, GridCoord *gc,  BnzIntegr integr, int1d bvars):

  cdef:
    int n, ng=gc.ng, j2=gc.j2
    int nbvar=bvars.size

  flds = get_bvar_fld_list(gd, bvars)

  for n in range(nbvar):

    # if bvar_types[n]==CC_FLD:
    prolong_y(flds[n], 1, j2+1, ng, gc.Ntot)

    # elif bvar_types[n]== FC_FLD or bvar_types[n]==EC_FLD:
    #   prolong_y(flds[n], 1, j2+2, ng-1, gc.Ntot)


cdef void z1_grid_bc_outflow(GridData gd, GridCoord *gc,  BnzIntegr integr, int1d bvars):

  cdef:
    int n, ng=gc.ng, k1=gc.k1
    int nbvar=bvars.size

  flds = get_bvar_fld_list(gd, bvars)

  for n in range(nbvar):
    prolong_z(flds[n], 0, k1, ng, gc.Ntot)


cdef void z2_grid_bc_outflow(GridData gd, GridCoord *gc,  BnzIntegr integr, int1d bvars):

  cdef:
    int n, ng=gc.ng, k2=gc.k2
    int nbvar=bvars.size

  flds = get_bvar_fld_list(gd, bvars)

  for n in range(nbvar):

    # if bvar_types[n]==CC_FLD:
    prolong_z(flds[n], 1, k2+1, ng, gc.Ntot)

    # elif bvar_types[n]== FC_FLD or bvar_types[n]==EC_FLD:
    #   prolong_z(flds[n], 1, k2+2, ng-1, gc.Ntot)


# ------------------------------------------------------------------------------

# Reflective MHD BC.

cdef void x1_grid_bc_reflect(GridData gd, GridCoord *gc,  BnzIntegr integr, int1d bvars):

  cdef:
    int n, i1=gc.i1, ng=gc.ng
    int nbvar=bvars.size

  flds = get_bvar_fld_list(gd, bvars)

  for n in range(nbvar):

    if bvars[n]==VX or bvars[n]==BXC or bvar[n]==FCX:
      copy_reflect_layer_x(flds[n], -1, i1-ng, i1, ng, gc.Ntot)

    if bvars[n]==BXF:
      set_layer_x(flds[n], 0, i1, 1, gc.Ntot)
      copy_reflect_layer_x(flds[n], -1, i1-ng, i1+1, ng, gc.Ntot)

    else:
      copy_reflect_layer_x(flds[n], 1, i1-ng, i1, ng, gc.Ntot)


cdef void x2_grid_bc_reflect(GridData gd, GridCoord *gc,  BnzIntegr integr, int1d bvars):

  cdef:
    int n, i2=gc.i2, ng=gc.ng
    int nbvar=bvars.size

  flds = get_bvar_fld_list(gd, bvars)

  for n in range(nbvar):

    if bvars[n]==VX or bvars[n]==BXC or bvar[n]==FCX:
      copy_reflect_layer_x(flds[n], -1, i2+1, i2-ng+1, ng, gc.Ntot)

    if bvars[n]==BXF:
      set_layer_x(flds[n], 0, i2+1, 1, gc.Ntot)
      copy_reflect_layer_x(flds[n], -1, i2+2, i2-ng+2, ng-1, gc.Ntot) #!!!

    else:
      copy_reflect_layer_x(flds[n], 1, i2+1, i2-ng+1, ng, gc.Ntot)



cdef void y1_grid_bc_reflect(GridData gd, GridCoord *gc,  BnzIntegr integr, int1d bvars):

  cdef:
    int n, j1=gc.j1, ng=gc.ng
    int nbvar=bvars.size

  flds = get_bvar_fld_list(gd, bvars)

  for n in range(nbvar):

    if bvars[n]==VY or bvars[n]==BYC or bvar[n]==FCY:
      copy_reflect_layer_y(flds[n], -1, j1-ng, j1, ng, gc.Ntot)

    if bvars[n]==BYF:
      set_layer_y(flds[n], 0, j1, 1, gc.Ntot)
      copy_reflect_layer_y(flds[n], -1, j1-ng, j1+1, ng, gc.Ntot)

    else:
      copy_reflect_layer_y(flds[n], 1, j1-ng, j1, ng, gc.Ntot)



cdef void y2_grid_bc_reflect(GridData gd, GridCoord *gc,  BnzIntegr integr, int1d bvars):

  cdef:
    int n, j2=gc.j2, ng=gc.ng
    int nbvar=bvars.size

  flds = get_bvar_fld_list(gd, bvars)

  for n in range(nbvar):

    if bvars[n]==VY or bvars[n]==BYC or bvar[n]==FCY:
      copy_reflect_layer_y(flds[n], -1, j2+1, j2-ng+1, ng, gc.Ntot)

    if bvars[n]==BYF:
      set_layer_y(flds[n], 0, j2+1, 1, gc.Ntot)
      copy_reflect_layer_y(flds[n], -1, j2+2, j2-ng+2, ng-1, gc.Ntot) #!!!

    else:
      copy_reflect_layer_y(flds[n], 1, j2+1, j2-ng+1, ng, gc.Ntot)



cdef void z1_grid_bc_reflect(GridData gd, GridCoord *gc,  BnzIntegr integr, int1d bvars):

  cdef:
    int n, ng=gc.ng, k1=gc.k1
    int nbvar=bvars.size

  flds = get_bvar_fld_list(gd, bvars)

  for n in range(nbvar):

    if bvars[n]==VZ or bvars[n]==BZC or bvar[n]==FCZ:
      copy_reflect_layer_z(flds[n], -1, k1-ng, k1, ng, gc.Ntot)

    if bvars[n]==BZF:
      set_layer_z(flds[n], 0, k1, 1, gc.Ntot)
      copy_reflect_layer_z(flds[n], -1, k1-ng, k1+1, ng, gc.Ntot)

    else:
      copy_reflect_layer_z(flds[n], 1, k1-ng, k1, ng, gc.Ntot)


cdef void z2_grid_bc_reflect(GridData gd, GridCoord *gc,  BnzIntegr integr, int1d bvars):

  cdef:
    int n, ng=gc.ng, k2=gc.k2
    int nbvar=bvars.size

  flds = get_bvar_fld_list(gd, bvars)

  for n in range(nbvar):

    if bvars[n]==VZ or bvars[n]==BZC or bvar[n]==FCZ:
      copy_reflect_layer_z(flds[n], -1, k2+1, k2-ng+1, ng, gc.Ntot)

    if bvars[n]==BZF:
      set_layer_z(flds[n], 0, k2+1, 1, gc.Ntot)
      copy_reflect_layer_z(flds[n], -1, k2+2, k2-ng+2, ng-1, gc.Ntot) #!!!

    else:
      copy_reflect_layer_z(flds[n], 1, k2+1, k2-ng+1, ng, gc.Ntot)



# ------------------------------------------------------------------------------

cdef void r1_grid_bc_sph(GridData gd, GridCoord *gc,  BnzIntegr integr, int1d bvars):

  cdef:
    int n, i1=gc.i1, ng=gc.ng
    int nbvar=bvars.size

  flds = get_bvar_fld_list(gd, bvars)

  for n in range(nbvar):

    if bvars[n]==VX or bvars[n]==BXC or bvar[n]==FCX:
      copy_layer_r_sph(flds[n], -1, i1-ng, i1, ng, gc.Ntot,gc.Nact)

    if bvars[n]==BXF:
      copy_layer_r_sph(flds[n], -1, i1-ng, i1+1, ng, gc.Ntot,gc.Nact)

    else:
      copy_layer_r_sph(flds[n], 1, i1-ng, i1, ng, gc.Ntot,gc.Nact)


cdef void r1_grid_bc_cyl(GridData gd, GridCoord *gc,  BnzIntegr integr, int1d bvars):

  cdef:
    int n, ng=gc.ng, i1=gc.i1
    int nbvar=bvars.size

  flds = get_bvar_fld_list(gd, bvars)

  for n in range(nbvar):

    if bvars[n]==VX or bvars[n]==BXC or bvar[n]==FCX:
      copy_layer_r_cyl(flds[n], -1, i1-ng, i1, ng, gc.Ntot, gc.Nact)

    if bvars[n]==BXF:
      # set_layer_x(flds[n], 0, i1, 1, gc.Ntot)
      copy_layer_r_cyl(flds[n], -1, i1-ng, i1+1, ng, gc.Ntot, gc.Nact)

    else:
      copy_layer_r_cyl(flds[n], 1, i1-ng, i1, ng, gc.Ntot, gc.Nact)


cdef void th1_grid_bc_sph(GridData gd, GridCoord *gc,  BnzIntegr integr, int1d bvars):

  cdef:
    int n, ng=gc.ng, j1=gc.j1
    int nbvar=bvars.size

  flds = get_bvar_fld_list(gd, bvars)

  for n in range(nbvar):

    if bvars[n]==VY or bvars[n]==BYC or bvar[n]==FCY:
      copy_layer_th_sph(flds[n], -1, j1-ng, j1, ng, gc.Ntot,gc.Nact)

    if bvars[n]==BYF:
      # set_layer_y(flds[n], 0, j1, 1, gc.Ntot)
      copy_layer_th_sph(flds[n], -1, j1-ng, j1+1, ng, gc.Ntot,gc.Nact)

    else:
      copy_layer_th_sph(flds[n], 1, j1-ng, j1, ng, gc.Ntot,gc.Nact)


cdef void th2_grid_bc_sph(GridData gd, GridCoord *gc,  BnzIntegr integr, int1d bvars):

  cdef:
    int n, ng=gc.ng, j2=gc.j2
    int nbvar=bvars.size

  flds = get_bvar_fld_list(gd, bvars)

  for n in range(nbvar):

    if bvars[n]==VY or bvars[n]==BYC or bvar[n]==FCY:
      copy_layer_th_sph(flds[n], -1, j2+1, j2-ng+1, ng, gc.Ntot,gc.Nact)

    if bvars[n]==BYF:
      # set_layer_y(flds[n], 0, j2+1, 1, gc.Ntot)
      copy_layer_th_sph(flds[n], -1, j2+2, j2-ng+2, ng-1, gc.Ntot,gc.Nact)

    else:
      copy_layer_th_sph(flds[n], 1, j2+1, j2-ng+1, ng, gc.Ntot,gc.Nact)



# ==============================================================================

IF MPI:

  cdef void pack_grid_all(GridData gd, GridCoord *gc, int1d bvars,
                          real1d buf, int ax, int side):

    cdef:
      int n
      long offset
      int nx=gc.Ntot[0], ny=gc.Ntot[1], nz=gc.Ntot[2], ng=gc.ng
      int i1=gc.i1, i2=gc.i2, j1=gc.j1, j2=gc.j2, k1=gc.k1, k2=gc.k2
      int nbvar = bvars.size
      int1d lims, pack_order

    flds = get_bvar_fld_list(gd, bvars)

    pack_order = np.ones(3, dtype=np.intc)
    sign=1
    offset=0

    # First treat special boundaries in curvilinear geometry.

    if gc.geom==CG_SPH:

      # r=0 boundary
      if ax==0 and side==0 and gc.lf[0][i1]==0.:

        # f(-r,theta,phi)=f(r,pi-theta,phi+pi) -> reflect along r and theta
        pack_order[0] = -1
        pack_order[1] = -1

        for n in range(nbvar):

          if bvars[n]==BXF:
            lims = np.array([i1+1,i1+ng, 0,ny-1, 0,nz-1], dtype=np.intc)
          else:
            lims = np.array([i1,i1+ng-1, 0,ny-1, 0,nz-1], dtype=np.intc)

          # reflect vector r- and theta-components
          if ((bvars[n]==VX or bvars[n]==BXC or bvars[n]==BXF or bvars[n]==FCX) or
              (bvars[n]==VY or bvars[n]==BYC or bvars[n]==BYF or bvars[n]==FCY)):
            sign=-1
          else:
            sign=1

          pack(flds[n], buf, &offset, lims, pack_order, sign)

        return

      # pole theta=0
      elif ax==1 and side==0 and gc.lf[1][j1]==0.:

        # f(r,-theta,phi)=f(r,theta,phi+pi) -> reflect along theta
        pack_order[1] = -1

        for n in range(nbvar):

          if bvars[n]==BYF:
            lims = np.array([0,nx-1, j1+1,j1+ng, 0,nz-1], dtype=np.intc)
          else:
            lims = np.array([0,nx-1, j1,j1+ng-1, 0,nz-1], dtype=np.intc)

          # reflect vector theta-components
          if bvars[n]==VY or bvars[n]==BYC or bvars[n]==BYF or bvars[n]==FCY:
            sign=-1
          else:
            sign=1

          pack(flds[n], buf, &offset, lims, pack_order, sign)

        return

      # pole theta=pi
      elif ax==1 and side==1 and gc.lf[1][j2+1]==B_PI:

        # f(r,pi+theta,phi)=f(r,pi-theta,phi+pi) -> reflect along theta
        pack_order[1] = -1

        for n in range(nbvar):

          if bvars[n]==BYF:
            # need only ng-1 layers of ghosts
            lims = np.array([0,nx-1, j2-ng+2,j2, 0,nz-1], dtype=np.intc)
          else:
            lims = np.array([0,nx-1, j2-ng+1,j2, 0,nz-1], dtype=np.intc)

          if bvars[n]==VY or bvars[n]==BYC or bvars[n]==BYF or bvars[n]==FCY:
            sign=-1
          else:
            sign=1

          pack(flds[n], buf, &offset, lims, pack_order, sign)

        return

    elif bc.geom==CG_CYL:

      # r=0 boundary
      if ax==0 and side==0 and gc.lf[0][i1]==0.:

        # f(-r,phi,z)=f(r,phi+pi,z) -> reflect along r
        pack_order[0] = -1

        for n in range(nbvar):

          if bvars[n]==BXF:
            lims = np.array([i1+1,i1+ng, 0,ny-1, 0,nz-1], dtype=np.intc)
          else:
            lims = np.array([i1,i1+ng-1, 0,ny-1, 0,nz-1], dtype=np.intc)

          # reflect vector r-components
          if bvars[n]==VX or bvars[n]==BXC or bvars[n]==BXF or bvars[n]==FCX:
            sign=-1
          else:
            sign=1

          pack(flds[n], buf, &offset, lims, pack_order, sign)

        return

    # Now treat ordinary MPI boundaries.

    if side==0:
      if ax==0:   lims = np.array([i1,i1+ng-1, 0,ny-1, 0,nz-1], dtype=np.intc)
      elif ax==1: lims = np.array([0,nx-1, j1,j1+ng-1, 0,nz-1], dtype=np.intc)
      elif ax==2: lims = np.array([0,nx-1, 0,ny-1, k1,k1+ng-1], dtype=np.intc)
    elif side==1:
      if ax==0:   lims = np.array([i2-ng+1,i2, 0,ny-1, 0,nz-1], dtype=np.intc)
      elif ax==1: lims = np.array([0,nx-1, j2-ng+1,j2, 0,nz-1], dtype=np.intc)
      elif ax==2: lims = np.array([0,nx-1, 0,ny-1, k2-ng+1,k2], dtype=np.intc)

    for n in range(nbvar):
      pack(flds[n], buf, &offset, lims, pack_order, 1)

    return


  # -----------------------------------------------------------------------------

  cdef void unpack_grid_all(GridData gd, GridCoord *gc,  int1d bvars,
                            real1d buf, int ax, int side):

    cdef:
      int n
      long offset
      int nx=gc.Ntot[0], ny=gc.Ntot[1], nz=gc.Ntot[2], ng=gc.ng
      int i1=gc.i1, i2=gc.i2, j1=gc.j1, j2=gc.j2, k1=gc.k1, k2=gc.k2
      int nbvar = bvars.size
      int1d lims

    flds = get_bvar_fld_list(gd, bvars)
    offset=0

    # First treat special boundaries in curvilinear geometry.

    if gc.geom==CG_SPH:

      # pole theta=pi
      if ax==1 and side==1 and gc.lf[1][j2+1]==B_PI:

        for n in range(nbvar):

          if bvars[n]==BYF:
            lims = np.array([0,nx-1, j2+2,j2+ng, 0,nz-1], dtype=np.intc)
          else:
            lims = np.array([0,nx-1, j2+1,j2+ng, 0,nz-1], dtype=np.intc)

          unpack(flds[n], buf, &offset, lims)

        return

    # Now treat ordinary MPI boundaries.

    if side==0:
      if ax==0:   lims = np.array([i1-ng,i1-1, 0,ny-1, 0,nz-1], dtype=np.intc)
      elif ax==1: lims = np.array([0,nx-1, j1-ng,j1-1, 0,nz-1], dtype=np.intc)
      elif ax==2: lims = np.array([0,nx-1, 0,ny-1, k1-ng,k1-1], dtype=np.intc)
    elif side==1:
      if ax==0:   lims = np.array([i2+1,i2+ng, 0,ny-1, 0,nz-1], dtype=np.intc)
      elif ax==1: lims = np.array([0,nx-1, j2+1,j2+ng, 0,nz-1], dtype=np.intc)
      elif ax==2: lims = np.array([0,nx-1, 0,ny-1, k2+1,k2+ng], dtype=np.intc)

    for n in range(nbvar):
      unpack(flds[n], buf, &offset, lims)

    return
