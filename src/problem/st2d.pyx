# -*- coding: utf-8 -*-

IF MPI:
  from mpi4py import MPI as mpi
  from mpi4py cimport MPI as mpi

import numpy as np
cimport numpy as np

from libc.math cimport sqrt,floor,ceil,log,exp,sin,cos,pow,fabs,fmin,fmax
from libc.stdlib cimport rand, RAND_MAX, srand

from bnz.coord cimport lind2gcrd, lind2gcrd_x,lind2gcrd_y,lind2gcrd_z
from bnz.read_config import read_user_param

IF SPREC:
  np_real = np.float32
ELSE:
  np_real = np.float64


cdef inline void transform(real *ax, real *ay, real *az):

  cdef:
    real ax_ = (ax[0]  - 2*ay[0])/sqrt(5)
    real ay_ = (2*ax[0] + ay[0])/sqrt(5)
    real az_ = az[0]

  ax[0]=ax_
  ay[0]=ay_
  az[0]=az_

cdef inline void transform_inv(real *ax, real *ay, real *az):

  cdef:
    real ax_ = (ax[0]  + 2*ay[0])/sqrt(5)
    real ay_ = (-2*ax[0] + ay[0])/sqrt(5)
    real az_ = az[0]

  ax[0]=ax_
  ay[0]=ay_
  az[0]=az_


cdef void set_problem(BnzSim sim):


  cdef:
    BnzGrid grid = sim.grid
    GridParams gp = grid.params
    real4d W = grid.data.W     # array of primitive variables
    real4d B = grid.data.B     # array of face-centered magnetic field
    BnzPhysics phys = sim.phys

  cdef:
    ints i,j,k, kp1
    real x=0,y=0,z=0, xf,yf,zf
    real xt,yt,zt, xft,yft,zft
    real bx1,bx2,by1,by2,bz1,bz2, vx1,vy1,vz1, vx2,vy2,vz2


  cdef ints Nza, k2a
  IF D3D:
    Nza = gp.Ntot[2]+1
    k2a = gp.k2+1
  ELSE:
    Nza = gp.Ntot[2]
    k2a = gp.k2

  cdef np.ndarray[real, ndim=4] A = np.zeros(
                (3, Nza, gp.Ntot[1]+1, gp.Ntot[0]+1), dtype=np.float32 )


  #-------------------------

  bx1=2./sqrt(4*M_PI)
  bx2=2./sqrt(4*M_PI)

  by1=3.6/sqrt(4*M_PI)
  by2=4./sqrt(4*M_PI)

  bz1=2./sqrt(4*M_PI)
  bz2=2./sqrt(4*M_PI)

  rho1=1.08
  vx1=1.2
  vy1=0.01
  vz1=0.5
  p1=0.95

  transform(&vx1, &vy1, &vz1)
  transform(&bx1, &by1, &bz1)

  rho2=1.
  vx2=0.
  vy2=0.
  vz2=0.
  p2=1.

  transform(&bx2, &by2, &bz2)

  for k in range(gp.k1, k2a+1):
    for j in range(gp.j1, gp.j2+2):
      for i in range(gp.i1, gp.i2+2):

        lind2gcrd(&x,&y,&z, i,j,k, gp)

        x -= 0.5*gp.Lglob[0]
        y -= 0.5*gp.Lglob[1]
        z -= 0.5*gp.Lglob[2]
        xf = x - 0.5*gp.dl[0]
        yf = y - 0.5*gp.dl[1]
        zf = z - 0.5*gp.dl[2]

        xt,yt,zt = x,y,z
        xft,yft,zft = xf,yf,zf

        transform_inv(&xft, &yft, &zft)
        transform_inv(&xt, &yt, &zt)

        if xft < 0:
          A[2,k,j,i] = bx1 * yf - by1 * xf
        else:
          A[2,k,j,i] = bx2 * yf - by2 * xf

        if xt < 0:
          A[0,k,j,i] = - bz1 * yf
        else:
          A[0,k,j,i] = - bz2 * yf


  for k in range(gp.k1, gp.k2+1):
    for j in range(gp.j1, gp.j2+1):
      for i in range(gp.i1, gp.i2+1):

        lind2gcrd(&x,&y,&z, i,j,k, gp)

        x -= 0.5*gp.Lglob[0]
        y -= 0.5*gp.Lglob[1]
        z -= 0.5*gp.Lglob[2]

        transform_inv(&x, &y, &z)

        if x < 0:

          W[RHO,k,j,i] = rho1
          W[PR,k,j,i] = p1
          W[VX,k,j,i] = vx1
          W[VY,k,j,i] = vy1
          W[VZ,k,j,i] = vz1

        else:

          W[RHO,k,j,i] = rho2
          W[PR,k,j,i] = p2

          W[VX,k,j,i] = vx2
          W[VY,k,j,i] = vy2
          W[VZ,k,j,i] = vz2

        IF D3D: kp1=k+1
        ELSE: kp1 = k

        B[0,k,j,i] =  ( (A[2,k,j+1,i] - A[2,k,j,i]) * gp.dli[1]
                      - (A[1,kp1,j,i] - A[1,k,j,i]) * gp.dli[2] )
        B[1,k,j,i] = ( - (A[2,k,j,i+1] - A[2,k,j,i]) * gp.dli[0]
                       + (A[0,kp1,j,i] - A[0,k,j,i]) * gp.dli[2] )
        B[2,k,j,i] = ( (A[1,k,j,i+1] - A[1,k,j,i]) * gp.dli[0]
                     - (A[0,k,j+1,i] - A[0,k,j,i]) * gp.dli[1] )



# ============================================================



cdef list create_bvar_fld_list(GridData gd, ints[::1] bvars):

  cdef ints bvar
  flist=[]

  for bvar in bvars:

    IF not MFIELD:
      flist.append(gd.W[bvar,...])
    ELSE:
      if bvar<NWAVES:
        flist.append(gd.W[bvar,...])
      elif bvar < NWAVES+3:
        flist.append(gd.B[bvar-NWAVES,...])

  return flist


# ---------------------------------------------------------------


cdef void x1_bc_grid_user(BnzSim sim, ints1d bvars):

  cdef:
    BnzGrid grid = sim.grid
    GridParams gp = grid.params
    GridData gd = grid.data

  cdef int i,j,k, bvar
  cdef real rho1,vx1,vy1,vz1,p1,bx1,by1,bz1

  bx1=2./sqrt(4*M_PI)
  by1=3.6/sqrt(4*M_PI)
  bz1=2./sqrt(4*M_PI)

  rho1=1.08
  vx1=1.2
  vy1=0.01
  vz1=0.5
  p1=0.95

  transform(&vx1, &vy1, &vz1)
  transform(&bx1, &by1, &bz1)

  for k in range(gp.k1, gp.k2+1):
    for j in range(gp.j1, gp.j2+1):
      for i in range(gp.i1):
        for bvar in bvars:

          if bvar==RHO:
            gd.W[RHO,k,j,i] = rho1
          elif bvar==VX:
            gd.W[VX,k,j,i] = vx1
          elif bvar==VY:
            gd.W[VY,k,j,i] = vy1
          elif bvar==VZ:
            gd.W[VZ,k,j,i] = vz1
          elif bvar==PR:
            gd.W[PR,k,j,i] = p1
          elif bvar==BXC:
            gd.W[BX,k,j,i] = bx1
          elif bvar==BYC:
            gd.W[BY,k,j,i] = by1
          elif bvar==BZC:
            gd.W[BZ,k,j,i] = bz1
          elif bvar==BXF:
            gd.B[0,k,j,i] = bx1
          elif bvar==BYF:
            gd.B[1,k,j,i] = by1
          elif bvar==BZF:
            gd.B[2,k,j,i] = bz1

  return


cdef void x2_bc_grid_user(BnzSim sim, ints1d bvars):

  cdef:
    BnzGrid grid = sim.grid
    GridParams gp = grid.params
    GridData gd = grid.data

  cdef int i,j,k, bvar
  cdef real rho2,vx2,vy2,vz2,p2,bx2,by2,bz2

  bx2=2./sqrt(4*M_PI)
  by2=4./sqrt(4*M_PI)
  bz2=2./sqrt(4*M_PI)

  rho2=1.
  vx2=0.
  vy2=0.
  vz2=0.
  p2=1.

  transform(&vx2, &vy2, &vz2)
  transform(&bx2, &by2, &bz2)

  for k in range(gp.k1, gp.k2+1):
    for j in range(gp.j1, gp.j2+1):
      for i in range(gp.i2+1, gp.Ntot[0]):
        for bvar in bvars:

          if bvar==RHO:
            gd.W[RHO,k,j,i] = rho2
          elif bvar==VX:
            gd.W[VX,k,j,i] = vx2
          elif bvar==VY:
            gd.W[VY,k,j,i] = vy2
          elif bvar==VZ:
            gd.W[VZ,k,j,i] = vz2
          elif bvar==PR:
            gd.W[PR,k,j,i] = p2
          elif bvar==BXC:
            gd.W[BX,k,j,i] = bx2
          elif bvar==BYC:
            gd.W[BY,k,j,i] = by2
          elif bvar==BZC:
            gd.W[BZ,k,j,i] = bz2
          elif bvar==BXF:
            gd.B[0,k,j,i] = bx2
          elif bvar==BYF:
            gd.B[1,k,j,i] = by2
          elif bvar==BZF:
            gd.B[2,k,j,i] = bz2

  return


cdef void y1_bc_grid_user(BnzSim sim, ints1d bvars):

  cdef:
    BnzGrid grid = sim.grid
    GridParams gp = grid.params
    GridData gd = grid.data

  cdef int i,j,k, n, i_

  cdef ints Nbvar=bvars.size
  flds = create_bvar_fld_list(gd, bvars)

  for n in range(Nbvar):
    for k in range(gp.k1, gp.k2+1):
      for j in range(gp.j1):
        for i in range(gp.Ntot[0]):

          i_ = i-2*gp.Nact[1]
          if i_ < 0:
            i_ = 0

          flds[n][k,j,i] = flds[n][k, gp.j2-gp.j1+j+1, i_]

  return


cdef void y2_bc_grid_user(BnzSim sim, ints1d bvars):

  cdef:
    BnzGrid grid = sim.grid
    GridParams gp = grid.params
    GridData gd = grid.data

  cdef int i,j,k, ir,jr, n, i_

  cdef ints Nbvar=bvars.size
  flds = create_bvar_fld_list(gd, bvars)

  for n in range(Nbvar):
    for k in range(gp.k1, gp.k2+1):
      for j in range(gp.j2+1, gp.Ntot[1]):
        for i in range(gp.Ntot[0]):

          i_ = i+2*gp.Nact[1]
          if i_ > gp.Ntot[0]-1:
            i_ = gp.Ntot[0]-1

          flds[n][k,j,i] = flds[n][k, gp.j1+j-gp.j2-1, i_]

  return




# =====================================================================

cdef void do_user_work_cons(real4d U1, real4d B1, real4d U0, real4d B0,
                       ints lims[6], BnzSim sim, double dt):
  return


cdef void set_bc_grid_ptrs_user(BnzBC bc):

  bc.bc_grid_funcs[0][0] = x1_bc_grid_user
  bc.bc_grid_funcs[0][1] = x2_bc_grid_user
  bc.bc_grid_funcs[1][0] = y1_bc_grid_user
  bc.bc_grid_funcs[1][1] = y2_bc_grid_user

  return


IF PIC or MHDPIC:
  cdef void set_bc_prt_ptrs_user(BnzBC bc):
    # bc.bc_prt_funcs[0][0] = x1_bc_prt
    return

cdef void set_phys_ptrs_user(BnzPhysics phys):
  #phys.grav_pot_func = grav_pot
  return


# Set user history variables and particle selection function.

cdef void set_output_user(BnzOutput output):
  # set up to NHST_U (8 by default) function pointers
  # output.hst_funcs_u[0] = hst_var1
  # output.hst_names_u[0] = "B2h"
  # IF PIC:
  #   output.prt_sel_func = select_particle
  return
