# -*- coding: utf-8 -*-

IF MPI:
  from mpi4py import MPI as mpi
  from mpi4py cimport MPI as mpi

import numpy as np
cimport numpy as np

from libc.math cimport sqrt,floor,ceil,log,exp,sin,cos,pow,fabs,fmin,fmax

from bnz.coord cimport lind2gcrd, lind2gcrd_x,lind2gcrd_y,lind2gcrd_z
from bnz.read_config import read_user_param

IF SPREC:
  np_real = np.float32
ELSE:
  np_real = np.float64



cdef void set_problem(BnzSim sim):

  cdef:
    BnzGrid grid = sim.grid
    GridParams gp = grid.params
    GridData gd = grid.data
    real4d W = grid.data.W
    real4d B = grid.data.B
    BnzPhysics phys = sim.phys

  cdef:
    ints i,j,k
    real x=0,y=0,z=0
    double xh,yh,zh
    double x1, w
    double rho0, v10,v20,v30, B10,B20,B30, en0,p0
    double v1,v2,v3, A1,A2,A3, en
    double eps
    double sa,ca,sb,cb
    double alpha

  cdef:
    np.ndarray[real, ndim=4] A
    np.ndarray[real, ndim=2] T,Ti
    np.ndarray[real, ndim=1] Rs,Ra,Rsm,Rfm,Rv, R,R1, A0, vlab,blab

  #----------------------------------------------------------------

  cdef double beta = read_user_param('beta', 'f', sim.output.usr_dir)

  # Nx/Ny=2 should be set for this test, Lx and Ly are rescaled below!

  # gas gamma
  phys.gam = 5./3
  #wave amplitude
  eps = 1e-6

  # orientation of the wave vector

  # coordinate rotation around y
  sa = 2./3 if gd.Nact[2]>1 else 0.
  ca = sqrt(5)/3 if gd.Nact[2]>1 else 1.
  # set Lx=3, Ly=Lz=1.5 in 3D; Lx=2, Ly=Lz=1 in 2D

  # coordinate rotation around z
  sb = 2./sqrt(5)
  cb = 1./sqrt(5)

  # background

  rho0 = 1.
  p0 = 0.6 #* beta/1.2

  # background magnetic field and velocity in wave frame
  v10 = 0. # 1 for entropy wave
  B10 = 1.
  B20 = 1.5
  B30 = 0.

  # rescale wavelength in 2D to keep the wave periodic
  per = 1. if gd.Nact[2]>1 else 2./sqrt(5)

  # eigenvectors of different MHD modes

  Rs = np.array([1, -1, 1, 1, 1.5, 0, 0, 0], dtype=np_real)
  Ra = np.array([0,
                 0, -1./3, 2*sqrt(2)/3,
                 0, -1./3, 2*sqrt(2)/3,
                 0], dtype=np_real)
  Ra_ = np.array([0,
                0, 0, -1,
                0, 0,  1,
                0], dtype=np_real)
  Rsm = np.array([2,
                 -1, -4*sqrt(2)/3, -2./3,
                  0, -2*sqrt(2)/3, -1./3,
                1.5], dtype=np_real) / sqrt(5)
  Rfm = np.array([1,
                 -2,  2*sqrt(2)/3, 1./3,
                  0,  4*sqrt(2)/3, 2./3,
                4.5], dtype=np_real) / sqrt(5)
  Rv = np.array([1, 1, 0, 0,
                 0, 0, 0, 0.5], dtype=np_real)

  # choose the MHD mode
  R = Ra


  # rotation matrix and its inverse

  T =  np.array([[ca*cb, -sb, -sa*cb],
                 [ca*sb,  cb, -sa*sb],
                 [sa,     0.,  ca   ]])

  Ti = np.array([[ ca*cb,  ca*sb,  sa],
                 [   -sb,     cb,  0.],
                 [-sa*cb, -sa*sb,  ca]])

  # magnetic-field polarization vector in lab frame
  R1 = np.dot(T, R[4:7])

  # print R1

  # calculate vector-potential components in lab frame
  A0 = np.zeros(3)
  A0[0] = -R1[2] / Ti[0,1] / (-2*M_PI)
  A0[1] = 0.
  A0[2] =  R1[0] / Ti[0,1] / (-2*M_PI)

  # set vector-potential array

  A = np.zeros((3, gp.Ntot[2]+1, gp.Ntot[1]+1, gp.Ntot[0]+1), dtype=np_real)

  # phase prefactor, including rescaled wavelength
  alpha = 2*M_PI / per
  A0 /= per

  # velocity and magnetic field in lab frame
  vlab = np.zeros(3)
  blab = np.zeros(3)


  for k in range(gp.k1, gp.k2+2):
    for j in range(gp.j1, gp.j2+2):
      for i in range(gp.i1, gp.i2+2):

        lind2gcrd(&x,&y,&z, i,j,k, gp)

        xh = x - 0.5*gp.dl[0]
        yh = y - 0.5*gp.dl[1]
        zh = z - 0.5*gp.dl[2]

        # Ax

        x1 = x * Ti[0,0] + yh * Ti[0,1] + zh * Ti[0,2]
        w = eps*cos(alpha * x1)
        A[0,k,j,i] = w * A0[0]

        # Az

        x1 = xh * Ti[0,0] + yh * Ti[0,1] + z * Ti[0,2]
        w = eps*cos(alpha * x1)
        A[2,k,j,i] = w * A0[2]


  for k in range(gp.k1, gp.k2+1):
    for j in range(gp.j1, gp.j2+1):
      for i in range(gp.i1, gp.i2+1):

        lind2gcrd(&x,&y,&z, i,j,k, gp)

        x1 = x * Ti[0,0] + y * Ti[0,1] + z * Ti[0,2]
        w = eps * sin(alpha * x1)

        # variables in wave frame

        W[RHO,k,j,i] = rho0 + R[0]*w

        v1 = v10 + R[1]*w / rho0
        v2 =       R[2]*w / rho0
        v3 =       R[3]*w / rho0

        # convert velocities to lab frame
        vlab= np.dot(T, [v1,v2,v3])
        W[VX,k,j,i] = vlab[0]
        W[VY,k,j,i] = vlab[1]
        W[VZ,k,j,i] = vlab[2]

        W[PR,k,j,i] = p0 + (phys.gam-1) * R[7]*w
        IF CGL: W[PPD,k,j,i] = W[PR,k,j,i]

        # convert background magnetic field to lab frame
        blab = np.dot(T, [B10,B20,B30])
        B[0,k,j,i] = blab[0]
        B[1,k,j,i] = blab[1]
        B[2,k,j,i] = blab[2]


        # add magnetic fluctuation from lab vector-potential

        B[0,k,j,i] +=     (A[2,k,j+1,i] - A[2,k,j,i]) * gp.dli[1]
        B[1,k,j,i] += ( - (A[2,k,j,i+1] - A[2,k,j,i]) * gp.dli[0]
                        + (A[0,k+1,j,i] - A[0,k,j,i]) * gp.dli[2] )
        B[2,k,j,i] +=   - (A[0,k,j+1,i] - A[0,k,j,i]) * gp.dli[1]



# =====================================================================

cdef void do_user_work_cons(real4d U1, real4d B1, real4d U0, real4d B0,
                       ints lims[6], BnzSim sim, double dt):
  return


cdef void set_bc_grid_ptrs_user(BnzBC bc):
    # bc.bc_grid_funcs[0][0] = x1_bc_grid
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
