#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True
# -*- coding: utf-8 -*-

import numpy as np
cimport numpy as np

from libc.math cimport M_PI, sqrt,floor,ceil,log,exp,sin,cos,pow,fabs,fmin,fmax
from libc.stdlib cimport rand, RAND_MAX, srand

from src.coord cimport ind2crdx_u, ind2crdy_u, ind2crdz_u

cdef void set_user_problem(Domain dom,
    np.ndarray[float, ndim=3] rho,
    np.ndarray[float, ndim=4] v,
    np.ndarray[float, ndim=3] p,
    np.ndarray[float, ndim=3] ppd,
    np.ndarray[float, ndim=3] pe,
    np.ndarray[float, ndim=3] pscal,
    np.ndarray[float, ndim=4] B):

  cdef:

    uint Nx = dom.Nx
    uint Ny = dom.Ny
    uint Nz = dom.Nz
    double Lx = dom.Lx
    double Ly = dom.Ly
    double Lz = dom.Lz
    double dx = dom.dx
    Parameters *pm = &(dom.params)

    int i,j,m
    double x,y,z, rc, rhoc, rhohot, R, Brms

  rc = 0.12 * 0.4
  rhoc = 2.
  rhohot = 0.25
  pm.gam=5./3

  #==================================================================
  #  L0 = 2.5 Mpc
  #  v0 = 1000 km/s
  #  T0 = 2.4*10^9 yr
  #  Temp0 = 6.4 KeV
  #  rho0 = 5*10^(-27) g cm^(-3) -> n0=3*10^(-3) cm^(-3)
  #  kappa0_Sp = 0.04
  #==================================================================

  for i in range(Nx):
    for j in range(Ny):
      for m in range(Nz):

        x = ind2crdx_u(i,dx)
        y = ind2crdy_u(j,dx)
        z = ind2crdz_u(m,dx)

        R = sqrt((x-0.5*Lx)**2 +
                 (y-0.5*Ly)**2 +
                 (z-0.5*Lz)**2)+1e-20

        if R < rc*sqrt(3):
          rho[i,j,m] = rhoc/(1+(R/rc)**2)
          p[i,j,m] = rhoc/pm.gam/(1+(R/rc)**2)

        else:
          rho[i,j,m] = rhohot
          p[i,j,m] = 0.25* rhoc/pm.gam
          v[i,j,m,0] = sqrt(2.)

         # B[i,j,m,1] = sqrt(2*0.25*rhoc/pm.gam/pm.beta)
         # if j==Ny-1: B[i,Ny,m,1] = B[i,Ny-1,m,1]

  Brms = sqrt(2*0.25*rhoc / pm.gam / pm.beta)
  # B[:Nx,:Ny,:Nz,:] = utils.gen_divfree3dvec(Nx,Ny,Nz,dx, Brms, s=11./3,
  #                             kmin=4./Nx, kmax=8./Nx, nt=dom.nt)
  for i in range(Nx):
    for j in range(Ny):
      for m in range(Nz):
        B[Nx,j,m,0] = B[0,j,m,0]
        B[i,Ny,m,1] = B[i,0,m,1]
        B[i,j,Nz,2] = B[i,j,0,2]

  dom.aux.Binit = B.copy()

     # rho[i,j]=1
     # p[i,j]=1
     # dom.gam = 5./3
     # B[i,j,0] = -1./sqrt(0.5*pm.beta)
     # B[i,j,1] = 0
     # v[i,j,0] = 0
     # v[i,j,1] = 1
     #
     # if fabs((i+0.5)*dx - 0.5) <= 0.1 and j*dx>=0.25:
     #      v[i,j,1] = 0

  dom.BCFlag_x1=3
  dom.BCFlag_x2=1
  dom.BCFlag_y1=1
  dom.BCFlag_y2=1
  dom.BCFlag_z1=1
  dom.BCFlag_z2=1

#----------------------------------------

cdef void set_user_output(Domain dom):
  return

#----------------------------------------

cdef void x1_bc_mhd_user(Domain dom) nogil:

  cdef:
    uint Nx = dom.Nx
    int i1 = dom.i1, i2 = dom.i2
    int j1 = dom.j1, j2 = dom.j2
    int m1 = dom.m1, m2 = dom.m2
    int ng = dom.ng

    Parameters pm = dom.params
    double gam = pm.gam
    double beta = pm.beta

    int i,j,m,k,g, j21,m21

    double v1,v2,v3, B1,B2L,B2R,B3

    double R, rhoc,rhohot
    int p, p1
    double di,fr

  rhohot=0.25
  rhoc=2

  di = sqrt(2.)*dom.t/dom.dx
  p = <int>floor(di)
  p1 = <int>ceil(di)
  fr = di - p

  for m in range(m1, m2+1):
    for j in range(j1, j2+1):
      for g in range(ng):

        dom.U[g,j,m,RHO] = rhohot
        dom.U[g,j,m,MX] = rhohot*sqrt(2.)
        dom.U[g,j,m,MY] = 0
        dom.U[g,j,m,MZ] = 0

       dom.U[g,j,m,BX] = 0
       dom.U[g,j,m,BY] = sqrt(2*0.25*rhoc/gam/beta)
       dom.U[g,j,m,BZ] = 0
       dom.U[g,j,m,EN] = ( rhohot + 0.25*rhoc/gam/(gam-1)
            + 0.5*dom.U[g,j,m,BY]**2 + 0.25*rhoc/gam/beta )

        # dom.U[g,j,m,EN] = rhohot + 0.25*rhoc/gam/(gam-1) +\
        #     0.5*(dom.U[g,j,m,BX]**2 + dom.U[g,j,m,BY]**2 + dom.U[g,j,m,BZ]**2)

    for j in range(j1, j2+1):
      for g in range(ng-1):

        B[ng-g-1,j,m,0] = 0
        # if p+g >= Nx: p = p - (p+g)/Nx*Nx
        # if p1+g >= Nx: p1 = p1 - (p1+g)/Nx*Nx
        # dom.B[i1-g-1,j,m,0] = (1-fr)*dom.aux.Binit[Nx-1-p-g,  j-j1, m-m1, 0] + \
        #                           fr*dom.aux.Binit[Nx-1-p1-g, j-j1, m-m1, 0]

    for j in range(j1, j2+2):
      for g in range(ng):

        B[ng-g-1,j,m,1] = sqrt(2 * 0.25*rhoc/gam/beta)
        # if p+g >= Nx: p = p - (p+g)/Nx*Nx
        # if p1+g >= Nx: p1 = p1 - (p1+g)/Nx*Nx
        # dom.B[i1-g-1,j,m,1] = (1-fr)*dom.aux.Binit[Nx-1-p-g,  j-j1, m-m1, 1] + \
        #                           fr*dom.aux.Binit[Nx-1-p1-g, j-j1, m-m1, 1]

  for m in range(m1, m2+2):
    for j in range(j1, j2+1):
      for g in range(ng):

       B[ng-g-1,j,m,2] = 0
       # if p+g >= Nx: p = p - (p+g)/Nx*Nx
       # if p1+g >= Nx: p1 = p1 - (p1+g)/Nx*Nx
       # dom.B[i1-g-1,j,m,2] = (1-fr)*dom.aux.B0[Nx-1-p-g,  j-j1, m-m1, 2] + \
       #                           fr*dom.aux.B0[Nx-1-p1-g, j-j1, m-m1, 2]

cdef void x2_bc_mhd_user(Domain dom) nogil:
  return

cdef void y1_bc_mhd_user(Domain dom) nogil:
  return
cdef void y2_bc_mhd_user(Domain dom) nogil:
  return

cdef void z1_bc_mhd_user(Domain dom) nogil:
  return
cdef void z2_bc_mhd_user(Domain dom) nogil:
  return

IF PIC:

  cdef void x1_bc_ex_user(Domain dom) nogil:
    return
  cdef void x2_bc_ex_user(Domain dom) nogil:
    return

  cdef void y1_bc_ex_user(Domain dom) nogil:
    return
  cdef void y2_bc_ex_user(Domain dom) nogil:
    return

  cdef void z1_bc_ex_user(Domain dom) nogil:
    return
  cdef void z2_bc_ex_user(Domain dom) nogil:
    return


  cdef void x1_bc_prt_user(Domain dom) nogil:
    return
  cdef void x2_bc_prt_user(Domain dom) nogil:
    return

  cdef void y1_bc_prt_user(Domain dom) nogil:
    return
  cdef void y2_bc_prt_user(Domain dom) nogil:
    return

  cdef void z1_bc_prt_user(Domain dom) nogil:
    return
  cdef void z2_bc_prt_user(Domain dom) nogil:
    return


cdef double grav_pot(double g, float x, float y, float z,
                     double Lx,double Ly,double Lz) nogil:

  cdef double Rx = x-0.5*Lx
  cdef double Ry = y-0.5*Ly
  cdef double Rz = z-0.5*Lz
  cdef double R = sqrt(Rx**2+Ry**2+Rz**2)
  cdef double rc=0.12*0.4
  cdef double gam=5./3

  if R < rc*sqrt(3):
      return 1./gam * log(1+(R/rc)**2)
  else:
      return 1./gam * log(4)
      #        return (-1.5*sqrt(3)*rc/R + log(4)+1.5) / gam

  # return g*y
