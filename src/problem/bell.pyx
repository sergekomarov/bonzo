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
    np.ndarray[real, ndim=3] rho,
    np.ndarray[real, ndim=4] v,
    np.ndarray[real, ndim=3] p,
    np.ndarray[real, ndim=3] ppd,
    np.ndarray[real, ndim=3] pe,
    np.ndarray[real, ndim=3] pscal,
    np.ndarray[real, ndim=4] B):

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

    int ppc1, Nxp,Nyp, nx,ny,n
    double gdrift
    real xp,yp,zp,gp, up,vp,wp, dxp

    double B0,rho0,ua, k0,th, dB, eps, Jcr

  #------------------------

  rho0 = 1.
  B0 = 1.
  dB = 1e-3
  ua = B0/sqrt(rho0)

  th = M_PI/6
  eps = sin(th)  # ua/udrift
  pm.udrift = ua/eps

  k0 = 2*M_PI/Lx
  Jcr = 2*B0*k0  # from dispersion relation

  pm.sol = 100*pm.udrift
  pm.qomc = 1e-6*k0
  pm.ncr = dom.ppc * Jcr / (pm.qomc * pm.udrift)

  for i in range(Nx):
    for j in range(Ny):
      for m in range(Nz):

        x = ind2crdx_u(i,dx)
        y = ind2crdy_u(j,dx)
        z = ind2crdz_u(m,dx)

        rho[i,j,m] = rho0
        B[i,j,m,0] = B0   # gives u_A=1

        B[i,j,m,1] = dB*cos(k0*x)
        B[i,j,m,2] = dB*sin(k0*x)

        v[i,j,m,1] =  dB*ua/B0*sin(k0*x-th)
        v[i,j,m,2] = -dB*ua/B0*cos(k0*x-th)

        p[i,j,m] = rho0

        if i==Nx-1: B[Nx,j,m,0] = B[0,j,m,0]
        if Ny>1 and j==Ny-1: B[i,Ny,m,1] = B[i,0,m,1]
        if Nz>1 and m==Nz-1: B[i,j,Nz,2] = B[i,j,0,2]

  gdrift = 1./sqrt(1.-(pm.udrift/pm.sol)**2)

  # ppc1 = <int>sqrt(dom.ppc)

  Nxp = dom.ppc*Nx
  dxp = dx/dom.ppc

  zp = 0.5*dx
  yp = 0.5*dx

  for n in range(Nxp):

    xp = (n+0.5)*dxp

    # using coordinates local to MPI block
    # use loc2glob_crd(dom, &xloc,&yloc,&zloc, &xglob,&yglob,&zglob)
    # to convert

    dom.prts[n].x = xp
    dom.prts[n].y = yp
    dom.prts[n].z = zp

    dom.prts[n].u = gdrift/pm.sol * pm.udrift
    dom.prts[n].v = 0
    dom.prts[n].w = 0
    dom.prts[n].g = gdrift

    IF MPI:
      dom.prts[n].id = dom.blocks.id * dom.Np + n
    ELSE:
      dom.prts[n].id = n

  dom.BCFlag_x1=0
  dom.BCFlag_x2=0
  dom.BCFlag_y1=0
  dom.BCFlag_y2=0
  dom.BCFlag_z1=0
  dom.BCFlag_z2=0

#----------------------------------------

cdef void set_user_output(Domain dom):
  return


cdef void x1_bc_mhd_user(Domain dom) nogil:
  return
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


cdef double grav_pot(double g, real x, real y, real z,
                     double Lx,double Ly,double Lz) nogil:

  cdef double Rx = x-0.5*Lx
  cdef double Ry = y-0.5*Ly
  cdef double Rz = z-0.5*Lz
  cdef double R = sqrt(Rx**2+Ry**2+Rz**2)
  cdef double rc=0.12*0.4
  cdef double gam=5./3

  # if R < rc*sqrt(3):
  #     return 1./gam * log(1+(R/rc)**2)
  # else:
  #     return 1./gam * log(4)
  #     #        return (-1.5*sqrt(3)*rc/R + log(4)+1.5) / gam

  return g*y
