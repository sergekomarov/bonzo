#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True
#cython: language_level=2
# -*- coding: utf-8 -*-

import numpy as np
cimport numpy as np

from libc.math cimport M_PI, sqrt,floor,ceil,log,exp,sin,cos,pow,fabs,fmin,fmax
from libc.stdlib cimport rand, RAND_MAX, srand

from src.coord cimport ind2crdx_u, ind2crdy_u, ind2crdz_u
from src.utils import gen_divfree3dvec

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
    double x,y,z

  #-------------------------

  for i in range(Nx):
    for j in range(Ny):
      for m in range(Nz):

        x = ind2crdx_u(i,dx)

        rho[i,j,m] = 1 + pm.tvar*cos(2*M_PI*x)
        p[i,j,m] = 1.
        IF BRAG1 or BRAG2:
          ppd[i,j,m] = p[i,j,m]

        B[i,j,m,0] = 1./sqrt(0.5*pm.beta)
        if i==Nx-1: B[Nx,j,m,0] = 1./sqrt(0.5*pm.beta)

  if pm.vrms != 0.:
    # v = gen_divfree3dvec(Nx,Ny,Nz,dx, rms=pm.vrms,
    #             s=pm.s, kmin=pm.kmin, kmax=pm.kmax,
    #             nt=dom.nt)
  pm.g = 0

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


cdef double grav_pot(double g, float x, float y, float z,
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
