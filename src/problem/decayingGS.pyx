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

from src.coord cimport ind2crd
from src.utils import gen_VGS

cdef void set_user_problem(Domain dom,
    np.ndarray[real, ndim=3] rho,
    np.ndarray[real, ndim=4] v,
    np.ndarray[real, ndim=3] p,
    np.ndarray[real, ndim=3] ppd,
    np.ndarray[real, ndim=3] pe,
    np.ndarray[real, ndim=3] pscal,
    np.ndarray[real, ndim=4] B):

  cdef:

    ints Nx = dom.N[0]
    ints Ny = dom.N[1]
    ints Nz = dom.N[2]
    double Lx = dom.L[0]
    double Ly = dom.L[1]
    double Lz = dom.L[2]
    double dx = dom.dx
    Parameters *pm = &(dom.params)

    ints i,j,m,k
    double x=0,y=0,z=0

  #-------------------------------

  for i in range(Nx):
    for j in range(Ny):
      for m in range(Nz):

        ind2crd(dom, i,j,m, &x,&y,&z)

        rho[i,j,m] = 1.
        p[i,j,m] = 0.5*pm.beta
        IF BRAG1 or BRAG2:
          ppd[i,j,m] = p[i,j,m]

        pscal[i,j,m] = 1. + pm.tvar * cos(2*M_PI / Ly * y)

        B[i,j,m,0] = 1.
        if i==Nx-1: B[Nx,j,m,0] = 1.

  if pm.vrms != 0.:
    v1 = np.load('VGS.npy')

  # print type(v)

    # v1 = gen_VGS(Nx,Ny,Nz, vrms=pm.vrms,
    #             Linj_cells=<int>(pm.lmax/dx),
    #             Lmin_cells=<int>(pm.lmin/dx),
    #             nt=dom.nt)

  for i in range(Nx):
    for j in range(Ny):
      for m in range(Nz):
        for k in range(3):
          v[i,j,m,k] = v1[i,j,m,k]

  dom.BCFlag_x1=0
  dom.BCFlag_x2=0
  dom.BCFlag_y1=0
  dom.BCFlag_y2=0
  dom.BCFlag_z1=0
  dom.BCFlag_z2=0



#========================================

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
                     double[::1] L) nogil:

  cdef double Rx = x-0.5*L[0]
  cdef double Ry = y-0.5*L[1]
  cdef double Rz = z-0.5*L[2]
  cdef double R = sqrt(Rx**2+Ry**2+Rz**2)
  cdef double rc=0.12*0.4
  cdef double gam=5./3

  # if R < rc*sqrt(3):
  #     return 1./gam * log(1+(R/rc)**2)
  # else:
  #     return 1./gam * log(4)
  #     #        return (-1.5*sqrt(3)*rc/R + log(4)+1.5) / gam

  return g*y
