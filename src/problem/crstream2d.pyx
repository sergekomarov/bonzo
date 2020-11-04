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

from src.coord cimport ind2crdx_u, ind2crdy_u, ind2crdz_u#, loc2glob_crd
from src.particle.init_particle cimport init_maxw_table, init_powlaw_table, distr_prt

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

    int ppc1, Nxp,Nyp, nx,ny,n
    double gdrift
    uint pdf_sz = 500
    float xp,yp,zp,gp, up,vp,wp, dxp

    # arrays used to generate particle distribution
    float[::1] gamma_table = np.zeros(pdf_sz, dtype=np.float32)
    float[::1] pdf_table = np.zeros(pdf_sz, dtype=np.float32)

  #------------------------

  for i in range(Nx):
    for j in range(Ny):
      for m in range(Nz):

        rho[i,j,m] = 1.
        B[i,j,m,0] = pm.bpar
        p[i,j,m] = 0.5 * pm.beta
        if i==Nx-1: B[Nx,j,m,0] = B[0,j,m,0]

  #sol=10 #cf=sqrt(2*beta*gamma)
  gdrift = 1./sqrt(1.-(pm.udrift/pm.sol)**2)

  init_maxw_table(gamma_table, pdf_table, pm.delgam)
  # np.save('gamma_table.npy', np.asarray(gamma_table))
  # np.save('pdf_table.npy', np.asarray(pdf_table))

  ppc1 = <int>sqrt(dom.ppc)

  Nxp = ppc1*Nx
  Nyp = ppc1*Ny
  dxp = dx/ppc1

  zp = 0.5*Nz*dx
  for nx in range(Nxp):
    xp = (nx+0.5)*dxp
    for ny in range(Nyp):
      yp = (ny+0.5)*dxp

      n = Nyp * nx + ny

      # using coordinates local to MPI block
      # use loc2glob_crd(dom, &xloc,&yloc,&zloc, &xglob,&yglob,&zglob)
      # to convert

      dom.prts[n].x = xp
      dom.prts[n].y = yp
      dom.prts[n].z = zp

      distr_prt(&up, &vp, &wp, &gp,
                gamma_table, pdf_table,
                gdrift, pm.sol)
      dom.prts[n].u = up
      dom.prts[n].v = vp
      dom.prts[n].w = wp
      dom.prts[n].g = gp

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


#------------------------------------------------------------

# Set user history variables and particle selection function.

cdef void set_user_output(Domain dom):

  # set up to NHST_U (8 by default) function pointers
  dom.output.hst_funcs_u[0] = hst_var1
  dom.output.hst_names_u[0] = "B2h"

  IF PIC:
    dom.output.prt_sel_func = select_particle


# define function "double (Domain)" to calculate history variable

cdef double hst_var1(Domain dom) nogil:

  cdef int i,j,m
  cdef float4d U = dom.U
  cdef double B2h,B2h1 = 0.

  for i in range(dom.i1,dom.i2+1):
    for j in range(dom.j1,dom.j2+1):
      for m in range(dom.m1,dom.m2+1):
        B2h += U[i,j,m,BX]**2 + U[i,j,m,BY]**2 + U[i,j,m,BZ]**2

  IF MPI:
    B2h1 = B2h
    mpi.Allreduce(&B2h1, &B2h, op=mpi.SUM)
    B2h /= mpi.COMM_WORLD.Get_size()

  return B2h / (2*dom.Nx*dom.Ny*dom.Nz)

IF PIC:

  cdef int select_particle(Particle p) nogil:

    if p.id % 100 == 0:
      return 1
    else:
      return 0

#-------------------------------------------------------------------------------


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
