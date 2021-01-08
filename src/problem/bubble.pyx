#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True
# -*- coding: utf-8 -*-

import numpy as np
cimport numpy as np

from libc.math cimport M_PI, sqrt,floor,ceil,log,exp,sin,cos,pow,fabs,fmin,fmax
from libc.stdlib cimport rand, RAND_MAX, srand

from src.particle.init_particle cimport init_maxw_table, init_powlaw_table, distr_prt

cdef void set_user_problem(Domain dom,
    np.ndarray[float, ndim=3] rho,
    np.ndarray[float, ndim=4] v,
    np.ndarray[float, ndim=3] ppl,
    np.ndarray[float, ndim=3] ppd,
    np.ndarray[float, ndim=3] pe,
    np.ndarray[float, ndim=3] pscal,
    np.ndarray[float, ndim=4] B):

  cdef uint Nx = dom.Nx
  cdef uint Ny = dom.Ny
  cdef uint Nz = dom.Nz
  cdef double Lx = dom.Lx
  cdef double Ly = dom.Ly
  cdef double Lz = dom.Lz
  cdef double dx = dom.dx
  cdef uint ng = dom.ng
  cdef uint i1=dom.i1, i2=dom.i2
  cdef uint j1=dom.j1, j2=dom.j2
  cdef uint m1=dom.m1, m2=dom.m2
  cdef uint Nx1 = Nx+1
  cdef uint Ny1 = Ny+1 if Ny>1 else 1
  cdef uint Nz1 = Nz+1 if Nz>1 else 1
  cdef double Lxi = 1./Lx
  cdef double Lyi = 1./Ly if Ly>0 else 0.
  cdef double Lzi = 1./Lz if Lz>0 else 0.

  cdef Parameters *pm = &(dom.params)

  cdef int i,j,m, ig,jg,mg
  cdef int im,ip, jm,jp, mm,mp

  # oblique shock tube
  cdef double x1,x2,x3,A3
  cdef double v1,v2,v3,B1,B2L,B2R,B3,B2

  # particles
  cdef int ppc1, Nxp,Nyp, nx,ny,n
  cdef double gdrift
  cdef uint pdf_sz = 500
  cdef float xp,yp,zp,gp, up,vp,wp, dxp

  cdef double Em,Ek

  # cold front
  cdef double rhoc,rc,rhohot,Brms, rho1,rho2
  cdef double R,bt

  rho1=1.
  rho2=2.

  cdef np.ndarray[float, ndim=2] Az = np.zeros((Nx1,Ny1), dtype=np.float32)
  cdef np.ndarray[float, ndim=4] A = np.zeros((Nx1,Ny1,Nz1,3), dtype=np.float32)
  cdef np.ndarray[float, ndim=1] Ev = np.zeros(8, dtype=np.float32)

  # arrays used to generate particle distribution
  cdef float[::1] gamma_table = np.zeros(pdf_sz, dtype=np.float32)
  cdef float[::1] pdf_table = np.zeros(pdf_sz, dtype=np.float32)

  #KH instability
  if pm.problem == KH_HD or pm.problem == KH_MHD:

    for i in range(Nx):
      for j in range(Ny):
        for m in range(Nz):

          R = sqrt((j+0.5-0.5*Ny)**2+(m+0.5-0.5*Nz)**2)
          if  R > Ny/4:
            v[i,j,m,0] = 0.5
            rho[i,j,m] = rho1
          else:
            v[i,j,m,0] = -0.5
            rho[i,j,m] = rho2

          v[i,j,m,0] += 0.01* np.random.normal(0,1)

          p[i,j,m] = 2.5
          IF BRAG1 or BRAG2:
            ppd[i,j,m] = p[i,j,m]
          IF BRAG2:
            ppl[i,j,m] = p[i,j,m]

          if pm.problem == KH_MHD:

            B[i,j,m,0] = 1./sqrt(0.5*pm.beta)
            if i==Nx-1: B[Nx,j,m,0] = B[Nx-1,j,m,0]

    pm.g = 0

    dom.BCFlag_x1=0
    dom.BCFlag_x2=0
    dom.BCFlag_y1=0
    dom.BCFlag_y2=0
    dom.BCFlag_z1=0
    dom.BCFlag_z2=0

  #RT instability

  elif pm.problem == RT_HD or pm.problem == RT_MHD:

    for i in range(Nx):
      for j in range(Ny):
        for m in range(Nz):

          v[i,j,m,1] += 0.01* (1+cos(2*M_PI*(i+0.5-0.5*Nx)*dx*Lxi)) *\
                              (1+cos(2*M_PI*(j+0.5-0.5*Ny)*dx*Lyi)) *\
                              (1+cos(2*M_PI*(m+0.5-0.5*Nz)*dx*Lzi)) / 8

          if j < Ny/2:
            rho[i,j,m] = rho1
          else:
            rho[i,j,m] = rho2

          p[i,j,m] = 2.5 - rho[i,j,m]*pm.g*(j+0.5-0.5*Ny)*dx
          IF BRAG1 or BRAG2:
            ppd[i,j,m] = p[i,j,m]
          IF BRAG2:
            ppl[i,j,m] = p[i,j,m]

          if pm.problem == RT_MHD:
            B[i,j,m,0] = 1./sqrt(0.5*pm.beta)
            if i==Nx-1: B[Nx,j,m,0] = B[Nx-1,j,m,0]


    dom.BCFlag_x1=0
    dom.BCFlag_x2=0
    dom.BCFlag_y1=2
    dom.BCFlag_y2=2
    dom.BCFlag_z1=0
    dom.BCFlag_z2=0

  #blast
  elif pm.problem == blast_HD or pm.problem == blast_MHD:

    for i in range(Nx):
      for j in range(Ny):
        for m in range(Nz):

          rho[i,j,m]=1.

          R = sqrt((i+0.5-0.5*Nx)**2 + (j+0.5-0.5*Ny)**2 +
                   (m+0.5-0.5*Nz)**2)*dx
          if R < 0.1: p[i,j,m] = 10
          else: p[i,j,m] = 0.1

          IF BRAG1 or BRAG2:
            ppd[i,j,m] = p[i,j,m]
          IF BRAG2:
            ppl[i,j,m] = p[i,j,m]

          if pm.problem == blast_MHD:
            bt = 1./sqrt(1.5*pm.beta)
            B[i,j,m,0] = bt
            B[i,j,m,1] = bt
            B[i,j,m,2] = bt
            if i==Nx-1: B[Nx,j,m,0] = bt
            if j==Ny-1 and Ny>1: B[i,Ny,m,1] = bt
            if m==Nz-1 and Nz>1: B[i,j,Nz,2] = bt

    pm.g = 0

    dom.BCFlag_x1=0
    dom.BCFlag_x2=0
    dom.BCFlag_y1=0
    dom.BCFlag_y2=0
    dom.BCFlag_z1=0
    dom.BCFlag_z2=0

  elif pm.problem == mixing:

    for i in range(Nx):
      for j in range(Ny):
        for m in range(Nz):

          rho[i,j,m] = 1 + pm.tvar*cos(2*M_PI*(i+0.5)*dx)
          p[i,j,m] = 1.
          IF BRAG1 or BRAG2:
            ppd[i,j,m] = p[i,j,m]
          IF BRAG2:
            ppl[i,j,m] = p[i,j,m]

          B[i,j,m,0] = 1./sqrt(0.5*pm.beta)
          if i==Nx-1: B[Nx,j,m,0] = 1./sqrt(0.5*pm.beta)

    # if pm.vrms != 0.:
    #   v = utils.gen_divfree3dvec(Nx,Ny,Nz,dx, rms=pm.vrms,
    #               s=pm.s, kmin=pm.kmin, kmax=pm.kmax,
    #               nt=dom.nt)
    pm.g = 0

    dom.BCFlag_x1=0
    dom.BCFlag_x2=0
    dom.BCFlag_y1=0
    dom.BCFlag_y2=0
    dom.BCFlag_z1=0
    dom.BCFlag_z2=0

  elif pm.problem == OTVortex:

    for i in range(Nx+1):
      for j in range(Ny+1):

        Az[i,j] = cos(4*M_PI*i*dx)/(4*M_PI) + cos(2*M_PI*j*dx)/(2*M_PI)

    for i in range(Nx):
      for j in range(Ny):
        for m in range(Nz):

          rho[i,j,m] = 25./9
          p[i,j,m] = 5./3
          IF BRAG1 or BRAG2:
            ppd[i,j,m] = p[i,j,m]
          IF BRAG2:
            ppl[i,j,m] = p[i,j,m]

          v[i,j,m,0] = -sin(2*M_PI*(j+0.5)*dx)
          v[i,j,m,1] =  sin(2*M_PI*(i+0.5)*dx)

          B[i,j,m,0] =  (Az[i,j+1] - Az[i,j])/dx / sqrt(pm.beta)
          B[i,j,m,1] = -(Az[i+1,j] - Az[i,j])/dx / sqrt(pm.beta)
          if i==Nx-1: B[Nx,j,m,0] =  (Az[Nx,j+1] - Az[Nx,j])/dx / sqrt(pm.beta)
          if j==Ny-1: B[i,Ny,m,1] = -(Az[i+1,Ny] - Az[i,Ny])/dx / sqrt(pm.beta)

    pm.gam = 5./3
    pm.g = 0

    dom.BCFlag_x1=0
    dom.BCFlag_x2=0
    dom.BCFlag_y1=0
    dom.BCFlag_y2=0
    dom.BCFlag_z1=0
    dom.BCFlag_z2=0

  elif pm.problem == FLadv2d:

    for i in range(Nx+1):
      for j in range(Ny+1):

        R = sqrt((i-0.5*Nx)**2+(j-0.5*Ny)**2)*dx

        #vector potential is defined at zone corners
        #i->i-1/2, j->j-1/2
        Az[i,j] = fmax(1e-3*(0.15-R), 0)

    for i in range(Nx):
      for j in range(Ny):
        for m in range(Nz):

          v[i,j,m,0] = 0.5*sqrt(3)
          v[i,j,m,1] = 0.5
          v[i,j,m,2] = 0.5
          rho[i,j,m] = 1
          p[i,j,m] = 1

          B[i,j,m,0] =  (Az[i,j+1] - Az[i,j])/dx
          B[i,j,m,1] = -(Az[i+1,j] - Az[i,j])/dx
          if i==Nx-1: B[Nx,j,m,0] =  (Az[Nx,j+1] - Az[Nx,j])/dx
          if j==Ny-1: B[i,Ny,m,1] = -(Az[i+1,Ny] - Az[i,Ny])/dx

    pm.gam = 5./3
    pm.g = 0

    dom.BCFlag_x1=0
    dom.BCFlag_x2=0
    dom.BCFlag_y1=0
    dom.BCFlag_y2=0
    dom.BCFlag_z1=0
    dom.BCFlag_z2=0

  elif pm.problem == linwave:

    for i in range(Nx):
      for j in range(Ny):
        for m in range(Nz):

          #Alfven wave
          rho[i,j,m]=1
          p[i,j,m]=3./5
          v[i,j,m,2] = -1e-6*cos(2*M_PI*(i+0.5)*dx)
#                        v[i,j,m,0]= 1.
          B[i,j,m,0]=1.
          B[i,j,m,1]=3./2
          B[i,j,m,2] =  1e-6*cos(2*M_PI*(i+0.5)*dx)
          if i==Nx-1: B[Nx,j,m,0] = B[Nx-1,j,m,0]
          if j==Ny-1 and Ny>1: B[i,Ny,m,1] = B[i,Ny-1,m,1]
          if m==Nz-1 and Nz>1: B[i,j,Nz,2] = B[i,j,Nz-1,2]

    pm.gam = 5./3
    pm.g = 0

    dom.BCFlag_x1=0
    dom.BCFlag_x2=0
    dom.BCFlag_y1=0
    dom.BCFlag_y2=0
    dom.BCFlag_z1=0
    dom.BCFlag_z2=0

  elif pm.problem == FLadv3d:

      #Lx=Ly=1, Lz=2 have to be set for this test

    for i in range(Nx+1):
      for j in range(Ny+1):
        for m in range(Nz+1):

          x1 = (2*(i-0.5*Nx) + (m-0.5*Nz))*dx/sqrt(5)
          x2 = (j-0.5*Ny)*dx
          x3 = (-(i-0.5*Nx) + 2*(m-0.5*Nz))*dx/sqrt(5)

          if x1 > 1./sqrt(5): x1 -= 2./sqrt(5)
          if x1 < -1./sqrt(5): x1 += 2./sqrt(5)

          R = sqrt(x1**2+x2**2)
          A3 = fmax(1e-3*(0.3-R), 0)
          A[i,j,m,0] = -A3/sqrt(5)
          A[i,j,m,1] = 0
          A[i,j,m,2] = 2*A3/sqrt(5)

    for i in range(Nx):
      for j in range(Ny):
        for m in range(Nz):

          v[i,j,m,0] = 1
          v[i,j,m,1] = 1
          v[i,j,m,2] = 2
          rho[i,j,m] = 1
          p[i,j,m] = 1

          B[i,j,m,0] =  (A[i,j+1,m,2] - A[i,j,m,2] - A[i,j,m+1,1] + A[i,j,m,1])/dx
          B[i,j,m,1] = -(A[i+1,j,m,2] - A[i,j,m,2] - A[i,j,m+1,0] + A[i,j,m,0])/dx
          B[i,j,m,2] =  (A[i+1,j,m,1] - A[i,j,m,1] - A[i,j+1,m,0] + A[i,j,m,0])/dx
          if i==Nx-1:
            B[Nx,j,m,0] =  (A[Nx,j+1,m,2] - A[Nx,j,m,2] - A[Nx,j,m+1,1] + A[Nx,j,m,1])/dx
          if j==Ny-1:
            B[i,Ny,m,1] = -(A[i+1,Ny,m,2] - A[i,Ny,m,2] - A[i,Ny,m+1,0] + A[i,Ny,m,0])/dx
          if m==Nz-1:
            B[i,j,Nz,2] =  (A[i+1,j,Nz,1] - A[i,j,Nz,1] - A[i,j+1,Nz,0] + A[i,j,Nz,0])/dx

    pm.gam = 5./3
    pm.g = 0

    dom.BCFlag_x1=0
    dom.BCFlag_x2=0
    dom.BCFlag_y1=0
    dom.BCFlag_y2=0
    dom.BCFlag_z1=0
    dom.BCFlag_z2=0

  elif pm.problem == ST1D:

    for i in range(Nx):
      for j in range(Ny):
        for m in range(Nz):
          if i <= Nx/2:
            rho[i,j,m] = 1.08
            v[i,j,m,0] = 1.2
            v[i,j,m,1] = 0.01
            v[i,j,m,2] = 0.5
            p[i,j,m] = 0.95
            B[i,j,m,0] = 2./sqrt(4*M_PI)
            B[i,j,m,1] = 3.6/sqrt(4*M_PI)
            B[i,j,m,2] = 2./sqrt(4*M_PI)
          else:
            rho[i,j,m] = 1.
            v[i,j,m,0] = 0
            v[i,j,m,1] = 0
            v[i,j,m,2] = 0
            p[i,j,m] = 1.
            B[i,j,m,0] = 2./sqrt(4*M_PI)
            B[i,j,m,1] = 4./sqrt(4*M_PI)
            B[i,j,m,2] = 2./sqrt(4*M_PI)

           # if i <= Nx/2:
           #     rho[i,j,m] = 1.
           #     v[i,j,m,0] = 0
           #     v[i,j,m,1] = 0
           #     v[i,j,m,2] =0
           #     p[i,j,m] = 1
           #     B[i,j,m,0] = 0.75
           #     B[i,j,m,1] = 1
           #     B[i,j,m,2] = 0
           # else:
           #     rho[i,j,m] = 0.125
           #     v[i,j,m,0] = 0
           #     v[i,j,m,1] = 0
           #     v[i,j,m,2] = 0
           #     p[i,j,m] = 0.1
           #     B[i,j,m,0] = 0.75
           #     B[i,j,m,1] = -1
           #     B[i,j,m,2] = 0
           #
           # if i <= Nx/2:
           #     rho[i,j,m] = 1.
           #     v[i,j,m,0] = 10
           #     v[i,j,m,1] = 0
           #     v[i,j,m,2] =0
           #     p[i,j,m] = 20
           #     B[i,j,m,0] = 5./sqrt(4*M_PI)
           #     B[i,j,m,1] = 5./sqrt(4*M_PI)
           #     B[i,j,m,2] = 0
           # else:
           #     rho[i,j,m] = 1.
           #     v[i,j,m,0] = -10
           #     v[i,j,m,1] = 0
           #     v[i,j,m,2] = 0
           #     p[i,j,m] = 1
           #     B[i,j,m,0] = 5./sqrt(4*M_PI)
           #     B[i,j,m,1] = 5./sqrt(4*M_PI)
           #     B[i,j,m,2] = 0

          if i==Nx-1: B[Nx,j,m,0] = B[Nx-1,j,m,0]
          if j==Ny-1 and Ny>1: B[i,Ny,m,1] = B[i,Ny-1,m,1]
          if m==Nz-1 and Nz>1: B[i,j,Nz,2] = B[i,j,Nz-1,2]

    pm.gam = 5./3
    pm.g=0

    dom.BCFlag_x1=1
    dom.BCFlag_x2=1
    dom.BCFlag_y1=0
    dom.BCFlag_y2=0
    dom.BCFlag_z1=0
    dom.BCFlag_z2=0

  elif pm.problem == ST2D:

    B1=2./sqrt(4*M_PI)
    B2L=3.6/sqrt(4*M_PI)
    B2R=4./sqrt(4*M_PI)
    B3=2./sqrt(4*M_PI)

    for i in range(Nx+1):
      for j in range(Ny+1):

        x1 = (i-Nx/2 + 2*(j-Ny/2))*dx/sqrt(5)
        x2 = (-2*(i-Nx/2) + j-Ny/2)*dx/sqrt(5)

        if x1 < 0:
          Az[i,j] = B1*x2 - B2L*x1
        else:
          Az[i,j] = B1*x2 - B2R*x1

    for i in range(Nx):
      for j in range(Ny):
        for m in range(Nz):

          if i < Nx/2 - 2*(j-Ny/2):# - 4*(m-Nz/2):

            rho[i,j,m] = 1.08
            p[i,j,m] = 0.95

            v1=1.2
            v2=0.01
            v3=0.5

            v[i,j,m,0] = (v1-2*v2)/sqrt(5)
            v[i,j,m,1] = (2*v1+v2)/sqrt(5)
            v[i,j,m,2] = v3

           # B[i,j,m,0] = (Bx-2*ByL)/sqrt(5)
           # B[i,j,m,1] = (2*Bx+ByL)/sqrt(5)
           # B[i,j,m,2] = Bz

          else:

            rho[i,j,m] = 1.
            p[i,j,m] = 1.

            v[i,j,m,0] = 0
            v[i,j,m,1] = 0
            v[i,j,m,2] = 0

             # B[i,j,m,0] = (Bx-2*ByR)/sqrt(5)
             # B[i,j,m,1] = (2*Bx+ByR)/sqrt(5)
             # B[i,j,m,2] = Bz

          B[i,j,m,0] =  (Az[i,j+1] - Az[i,j])/dx
          B[i,j,m,1] = -(Az[i+1,j]-Az[i,j])/dx
          B[i,j,m,2] = B3
          if i==Nx-1: B[Nx,j,m,0] =  (Az[Nx,j+1] - Az[Nx,j])/dx
          if j==Ny-1: B[i,Ny,m,1] = -(Az[i+1,Ny] - Az[i,Ny])/dx
          if m==Nz-1 and Nz>1: B[i,j,Nz,2] = B3

    pm.gam = 5./3
    pm.g=0

    dom.BCFlag_x1=3
    dom.BCFlag_x2=3
    dom.BCFlag_y1=3
    dom.BCFlag_y2=3
    dom.BCFlag_z1=0
    dom.BCFlag_z2=0


  elif pm.problem == coldfront:

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

          R = sqrt(((i+0.5)*dx-0.5*Lx)**2 +
                   ((j+0.5)*dx-0.5*Ly)**2 +
                   ((m+0.5)*dx-0.5*Lz)**2)+1e-20

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

    dom.aux.B0 = B.copy()

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

    #gravitational potential is defined separately below

  elif pm.problem == CRstream2d:

    for i in range(Nx):
      for j in range(Ny):
        for m in range(Nz):

          rho[i,j,m] = 1.
          B[i,j,m,0] = pm.bpar
          p[i,j,m] = 0.5 * pm.beta
          if i==Nx-1: B[Nx,j,m,0] = B[Nx-1,j,m,0]

    #sol=10 #cf=sqrt(2*beta*gamma)
    gdrift = 1./sqrt(1.-(pm.vdrift/pm.sol)**2)

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

  elif pm.problem == BragAlfven1d:

    for i in range(Nx):
      for j in range(Ny):
        for m in range(Nz):

          rho[i,j,m] = 1.
          B[i,j,m,0] = 1.
          p[i,j,m] = 0.5 * pm.beta
          ppd[i,j,m] = p[i,j,m] #0.5 * p[i,j,m]  #ion pressure twice less than total p.
          ppl[i,j,m] = p[i,j,m]

          B[i,j,m,2] = -0.5*cos(2*M_PI*(i+0.5)*dx)

          if i==Nx-1: B[Nx,j,m,0] = B[Nx-1,j,m,0]
          if m==Nz-1 and Nz>1: B[i,j,Nz,2] = B[i,j,Nz-1,2]

  elif pm.problem == BragSound1d:

    for i in range(Nx):
      for j in range(Ny):
        for m in range(Nz):

          rho[i,j,m] = 1.
          B[i,j,m,0] = 1.
          p[i,j,m] = 0.5 * pm.beta
          ppd[i,j,m] = p[i,j,m] #0.5 * p[i,j,m]  #ion pressure twice less than total p.
          ppl[i,j,m] = p[i,j,m]

          B[i,j,m,2] = -0.5*cos(2*M_PI*(i+0.5)*dx)

          if i==Nx-1: B[Nx,j,m,0] = B[Nx-1,j,m,0]
          if m==Nz-1 and Nz>1: B[i,j,Nz,2] = B[i,j,Nz-1,2]


 # elif pm.problem == bubble:
 #
 #    rc = 0.12
 #    pm.gam=5./3
 #
 #    #==================================================================
 #    #  L0 = 2.5 Mpc
 #    #  v0 = 1000 km/s
 #    #  T0 = 2.4*10^9 yr
 #    #  Temp0 = 6.4 KeV
 #    #  rho0 = 5*10^(-27) g cm^(-3) -> n0=3*10^(-3) cm^(-3)
 #    #  kappa0_Sp = 0.04
 #    #==================================================================
 #
 #    for i in range(Nx):
 #      for j in range(Ny):
 #        for m in range(Nz):
 #
 #          R = sqrt(((i+0.5)*dx-0.5)**2 +
 #                   ((j+0.5)*dx-0.5*Ly)**2 +
 #                   ((m+0.5)*dx-0.5*Lz)**2)
 #
 #          if R < rc:
 #            v[i,j,m,0] = 0.
 #          else:
 #            v[i,j,m,0] = 0.5
 #
 #          v[i,j,m,1] = 0
 #          v[i,j,m,2] = 0
 #          rho[i,j,m] = 1
 #
 #          i1=i+ng
 #          j1=j+ng
 #          m1=m+ng
 #
 #          p[i,j,m] = 1
 #
 #       # B[i,j,m,0] = 0
 #       # B[i,j,m,1] = sqrt(2./pm.beta)
 #       # B[i,j,m,2] = 0
 #
 #    Brms = sqrt(2./pm.beta)
 #    # B[:Nx,:Ny,:Nz,:] = utils.gen_divfree3dvec(Nx,Ny,Nz,dx, Brms, s=11./3,
 #    #                             kmin=16./Ny, kmax=24./Ny, nt=dom.nt)
 #    for i in range(Nx):
 #      for j in range(Ny):
 #        for m in range(Nz):
 #          B[Nx,j,m,0] = B[0,j,m,0]
 #          B[i,Ny,m,1] = B[i,0,m,1]
 #          B[i,j,Nz,2] = B[i,j,0,2]
 #
 #    dom.aux.B0 = B.copy()
 #
 #     # rho[i,j]=1
 #     # p[i,j]=1
 #     # pm.gam = 5./3
 #     # B[i,j,0] = -1./sqrt(0.5*pm.beta)
 #     # B[i,j,1] = 0
 #     # if i==Nx-1: B[Nx,j,0] = B[Nx-1,j,m,0]
 #     # if j==Ny-1: B[i,Ny,1] = B[i,Ny-1,m,1]
 #     # v[i,j,0] = 0
 #     # v[i,j,1] = 1
 #     #
 #     # if fabs((i+0.5)*dx - 0.5) <= 0.1 and j*dx>=0.25:
 #     #      v[i,j,1] = 0
 #
 #    dom.BCFlag_x1=3
 #    dom.BCFlag_x2=3
 #    dom.BCFlag_y1=3
 #    dom.BCFlag_y2=3
 #    dom.BCFlag_z1=3
 #    dom.BCFlag_z2=3

  else:

    for i in range(Nx):
      for j in range(Ny):
        for m in range(Nz):

          rho[i,j,m] = 1.
          p[i,j,m] = 1.

          B[i,j,m,1] = 1.
          if j==Ny-1: B[i,Ny,m,1] = B[i,Ny-1,m,1]


cdef void x1_bc_mhd_user(Domain dom) nogil:

  # ghost cells for U5,U6,U7 (magnetic field at cell centers) are
  # assigned separately by interpolation of B from cell interfaces
  # after setting B ghost cells

  # important to set B ghost cells first!

  cdef uint Nx = dom.Nx
  cdef int i1 = dom.i1, i2 = dom.i2
  cdef int j1 = dom.j1, j2 = dom.j2
  cdef int m1 = dom.m1, m2 = dom.m2
  cdef int ng = dom.ng

  cdef Parameters pm = dom.params
  cdef double gam = pm.gam
  cdef double beta = pm.beta

  cdef int i,j,m,k,g, j21,m21

  cdef double v1,v2,v3, B1,B2L,B2R,B3

  cdef double R, rhoc,rhohot
  cdef int p, p1
  cdef double di,fr

  if pm.problem == ST2D:

    B1=2./sqrt(4*M_PI)
    B2L=3.6/sqrt(4*M_PI)
    B3=2./sqrt(4*M_PI)

    for m in range(m1, m2+1):
      for j in range(j1, j2+1):
        for g in range(ng):

          dom.U[g, j,m,RHO] = 1.08

          v1=1.2
          v2=0.01
          v3=0.5

          dom.U[g, j,m,MX] = dom.U[g, j,m,RHO]*(v1-2*v2)/sqrt(5)
          dom.U[g, j,m,MY] = dom.U[g, j,m,RHO]*(2*v1+v2)/sqrt(5)
          dom.U[g, j,m,MZ] = dom.U[g, j,m,RHO]*v3

          dom.U[g, j,m,BX] = (B1-2*B2L)/sqrt(5)
          dom.U[g, j,m,BY] = (2*B1+B2L)/sqrt(5)
          dom.U[g, j,m,BZ] = B3

          dom.U[g, j,m,EN] =\
              0.5*(dom.U[g, j,m,MX]**2 + dom.U[g, j,m,MY]**2 + dom.U[g, j,m,MZ]**2)/dom.U[g, j,m,RHO] +\
              0.5*(dom.U[g, j,m,BX]**2 + dom.U[g, j,m,BY]**2 + dom.U[g, j,m,BZ]**2) +\
              0.95/(gam-1)

      for j in range(j1, j2+1):
        for g in range(1,ng):
          dom.B[g, j,m,0] = (B1-2*B2L)/sqrt(5)

      for j in range(j1, j2+2):
        for g in range(ng):
          dom.B[g, j,m,1] = (2*B1+B2L)/sqrt(5)

    for m in range(m1, m2+2):
      for j in range(j1, j2+1):
        for g in range(ng):
          dom.B[g, j,m,2] = B3

  elif pm.problem == coldfront:

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

         # dom.U[g,j,m,5] = 0
         # dom.U[g,j,m,6] = sqrt(2*0.25*rhoc/gam/beta)
         # dom.U[g,j,m,7] = 0
         # dom.U[g,j,m,4] = rhohot + 0.25*rhoc/gam/(gam-1) + 0.5*dom.U[g,j,m,6]**2
         #     +0.25*rhoc/gam/beta

          #!!!!!!!!!!!!!!!!!!!!!!!!!!
          dom.U[g,j,m,EN] = rhohot + 0.25*rhoc/gam/(gam-1) +\
              0.5*(dom.U[g,j,m,BX]**2 + dom.U[g,j,m,BY]**2 + dom.U[g,j,m,BZ]**2)

      for j in range(j1, j2+1):
        for g in range(ng-1):
          # B[ng-g-1,j,m,1] = sqrt(2 * 0.25*rhoc/gam/beta)
          if p+g >= Nx: p = p - (p+g)/Nx*Nx
          if p1+g >= Nx: p1 = p1 - (p1+g)/Nx*Nx
          dom.B[i1-g-1,j,m,0] = (1-fr)*dom.aux.B0[Nx-1-p-g,  j-j1, m-m1, 0] + \
                                    fr*dom.aux.B0[Nx-1-p1-g, j-j1, m-m1, 0]

      for j in range(j1, j2+2):
        for g in range(ng):
          # B[ng-g-1,j,m,0] = 0
          if p+g >= Nx: p = p - (p+g)/Nx*Nx
          if p1+g >= Nx: p1 = p1 - (p1+g)/Nx*Nx
          dom.B[i1-g-1,j,m,1] = (1-fr)*dom.aux.B0[Nx-1-p-g,  j-j1, m-m1, 1] + \
                                    fr*dom.aux.B0[Nx-1-p1-g, j-j1, m-m1, 1]

    for m in range(m1, m2+2):
      for j in range(j1, j2+1):
        for g in range(ng):
         # B[ng-g-1,j,m,2] = 0
         if p+g >= Nx: p = p - (p+g)/Nx*Nx
         if p1+g >= Nx: p1 = p1 - (p1+g)/Nx*Nx
         dom.B[i1-g-1,j,m,2] = (1-fr)*dom.aux.B0[Nx-1-p-g,  j-j1, m-m1, 2] + \
                                   fr*dom.aux.B0[Nx-1-p1-g, j-j1, m-m1, 2]

cdef void x2_bc_mhd_user(Domain dom) nogil:

  cdef int i1 = dom.i1, i2 = dom.i2
  cdef int j1 = dom.j1, j2 = dom.j2
  cdef int m1 = dom.m1, m2 = dom.m2
  cdef int ng = dom.ng

  cdef Parameters pm = dom.params
  cdef double gam = pm.gam
  cdef double beta = pm.beta

  cdef uint i,j,m,k,g
  cdef double v1,v2,v3, B1,B2L,B2R,B3

  if pm.problem == ST2D:

    B1=2./sqrt(4*M_PI)
    B2R=4./sqrt(4*M_PI)
    B3=2./sqrt(4*M_PI)

    for m in range(m1, m2+1):
      for i in range(i2+1, i2+ng+1):
        for j in range(j1, j2+1):

          dom.U[i,j,m,RHO] = 1.

          dom.U[i,j,m,MX] = 0
          dom.U[i,j,m,MY] = 0
          dom.U[i,j,m,MZ] = 0

          dom.U[i,j,m,BX] = (B1-2*B2R)/sqrt(5)
          dom.U[i,j,m,BY] = (2*B1+B2R)/sqrt(5)
          dom.U[i,j,m,BZ] = B3

          dom.U[i,j,m,EN] =\
              0.5*(dom.U[i,j,m,MX]**2 + dom.U[i,j,m,MY]**2 + dom.U[i,j,m,MZ]**2) / dom.U[i,j,m,0] +\
              0.5*(dom.U[i,j,m,BX]**2 + dom.U[i,j,m,BY]**2 + dom.U[i,j,m,BZ]**2) +\
                1./(gam-1)

      for j in range(j1, j2+1):
        for i in range(i2+2, i2+ng+1):
          dom.B[i,j,m,0] = (B1-2*B2R)/sqrt(5)

      for j in range(j1, j2+2):
        for i in range(i2+1, i2+ng+1):
          dom.B[i,j,m,1] = (2*B1+B2R)/sqrt(5)

    for m in range(m1, m2+2):
      for j in range(j1, j2+1):
        for i in range(i2+1, i2+ng+1):
          dom.B[i,j,m,2] = B3

cdef void y1_bc_mhd_user(Domain dom) nogil:

  cdef int i1 = dom.i1, i2 = dom.i2
  cdef int j1 = dom.j1, j2 = dom.j2
  cdef int m1 = dom.m1, m2 = dom.m2
  cdef int ng = dom.ng
  cdef uint Ny = dom.Ny

  cdef Parameters pm = dom.params
  cdef double gam = pm.gam
  cdef double beta = pm.beta

  cdef uint i,j,m,k,g, i22
  cdef double v1,v2,v3, B1,B2L,B2R,B3

  if pm.problem == ST2D:

    B1=2./sqrt(4*M_PI)
    B2L=3.6/sqrt(4*M_PI)
    B3=2./sqrt(4*M_PI)

    i22 = i2+ng+1

    for m in range(m1, m2+1):

      for i in range(2*Ny+i1, i22):
        for g in range(ng):
          for k in range(dom.Ncons):
            dom.U[i, j1-g-1, m,k] = dom.U[i+2*Ny, j2-g, m,k]

      for i in range(2*Ny+i1, i22):
        for g in range(ng):
          dom.B[i, j1-g-1, m,0] = dom.B[i+2*Ny, j2-g, m,0]

      for i in range(2*Ny+i1, i22):
        for g in range(ng-1):
          dom.B[i, j1-g-1, m,1] = dom.B[i+2*Ny, j2-g, m,1]

    for m in range(m1, m2+2):
      for i in range(2*Ny+i1, i22):
        for g in range(ng):
          dom.B[i, j1-g-1, m,2] = dom.B[i+2*Ny, j2-g, m,2]

    for m in range(m1, m2+1):
      for i in range(i1-ng, 2*Ny+i1):
        for j in range(ng):

          dom.U[i,j,m,RHO] = 1.08

          v1=1.2
          v2=0.01
          v3=0.5

          dom.U[i,j,m,MX] = dom.U[i,j,m,RHO]*(v1-2*v2)/sqrt(5)
          dom.U[i,j,m,MY] = dom.U[i,j,m,RHO]*(2*v1+v2)/sqrt(5)
          dom.U[i,j,m,MZ] = dom.U[i,j,m,RHO]*v3

          dom.U[i,j,m,BX] = (B1-2*B2L)/sqrt(5)
          dom.U[i,j,m,BY] = (2*B1+B2L)/sqrt(5)
          dom.U[i,j,m,BZ] = B3

          dom.U[i,j,m,EN] =\
              0.5*(dom.U[i,j,m,MX]**2 + dom.U[i,j,m,MY]**2 + dom.U[i,j,m,MZ]**2)/dom.U[i,j,m,RHO] +\
              0.5*(dom.U[i,j,m,BX]**2 + dom.U[i,j,m,BY]**2 + dom.U[i,j,m,BZ]**2) +\
              0.95/(gam-1)

      for i in range(i1-ng+1, 2*Ny+i1):
        for j in range(ng):
          dom.B[i,j,m,0] = (B1-2*B2L)/sqrt(5)

      for i in range(i1-ng, 2*Ny+i1):
        for j in range(1,ng):
          dom.B[i,j,m,1] = (2*B1+B2L)/sqrt(5)

    for m in range(m1, m2+2):
      for i in range(i1-ng, 2*Ny+i1):
        for j in range(ng):
          dom.B[i,j,m,2] = B3

cdef void y2_bc_mhd_user(Domain dom) nogil:

  cdef int i1 = dom.i1, i2 = dom.i2
  cdef int j1 = dom.j1, j2 = dom.j2
  cdef int m1 = dom.m1, m2 = dom.m2
  cdef int ng = dom.ng
  cdef uint Ny = dom.Ny

  cdef Parameters pm = dom.params
  cdef double gam = pm.gam
  cdef double beta = pm.beta

  cdef uint i,j,m,k,g, i22,i22ny
  cdef double v1,v2,v3, B1,B2L,B2R,B3

  if pm.problem == ST2D:

    B1=2./sqrt(4*M_PI)
    B2R=4./sqrt(4*M_PI)
    B3=2./sqrt(4*M_PI)

    i22 = i2+ng+1
    i22ny = i2+1-2*Ny

    for m in range(m1, m2+1):

      for i in range(i1-ng, i22ny):
        for g in range(ng):
          for k in range(dom.Ncons):
            dom.U[i, j2+g+1, m,k] = dom.U[i+2*Ny, j1+g, m,k]

      for i in range(i1-ng+1, i22ny):
        for g in range(ng):
          dom.B[i, j2+g+1, m,0] = dom.B[i+2*Ny, j1+g, m,0]

      for i in range(i1-ng, i22ny):
        for g in range(1,ng):
          dom.B[i, j2+g+1, m,1] = dom.B[i+2*Ny, j1+g, m,1]

    for m in range(m1, m2+2):
      for i in range(i1-ng, i22ny):
        for g in range(ng):
          dom.B[i, j2+g+1, m,2] = dom.B[i+2*Ny, j1+g, m,2]

    for m in range(m1, m2+1):
      for i in range(i22ny, i22):
        for j in range(j2+1, j2+ng+1):

          dom.U[i,j,m,RHO] = 1.

          dom.U[i,j,m,MX] = 0
          dom.U[i,j,m,MY] = 0
          dom.U[i,j,m,MZ] = 0

          dom.U[i,j,m,BX] = (B1-2*B2R)/sqrt(5)
          dom.U[i,j,m,BY] = (2*B1+B2R)/sqrt(5)
          dom.U[i,j,m,BZ] = B3

          dom.U[i,j,m,EN] =\
              0.5*(dom.U[i,j,m,MX]**2 + dom.U[i,j,m,MY]**2 + dom.U[i,j,m,MZ]**2) / dom.U[i,j,m,RHO] +\
              0.5*(dom.U[i,j,m,BX]**2 + dom.U[i,j,m,BY]**2 + dom.U[i,j,m,BZ]**2) +\
                1./(gam-1)

      for i in range(i22ny, i22):
        for j in range(j2+1, j2+ng+1):
          dom.B[i,j,m,0] = (B1-2*B2R)/sqrt(5)

      for i in range(i22ny, i22):
        for j in range(j2+2, j2+ng+1):
          dom.B[i,j,m,1] = (2*B1+B2R)/sqrt(5)

    for m in range(m1, m2+2):
      for i in range(i22ny, i22):
        for j in range(j2+1, j2+ng+1):
          dom.B[i,j,m,2] = B3

cdef void z1_bc_mhd_user(Domain dom) nogil:
  return
cdef void z2_bc_mhd_user(Domain dom) nogil:
  return

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
