# -*- coding: utf-8 -*-

IF MPI:
  from mpi4py import MPI as mpi
  from mpi4py cimport MPI as mpi

import numpy as np
cimport numpy as np

from libc.math cimport sqrt,sin,cos,fabs,log,exp,pow, fabs,fmin,fmax
from libc.stdlib cimport rand, RAND_MAX, srand

from scipy.fftpack import ifftn
from util cimport rand01

IF SPREC:
  np_real = np.float32
ELSE:
  np_real = np.float64


# ---------------------------------------------------------

cdef np.ndarray[double, ndim=4] gen_sol2d(int Nx, int Ny,
              double rms, int Linj_cells, int Lmin_cells,
              double dli[3]):

  # Generate random solenoidal 2d vector field.

  cdef:
    int i,j
    int im,jm, ip,jp
    double z, wn,wnx,wny,wnz
    double w1,w2, norm, wnb,wnmax

  cdef:
    np.ndarray[double, ndim=2] Akr = np.zeros((Ny,Nx))
    np.ndarray[double, ndim=2] Aki = np.zeros((Ny,Nx))
    np.ndarray[complex, ndim=2] A = np.zeros((Ny,Nx),dtype=complex)
    np.ndarray[double, ndim=3] V = np.zeros((2,Ny,Nx))

  np.random.seed()
  srand(np.random.randint(RAND_MAX))

  wnb = <double>Nx / Linj_cells
  wnmax = <double>Nx / Lmin_cells

  for j in range(Ny):
    for i in range(Nx):

      wnx = <double>i
      wny = <double>j
      if i > Nx/2: wnx = wnx-Nx
      if j > Ny/2: wny = wny-Ny

      wn = sqrt(wnx**2 + wny**2) / wnb + 1e-20
      if wn <= wnmax:
        z = wn**(-0.5) * (1 + wn**2)**(0.5*(-7./3 + 0.5))
      else:
        z=0

      w1 = rand01()
      w2 = rand01()

      Akr[j,i] = z * sqrt(-2*log(w1)) * cos(2*B_PI*w2)
      Aki[j,i] = z * sqrt(-2*log(w1)) * sin(2*B_PI*w2)

  Akr[0,0] = 0
  Aki[0,0] = 0


  Ak = Akr + 1j*Aki
  A = ifftn(Ak, axes=(0,1))
  A *= sqrt(Nx*Ny)

  print( '\n<A>f=',sqrt(0.5*(abs(Ak)**2).mean()),
         '<A>r=',  sqrt(0.5*(abs(A)**2 ).mean()) )

  Ar = A.real

  np.save('Ar.npy',Ar)

  for j in range(Ny):
    for i in range(Nx):

      ip = i+1 if i<Nx-1 else 0
      jp = j+1 if j<Ny-1 else 0

      V[0,j,i] =  (Ar[jp,i] - Ar[j,i]) * dli[1]
      V[1,j,i] = -(Ar[j,ip] - Ar[j,i]) * dli[0]

#        divB = 1./(2*h)*(V[0,1:-1,1:-1,2:]-V[0,1:-1,1:-1,:-2]+
#                     V[1,1:-1,2:,1:-1]-V[1,1:-1,:-2,1:-1]+
#                     V[2,2:,1:-1,1:-1]-V[2,:-2,1:-1,1:-1])
#        np.save('divB.npy',divB)

  norm = sqrt((V[0,:,:]**2 + V[1,:,:]**2).mean())
  V *= rms/norm
  print('rms =', sqrt((V[0,:,:]**2 + V[1,:,:]**2).mean()), '\n')

  np.save('V.npy',V)

  return V


# ---------------------------------------------------------------

cdef np.ndarray[double, ndim=4] gen_sol3d(int Nx, int Ny, int Nz,
              double rms, int Linj_cells, int Lmin_cells,
              double dli[3]):

  # Generate random solenoidal 3d vector field.

  cdef:
    int i,j,k
    int im,jm,km, ip,jp,kp
    double z, wn,wnx,wny,wnz, wnb,wnmax
    double phir,phii, zr,zi, w1,w2, A0r,A0i, norm

  cdef:
    np.ndarray[double, ndim=4] Akr = np.zeros((3,Nz,Ny,Nx))
    np.ndarray[double, ndim=4] Aki = np.zeros((3,Nz,Ny,Nx))
    np.ndarray[double, ndim=4] V = np.zeros((3,Nz,Ny,Nx))
    np.ndarray[complex, ndim=4] A = np.zeros((3,Nz,Ny,Nx),dtype=complex)

  wnb = <double>Nx / Linj_cells
  wnmax = <double>Nx / Lmin_cells

  np.random.seed()
  srand(np.random.randint(RAND_MAX))

  for k in range(Nz):
    for j in range(Ny):
      for i in range(Nx):

        wnx = <double>i
        wny = <double>j
        wnz = <double>k
        if i > Nx/2: wnx = wnx-Nx
        if j > Ny/2: wny = wny-Ny
        if k > Nz/2: wnz = wnz-Nz

        wn = sqrt(wnx**2 + wny**2 + wnz**2) / wnb + 1e-20
        if wn <= wnmax:
          z = wn**(-0.5) * (1 + wn**2)**(0.5*(-17./6 + 0.5))
        else:
          z=0.

        phir = 2*B_PI*rand01()
        phii = 2*B_PI*rand01()

        zr = 2*rand01() - 1
        zi = 2*rand01() - 1

        w1 = rand01()
        w2 = rand01()
        A0r = z * sqrt(-2*log(w1)) * cos(2*B_PI*w2)
        A0i = z * sqrt(-2*log(w1)) * sin(2*B_PI*w2)

        Akr[0,k,j,i] = A0r * cos(phir) * sqrt(1-zr**2)
        Aki[0,k,j,i] = A0i * cos(phii) * sqrt(1-zi**2)

        Akr[1,k,j,i] = A0r * sin(phir) * sqrt(1-zr**2)
        Aki[1,k,j,i] = A0i * sin(phii) * sqrt(1-zi**2)

        Akr[2,k,j,i] = A0r * zr
        Aki[2,k,j,i] = A0i * zi

  for i in range(3):

    Akr[i,0,0,0] = 0
    Aki[i,0,0,0] = 0

  Ak = Akr + 1j*Aki
  A = ifftn(Ak, axes=(1,2,3))
  A *= sqrt(Nx*Ny*Nz)

  print(
    '<A>f=', sqrt(0.5*(abs(Ak[0,...])**2 + abs(Ak[1,...])**2 + abs(Ak[2,...])**2).mean()),
    '<A>r=', sqrt(0.5*(abs( A[0,...])**2 + abs( A[1,...])**2 + abs( A[2,...])**2).mean())
    )

  Ar = A.real

  np.save('A3r.npy',Ar)

  for k in range(Nz):
    for j in range(Ny):
      for i in range(Nx):

        ip = i+1 if i<Nx-1 else 0
        jp = j+1 if j<Ny-1 else 0
        kp = k+1 if k<Nz-1 else 0

        V[0,k,j,i] =  ((Ar[2,k,jp,i] - Ar[2,k,j,i]) * dli[1] -
                       (Ar[1,kp,j,i] - Ar[1,k,j,i]) * dli[2])
        V[1,k,j,i] = -((Ar[2,k,j,ip] - Ar[2,k,j,i]) * dli[0]  -
                       (Ar[0,kp,j,i] - Ar[0,k,j,i]) * dli[2])
        V[2,k,j,i] =  ((Ar[1,k,j,ip] - Ar[1,k,j,i]) * dli[0] -
                       (Ar[0,k,jp,i] - Ar[0,k,j,i]) * dli[1])
  #        divB = 1./(2*h)*(V[0,1:-1,1:-1,2:]-V[0,1:-1,1:-1,:-2]+
  #                     V[1,1:-1,2:,1:-1]-V[1,1:-1,:-2,1:-1]+
  #                     V[2,2:,1:-1,1:-1]-V[2,:-2,1:-1,1:-1])
  #        np.save('divB.npy',divB)

  norm = sqrt((V[0,...]**2 + V[1,...]**2 + V[2,...]**2).mean())
  V *= rms/norm
  print('rms =', sqrt((V[0,...]**2 + V[1,...]**2 + V[2,...]**2).mean()))

  return V


# -----------------------------------------------------------------------

cpdef gen_fld3d(int Nx, int Ny, int Nz,
                int Linj_cells, int Lmin_cells):

  # Generate random 3D vector field.

  cdef:
    int i,j,k
    double z, wn,wnx,wny,wnz, wnb,wnmax
    double phir,phii, zr,zi, w1,w2, A0r,A0i

  cdef:
    np.ndarray[double, ndim=4] Akr = np.zeros((3,Nz,Ny,Nx))
    np.ndarray[double, ndim=4] Aki = np.zeros((3,Nz,Ny,Nx))
    np.ndarray[complex, ndim=4] A = np.zeros((3,Nz,Ny,Nx),dtype=complex)

  wnb = <double>Nx / Linj_cells
  wnmax = <double>Nx / Lmin_cells

  np.random.seed()
  srand(np.random.randint(RAND_MAX))

  for k in range(Nz):
    for j in range(Ny):
      for i in range(Nx):

        wnx = <double>i
        wny = <double>j
        wnz = <double>k
        if i > Nx/2: wnx = wnx-Nx
        if j > Ny/2: wny = wny-Ny
        if k > Nz/2: wnz = wnz-Nz

        wn = sqrt(wnx**2 + wny**2 + wnz**2) / wnb + 1e-20
        if wn <= wnmax:
          z = wn**(-0.5) * (1 + wn**2)**(0.5*(-17./6 + 0.5))
        else:
          z=0.

        phir = 2*B_PI*rand01()
        phii = 2*B_PI*rand01()

        zr = 2*rand01() - 1
        zi = 2*rand01() - 1

        w1 = rand01()
        w2 = rand01()
        A0r = z * sqrt(-2*log(w1)) * cos(2*B_PI*w2)
        A0i = z * sqrt(-2*log(w1)) * sin(2*B_PI*w2)

        Akr[0,k,j,i] = A0r * cos(phir) * sqrt(1-zr**2)
        Aki[0,k,j,i] = A0i * cos(phii) * sqrt(1-zi**2)

        Akr[1,k,j,i] = A0r * sin(phir) * sqrt(1-zr**2)
        Aki[1,k,j,i] = A0i * sin(phii) * sqrt(1-zi**2)

        Akr[2,k,j,i] = A0r * zr
        Aki[2,k,j,i] = A0i * zi

  for i in range(3):

    Akr[i,0,0,0] = 0
    Aki[i,0,0,0] = 0

  Ak = Akr + 1j*Aki
  A = ifftn(Ak, axes=(1,2,3))
  A *= sqrt(Nx*Ny*Nz)

  print(
    '<A>f=', sqrt(0.5*(abs(Ak[0,...])**2 + abs(Ak[1,...])**2 + abs(Ak[2,...])**2).mean()),
    '<A>r=', sqrt(0.5*(abs( A[0,...])**2 + abs( A[1,...])**2 + abs( A[2,...])**2).mean())
    )

  Ar = A.real

  # np.save('A3.npy',Ar)

  return Ar


# -----------------------------------------------------------------------

cpdef gen_scal_fld_3d(int Nx, int Ny, int Nz, double rms, double p,
                      int Linj_cells, int Lmin_cells):

  # Generate random scalar 3D field.

  cdef:
    int i,j,k
    double z, wn,wnx,wny,wnz, wnb,wnmax
    double phir,phii, zr,zi, w1,w2

  cdef:
    np.ndarray[double, ndim=3] Akr = np.zeros((Nz,Ny,Nx))
    np.ndarray[double, ndim=3] Aki = np.zeros((Nz,Ny,Nx))
    np.ndarray[complex, ndim=3] A = np.zeros((Nz,Ny,Nx),dtype=complex)

  wnb = <double>Nx / Linj_cells
  wnmax = <double>Nx / Lmin_cells

  np.random.seed()
  srand(np.random.randint(RAND_MAX))

  for k in range(Nz):
    for j in range(Ny):
      for i in range(Nx):

        wnx = <double>i
        wny = <double>j
        wnz = <double>k
        if i > Nx/2: wnx = wnx-Nx
        if j > Ny/2: wny = wny-Ny
        if k > Nz/2: wnz = wnz-Nz

        wn = sqrt(wnx**2 + wny**2 + wnz**2) / wnb + 1e-20
        if wn <= wnmax:
          z = wn**(-0.5) * (1 + wn**2)**(0.5*(-0.5*p-1 + 0.5))
        else:
          z=0.

        phir = 2*B_PI*rand01()
        phii = 2*B_PI*rand01()

        zr = 2*rand01() - 1
        zi = 2*rand01() - 1

        w1 = rand01()
        w2 = rand01()
        Akr[k,j,i] = z * sqrt(-2*log(w1)) * cos(2*B_PI*w2)
        Aki[k,j,i] = z * sqrt(-2*log(w1)) * sin(2*B_PI*w2)

  Akr[0,0,0] = 0
  Aki[0,0,0] = 0

  Ak = Akr + 1j*Aki
  A = ifftn(Ak, axes=(1,2,3))
  A *= sqrt(Nx*Ny*Nz)

  print( '<A>f=', sqrt(0.5*(abs(Ak)**2).mean()),
         '<A>r=', sqrt(0.5*(abs( A)**2).mean()) )

  Ar = A.real

  norm = sqrt((Ar**2).mean())
  Ar *= rms/norm

  return Ar


# ------------------------------------------------------------------

cpdef gen_GS(int Nx, int Ny, int Nz,
             double vrms, int Linj_cells, int Lmin_cells):

  # Generate vector field with Goldreich-Shridhar spectrum,
  # vectors are perpendicular to x axis.

  cdef:
    int i,j,k
    double z, wn,wnyz,wnx,wny,wnz, wnb,wnmax
    double phir,phii, w1,w2, V0r,V0i, norm
    double one6th = 1./6
    double two3rd = 2./3
    double elev12th = 11./12

  cdef:
    np.ndarray[real, ndim=4] Vkr = np.zeros((3,Nz,Ny,Nx))
    np.ndarray[real, ndim=4] Vki = np.zeros((3,Nz,Ny,Nx))
    np.ndarray[real, ndim=4] Vr = np.zeros((3,Nz,Ny,Nx))

  np.random.seed()
  srand(np.random.randint(RAND_MAX))

  wnb = <double>Nx / Linj_cells
  wnmax = <double>Nx / Lmin_cells

  for k in range(Nz):
    for j in range(Ny):
      for i in range(Nx):

        wnx = <double>i
        wny = <double>j
        wnz = <double>k
        if i > Nx/2: wnx = wnx-Nx
        if j > Ny/2: wny = wny-Ny
        if k > Nz/2: wnz = wnz-Nz

        wnyz = sqrt(wny**2 + wnz**2)+1e-20
        wnx =  wnx/wnb
        wnyz = wnyz/wnb
        if wnyz < wnmax:
          z = wnyz**one6th / (1 + wnyz**2)**elev12th * exp(-0.5*fabs(wnx) / wnyz**two3rd)
        else:
          z=0

        phir = 2*B_PI*rand01()
        phii = 2*B_PI*rand01()

        w1 = rand01()
        w2 = rand01()
        V0r = z * sqrt(-2*log(w1)) * cos(2*B_PI*w2)
        V0i = z * sqrt(-2*log(w1)) * sin(2*B_PI*w2)

        Vkr[1,k,j,i] = V0r * cos(phir)
        Vki[1,k,j,i] = V0i * cos(phii)

        Vkr[2,k,j,i] = V0r * sin(phir)
        Vki[2,k,j,i] = V0i * sin(phii)

  for i in range(3):

    Vkr[i,0,0,0] = 0
    Vki[i,0,0,0] = 0

  Vk = Vkr + 1j*Vki
  V = ifftn(Vk, axes=(1,2,3))

  V *= sqrt(Nx*Ny*Nz)

  print(
    '\n<V>f =', sqrt(0.5*(abs(Vk[0,...])**2 + abs(Vk[1,...])**2 + abs(Vk[2,...])**2).mean()),
      '<V>r =', sqrt(0.5*(abs( V[0,...])**2 +  abs(V[1,...])**2 +  abs(V[2,...])**2).mean())
    )

  Vr = V.real

  norm = sqrt((Vr[0,...]**2 + Vr[1,...]**2 + Vr[2,...]**2).mean())
  Vr *= vrms/norm
  print( 'rms =', sqrt((Vr[0,...]**2 + Vr[1,...]**2 + Vr[2,...]**2).mean()) )

  np.save('GS.npy',Vr)

  return Vr
