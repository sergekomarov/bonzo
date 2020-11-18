# -*- coding: utf-8 -*-

from mpi4py import MPI as mpi
from mpi4py cimport MPI as mpi

import numpy as np
cimport numpy as np

from libc.math cimport sqrt,floor,ceil,log,exp,sin,cos,pow,fabs,fmin,fmax
from libc.stdlib cimport rand, RAND_MAX, srand
from libc.stdlib cimport malloc, calloc, free
from libc.stdio cimport stdout, printf

from scipy.fftpack import ifftn

IF SPREC:
  np_real = np.float32
ELSE:
  np_real = np.float64



cdef inline ints maxi(ints a, ints b) nogil:
  return a if a>b else b

cdef inline ints mini(ints a, ints b) nogil:
  return a if a<b else b

cdef inline real sign(real a) nogil:
  return -1. if a<0. else (1. if a>0. else 0.)

cdef inline real sqr(real a) nogil:
  return a*a

cdef inline real cube(real a) nogil:
  return a*a*a


# ===================================================================

# Calculate time difference in milliseconds.

cdef double timediff(timeval tstart, timeval tstop) nogil:

  return (1000 * (tstop.tv_sec  - tstart.tv_sec)
        + 1e-3 * (tstop.tv_usec - tstart.tv_usec))


# ==================================================

# Generate a random double in [0,1).

cdef inline double rand01() nogil:
  return <double>rand()/RAND_MAX


# ================================================================

# Allocate 2D C array.

cdef void** calloc_2d_array(ints n1, ints n2, ints size) nogil:

  cdef void **array
  cdef ints i

  array = <void **>calloc(n1, sizeof(void*))
  array[0] = <void *>calloc(n1 * n2, size)

  for i in range(1,n1):
    array[i] = <void *>(<unsigned char *>array[0] + i * n2 * size)

  return array


cdef void free_2d_array(void *array) nogil:

  cdef void **ta = <void **>array

  free(ta[0])
  free(array)


# ================================================================

# Allocate 3D Cs.

cdef void*** calloc_3d_array(ints n1, ints n2, ints n3, ints size) nogil:

  cdef void ***array
  cdef ints i,j

  array = <void ***>calloc(n1, sizeof(void**))
  array[0] = <void **>calloc(n1 * n2, sizeof(void*))

  for i in range(1,n1):
    array[i] = <void **>(<unsigned char *>array[0] + i * n2 * sizeof(void*))

  array[0][0] = <void *>calloc(n1 * n2 * n3, size)

  for j in range(1,n2):
    array[0][j] = <void **>(<unsigned char *>array[0][j-1] + n3 * size)

  for i in range(1,n1):
    array[i][0] = <void **>(<unsigned char *>array[i-1][0] + n2 * n3 * size)
    for j in range(1,n2):
      array[i][j] = <void **>(<unsigned char *>array[i][j-1] + n3 * size)

  return array


cdef void free_3d_array(void *array) nogil:

  cdef void ***ta = <void ***>array

  free(ta[0][0])
  free(ta[0])
  free(array)



#================================================================================

# Allocate 4D C array from Cython array.

cdef void**** calloc_4d_array(ints n1, ints n2, ints n3, ints n4, ints size) nogil:

  cdef ints i
  cdef void ****array = <void****>calloc(n1, sizeof(void***))

  for i in range(n1):
    array[i] = calloc_3d_array(n2,n3,n4, size)

  return array

cdef void free_4d_array(void ****array, ints n1) nogil:

  cdef ints i
  for i in range(n1):
    free_3d_array(array[i])



# ================================================================

# Allocate 2D C array with rows of different lengths.
# *n2 needs to have n1 elements
# n1 is number of dimensions
# *n2 numbers of elements in each dimension

# cdef void** calloc_2dv_array(ints n1, ints *n2, ints size) nogil:
#
#   cdef void **array
#   cdef ints i, offset, nall
#
#   nall=0
#   for i in range(n1):
#     nall += n2[i]
#
#   array = <void **>calloc(n1, sizeof(void*))
#   array[0] = <void *>calloc(nall, size)
#
#   offset = n2[0]*size
#   for i in range(1,n1):
#     array[i] = <void *>(<unsigned char *>array[0] + offset)
#     offset += n2[i]*size
#
#   return array

cdef void** calloc_2dv_array(ints n1, ints *n2, ints size) nogil:

  cdef void **array
  cdef ints iz

  array = <void **>calloc(n1, sizeof(void*))
  for i in range(n1):
    array[i] = <void *>calloc(n2[i], size)

  return array

# ================================================================

# Allocate 3D C array with rows of different lengths.
# *n3 needs to have n1 elements
# n1 is number of dimensions
# n2 number of coefficients
# *n3 numbers of elements in each dimension

cdef void*** calloc_3dv_array(ints n1, ints n2, ints *n3, ints size) nogil:

  cdef void ***array
  cdef ints i,j

  array = <void ***>calloc(n1, sizeof(void**))
  for i in range(n1):
    array[i] = <void **>calloc(n2, sizeof(void*))
    array[i][0] = <void *>calloc(n2*n3[i], size)

    for j in range(1,n2):
      array[i][j] = <void *>(<unsigned char *>array[i][0] + j*n3[i]*size)

  return array



#========================================================================

cdef void copy_2d_array(real **dest, real **src, ints nx, ints ny) nogil:

  cdef ints i,j
  for i in range(nx):
    for j in range(ny):
      dest[i][j] = src[i][j]


# =======================================================================

cdef void swap_2d_array_ptrs(real **A, real **B, ints nx) nogil:

  cdef:
    ints i
    # real ** tmp
    real *tmp

  for i in range(nx):

    tmp = A[i]
    A[i] = B[i]
    B[i] = tmp

  tmp = A
  A = B
  B = tmp

# =======================================================================

cdef void swap_array_ptrs(void *A, void *B) nogil:

  cdef void *tmp = A
  A = B
  B = tmp


#==========================================================================

# Shallow copy of a 3D memoryview as an array of C pointers.

cdef real*** memview2carray_3d(real3d A, ints n1, ints n2) nogil:

  cdef:
    ints j,k
    real ***B

  B = <real***>calloc_2d_array(n1,n2, sizeof(real*))

  for k in range(n1):
    for j in range(n2):
      B[k][j] = &A[k][j][0]

  return B


#==========================================================================

# Shallow copy of a 4D memoryview as an array of C pointers.

cdef real**** memview2carray_4d(real4d A, ints n1, ints n2, ints n3) nogil:

  cdef:
    ints k,j,n
    real ****B

  B = <real****>calloc_3d_array(n1,n2,n3, sizeof(real*))

  for n in range(n1):
    for k in range(n2):
      for j in range(n3):
        B[n][k][j] = &A[n][k][j][0]

  return B


#================================================================================

# Allocate 4D C array from Cython array.

# cdef real**** calloc_from_memview_4d(ints n1, ints n2, ints n3, ints n4):
#
#   cdef:
#     ints j,k,n
#     real**** A = <real****>calloc_3d_array(n1,n2,n3, sizeof(real*))
#     real4d Acy = np.zeros((n1,n2,n3,n4), dtype=np_real)
#
#   for n in range(n1):
#     for k in range(n2):
#       for j in range(n3):
#         A[n][k][j] = &Acy[n][k][j][0]
#
#   return A


#================================================================================

# Allocate 3D C array from Cython memoryview.

# cdef real*** calloc_from_memview_3d(ints n1, ints n2, ints n3):
#
#   cdef:
#     ints j,k
#     real*** A = <real***>calloc_2d_array(n1,n2, sizeof(real*))
#     real3d Acy = np.zeros((n1,n2,n3), dtype=np_real)
#
#   for k in range(n1):
#     for j in range(n2):
#         A[k][j] = &Acy[k][j][0]
#
#   return A


# ====================================================================

# Output to stdout only if root process.

cdef void print_root(char *fmt, ...):

  cdef:
    va_list args
    int rank = 0

  IF MPI: rank = mpi.COMM_WORLD.Get_rank()

  if rank==0:
    va_start(args, fmt)
    vfprintf(stdout, fmt, args)
    va_end(args)

  return
  

# =======================================================================

# Generate vector field with Goldreich-Shridhar spectrum,
# vectors are perpendicular to x axis.


cpdef gen_GS(ints Nx, ints Ny, ints Nz,
            double vrms, ints Linj_cells, ints Lmin_cells):

  cdef:
    ints i,j,k
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


  # Vk = pyfftw.empty_aligned((3,Nz,Ny,Nx), dtype='complex128', n=16)
  # V =  pyfftw.empty_aligned((3,Nz,Ny,Nx), dtype='complex128', n=16)
  # ifft_V = pyfftw.FFTW(Vk, V, axes=(1,2,3),
  #                     direction='FFTW_BACKWARD')
  #                     #, threads=nt)
  # Vk[:] = np.asarray(Vkr) + 1j*np.asarray(Vki)
  # ifft_V()
  Vk = Vkr + 1j*Vki
  V = ifftn(Vk, axes=(1,2,3))

  # V *= sqrt(Nx*Ny*Nz)
  V *= sqrt(Nx*Ny*Nz)

  print '\n<V>f=',sqrt(0.5*(abs(Vk[0,...])**2 + abs(Vk[1,...])**2 + abs(Vk[2,...])**2).mean()),\
          '<V>r=',sqrt(0.5*(abs( V[0,...])**2 +  abs(V[1,...])**2 +  abs(V[2,...])**2).mean())

  Vr = V.real

  norm = sqrt((Vr[0,...]**2 + Vr[1,...]**2 + Vr[2,...]**2).mean())
  Vr *= vrms/norm
  print 'rms =', sqrt((Vr[0,...]**2 + Vr[1,...]**2 + Vr[2,...]**2).mean())

  np.save('GS.npy',Vr)

  return Vr



# =========================================================

# Generate random solenoidal 2d vector field.

cdef np.ndarray[double, ndim=4] gen_sol2d(ints Nx, ints Ny,
              double rms, ints Linj_cells, ints Lmin_cells,
              double dli[3]):

  cdef:
    ints i,j
    ints im,jm, ip,jp
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

  print '\n<A>f=',sqrt(0.5*(abs(Ak)**2).mean()),\
          '<A>r=',sqrt(0.5*(abs(A)**2).mean())

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
  print 'rms =', sqrt((V[0,:,:]**2 + V[1,:,:]**2).mean()), '\n'

  np.save('V.npy',V)

  return V



# =============================================

# Generate random solenoidal 3d vector field.


cdef np.ndarray[double, ndim=4] gen_sol3d(ints Nx, ints Ny, ints Nz,
              double rms, ints Linj_cells, ints Lmin_cells,
              double dli[3]):

  cdef:
    ints i,j,k
    ints im,jm,km, ip,jp,kp
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


  # Ak = pyfftw.empty_aligned((3,Nz,Ny,Nx), dtype='complex128', n=16)
  # A =  pyfftw.empty_aligned((3,Nz,Ny,Nx), dtype='complex128', n=16)
  # ifft_A = pyfftw.FFTW(Ak, A, axes=(1,2,3),
  #                     direction='FFTW_BACKWARD',
  #                     threads=nt)
  # Ak[:] = np.asarray(Akr) + 1j*np.asarray(Aki)
  # ifft_A()
  # A *= sqrt(Nx*Ny*Nz)

  Ak = Akr + 1j*Aki
  A = ifftn(Ak, axes=(1,2,3))
  A *= sqrt(Nx*Ny*Nz)

  print '<A>f=',sqrt(0.5*(abs(Ak[0,...])**2 + abs(Ak[1,...])**2 + abs(Ak[2,...])**2).mean()),\
        '<A>r=',sqrt(0.5*(abs(A[0,...])**2 + abs(A[1,...])**2 + abs(A[2,...])**2).mean())

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
  print 'rms =', sqrt((V[0,...]**2 + V[1,...]**2 + V[2,...]**2).mean())

  return V



# =====================================================

# Generate random 3D vector field.

cpdef gen_fld3d(ints Nx, ints Ny, ints Nz,
            ints Linj_cells, ints Lmin_cells):

  cdef:
    ints i,j,k
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


  # Ak = pyfftw.empty_aligned((3,Nz,Ny,Nx), dtype='complex128', n=16)
  # A =  pyfftw.empty_aligned((3,Nz,Ny,Nx), dtype='complex128', n=16)
  # ifft_A = pyfftw.FFTW(Ak, A, axes=(1,2,3),
  #                     direction='FFTW_BACKWARD',
  #                     threads=nt)
  # Ak[:] = np.asarray(Akr) + 1j*np.asarray(Aki)
  # ifft_A()
  # A *= sqrt(Nx*Ny*Nz)

  Ak = Akr + 1j*Aki
  A = ifftn(Ak, axes=(1,2,3))
  A *= sqrt(Nx*Ny*Nz)

  print '<A>f=',sqrt(0.5*(abs(Ak[0,...])**2 + abs(Ak[1,...])**2 + abs(Ak[2,...])**2).mean()),\
        '<A>r=',sqrt(0.5*(abs(A[0,...])**2 + abs(A[1,...])**2 + abs(A[2,...])**2).mean())

  Ar = A.real

  # np.save('A3.npy',Ar)

  return Ar



# ====================================================================

# Generate random scalar 3D field.

cpdef gen_scal_fld_3d(ints Nx, ints Ny, ints Nz, double rms, double p,
            ints Linj_cells, ints Lmin_cells):

  cdef:
    ints i,j,k
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


  # Ak = pyfftw.empty_aligned((3,Nz,Ny,Nx), dtype='complex128', n=16)
  # A =  pyfftw.empty_aligned((3,Nz,Ny,Nx), dtype='complex128', n=16)
  # ifft_A = pyfftw.FFTW(Ak, A, axes=(1,2,3),
  #                     direction='FFTW_BACKWARD',
  #                     threads=nt)
  # Ak[:] = np.asarray(Akr) + 1j*np.asarray(Aki)
  # ifft_A()
  # A *= sqrt(Nx*Ny*Nz)

  Ak = Akr + 1j*Aki
  A = ifftn(Ak, axes=(1,2,3))
  A *= sqrt(Nx*Ny*Nz)

  print '<A>f=',sqrt(0.5*(abs(Ak)**2).mean()),\
        '<A>r=',sqrt(0.5*(abs(A)**2).mean())

  Ar = A.real

  norm = sqrt((Ar**2).mean())
  Ar *= rms/norm


  return Ar
