# -*- coding: utf-8 -*-

IF MPI:
  from mpi4py import MPI as mpi
  from mpi4py cimport MPI as mpi

import numpy as np
cimport numpy as np

from libc.stdlib cimport rand, RAND_MAX, srand
from libc.stdlib cimport malloc, calloc, free
from libc.stdio cimport stdout, printf

from scipy.fftpack import ifftn

IF SPREC:
  np_real = np.float32
ELSE:
  np_real = np.float64


cdef void print_root(char *fmt, ...):

  # Output to stdout only if root.

  cdef:
    va_list args
    int rank = 0

  IF MPI: rank = mpi.COMM_WORLD.Get_rank()

  if rank==0:
    va_start(args, fmt)
    vfprintf(stdout, fmt, args)
    va_end(args)

  return

cdef double timediff(timeval tstart, timeval tstop) nogil:
  # Calculate time difference in milliseconds.

  return (1000 * (tstop.tv_sec  - tstart.tv_sec)
        + 1e-3 * (tstop.tv_usec - tstart.tv_usec))

cdef inline real rand01() nogil:
  # Generate a random real in [0,1).
  return <real>rand()/RAND_MAX

# -------------------------------------------------------------------

cdef void** calloc_2d_array(size_t n1, size_t n2, size_t size) nogil:

  # Allocate 2D contiguous array.

  cdef void **arr
  cdef size_t i

  arr = <void **>calloc(n1, sizeof(void*))
  arr[0] = <void *>calloc(n1 * n2, size)

  for i in range(1,n1):
    arr[i] = <void *>(<unsigned char *>arr[0] + i * n2 * size)

  return arr


cdef void free_2d_array(void *arr) nogil:

  cdef void **ta = <void **>arr
  free(ta[0])
  free(arr)


# ------------------------------------------------------------------

cdef void*** calloc_3d_array(size_t n1, size_t n2,
                             size_t n3, size_t size) nogil:

  # Allocate 3D contiguous array.

  cdef void ***arr
  cdef size_t i,j

  arr = <void ***>calloc(n1, sizeof(void**))
  arr[0] = <void **>calloc(n1 * n2, sizeof(void*))

  for i in range(1,n1):
    arr[i] = <void **>(<unsigned char *>arr[0] + i * n2 * sizeof(void*))

  arr[0][0] = <void *>calloc(n1 * n2 * n3, size)

  for j in range(1,n2):
    arr[0][j] = <void **>(<unsigned char *>arr[0][j-1] + n3 * size)

  for i in range(1,n1):
    arr[i][0] = <void **>(<unsigned char *>arr[i-1][0] + n2 * n3 * size)
    for j in range(1,n2):
      arr[i][j] = <void **>(<unsigned char *>arr[i][j-1] + n3 * size)

  return arr


cdef void free_3d_array(void *arr) nogil:

  cdef void ***ta = <void ***>arr

  free(ta[0][0])
  free(ta[0])
  free(arr)


# ------------------------------------------------------------------

cdef void**** calloc_4d_array(size_t n1, size_t n2, size_t n3,
                              size_t n4, size_t size) nogil:

  # Allocate 4D array contiguous in the inner 3 dimensions.

  cdef size_t i
  cdef void ****arr = <void****>calloc(n1, sizeof(void***))

  for i in range(n1):
    arr[i] = calloc_3d_array(n2,n3,n4, size)

  return arr

cdef void free_4d_array(void *arr, size_t n1) nogil:

  cdef size_t i
  for i in range(n1): free_3d_array(arr[i])
  free(arr)


# ------------------------------------------------------------------

# cdef void** calloc_2dv_array(int n1, int *n2, int size) nogil:
#
  # Allocate 2D C array with rows of different lengths.
  # *n2 needs to have n1 elements
  # n1 is number of dimensions
  # *n2 numbers of elements in each dimension
#
#   cdef void **arr
#   cdef int i, offset, nall
#
#   nall=0
#   for i in range(n1):
#     nall += n2[i]
#
#   arr = <void **>calloc(n1, sizeof(void*))
#   arr[0] = <void *>calloc(nall, size)
#
#   offset = n2[0]*size
#   for i in range(1,n1):
#     arr[i] = <void *>(<unsigned char *>arr[0] + offset)
#     offset += n2[i]*size
#
#   return arr

# cdef void** calloc_2dv_array(int n1, int *n2, int size) nogil:
#
#   cdef void **arr
#   cdef int iz
#
#   arr = <void **>calloc(n1, sizeof(void*))
#   for i in range(n1):
#     arr[i] = <void *>calloc(n2[i], size)
#
#   return arr

# ------------------------------------------------------------------

# cdef void*** calloc_3dv_array(int n1, int n2, int *n3, int size) nogil:

  # Allocate 3D C array with rows of different lengths.
  # *n3 needs to have n1 elements
  # n1 is number of dimensions
  # n2 number of coefficients
  # *n3 numbers of elements in each dimension
#
#   cdef void ***arr
#   cdef int i,j
#
#   arr = <void ***>calloc(n1, sizeof(void**))
#   for i in range(n1):
#     arr[i] = <void **>calloc(n2, sizeof(void*))
#     arr[i][0] = <void *>calloc(n2*n3[i], size)
#
#     for j in range(1,n2):
#       arr[i][j] = <void *>(<unsigned char *>arr[i][0] + j*n3[i]*size)
#
#   return arr


# ----------------------------------------------------------------------------

cdef void copy_2d_array(void **dest, void **src, size_t n1, size_t n2) nogil:

  cdef size_t i,j
  for i in range(n1):
    for j in range(n2):
      dest[i][j] = src[i][j]

cdef void swap_array_ptrs(void *A, void *B) nogil:

  cdef void *tmp = A
  A = B
  B = tmp


# -------------------------------------------------------------------

cdef real*** memview2carray_3d(real3d A, size_t n1, size_t n2) nogil:

  # Shallow copy of a 3D memoryview as an array of C pointers.

  cdef:
    size_t j,k
    real ***B

  B = <real***>calloc_2d_array(n1,n2, sizeof(real*))

  for k in range(n1):
    for j in range(n2):
      B[k][j] = &A[k][j][0]

  return B


cdef real**** memview2carray_4d(real4d A, size_t n1, size_t n2, size_t n3) nogil:

  # Shallow copy of a 4D memoryview as an array of C pointers.

  cdef:
    size_t k,j,n
    real ****B

  B = <real****>calloc_3d_array(n1,n2,n3, sizeof(real*))

  for n in range(n1):
    for k in range(n2):
      for j in range(n3):
        B[n][k][j] = &A[n][k][j][0]

  return B
