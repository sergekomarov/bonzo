# -*- coding: utf-8 -*-

import numpy as np
cimport numpy as np

from defs_cy cimport *

from libc.stdio cimport FILE

cdef extern from "stdarg.h" nogil:
  ctypedef struct va_list:
    pass
  void va_start(va_list, void* arg)
  void va_end(va_list)

cdef extern from "stdio.h" nogil:
  int vfprintf(FILE*, char*, va_list)


cdef ints maxi(ints, ints) nogil
cdef ints mini(ints, ints) nogil
cdef real sign(real) nogil
cdef real sqr(real) nogil
cdef real cube(real) nogil

cdef double rand01() nogil


cdef double timediff(timeval tstart, timeval tstop) nogil

cdef void** calloc_2d_array(ints,ints, ints) nogil
cdef void free_2d_array(void*) nogil

cdef void*** calloc_3d_array(ints,ints,ints, ints) nogil
cdef void free_3d_array(void*) nogil

cdef void**** calloc_4d_array(ints,ints,ints,ints, ints) nogil
cdef void free_4d_array(void****, ints) nogil

cdef void**  calloc_2dv_array(ints n1,          ints *n2, ints size) nogil
cdef void*** calloc_3dv_array(ints n1, ints n2, ints *n3, ints size) nogil

cdef void copy_2d_array(real**,real**, ints,ints) nogil
# cdef void swap_2d_array_ptrs(real**, real**, ints) nogil
cdef void swap_2d_array_ptrs(real**, real**) nogil
cdef void swap_array_ptrs(void*, void*) nogil

cdef real***  memview2carray_3d(real3d, ints, ints) nogil
cdef real**** memview2carray_4d(real4d, ints, ints, ints) nogil

# cdef real**** calloc_from_memview_4d(ints, ints, ints, ints)
# cdef real***  calloc_from_memview_3d(ints, ints, ints)

# cdef ints flat(ints,ints,ints,ints, ints[3]) nogil

cdef void print_root(char*, ...)

cdef np.ndarray[double, ndim=4] gen_sol2d(ints,ints,      double, ints,ints, double[3])
cdef np.ndarray[double, ndim=4] gen_sol3d(ints,ints,ints, double, ints,ints, double[3])
