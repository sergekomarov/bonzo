# -*- coding: utf-8 -*-

from libc.stdio cimport FILE
from defs cimport *

cdef extern from "stdarg.h" nogil:
  ctypedef struct va_list:
    pass
  void va_start(va_list, void* arg)
  void va_end(va_list)

cdef extern from "stdio.h" nogil:
  int vfprintf(FILE*, char*, va_list)

cdef double timediff(timeval tstart, timeval tstop) nogil
cdef void print_root(char*, ...)
cdef real rand01() nogil

cdef void** calloc_2d_array(size_t,size_t, size_t) nogil
cdef void free_2d_array(void*) nogil

cdef void*** calloc_3d_array(size_t,size_t,size_t, size_t) nogil
cdef void free_3d_array(void*) nogil

cdef void**** calloc_4d_array(size_t,size_t,size_t,size_t, size_t) nogil
cdef void free_4d_array(void*, size_t) nogil

# cdef void**  calloc_2dv_array(int n1,         int *n2, int size) nogil
# cdef void*** calloc_3dv_array(int n1, int n2, int *n3, int size) nogil

cdef void copy_2d_array(void**,void**, size_t,size_t) nogil
cdef void swap_array_ptrs(void*, void*) nogil

cdef real***  memview2carray_3d(real3d, size_t, size_t) nogil
cdef real**** memview2carray_4d(real4d, size_t, size_t, size_t) nogil
