# -*- coding: utf-8 -*-

from bnz.defs_cy cimport *

cdef void copy_layer_x(real3d, ints,ints, int, ints[3]) nogil
cdef void copy_layer_y(real3d, ints,ints, int, ints[3]) nogil
cdef void copy_layer_z(real3d, ints,ints, int, ints[3]) nogil

cdef void copy_reflect_layer_x(real3d, int, ints,ints, int, ints[3]) nogil
cdef void copy_reflect_layer_y(real3d, int, ints,ints, int, ints[3]) nogil
cdef void copy_reflect_layer_z(real3d, int, ints,ints, int, ints[3]) nogil

cdef void copy_add_layer_x(real3d, ints,ints, int, ints[3]) nogil
cdef void copy_add_layer_y(real3d, ints,ints, int, ints[3]) nogil
cdef void copy_add_layer_z(real3d, ints,ints, int, ints[3]) nogil

cdef void copy_add_reflect_layer_x(real3d, int, ints,ints, int, ints[3]) nogil
cdef void copy_add_reflect_layer_y(real3d, int, ints,ints, int, ints[3]) nogil
cdef void copy_add_reflect_layer_z(real3d, int, ints,ints, int, ints[3]) nogil

cdef void set_layer_x(real3d, double, ints, int, ints[3]) nogil
cdef void set_layer_y(real3d, double, ints, int, ints[3]) nogil
cdef void set_layer_z(real3d, double, ints, int, ints[3]) nogil

cdef void prolong_x(real3d, int, ints, int, ints[3]) nogil
cdef void prolong_y(real3d, int, ints, int, ints[3]) nogil
cdef void prolong_z(real3d, int, ints, int, ints[3]) nogil

cdef void copy_layer_r_sph(real3d, int, ints,ints, int, ints[3], ints[3]) nogil
cdef void copy_layer_r_cyl(real3d, int, ints,ints, int, ints[3], ints[3]) nogil
cdef void copy_layer_th_sph(real3d, int, ints,ints, int, ints[3], ints[3]) nogil

cdef void copy_add_layer_r_sph(real3d, int, ints,ints, int, ints[3], ints[3]) nogil
cdef void copy_add_layer_r_cyl(real3d, int, ints,ints, int, ints[3], ints[3]) nogil
cdef void copy_add_layer_th_sph(real3d, int, ints,ints, int, ints[3], ints[3]) nogil

IF MPI:

  cdef void pack(      real3d, real1d, ints*, ints*, int*, int)
  cdef void unpack(    real3d, real1d, ints*, ints*)
  cdef void unpack_add(real3d, real1d, ints*, ints*)
