# -*- coding: utf-8 -*-

from bnz.defs_cy cimport *

cdef void init_maxw_table(real[::1], real[::1], double)
cdef void init_powlaw_table(real[::1], real[::1],
                double, double, double)
cdef void distr_prt(real*, real*, real*, real*,
              real[::1], real[::1], double, double)


cdef void getweight1(ints*,ints*,ints*,
                     real,real,real, real***,
                     double[3], int) nogil

cdef void getweight2(ints*,ints*,ints*,
                     real,real,real, real***,
                     double[3], int) nogil

cdef void clearF(real4d, ints[3]) nogil
# cdef void reduceF(real4d, ints[3], int) nogil
