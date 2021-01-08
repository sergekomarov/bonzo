# -*- coding: utf-8 -*-

from bnz.defs cimport *
from bnz.coordinates.coord cimport GridCoord

cdef real tr_grad_lim_xy(real3d, int,int,int, GridCoord*) nogil
cdef real tr_grad_lim_xz(real3d, int,int,int, GridCoord*) nogil
cdef real tr_grad_lim_yx(real3d, int,int,int, GridCoord*) nogil
cdef real tr_grad_lim_yz(real3d, int,int,int, GridCoord*) nogil
cdef real tr_grad_lim_zx(real3d, int,int,int, GridCoord*) nogil
cdef real tr_grad_lim_zy(real3d, int,int,int, GridCoord*) nogil

cdef real tr_grad_xy(real3d, int,int,int, GridCoord*) nogil
cdef real tr_grad_xz(real3d, int,int,int, GridCoord*) nogil
cdef real tr_grad_yx(real3d, int,int,int, GridCoord*) nogil
cdef real tr_grad_yz(real3d, int,int,int, GridCoord*) nogil
cdef real tr_grad_zx(real3d, int,int,int, GridCoord*) nogil
cdef real tr_grad_zy(real3d, int,int,int, GridCoord*) nogil
