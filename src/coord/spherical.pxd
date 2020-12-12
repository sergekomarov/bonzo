# -*- coding: utf-8 -*-

from bnz.defs_cy cimport *
from coord_cy cimport GridCoord

cdef void set_geometry_(GridCoord*)
cdef void free_geom_data_(GridCoord*)

cdef void add_geom_src_terms_(real4d,real4d, real4d,real4d,real4d,
                              GridCoord*, int*, real) nogil

cdef real get_edge_len_x_(GridCoord*, int,int,int) nogil
cdef real get_edge_len_y_(GridCoord*, int,int,int) nogil
cdef real get_edge_len_z_(GridCoord*, int,int,int) nogil

cdef real get_centr_len_x_(GridCoord*, int,int,int) nogil
cdef real get_centr_len_x_(GridCoord*, int,int,int) nogil
cdef real get_centr_len_x_(GridCoord*, int,int,int) nogil

cdef real get_face_area_x_(GridCoord*, int,int,int) nogil
cdef real get_face_area_y_(GridCoord*, int,int,int) nogil
cdef real get_face_area_z_(GridCoord*, int,int,int) nogil

cdef real get_cell_vol_(GridCoord*, int,int,int) nogil
