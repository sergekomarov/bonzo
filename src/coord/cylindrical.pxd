# -*- coding: utf-8 -*-

from bnz.defs_cy cimport *
from grid cimport *

cdef void set_geometry(GridCoord*)

cdef real get_edge_len_x_(GridCoord*, ints,ints,ints) nogil
cdef real get_edge_len_y_(GridCoord*, ints,ints,ints) nogil
cdef real get_edge_len_z_(GridCoord*, ints,ints,ints) nogil

cdef real get_centr_len_x_(GridCoord*, ints,ints,ints) nogil
cdef real get_centr_len_x_(GridCoord*, ints,ints,ints) nogil
cdef real get_centr_len_x_(GridCoord*, ints,ints,ints) nogil

cdef real get_face_area_x_(GridCoord*, ints,ints,ints) nogil
cdef real get_face_area_y_(GridCoord*, ints,ints,ints) nogil
cdef real get_face_area_z_(GridCoord*, ints,ints,ints) nogil

cdef real get_cell_vol_(GridCoord*, ints,ints,ints) nogil
