# -*- coding: utf-8 -*-

from bnz.defs_cy cimport *
from grid cimport *

cdef void init_coordinates(BnzGrid)

cdef void add_geom_src_terms(GridCoord*, real4d,real4d,
                real4d,real4d,real4d, ints*, real) nogil

cdef void add_laplacian(GridCoord*, real4d, int) nogil

cdef real get_edge_len_x(GridCoord*, ints,ints,ints) nogil
cdef real get_edge_len_y(GridCoord*, ints,ints,ints) nogil
cdef real get_edge_len_z(GridCoord*, ints,ints,ints) nogil

cdef real get_centr_len_x(GridCoord*, ints,ints,ints) nogil
cdef real get_centr_len_x(GridCoord*, ints,ints,ints) nogil
cdef real get_centr_len_x(GridCoord*, ints,ints,ints) nogil

cdef real get_face_area_x(GridCoord*, ints,ints,ints) nogil
cdef real get_face_area_y(GridCoord*, ints,ints,ints) nogil
cdef real get_face_area_z(GridCoord*, ints,ints,ints) nogil

cdef real get_cell_vol(GridCoord*, ints,ints,ints) nogil
