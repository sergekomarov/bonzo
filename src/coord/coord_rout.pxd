# -*- coding: utf-8 -*-

from bnz.defs_cy cimport *
from grid cimport *

cdef void init_coord(BnzGrid)

cdef void add_geom_src_terms(GridCoord*, real4d,real4d,
                real4d,real4d,real4d, ints*, real) nogil

cdef void add_laplacian(GridCoord*, real4d, int) nogil

# cdef void lind2gcrd(real*,real*,real*, ints,ints,ints, GridParams) nogil
#
# cdef void lind2gcrd_x(real*, ints, GridParams) nogil
# cdef void lind2gcrd_y(real*, ints, GridParams) nogil
# cdef void lind2gcrd_z(real*, ints, GridParams) nogil
#
# cdef void lcrd2gcrd(real*, real*, real*, real, real, real, GridParams) nogil
#
# cdef void lcrd2gcrd_x(real*, real, GridParams) nogil
# cdef void lcrd2gcrd_y(real*, real, GridParams) nogil
# cdef void lcrd2gcrd_z(real*, real, GridParams) nogil
#
# cdef void lind2gind(ints*, ints*, ints*, ints, ints, ints, GridParams) nogil
#
# cdef void lind2gind_x(ints*, ints, GridParams) nogil
# cdef void lind2gind_y(ints*, ints, GridParams) nogil
# cdef void lind2gind_z(ints*, ints, GridParams) nogil
