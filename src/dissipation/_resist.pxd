# -*- coding: utf-8 -*-

from bnz.defs_cy cimport *
from bnz.data_struct cimport *

cdef void diffuse_implicit_res(BnzSim)
# cdef void update_nrg_res(Domain) nogil
cdef void diffuse_sts_res(BnzSim)
