# -*- coding: utf-8 -*-

from bnz.defs_cy cimport *
from bnz.data_struct cimport *

# cdef void update_nrg_res4(Domain) nogil
cdef void diffuse_sts_res4(BnzSim)
