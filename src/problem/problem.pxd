# -*- coding: utf-8 -*-

import numpy as np
cimport numpy as np

from bnz.defs_cy cimport *
from bnz.data_struct cimport *

cdef void set_problem(BnzSim)

cdef void do_user_work_cons(real4d,real4d, real4d,real4d, ints[6], BnzSim, double)

cdef void set_bc_grid_ptrs_user(BnzBC)
IF PIC or MHDPIC:
  cdef void set_bc_prt_ptrs_user(BnzBC)

cdef void set_phys_ptrs_user(BnzPhysics)

cdef void set_output_user(BnzOutput)
