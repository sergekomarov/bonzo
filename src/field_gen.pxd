# -*- coding: utf-8 -*-

import numpy as np
cimport numpy as np

from defs cimport *

cdef np.ndarray[double, ndim=4] gen_sol2d(int,int,     double, int,int, double[3])
cdef np.ndarray[double, ndim=4] gen_sol3d(int,int,int, double, int,int, double[3])
