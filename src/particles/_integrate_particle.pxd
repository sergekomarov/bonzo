#cython: language_level=2
# -*- coding: utf-8 -*-

from src.data_struct cimport *

# from openmp cimport omp_lock_t

cdef void feedback_predict(Domain)
cdef void feedback_correct(Domain)
