# -*- coding: utf-8 -*-

# Definitions of types and external C functions.

cdef extern from "sys/types.h" nogil:
  ctypedef int time_t
  ctypedef int subseconds_t

cdef extern from "sys/time.h" nogil:
  cdef struct timeval:
    time_t tv_sec
    subseconds_t tv_usec
  int gettimeofday(timeval*, void*)


#---------------------------------------------------

# Type shortcuts.

# Use this type for indexing.
ctypedef Py_ssize_t ints

# Define "real" depending on precision.

IF SPREC:
  ctypedef float real
  # np_real = np.float32
  ctypedef float[::1] real1d
  ctypedef float[:,::1] real2d
  ctypedef float[:,:,::1] real3d
  ctypedef float[:,:,:,::1] real4d
  ctypedef float[:,:,:,:,::1] real5d
ELSE:
  ctypedef double real
  # np_real = np.float64
  ctypedef double[::1] real1d
  ctypedef double[:,::1] real2d
  ctypedef double[:,:,::1] real3d
  ctypedef double[:,:,:,::1] real4d
  ctypedef double[:,:,:,:,::1] real5d

ctypedef ints[::1] ints1d
ctypedef ints[:,::1] ints2d
ctypedef ints[:,:,::1] ints3d

#---------------------------------------------------

# Import constants from the C definitions file.

cdef extern from "defs.h" nogil:
  double B_PI

cdef extern from "defs.h" nogil:
  int NVARS
  enum: XAX,YAX,ZAX

IF not PIC:

  cdef extern from "defs.h" nogil:

    int NWAVES

    enum: RHO, MX,MY,MZ, EN
    enum: VX,VY,VZ, PR
    enum: PSC
    enum: BXC,BYC,BZC
    enum: BXF,BYF,BZF
    enum: MU,NU,MUT,GAMT

  IF MFIELD:
    cdef extern from "defs.h" nogil:
      enum: BX,BY,BZ

  IF CGL:
    cdef extern from "defs.h" nogil:
      enum: LA
      enum: PPD

  IF TWOTEMP:
    cdef extern from "defs.h" nogil:
      enum: SE
      enum: PE

  IF MHDPIC:
    cdef extern from "defs.h" nogil:
      int NPRT_PROP

ELSE:

  cdef extern from "defs.h" nogil:
    enum: EX,EY,EZ, BX,BY,BZ, JX,JY,JZ
    int NPRT_PROP
