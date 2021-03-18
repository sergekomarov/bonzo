# -*- coding: utf-8 -*-

cdef extern from "sys/types.h" nogil:
  ctypedef int time_t
  ctypedef int subseconds_t

cdef extern from "sys/time.h" nogil:
  cdef struct timeval:
    time_t tv_sec
    subseconds_t tv_usec
  int gettimeofday(timeval*, void*)

#---------------------------------------------------

IF SPREC:
  ctypedef float real
  # np_real = np.float32
  ctypedef float[::1] real1d
  ctypedef float[:,::1] real2d
  ctypedef float[:,:,::1] real3d
  ctypedef float[:,:,:,::1] real4d
ELSE:
  ctypedef double real
  # np_real = np.float64
  ctypedef double[::1] real1d
  ctypedef double[:,::1] real2d
  ctypedef double[:,:,::1] real3d
  ctypedef double[:,:,:,::1] real4d

ctypedef int[::1] int1d
ctypedef int[:,::1] int2d
ctypedef int[:,:,::1] int3d

#---------------------------------------------------

cdef extern from "defs_c.h" nogil:

  real B_PI
  real SMALL_NUM

  int NVAR
  int NMODE
  enum: XAX,YAX,ZAX

  enum: RHO, MX,MY,MZ, EN
  enum: VX,VY,VZ, PR
  enum: PSC

  enum: BXC,BYC,BZC
  enum: BXF,BYF,BZF
  enum: FC0, FCX, FCY, FCZ

IF MFIELD:
  cdef extern from "defs_c.h" nogil:
    enum: BX,BY,BZ

IF CGL:
  cdef extern from "defs_c.h" nogil:
    enum: LA
    enum: PPD

IF TWOTEMP:
  cdef extern from "defs_c.h" nogil:
    enum: SE
    enum: PE

IF MHDPIC:
  cdef extern from "defs_c.h" nogil:
    int NPRT_PROP

#---------------------------------------------------

cdef extern from "defs_c.h" nogil:
  real SQR(real)
  real CUBE(real)
  real SQRT(real)
  real FABS(real)
  real SIGN(real)
  real LOG(real)
  real POW(real)
  real EXP(real)
  real FLOOR(real)
  real FMIN(real)
  real FMAX(real)
  int IMIN(int)
  int IMAX(int)
  real SIN(real)
  real COS(real)
