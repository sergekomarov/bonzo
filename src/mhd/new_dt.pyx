# -*- coding: utf-8 -*-

IF MPI:
  from mpi4py import MPI as mpi
  from mpi4py cimport MPI as mpi

import numpy as np
cimport numpy as np
from cython.parallel import prange, parallel, threadid

from libc.math cimport sqrt,floor,ceil,log,exp,sin,cos,pow,fabs,fmin,fmax
from libc.stdlib cimport malloc, calloc, free
from libc.stdio cimport printf

from bnz.utils cimport print_root, sqr
from bnz.coord.coordinates cimport get_cell_width_x, get_cell_width_y, get_cell_width_z


cdef double new_dt(real4d w, GridCoord *gc, BnzIntegr integr):

  cdef:

    ints i,j,k, n, ig,jg,kg
    ints id

    real vfx,vfy,vfz, cx,cy,cz, rhoi
    real dlmin
    real dt, dtmhd,dtp
    real bx, by, bz, b2
    real dsx,dsy,dsz

    real cds
    real cdsmax=0, b2max=0

  cdef:
    real1d cdsmaxloc=np.zeros(OMP_NT)
    real1d b2maxloc=np.zeros(OMP_NT)
    real *_w

  IF MPI:
    cdef:
      mpi.Comm comm = mpi.COMM_WORLD
      double[::1] var     = np.empty(1, dtype='f8')
      double[::1] var_max = np.empty(1, dtype='f8')


  with nogil, parallel(num_threads=OMP_NT):

    id = threadid()
    _w = <real*>calloc(NMODE, sizeof(real))

    for k in prange(gc.k1, gc.k2+1, schedule='dynamic'):

      cx, cy, cz = 0.,0.,0.
      bx, by, bz = 0.,0.,0.

      for j in range(gc.j1, gc.j2+1):
        for i in range(gc.i1, gc.i2+1):

          for n in range(NMODE):
            _w[n] = w[n,k,j,i]

          # if W[PR,k,j,i]<0:
          #   lind2gind(&ig,&jg,&kg, i,j,k, gp)
          #   printf('p=%f at (%i,%i,%i)\n', W[PR,k,j,i], ig,jg,kg)

          # if W[RHO,k,j,i]<0:
          #   lind2gind(&ig,&jg,&kg, i,j,k, gp)
          #   printf('rho=%f at (%i,%i,%i)\n', W[RHO,k,j,i], ig,jg,kg)

          IF MFIELD: bx,by,bz = _w[BX],_w[BY],_w[BZ]

          vfx = fms(_w, bx, integr.gam)
          cx = fabs(_w[VX]) + vfx

          IF D2D:
            vfy = fms(_w, by, integr.gam)
            cy = fabs(_w[VY]) + vfy

          IF D3D:
            vfz = fms(_w, bz, integr.gam)
            cz = fabs(_w[VZ]) + vfz

          dsx = get_cell_width_x(gc,i,j,k)
          dsy = get_cell_width_y(gc,i,j,k)
          dsz = get_cell_width_z(gc,i,j,k)

          cds = fmax(fmax(cx/dsx, cy/dsy), cz/dsz)

          if cds > cdsmaxloc[id]: cdsmaxloc[id] = cds

          IF MHDPIC:
            b2 = sqr(_w[BX]) + sqr(_w[BY]) + sqr(_w[BZ])
            if b2 > b2maxloc[id]: b2maxloc[id] = b2

    free(_w)


  #-----------------------------------------------

  for i in range(OMP_NT):
    if cdsmaxloc[i] > cdsmax: cdsmax = cdsmaxloc[i]
    IF MHDPIC:
      if b2maxloc[i] > b2max: b2max = b2maxloc[i]

  IF MPI:
    var[0] = cdsmax
    comm.Allreduce(var, var_max, op=mpi.MAX)
    cdsmax = var_max[0]
    IF MHDPIC:
      var[0] = b2max
      comm.Allreduce(var, var_max, op=mpi.MAX)
      b2max = var_max[0]

  dtmhd = cour / cdsmax
  IF MHDPIC:
    # ! only for uniform Cartesian grids
    dlmin = fmin(fmin(gc.dlf[0][0], gc.dlf[1][0]), gc.dlf[2][0])
    dtp = 0.9*fmin(1./(3 * integr.q_mc * sqrt(b2max)),
                  dlmin / integr.sol)

  print_root("MHD dt = %f\n", dtmhd)
  IF MHDPIC:
    print_root("Particle dt = %f\n", dtp)

  IF MHDPIC: dt = fmin(dtmhd, dtp)
  ELSE: dt = dtmhd

  return dt
