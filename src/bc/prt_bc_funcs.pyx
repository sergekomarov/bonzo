# -*- coding: utf-8 -*-

import numpy as np
cimport numpy as np
from cython.parallel import prange, parallel

IF SPREC:
  np_real = np.float32
ELSE:
  np_real = np.float64


# ===========================================================================

# Periodic BCs for particles.

cdef void x1_prt_bc_periodic(PrtData *pd, PrtProp *pp,
                             GridData gd, GridCoord *gc, BnzIntegr integr):

  cdef long n
  cdef real dx = gc.lmax[0]-gc.lmin[0]

  for n in range(pp.Np):
    if pd.x[n] < gc.lmin[0]:
      pd.x[n] = pd.x[n] + dx


cdef void x2_prt_bc_periodic(PrtData *pd, PrtProp *pp,
                             GridData gd, GridCoord *gc, BnzIntegr integr):

  cdef long n
  cdef real dx = gc.lmax[0]-gc.lmin[0]

  for n in range(pp.Np):
    if pd.x[n] >= gc.lmax[0]:
      pd.x[n] = pd.x[n] - dx


cdef void y1_prt_bc_periodic(PrtData *pd, PrtProp *pp,
                             GridData gd, GridCoord *gc, BnzIntegr integr):

  cdef long n
  cdef real dy = gc.lmax[1]-gc.lmin[1]

  for n in range(pp.Np):
    if pd.y[n] < gc.lmin[1]:
      pd.y[n] = pd.y[n] + dy


cdef void y2_prt_bc_periodic(PrtData *pd, PrtProp *pp,
                             GridData gd, GridCoord *gc, BnzIntegr integr):

  cdef long n
  cdef real dy = gc.lmax[1]-gc.lmin[1]

  for n in range(pp.Np):
    if pd.y[n] >= gc.lmax[1]:
      pd.y[n] = pd.y[n] - dy


cdef void z1_prt_bc_periodic(PrtData *pd, PrtProp *pp,
                             GridData gd, GridCoord *gc, BnzIntegr integr):

  cdef long n
  cdef real dz = gc.lmax[2]-gc.lmin[2]

  for n in range(pp.Np):
    if pd.z[n] < gc.lmin[2]:
      pd.z[n] = pd.z[n] + dz

cdef void z2_prt_bc_periodic(PrtData *pd, PrtProp *pp,
                             GridData gd, GridCoord *gc, BnzIntegr integr):

  cdef long n
  cdef real dz = gc.lmax[2]-gc.lmin[2]

  for n in range(pp.Np):
    if pd.z[n] >= gc.lmax[2]:
      pd.z[n] = pd.z[n] - dz



# =====================================================================

# Outflow BCs for particles.
# Remove particles as they recede 1 ghost cell away from active domain.

cdef void x1_prt_bc_outflow(PrtData *pd, PrtProp *pp,
                            GridData gd, GridCoord *gc, BnzIntegr integr):

  cdef long n

  for n in range(pp.Np):
    if pd.x[n] < gc.lmin[0]:
      delete_particle(pd,pp, n)
      n = n-1


cdef void x2_prt_bc_outflow(PrtData *pd, PrtProp *pp,
                            GridData gd, GridCoord *gc, BnzIntegr integr):

  cdef long n

  for n in range(pp.Np):
    if pd.x[n] >= gc.lmax[0]:
      delete_particle(pd,pp, n)
      n = n-1


cdef void y1_prt_bc_outflow(PrtData *pd, PrtProp *pp,
                            GridData gd, GridCoord *gc, BnzIntegr integr):

  cdef long n

  for n in range(pp.Np):
    if pd.y[n] < gc.lmin[1]:
      delete_particle(pd,pp, n)
      n = n-1


cdef void y2_prt_bc_outflow(PrtData *pd, PrtProp *pp,
                            GridData gd, GridCoord *gc, BnzIntegr integr):

  cdef long n

  for n in range(pp.Np):
    if pd.y[n] >= gc.lmax[1]:
      delete_particle(pd,pp, n)
      n = n-1


cdef void z1_prt_bc_outflow(PrtData *pd, PrtProp *pp,
                            GridData gd, GridCoord *gc, BnzIntegr integr):

  cdef long n

  for n in range(pp.Np):
    if pd.z[n] < gc.lmin[2]:
      delete_particle(pd,pp, n)
      n = n-1


cdef void z2_prt_bc_outflow(PrtData *pd, PrtProp *pp,
                            GridData gd, GridCoord *gc, BnzIntegr integr):

  cdef long n

  for n in range(pp.Np):
    if pd.z[n] >= gc.lmax[2]:
      delete_particle(pd,pp, n)
      n = n-1


# =================================================================

# Reflective BCs for particles.

cdef void x1_prt_bc_reflect(PrtData *pd, PrtProp *pp,
                            GridData gd, GridCoord *gc, BnzIntegr integr):

  cdef long n

  for n in range(pp.Np):
    if pd.x[n] < gc.lmin[0]:
      pd.x[n] = - pd.x[n]
      pd.u[n] = - pd.u[n]


cdef void x2_prt_bc_reflect(PrtData *pd, PrtProp *pp,
                            GridData gd, GridCoord *gc, BnzIntegr integr):

  cdef long n

  for n in range(pp.Np):
    if pd.x[n] >= gc.lmax[0]:
      pd.x[n] = 2.*gc.lmax[0] - pd.x[n]
      pd.u[n] = -pd.u[n]


cdef void y1_prt_bc_reflect(PrtData *pd, PrtProp *pp,
                            GridData gd, GridCoord *gc, BnzIntegr integr):

  cdef long n

  for n in range(pp.Np):
    if pd.y[n] < gc.lmin[1]:
      pd.y[n] = - pd.y[n]
      pd.v[n] = - pd.v[n]


cdef void y2_prt_bc_reflect(PrtData *pd, PrtProp *pp,
                            GridData gd, GridCoord *gc, BnzIntegr integr):

  cdef long n

  for n in range(pp.Np):
    if pd.y[n] >= gc.lmax[1]:
      pd.y[n] = 2.*gc.lmax[1] - pd.y[n]
      pd.v[n] = -pd.v[n]


cdef void z1_prt_bc_reflect(PrtData *pd, PrtProp *pp,
                            GridData gd, GridCoord *gc, BnzIntegr integr):

  cdef long n

  for n in range(pp.Np):
    if pd.z[n] < gc.lmin[2]:
      pd.z[n] = - pd.z[n]
      pd.w[n] = - pd.w[n]


cdef void z2_prt_bc_reflect(PrtData *pd, PrtProp *pp,
                            GridData gd, GridCoord *gc, BnzIntegr integr):

  cdef long n

  for n in range(pp.Np):
    if pd.z[n] >= gc.lmax[2]:
      pd.z[n] = 2.*gc.lmax[2] - pd.z[n]
      pd.w[n] = -pd.w[n]



# ====================================================================

cdef void delete_particle(PrtData *pd, PrtProp *pp, long n):

  cdef long Np = pp.Np

  pp.Np = Np-1
  pp.spc_props[pd.spc[n]].Np -= 1

  pd.x[n] = pd.x[Np-1]
  pd.y[n] = pd.y[Np-1]
  pd.z[n] = pd.z[Np-1]
  pd.u[n] = pd.u[Np-1]
  pd.v[n] = pd.v[Np-1]
  pd.w[n] = pd.w[Np-1]
  pd.g[n] = pd.g[Np-1]
  pd.m[n] = pd.m[Np-1]
  pd.spc[n] = pd.spc[Np-1]
  pd.id[n] = pd.id[Np-1]


# -----------------------------------------------------------------------

IF MPI:

  cdef void realloc_recvbuf(real2d recvbuf, long *recvbuf_size):

    recvbuf_size[0] *= 2
    recvbuf = np.zeros((2,recvbuf_size[0]), dtype=np_real)

  cdef void realloc_sendbuf(real2d sendbuf, long *sendbuf_size):

    sendbuf_size[0] *= 2
    sendbuf = np.zeros((2,sendbuf_size[0]), dtype=np_real)



  cdef long x1_pack_shift_prt(PrtData *pd, PrtProp *pp, real2d sendbuf,
                              long *sendbuf_size, real xmin, real xmax):

    cdef:
      long n,i
      real dx = xmax-xmin

    i=0

    for n in range(pp.Np):

      if pd.x[n] < xmin:

        sendbuf[0,i] = pd.x[n] + dx
        i=i+1
        sendbuf[0,i] = pd.y[n]
        i=i+1
        sendbuf[0,i] = pd.z[n]
        i=i+1
        sendbuf[0,i] = pd.u[n]
        i=i+1
        sendbuf[0,i] = pd.v[n]
        i=i+1
        sendbuf[0,i] = pd.w[n]
        i=i+1
        sendbuf[0,i] = pd.g[n]
        i=i+1
        sendbuf[0,i] = pd.m[n]
        i=i+1
        sendbuf[0,i] = pd.spc[n] + 0.1
        i=i+1
        sendbuf[0,i] = pd.id[n] + 0.1
        i=i+1

        # remove particle from this processor
        delete_particle(pd,pp, n)
        n = n-1

        if i+11 > sendbuf_size[0]:
          realloc_sendbuf(sendbuf, sendbuf_size)
          i = 0
          n = 0

    return i



  cdef long x2_pack_shift_prt(PrtData *pd, PrtProp *pp, real2d sendbuf,
                              long *sendbuf_size, real xmin, real xmax):

    cdef:
      long n,i
      real dx = xmax-xmin

    i=0

    for n in range(pp.Np):

      if pd.x[n] >= xmax:

        sendbuf[0,i] = pd.x[n] - dx
        i=i+1
        sendbuf[0,i] = pd.y[n]
        i=i+1
        sendbuf[0,i] = pd.z[n]
        i=i+1
        sendbuf[0,i] = pd.u[n]
        i=i+1
        sendbuf[0,i] = pd.v[n]
        i=i+1
        sendbuf[0,i] = pd.w[n]
        i=i+1
        sendbuf[0,i] = pd.g[n]
        i=i+1
        sendbuf[0,i] = pd.m[n]
        i=i+1
        sendbuf[0,i] = pd.spc[n] + 0.1
        i=i+1
        sendbuf[0,i] = pd.id[n] + 0.1
        i=i+1

        # remove particle from this processor
        delete_particle(pd,pp, n)
        n = n-1

        if i+11 > sendbuf_size[0]:
          realloc_sendbuf(sendbuf, sendbuf_size)
          i = 0
          n = 0

    return i


  cdef long y1_pack_shift_prt(PrtData *pd, PrtProp *pp, real2d sendbuf,
                            long *sendbuf_size, real ymin, real ymax):

    cdef:
      long n,i
      real dy = ymax-ymin

    i=0

    for n in range(pp.Np):

      if pd.y[n] < ymin:

        sendbuf[0,i] = pd.x[n]
        i=i+1
        sendbuf[0,i] = pd.y[n] + dy
        i=i+1
        sendbuf[0,i] = pd.z[n]
        i=i+1
        sendbuf[0,i] = pd.u[n]
        i=i+1
        sendbuf[0,i] = pd.v[n]
        i=i+1
        sendbuf[0,i] = pd.w[n]
        i=i+1
        sendbuf[0,i] = pd.g[n]
        i=i+1
        sendbuf[0,i] = pd.m[n]
        i=i+1
        sendbuf[0,i] = pd.spc[n] + 0.1
        i=i+1
        sendbuf[0,i] = pd.id[n] + 0.1
        i=i+1

        # remove particle from this processor
        delete_particle(pd,pp, n)
        n = n-1

        if i+11 > sendbuf_size[0]:
          realloc_sendbuf(sendbuf, sendbuf_size)
          i = 0
          n = 0

    return i


  cdef long y2_pack_shift_prt(PrtData *pd, PrtProp *pp, real2d sendbuf,
                              long *sendbuf_size, real ymin, real ymax):

    cdef:
      long n,i
      real dy = ymax-ymin

    i=0

    for n in range(pp.Np):

      if pd.y[n] >= ymax:

        sendbuf[0,i] = pd.x[n]
        i=i+1
        sendbuf[0,i] = pd.y[n] - dy
        i=i+1
        sendbuf[0,i] = pd.z[n]
        i=i+1
        sendbuf[0,i] = pd.u[n]
        i=i+1
        sendbuf[0,i] = pd.v[n]
        i=i+1
        sendbuf[0,i] = pd.w[n]
        i=i+1
        sendbuf[0,i] = pd.g[n]
        i=i+1
        sendbuf[0,i] = pd.m[n]
        i=i+1
        sendbuf[0,i] = pd.spc[n] + 0.1
        i=i+1
        sendbuf[0,i] = pd.id[n] + 0.1
        i=i+1

        # remove particle from this processor
        delete_particle(pd,pp, n)
        n = n-1

        if i+11 > sendbuf_size[0]:
          realloc_sendbuf(sendbuf, sendbuf_size)
          i = 0
          n = 0

    return i


  cdef long z1_pack_shift_prt(PrtData *pd, PrtProp *pp, real2d sendbuf,
                            long *sendbuf_size, real zmin, real zmax):

    cdef:
      long n,i
      real dz = zmax-zmin

    i=0

    for n in range(pp.Np):

      if pd.z[n] < zmin:

        sendbuf[0,i] = pd.x[n]
        i=i+1
        sendbuf[0,i] = pd.y[n]
        i=i+1
        sendbuf[0,i] = pd.z[n] + dz
        i=i+1
        sendbuf[0,i] = pd.u[n]
        i=i+1
        sendbuf[0,i] = pd.v[n]
        i=i+1
        sendbuf[0,i] = pd.w[n]
        i=i+1
        sendbuf[0,i] = pd.g[n]
        i=i+1
        sendbuf[0,i] = pd.m[n]
        i=i+1
        sendbuf[0,i] = pd.spc[n] + 0.1
        i=i+1
        sendbuf[0,i] = pd.id[n] + 0.1
        i=i+1

        # remove particle from this processor
        delete_particle(pd,pp, n)
        n = n-1

        if i+11 > sendbuf_size[0]:
          realloc_sendbuf(sendbuf, sendbuf_size)
          i = 0
          n = 0

    return i


  cdef long z2_pack_shift_prt(PrtData *pd, PrtProp *pp, real2d sendbuf,
                            long *sendbuf_size, real zmin, real zmax):

    cdef:
      long n,i
      real dz = zmax-zmin

    i=0

    for n in range(pp.Np):

      if pd.z[n] >= zmax:

        sendbuf[0,i] = pd.x[n]
        i=i+1
        sendbuf[0,i] = pd.y[n]
        i=i+1
        sendbuf[0,i] = pd.z[n] - dz
        i=i+1
        sendbuf[0,i] = pd.u[n]
        i=i+1
        sendbuf[0,i] = pd.v[n]
        i=i+1
        sendbuf[0,i] = pd.w[n]
        i=i+1
        sendbuf[0,i] = pd.g[n]
        i=i+1
        sendbuf[0,i] = pd.m[n]
        i=i+1
        sendbuf[0,i] = pd.spc[n] + 0.1
        i=i+1
        sendbuf[0,i] = pd.id[n] + 0.1
        i=i+1

        # remove particle from this processor
        delete_particle(pd,pp, n)
        n = n-1

        if i+11 > sendbuf_size[0]:
          realloc_sendbuf(sendbuf, sendbuf_size)
          i = 0
          n = 0

    return i


  cdef void unpack_prt(PrtData *pd, PrtProp *pp, real2d recvbuf, long recvbuf_size):

    cdef long n,i

    n = pp.Np
    i=0

    while i < recvbuf_size:

      pd.x[n] = recvbuf[0,i]
      i=i+1
      pd.y[n] = recvbuf[0,i]
      i=i+1
      pd.z[n] = recvbuf[0,i]
      i=i+1
      pd.u[n] = recvbuf[0,i]
      i=i+1
      pd.v[n] = recvbuf[0,i]
      i=i+1
      pd.w[n] = recvbuf[0,i]
      i=i+1
      pd.g[n] = recvbuf[0,i]
      i=i+1
      pd.m[n] = recvbuf[0,i]
      i=i+1
      pd.spc[n] = <int>recvbuf[0,i]
      i=i+1
      pd.id[n] = <long>recvbuf[0,i]
      i=i+1

      n += 1
      pp.Np += 1
      pp.spc_props[pd.spc[n]].Np += 1
