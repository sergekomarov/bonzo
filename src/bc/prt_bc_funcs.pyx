# -*- coding: utf-8 -*-

IF MPI:
  from mpi4py import MPI as mpi
  from mpi4py cimport MPI as mpi

import numpy as np
cimport numpy as np
from cython.parallel import prange, parallel

from libc.math cimport sqrt,floor,ceil,log,exp,sin,cos,pow,fabs,fmin,fmax

IF SPREC:
  np_real = np.float32
ELSE:
  np_real = np.float64


# ===========================================================================

# Periodic BCs for particles.

cdef void x1_prt_bc_periodic(BnzSim sim):

  cdef:
    ParticleData *pd = sim.grid.prts.data
    ints n, Np=sim.grid.prts.prop.Np
    real lmin = sim.grid.coord.lmin[0]
    real lmax = sim.grid.coord.lmax[0]
    real dx   = xmax-xmin

  for n in range(Np):
    if pd.x[n] < xmin:
      pd.x[n] = pd.x[n] + dx


cdef void x2_prt_bc_periodic(BnzSim sim):

  cdef:
    ParticleData *pd = sim.grid.prts.data
    ints n, Np=sim.grid.prts.prop.Np
    real xmin = sim.grid.coord.lmin[0]
    real xmax = sim.grid.coord.lmax[0]
    real dx   = xmax-xmin

  for n in range(Np):
    if pd.x[n] >= xmax:
      pd.x[n] = pd.x[n] - dx


cdef void y1_prt_bc_periodic(BnzSim sim):

  cdef:
    ParticleData *pd = sim.grid.prts.data
    ints n, Np=sim.grid.prts.prop.Np
    real ymin = sim.grid.coord.lmin[1]
    real ymax = sim.grid.coord.lmax[1]
    real dy   = ymax-ymin

  for n in range(Np):
    if pd.y[n] < ymin:
      pd.y[n] = pd.y[n] + dy


cdef void y2_prt_bc_periodic(BnzSim sim):

  cdef:
    ParticleData *pd = sim.grid.prts.data
    ints n, Np=sim.grid.prts.prop.Np
    real ymin = sim.grid.coord.lmin[1]
    real ymax = sim.grid.coord.lmax[1]
    real dy   = ymax-ymin

  for n in range(Np):
    if pd.y[n] >= ymax:
      pd.y[n] = pd.y[n] - dy


cdef void z1_prt_bc_periodic(BnzSim sim):

  cdef:
    ParticleData *pd = sim.grid.prts.data
    ints n, Np=sim.grid.prts.prop.Np
    real zmin = sim.grid.coord.lmin[2]
    real zmax = sim.grid.coord.lmax[2]
    real dz   = zmax-zmin

  for n in range(Np):
    if pd.z[n] < zmin:
      pd.z[n] = pd.z[n] + dz

cdef void z2_prt_bc_periodic(BnzSim sim):

  cdef:
    ParticleData *pd = sim.grid.prts.data
    ints n, Np=sim.grid.prts.prop.Np
    real zmin = sim.grid.coord.lmin[2]
    real zmax = sim.grid.coord.lmax[2]
    real dz   = zmax-zmin

  for n in range(Np):
    if pd.z[n] >= zmax:
      pd.z[n] = pd.z[n] - dz



# =====================================================================

# Outflow BCs for particles.
# Remove particles as they recede 1 ghost cell away from active domain.

cdef void x1_prt_bc_outflow(BnzSim sim):

  cdef:
    BnzParticles prts = sim.grid.prts
    ParticleData *pd = prts.data
    ints n, Np = prts.prop.Np
    real xmin = sim.grid.coord.lmin[0]

  for n in range(Np):
    if pd.x[n] < xmin:
      delete_particle(prts, n)
      n = n-1


cdef void x2_prt_bc_outflow(BnzSim sim):

  cdef:
    BnzParticles prts = sim.grid.prts
    ParticleData *pd = prts.data
    ints n, Np=prts.prop.Np
    real xmax = sim.grid.coord.lmax[0]

  for n in range(Np):
    if pd.x[n] >= xmax:
      delete_particle(prts, n)
      n = n-1


cdef void y1_prt_bc_outflow(BnzSim sim):

  cdef:
    BnzParticles prts = sim.grid.prts
    ParticleData *pd = prts.data
    ints n, Np=prts.prop.Np
    real ymin = sim.grid.coord.lmin[1]

  for n in range(Np):
    if pd.y[n] < ymin:
      delete_particle(prts, n)
      n = n-1


cdef void y2_prt_bc_outflow(BnzSim sim):

  cdef:
    BnzParticles prts = sim.grid.prts
    ParticleData *pd = prts.data
    ints n, Np=prts.prop.Np
    real ymax = sim.grid.coord.lmax[1]

  for n in range(Np):
    if pd.y[n] >= ymax:
      delete_particle(prts, n)
      n = n-1


cdef void z1_prt_bc_outflow(BnzSim sim):

  cdef:
    BnzParticles prts = sim.grid.prts
    ParticleData *pd = prts.data
    ints n, Np=prts.prop.Np
    real zmin = sim.grid.coord.lmin[2]

  for n in range(Np):
    if pd.z[n] < zmin:
      delete_particle(prts, n)
      n = n-1


cdef void z2_prt_bc_outflow(BnzSim sim):

  cdef:
    BnzParticles prts = sim.grid.prts
    ParticleData *pd = prts.data
    ints n, Np=prts.prop.Np
    real zmax = sim.grid.coord.lmax[2]

  for n in range(Np):
    if pd.z[n] >= zmax:
      delete_particle(prts, n)
      n = n-1


# =================================================================

# Reflective BCs for particles.

cdef void x1_prt_bc_reflect(BnzSim sim):

  cdef:
    ParticleData *pd = sim.grid.prts.data
    ints n, Np=sim.grid.prts.prop.Np
    real xmin = sim.grid.coord.lmin[0]

  for n in range(Np):
    if pd.x[n] < xmin:
      pd.x[n] = - pd.x[n]
      pd.u[n] = - pd.u[n]


cdef void x2_prt_bc_reflect(BnzSim sim):

  cdef:
    ParticleData *pd = sim.grid.prts.data
    ints n, Np=sim.grid.prts.prop.Np
    real xmax = sim.grid.coord.lmax[0]

  for n in range(Np):
    if pd.x[n] >= xmax:
      pd.x[n] = 2*xmax - pd.x[n]
      pd.u[n] = -pd.u[n]


cdef void y1_prt_bc_reflect(BnzSim sim):

  cdef:
    ParticleData *pd = sim.grid.prts.data
    ints n, Np=sim.grid.prts.prop.Np
    real ymin = sim.grid.coord.lmin[1]

  for n in range(Np):
    if pd.y[n] < ymin:
      pd.y[n] = - pd.y[n]
      pd.v[n] = - pd.v[n]


cdef void y2_prt_bc_reflect(BnzSim sim):

  cdef:
    ParticleData *pd = sim.grid.prts.data
    ints n, Np=sim.grid.prts.prop.Np
    real ymax = sim.grid.coord.lmax[1]

  for n in range(Np):
    if pd.y[n] >= ymax:
      pd.y[n] = 2*ymax - pd.y[n]
      pd.v[n] = -pd.v[n]


cdef void z1_prt_bc_reflect(BnzSim sim):

  cdef:
    ParticleData *pd = sim.grid.prts.data
    ints n, Np=sim.grid.prts.prop.Np
    real zmin = sim.grid.coord.lmin[2]

  for n in range(Np):
    if pd.z[n] < zmin:
      pd.z[n] = - pd.z[n]
      pd.w[n] = - pd.w[n]


cdef void z2_prt_bc_reflect(BnzSim sim):

  cdef:
    ParticleData *pd = sim.grid.prts.data
    ints n, Np=sim.grid.prts.prop.Np
    real zmax = sim.grid.coord.lmax[2]

  for n in range(Np):
    if pd.z[n] >= zmax:
      pd.z[n] = 2*zmax - pd.z[n]
      pd.w[n] = -pd.w[n]



# =============================================================

cdef void delete_particle(BnzParticles prts, ints n):

  cdef:
    ParticleData *pd = prts.data
    ints Np = prts.prop.Np

  prts.prop.Np = Np-1
  prts.prop.spc_props[pd.spc[n]].Np -= 1

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


# IF MPI:

  # cdef ints pack_shift_prt_x1(BnzParticles prts, real2d sendbuf, ints *sendbuf_size, double Lx):
  #
  #   cdef:
  #     ints n,i,nsel
  #     real2d sb = sendbuf
  #     ParticleData *pd = &(prts.data)
  #
  #   i=0
  #   nsel=0
  #
  #   for n in range(prts.Np):
  #
  #     if pd.x[n] < 0:
  #
  #       sb[1,i] = n
  #       nsel=nsel+1
  #
  #       sb[0,i] = pd.x[n] + Lx
  #       i=i+1
  #
  #   if i*NPRT_PROP > sendbuf_size[0]:
  #
  #     while i*NPRT_PROP > sendbuf_size[0]:
  #       realloc_sendbuf(sendbuf, sendbuf_size)
  #
  #     i=0
  #     nsel=0
  #
  #     for n in range(prts.Np):
  #       if pd.x[n] < 0:
  #         sb[1,i] = n
  #         nsel=nsel+1
  #         sb[0,i] = pd.x[n] + Lx
  #         i=i+1
  #
  #   for n in range(nsel):
  #     sb[0,i] = pd.y[sb[1,n]]
  #     i=i+1
  #
  #   for n in range(nsel):
  #     sb[0,i] = pd.z[sb[1,n]]
  #     i=i+1
  #
  #   for n in range(nsel):
  #     sb[0,i] = pd.u[sb[1,n]]
  #     i=i+1
  #
  #   for n in range(nsel):
  #     sb[0,i] = pd.v[sb[1,n]]
  #     i=i+1
  #
  #   for n in range(nsel):
  #     sb[0,i] = pd.w[sb[1,n]]
  #     i=i+1
  #
  #   for n in range(nsel):
  #     sb[0,i] = pd.g[sb[1,n]]
  #     i=i+1
  #
  #   for n in range(nsel):
  #     sb[0,i] = pd.m[sb[1,n]]
  #     i=i+1
  #
  #   for n in range(nsel):
  #     sb[0,i] = pd.spc[sb[1,n]]+0.1
  #     i=i+1
  #
  #   for n in range(nsel):
  #     sb[0,i] = pd.id[sb[1,n]]+0.1
  #     i=i+1
  #
  #   for n in range(nsel):
  #     delete_particle(prts, sb[1,n])
  #     IF PIC:
  #       prts.spc_props[pd.spc[n]].Np = prts.spc_props[pd.spc[n]].Np - 1
  #
  #   return i


IF MPI:

  cdef void realloc_recvbuf(real2d recvbuf, ints *recvbuf_size):

    recvbuf_size[0] *= 2
    recvbuf = np.zeros((2,recvbuf_size[0]), dtype=np_real)

  cdef void realloc_sendbuf(real2d sendbuf, ints *sendbuf_size):

    sendbuf_size[0] *= 2
    sendbuf = np.zeros((2,sendbuf_size[0]), dtype=np_real)



  cdef ints x1_pack_shift_prt(BnzParticles prts, real2d sendbuf,
                              ints *sendbuf_size, real xmin, real xmax):

    cdef:
      ints n,i
      real dx = xmax-xmin
      ParticleData *pd = prts.data

    i=0

    for n in range(prts.prop.Np):

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
        delete_particle(prts, n)
        n = n-1

        if i+11 > sendbuf_size[0]:
          realloc_sendbuf(sendbuf, sendbuf_size)
          i = 0
          n = 0

    return i



  cdef ints x2_pack_shift_prt(BnzParticles prts, real2d sendbuf,
                              ints *sendbuf_size, real xmin, real xmax):

    cdef:
      ints n,i
      real dx = xmax-xmin
      ParticleData *pd = prts.data

    i=0

    for n in range(prts.prop.Np):

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
        delete_particle(prts, n)
        n = n-1

        if i+11 > sendbuf_size[0]:
          realloc_sendbuf(sendbuf, sendbuf_size)
          i = 0
          n = 0

    return i


  cdef ints y1_pack_shift_prt(BnzParticles prts, real2d sendbuf,
                              ints *sendbuf_size, real ymin, real ymax):

    cdef:
      ints n,i
      real dy = ymax-ymin
      ParticleData *pd = prts.data

    i=0

    for n in range(prts.prop.Np):

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
        delete_particle(prts, n)
        n = n-1

        if i+11 > sendbuf_size[0]:
          realloc_sendbuf(sendbuf, sendbuf_size)
          i = 0
          n = 0

    return i


  cdef ints y2_pack_shift_prt(BnzParticles prts, real2d sendbuf,
                              ints *sendbuf_size, real ymin, real ymax):

    cdef:
      ints n,i
      real dy = ymax-ymin
      ParticleData *pd = prts.data

    i=0

    for n in range(prts.prop.Np):

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
        delete_particle(prts, n)
        n = n-1

        if i+11 > sendbuf_size[0]:
          realloc_sendbuf(sendbuf, sendbuf_size)
          i = 0
          n = 0

    return i


  cdef ints z1_pack_shift_prt(BnzParticles prts, real2d sendbuf,
                              ints *sendbuf_size, real zmin, real zmax):

    cdef:
      ints n,i
      real dz = zmax-zmin
      ParticleData *pd = prts.data

    i=0

    for n in range(prts.prop.Np):

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
        delete_particle(prts, n)
        n = n-1

        if i+11 > sendbuf_size[0]:
          realloc_sendbuf(sendbuf, sendbuf_size)
          i = 0
          n = 0

    return i


  cdef ints z2_pack_shift_prt(BnzParticles prts, real2d sendbuf,
                              ints *sendbuf_size, real zmin, real zmax):

    cdef:
      ints n,i
      real dz = zmax-zmin
      ParticleData *pd = prts.data

    i=0

    for n in range(prts.prop.Np):

      if pd.z[n] >= zmax:

        sendbuf[0,i] = pd.x[n]
        i=i+1
        sendbuf[0,i] = pd.y[n]
        i=i+1
        sendbuf[0,i] = pd.z[n] - Lz
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
        delete_particle(prts, n)
        n = n-1

        if i+11 > sendbuf_size[0]:
          realloc_sendbuf(sendbuf, sendbuf_size)
          i = 0
          n = 0

    return i



  cdef void unpack_prt(BnzParticles prts, real2d recvbuf, ints recvbuf_size):

    cdef:
      ints n,i
      ParticleData *pd = prts.data

    n = prts.prop.Np
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
      pd.spc[n] = <ints>recvbuf[0,i]
      i=i+1
      pd.id[n] = <ints>recvbuf[0,i]
      i=i+1

      n = n+1
      prts.prop.Np = prts.prop.Np+1
      prts.spc_props[pd.spc[n]].Np = prts.spc_props[pd.spc[n]].Np + 1
