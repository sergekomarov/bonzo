# -*- coding: utf-8 -*-

from bnz.defs_cy cimport *
from bnz.coord.grid cimport *

# Particle boundary conditions.

cdef class BnzSim

cdef void x1_prt_bc_periodic(BnzSim)
cdef void x2_prt_bc_periodic(BnzSim)
cdef void y1_prt_bc_periodic(BnzSim)
cdef void y2_prt_bc_periodic(BnzSim)
cdef void z1_prt_bc_periodic(BnzSim)
cdef void z2_prt_bc_periodic(BnzSim)

cdef void x1_prt_bc_outflow(BnzSim)
cdef void x2_prt_bc_outflow(BnzSim)
cdef void y1_prt_bc_outflow(BnzSim)
cdef void y2_prt_bc_outflow(BnzSim)
cdef void z1_prt_bc_outflow(BnzSim)
cdef void z2_prt_bc_outflow(BnzSim)

cdef void x1_prt_bc_reflective(BnzSim)
cdef void x2_prt_bc_reflective(BnzSim)
cdef void y1_prt_bc_reflective(BnzSim)
cdef void y2_prt_bc_reflective(BnzSim)
cdef void z1_prt_bc_reflective(BnzSim)
cdef void z2_prt_bc_reflective(BnzSim)

cdef void realloc_recvbuf(real2d, ints*)
cdef void realloc_sendbuf(real2d, ints*)

cdef void x1_pack_shift_prt(BnzParticles, real2d, ints*, real,real)
cdef void x2_pack_shift_prt(BnzParticles, real2d, ints*, real,real)
cdef void y1_pack_shift_prt(BnzParticles, real2d, ints*, real,real)
cdef void y2_pack_shift_prt(BnzParticles, real2d, ints*, real,real)
cdef void z1_pack_shift_prt(BnzParticles, real2d, ints*, real,real)
cdef void z2_pack_shift_prt(BnzParticles, real2d, ints*, real,real)

cdefv void unpack_prt(BnzParticles, real2d, ints)
