# -*- coding: utf-8 -*-

from bnz.defs_cy cimport *
from bnz.coord.grid cimport GridCoord, GridData
from bnz.particle.particle cimport PrtProp, PrtData
from bnz.mhd.integrate cimport BnzIntegr

# Prt boundary conditions.

cdef void x1_prt_bc_periodic(PrtData*, PrtProp*, GridData, GridCoord*, BnzIntegr)
cdef void x2_prt_bc_periodic(PrtData*, PrtProp*, GridData, GridCoord*, BnzIntegr)
cdef void y1_prt_bc_periodic(PrtData*, PrtProp*, GridData, GridCoord*, BnzIntegr)
cdef void y2_prt_bc_periodic(PrtData*, PrtProp*, GridData, GridCoord*, BnzIntegr)
cdef void z1_prt_bc_periodic(PrtData*, PrtProp*, GridData, GridCoord*, BnzIntegr)
cdef void z2_prt_bc_periodic(PrtData*, PrtProp*, GridData, GridCoord*, BnzIntegr)

cdef void x1_prt_bc_outflow(PrtData*, PrtProp*, GridData, GridCoord*, BnzIntegr)
cdef void x2_prt_bc_outflow(PrtData*, PrtProp*, GridData, GridCoord*, BnzIntegr)
cdef void y1_prt_bc_outflow(PrtData*, PrtProp*, GridData, GridCoord*, BnzIntegr)
cdef void y2_prt_bc_outflow(PrtData*, PrtProp*, GridData, GridCoord*, BnzIntegr)
cdef void z1_prt_bc_outflow(PrtData*, PrtProp*, GridData, GridCoord*, BnzIntegr)
cdef void z2_prt_bc_outflow(PrtData*, PrtProp*, GridData, GridCoord*, BnzIntegr)

cdef void x1_prt_bc_reflect(PrtData*, PrtProp*, GridData, GridCoord*, BnzIntegr)
cdef void x2_prt_bc_reflect(PrtData*, PrtProp*, GridData, GridCoord*, BnzIntegr)
cdef void y1_prt_bc_reflect(PrtData*, PrtProp*, GridData, GridCoord*, BnzIntegr)
cdef void y2_prt_bc_reflect(PrtData*, PrtProp*, GridData, GridCoord*, BnzIntegr)
cdef void z1_prt_bc_reflect(PrtData*, PrtProp*, GridData, GridCoord*, BnzIntegr)
cdef void z2_prt_bc_reflect(PrtData*, PrtProp*, GridData, GridCoord*, BnzIntegr)

cdef void realloc_recvbuf(real2d, long*)
cdef void realloc_sendbuf(real2d, long*)

cdef long x1_pack_shift_prt(PrtData*, PrtProp*, real2d, long*, real,real)
cdef long x2_pack_shift_prt(PrtData*, PrtProp*, real2d, long*, real,real)
cdef long y1_pack_shift_prt(PrtData*, PrtProp*, real2d, long*, real,real)
cdef long y2_pack_shift_prt(PrtData*, PrtProp*, real2d, long*, real,real)
cdef long z1_pack_shift_prt(PrtData*, PrtProp*, real2d, long*, real,real)
cdef long z2_pack_shift_prt(PrtData*, PrtProp*, real2d, long*, real,real)

cdefv void unpack_prt(PrtData*,PrtProp*, real2d, long)
