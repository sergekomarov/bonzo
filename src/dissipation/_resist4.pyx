# -*- coding: utf-8 -*-

import numpy as np
cimport numpy as np
from cython.parallel import prange, parallel

from libc.math cimport sqrt,floor,ceil,log,exp,sin,cos,pow,fabs,fmin,fmax
from libc.stdlib cimport rand, RAND_MAX, srand
from libc.stdlib cimport calloc,malloc, free

from bnz.utils cimport print_root#, curl1d,curl2d,curl3d
from bnz.mhd.ct cimport interp_b_field
cimport utils_diffuse
cimport bnz.bc.bc_grid as bc_grid


# ==============================================

# Evolve magnetic field by super-time-stepping.

cdef void diffuse_sts_res4(BnzSim sim):

  cdef:
    int n, s, ngh
    double dt_diff

  cdef:
    double[:,::1] sts_coeff
    ints[::1] bc_vars
    real4d B0=sim.grid.tmp.V0, Bm1=sim.grid.tmp.Vm1, MB0=sim.grid.tmp.MV0

  cdef ints rank=0
  IF MPI:
    rank=mpi.COMM_WORLD.Get_rank()


  ngh = sim.grid.ng/2

  sim.phys.eta4 = 0.5/sim.dt * (sim.grid.dl[0]/1.7)**4

  bc_vars = np.asarray([BX,BY,BZ])

  # dt_diff = get_dt_res(dom) * 0.25 * dom.dx**2 * dom.params.eta / dom.params.eta4

  s = 1 #utils_diffuse.get_s(dom.dt, dt_diff)
  print_root(rank, "\nresistivity, N_STS=%d ... ", s)

  sts_coeff = utils_diffuse.get_sts_coeff(s)

  for n in range(1,s+1):

    # need 2 layers of ghost cells to update energy
    # apply BC only if all layers from previous application have been used
    if (n-1)%ngh == 0: bc_grid.apply_bc_grid(sim, bc_vars)

    sts_iterate_res4(sim.grid, B0,Bm1,MB0, sts_coeff, sim.phys, n, sim.dt)

  bc_grid.apply_bc_grid(sim, bc_vars)

  # calculate cell-centered magnetic field
  interp_b_field(sim.grid.U, sim.grid)

  # need 2 ghost cells of (BX,BY,BZ) and 1 of (BXC,BYC,BZC)
  update_nrg_res4(sim.grid, B0, sim.phys, sim.dt)



# ===============================================================================

cdef void sts_iterate_res4(BnzGrid grid,
            real4d B0, real4d Bm1, real4d MB0,
            double[:,::1] sts_coeff, BnzPhysics phys,
            int n, double dt) nogil:

  cdef:
    ints i,j,m,k
    double a

    ints i1_2=grid.i1_2, i2_2=grid.i2_2
    ints j1_2=grid.j1_2, j2_2=grid.j2_2
    ints m1_2=grid.m1_2, m2_2=grid.m2_2

  cdef:
    real4d U = grid.U
    # temporary local array
    real4d MB = grid.tmp.MV


  Mres4(MB, grid,phys,dt)

  if n==1:
    for i in prange(i1_2, i2_2+1, nogil=True, num_threads=OMP_NT, schedule='dynamic'):
      for j in range(j1_2, j2_2+1):
        for m in range(m1_2, m2_2+1):
          for k in range(3):
            B0[i,j,m,k] =   U[i,j,m,BX+k]
            MB0[i,j,m,k] = MB[i,j,m,k]

  for i in prange(i1_2, i2_2+1, nogil=True, num_threads=OMP_NT, schedule='dynamic'):
    for j in range(j1_2, j2_2+1):
      for m in range(m1_2, m2_2+1):
        for k in range(3):

          a = U[i,j,m,BX+k]
          U[i,j,m,BX+k] =           ( sts_coeff[MU,  n]  *   U[i,j,m,BX+k] +
                                      sts_coeff[NU,  n]  * Bm1[i,j,m,k]
             + (1 - sts_coeff[MU,n] - sts_coeff[NU,  n]) *  B0[i,j,m,k]
                                    + sts_coeff[MUT, n]  *  MB[i,j,m,k]
                                    + sts_coeff[GAMT,n]  * MB0[i,j,m,k] )
          Bm1[i,j,m,k] = a


# ===============================================================

# Apply magnetic hyperdiffusion matrix operator to magnetic field

cdef real Mres4(real4d MB, BnzGrid grid,
                BnzPhysics phys, double dt) nogil:

  cdef:
    ints i,j,m,k

    ints i1_2 = grid.i1_2, i2_2=grid.i2_2
    ints j1_2 = grid.j1_2, j2_2=grid.j2_2
    ints m1_2 = grid.m1_2, m2_2=grid.m2_2

    double dteta = - phys.eta4 * dt

  cdef real4d U = grid.U


  with nogil, parallel(num_threads=OMP_NT):

    for i in prange(i1_2, i2_2+1, schedule='dynamic'):
      for j in range(j1_2, j2_2+1):
        for m in range(m1_2, m2_2+1):
          for k in range(BX,BZ+1):
            MB[i,j,m,k-BX] = ( U[i-2,j,m,k] - 4*U[i-1,j,m,k] + 6*U[i,j,m,k]
                           - 4*U[i+1,j,m,k] +   U[i+2,j,m,k] )

    IF D2D:
      for i in prange(i1_2, i2_2+1, schedule='dynamic'):
        for j in range(j1_2, j2_2+1):
          for m in range(m1_2, m2_2+1):
            for k in range(BX,BZ+1):
              MB[i,j,m,k-BX] = ( MB[i,j,m,k-BX]
                  + U[i,j-2,m,k] - 4*U[i,j-1,m,k] + 6*U[i,j,m,k]
                - 4*U[i,j+1,m,k] +   U[i,j+2,m,k] )

    IF D3D:
      for i in prange(i1_2, i2_2+1, schedule='dynamic'):
        for j in range(j1_2, j2_2+1):
          for m in range(m1_2, m2_2+1):
            for k in range(BX,BZ+1):
              MB[i,j,m,k-BX] = ( MB[i,j,m,k-BX]
                  + U[i,j,m-2,k] - 4*U[i,j,m-1,k] + 6*U[i,j,m,k]
                - 4*U[i,j,m+1,k] +   U[i,j,m+2,k] )

    for i in prange(i1_2, i2_2+1, schedule='dynamic'):
      for j in range(j1_2, j2_2+1):
        for m in range(m1_2, m2_2+1):
          for k in range(3):
            MB[i,j,m,k] = dteta * MB[i,j,m,k]



# =======================================================

# Calculate resistive energy release using Pointing flux.

cdef void update_nrg_res4(BnzGrid grid, real4d B0,
                          BnzPhysics phys, double dt) nogil:

  cdef:
    ints i,j,m,k
    double pe, p0, ppd_ppl0, ppd0, p, ppd_ppl, ppd, dp
    double rhoi
    double B2,B02,B3,B03, B0cx,B0cy,B0cz
    double eb,eb0, ek0, de, deb

  cdef:
    ints i1=grid.i1, i2=grid.i2
    ints j1=grid.j1, j2=grid.j2
    ints m1=grid.m1, m2=grid.m2
    double dxi=grid.dli[0], dyi=grid.dli[1], dzi=grid.dli[2]
    double dxi2=dxi**2, dyi2=dyi**2, dzi2=dzi**2

    double gamm1 = phys.gam-1
    double dteta = - phys.eta4 * dt

  cdef:
    real4d U = grid.U

    # temporary local arrays
    real4d Sp = grid.tmp.MV0
    real4d J = grid.tmp.MV
    real4d E = grid.tmp.Vm1


  IF D2D and D3D:

    for i in range(1, grid.Ntot[0]):
      for j in range(1, grid.Ntot[1]):
        for m in range(1, grid.Ntot[2]):
          J[i,j,m,0] = dyi*(U[i,j,m,BZ]-U[i,j-1,m,BZ]) - dzi*(U[i,j,m,BY]-U[i,j,m-1,BY])
          J[i,j,m,1] = dzi*(U[i,j,m,BX]-U[i,j,m-1,BX]) - dxi*(U[i,j,m,BZ]-U[i-1,j,m,BZ])
          J[i,j,m,2] = dxi*(U[i,j,m,BY]-U[i-1,j,m,BY]) - dyi*(U[i,j,m,BX]-U[i,j-1,m,BX])

    for i in range(2, grid.Ntot[0]-1):
      for j in range(2, grid.Ntot[1]-1):
        for m in range(2, grid.Ntot[2]-1):
          for k in range(3):

            E[i,j,m,k] = ( dxi2 * (J[i-1,j,m,k] - 2*J[i,j,m,k] + J[i+1,j,m,k])
                         + dyi2 * (J[i,j-1,m,k] - 2*J[i,j,m,k] + J[i,j+1,m,k])
                         + dzi2 * (J[i,j,m-1,k] - 2*J[i,j,m,k] + J[i,j,m+1,k]) )

    for i in range(i1, i2+2):
      for j in range(j1, j2+2):
        for m in range(m1, m2+2):

          Sp[i,j,m,0] = 0.25*((E[i,j,m,1] + E[i,j,m+1,1]) * (U[i-1,j,m,BZC] + U[i,j,m,BZC])
                            - (E[i,j,m,2] + E[i,j+1,m,2]) * (U[i-1,j,m,BYC] + U[i,j,m,BYC]))

          Sp[i,j,m,1] = 0.25*((E[i,j,m,2] + E[i+1,j,m,2]) * (U[i,j-1,m,BXC] + U[i,j,m,BXC])
                            - (E[i,j,m,0] + E[i,j,m+1,0]) * (U[i,j-1,m,BZC] + U[i,j,m,BZC]))

          Sp[i,j,m,2] = 0.25*((E[i,j,m,0] + E[i,j+1,m,0]) * (U[i,j,m-1,BYC] + U[i,j,m,BYC])
                            - (E[i,j,m,1] + E[i+1,j,m,1]) * (U[i,j,m-1,BXC] + U[i,j,m,BXC]))

  ELIF D2D:

    for i in range(1,grid.Ntot[0]):
      for j in range(1,grid.Ntot[1]):
        for m in range(m1,m2+1):
          J[i,j,m,0] =   dyi*(U[i,j,m,BZ]-U[i,j-1,m,BZ])
          J[i,j,m,1] = - dxi*(U[i,j,m,BZ]-U[i-1,j,m,BZ])
          J[i,j,m,2] =   dxi*(U[i,j,m,BY]-U[i-1,j,m,BY]) - dyi*(U[i,j,m,BX]-U[i,j-1,m,BX])

    for i in range(2, grid.Ntot[0]-1):
      for j in range(2, grid.Ntot[1]-1):
        for m in range(m1,m2+1):
          for k in range(3):

            E[i,j,m,k] = ( dxi2 * (J[i-1,j,m,k] - 2*J[i,j,m,k] + J[i+1,j,m,k])
                         + dyi2 * (J[i,j-1,m,k] - 2*J[i,j,m,k] + J[i,j+1,m,k]) )

    for i in range(i1,i2+2):
      for j in range(j1,j2+2):
        for m in range(m1,m2+1):

          Sp[i,j,m,0] = 0.25*(2*E[i,j,m,1] *                 (U[i-1,j,m,BZC] + U[i,j,m,BZC])
                             - (E[i,j,m,2] + E[i,j+1,m,2]) * (U[i-1,j,m,BYC] + U[i,j,m,BY]))

          Sp[i,j,m,1] = 0.25*( (E[i,j,m,2] + E[i+1,j,m,2]) * (U[i,j-1,m,BXC] + U[i,j,m,BXC])
                            - 2*E[i,j,m,0] *                 (U[i,j-1,m,BZC] + U[i,j,m,BZC]))

  ELSE:

    for i in range(1, grid.Ntot[0]):
      for j in range(j1,j2+1):
        for m in range(m1,m2+1):
          J[i,j,m,0] =   0.
          J[i,j,m,1] = - dxi*(U[i,j,m,BZ]-U[i-1,j,m,BZ])
          J[i,j,m,2] =   dxi*(U[i,j,m,BY]-U[i-1,j,m,BY])

    for i in range(2, grid.Ntot[1]-1):
      for j in range(j1,j2+1):
        for m in range(m1,m2+1):
          for k in range(3):
            E[i,j,m,k] = dxi2 * (J[i-1,j,m,k] - 2*J[i,j,m,k] + J[i+1,j,m,k])

    for i in range(i1, i2+2):
      for j in range(j1, j2+1):
        for m in range(m1, m2+1):

          Sp[i,j,m,0] = 0.5*(E[i,j,m,1] * (U[i-1,j,m,BZC] + U[i,j,m,BZC])
                          -  E[i,j,m,2] * (U[i-1,j,m,BYC] + U[i,j,m,BYC]))


  for i in prange(i1,i2+1, nogil=True, num_threads=OMP_NT, schedule='dynamic'):
    for j in range(j1,j2+1):
      for m in range(m1,m2+1):

        # ! Sp was initialized to zeros
        de = dxi * (Sp[i+1,j,m,0] - Sp[i,j,m,0])
        IF D2D: de = de + dyi * (Sp[i,j+1,m,1] - Sp[i,j,m,1])
        IF D3D: de = de + dzi * (Sp[i,j,m+1,2] - Sp[i,j,m,2])
        de = de * dteta

        IF TWOTEMP or CGL:

          rhoi = 1./U[i,j,m,RHO]
          B2 =   U[i,j,m,BXC]**2 + U[i,j,m,BYC]**2 + U[i,j,m,BZC]**2
          B0cx = B0[i,j,m,0]+B0[i+1,j,m,0]

          IF D2D: B0cy = B0[i,j,m,1]+B0[i,j+1,m,1]
          ELSE:   B0cy = 2*B0[i,j,m,1]

          IF D3D: B0cz = B0[i,j,m,2]+B0[i,j,m+1,2]
          ELSE:   B0cz = 2*B0[i,j,m,2]

          B02 = 0.25*(B0cx**2 + B0cy**2 + B0cz**2)

          eb0 = 0.5 * B02
          deb = 0.5 * (B2 - B02)
          dp = gamm1 * (de - deb)

          IF TWOTEMP:
            pe = exp(U[i,j,m,SE] * rhoi) * pow(U[i,j,m,RHO], phys.gam)
            U[i,j,m,SE] = U[i,j,m,RHO] * (log(pe+dp) - phys.gam * log(U[i,j,m,RHO]))

          ELIF CGL:

            ek0 = 0.5*(U[i,j,m,MX]**2 + U[i,j,m,MY]**2 + U[i,j,m,MZ]**2) * rhoi
            p0 = gamm1 * (U[i,j,m,EN] - eb0 - ek0)

            B03 = B02*sqrt(B02)
            B3 = B2*sqrt(B2)

            ppd_ppl0 = exp(U[i,j,m,LA] * rhoi) * B03 * rhoi**2
            ppd0 = p0 * 3*ppd_ppl0 / (1 + 2*ppd_ppl0)

            ppd = ppd0 + dp
            p = p0 + dp
            ppd_ppl = ppd / (3*p - 2*ppd)
            U[i,j,m,LA] = U[i,j,m,RHO] * log(ppd_ppl * U[i,j,m,RHO]**2 / B3)

        U[i,j,m,EN] = U[i,j,m,EN] + de
