# -*- coding: utf-8 -*-

IF MPI:
  from mpi4py import MPI as mpi
  from mpi4py cimport MPI as mpi

import numpy as np
cimport numpy as np
from cython.parallel import prange, parallel, threadid

from libc.math cimport M_PI, sqrt,floor,ceil,log,exp,sin,cos,pow,fabs,fmin,fmax
from libc.stdlib cimport rand, RAND_MAX, srand
from libc.stdlib cimport calloc,malloc, free

from bnz.utils cimport print_root, maxi,mini#, curl1d,curl2d,curl3d
cimport utils_diffuse
cimport bnz.bc.bc_grid as bc_grid
from bnz.mhd.ct cimport interp_b_field


# !!! DIFFUSION ROUTINES DO NOT SET Bx(i2+1,j,m), By(i,j2+1,m), Bz(i,j,m2+1)
# !!! THESE SHOULD BE SET BY APPLYING BOUNDARY CONDITIONS


# ==============================================

# Evolve magnetic field by super-time-stepping.

cdef void diffuse_sts_res(BnzSim sim):

  cdef:
    int n, s, ng
    double dt_diff

  cdef:
    double[:,::1] sts_coeff
    ints[::1] bc_vars
    # arrays shared over STS iterations
    real4d B0=sim.grid.tmp.V0, Bm1=sim.grid.tmp.Vm1, MB0=sim.grid.tmp.MV0

  cdef ints rank=0
  IF MPI:
    rank=mpi.COMM_WORLD.Get_rank()


  ng = sim.grid.ng

  dt_diff = get_dt_res(sim.grid, sim.phys, sim.method.cour_diff)

  s = utils_diffuse.get_s(sim.dt, dt_diff)
  print_root(rank, "\nresistivity, N_STS=%d ... ", s)

  sts_coeff = utils_diffuse.get_sts_coeff(s)

  # apply BC only to face-centered field
  bc_vars = np.asarray([BX,BY,BZ], dtype=np.intp)

  for n in range(1,s+1):

    # need 1 layer of ghost cells to do 1 iteration
    if (n-1)%ng==0: bc_grid.apply_bc_grid(sim, bc_vars)

    sts_iterate_res(sim.grid, B0,Bm1,MB0, sts_coeff, sim.phys, n, dt_diff)

  # need 1 layer of ghost cells to update energy
  # apply BC only if all layers from previous application have been used
  if (n-1)%ng == ng-1: bc_grid.apply_bc_grid(sim, bc_vars)

  # calculate cell-centered magnetic field
  interp_b_field(sim.grid.U, sim.grid)

  update_nrg_res(sim.grid, B0, sim.phys, sim.dt)



# ==============================================================================

cdef void sts_iterate_res(BnzGrid grid,
            real4d B0, real4d Bm1, real4d MB0,
            double[:,::1] sts_coeff, BnzPhysics phys,
            int n, double dt) nogil:

  cdef:
    ints i,j,m,k
    double a

    ints i1_1=grid.i1_1, i2_1=grid.i2_1
    ints j1_1=grid.j1_1, j2_1=grid.j2_1
    ints m1_1=grid.m1_1, m2_1=grid.m2_1

  cdef:
    real4d U = grid.U
    # temporary local arrays
    real4d MB = grid.tmp.MV


  Mres(MB, grid,phys,dt)

  if n==1:
    for i in prange(i1_1, i2_1+1, nogil=True, num_threads=OMP_NT, schedule='dynamic'):
      for j in range(j1_1, j2_1+1):
        for m in range(m1_1, m2_1+1):
          for k in range(3):
            B0[i,j,m,k] =   U[i,j,m,BX+k]
            MB0[i,j,m,k] = MB[i,j,m,k]

  for i in prange(i1_1, i2_1+1, nogil=True, num_threads=OMP_NT, schedule='dynamic'):
    for j in range(j1_1, j2_1+1):
      for m in range(m1_1, m2_1+1):
        for k in range(3):

          a = U[i,j,m,BX+k]
          U[i,j,m,BX+k] =           ( sts_coeff[MU,  n]  *   U[i,j,m,BX+k] +
                                      sts_coeff[NU,  n]  * Bm1[i,j,m,k]
             + (1 - sts_coeff[MU,n] - sts_coeff[NU,  n]) *  B0[i,j,m,k]
                                    + sts_coeff[MUT, n]  *  MB[i,j,m,k]
                                    + sts_coeff[GAMT,n]  * MB0[i,j,m,k] )
          Bm1[i,j,m,k] = a



# ===========================================================

# Apply magnetic diffusion matrix operator to magnetic field.

cdef real Mres(real4d MB, BnzGrid grid,
               BnzPhysics phys, double dt) nogil:

  cdef:
    ints i,j,m,k

    ints i1_1 = grid.i1_1, i2_1=grid.i2_1
    ints j1_1 = grid.j1_1, j2_1=grid.j2_1
    ints m1_1 = grid.m1_1, m2_1=grid.m2_1

    double dteta = phys.eta * dt

  cdef real4d U = grid.U


  with nogil, parallel(num_threads=OMP_NT):

    for i in prange(i1_1, i2_1+1, schedule='dynamic'):
      for j in range(j1_1, j2_1+1):
        for m in range(m1_1, m2_1+1):
          for k in range(BX,BZ+1):

            MB[i,j,m,k-BX] = U[i-1,j,m,k] - 2*U[i,j,m,k] + U[i+1,j,m,k]

    IF D2D:
      for i in prange(i1_1, i2_1+1, schedule='dynamic'):
        for j in range(j1_1, j2_1+1):
          for m in range(m1_1, m2_1+1):
            for k in range(BX,BZ+1):

              MB[i,j,m,k-BX] = MB[i,j,m,k-BX] + U[i,j+1,m,k] - 2*U[i,j,m,k] + U[i,j-1,m,k]

    IF D3D:
      for i in prange(i1_1, i2_1+1, schedule='dynamic'):
        for j in range(j1_1, j2_1+1):
          for m in range(m1_1, m2_1+1):
            for k in range(BX,BZ+1):

              MB[i,j,m,k-BX] = MB[i,j,m,k-BX] + U[i,j,m+1,k] - 2*U[i,j,m,k] + U[i,j,m-1,k]

    for i in prange(i1_1, i2_1+1, schedule='dynamic'):
      for j in range(j1_1, j2_1+1):
        for m in range(m1_1, m2_1+1):
          for k in range(3):
            MB[i,j,m,k] = dteta * MB[i,j,m,k]



# ========================================================

# Calculate resistive energy release using Pointing flux.

# !!! Can make it second-order in time by calculating time averaged m.f. first.

cdef void update_nrg_res(BnzGrid grid, real4d B0,
                         BnzPhysics phys, double dt) nogil:

  cdef:
    ints i,j,m
    double Jx,Jy,Jz
    double pe, p0, ppd_ppl0, ppd0, p, ppd_ppl, ppd, dp
    double rhoi
    double B2,B02,B3,B03, B0cx,B0cy,B0cz
    double eb,eb0, ek0, de, deb

  cdef:
    ints i1=grid.i1, i2=grid.i2
    ints j1=grid.j1, j2=grid.j2
    ints m1=grid.m1, m2=grid.m2
    double dxi=grid.dli[0], dyi=grid.dli[1], dzi=grid.dli[2]

    double gam = phys.gam
    double gamm1 = gam-1
    double dteta = phys.eta * dt

  cdef:
    real4d U = grid.U
    # temporary local array
    real4d Sp = grid.tmp.MV


  IF D2D and D3D:

    for i in range(i1, i2+2):
      for j in range(j1, j2+2):
        for m in range(m1, m2+2):

          Jy =   0.5*dzi*(U[i,j,m+1,BX] - U[i,j,m-1,BX]) - dxi*(U[i,j,m,BZC] - U[i-1,j,m,BZC])
          Jz = - 0.5*dyi*(U[i,j+1,m,BX] - U[i,j-1,m,BX]) + dxi*(U[i,j,m,BYC] - U[i-1,j,m,BYC])

          Sp[i,j,m,0] = ( Jy * 0.5*(U[i-1,j,m,BZC] + U[i,j,m,BZC])
                        - Jz * 0.5*(U[i-1,j,m,BYC] + U[i,j,m,BYC]) )

          Jx = - 0.5*dzi*(U[i,j,m+1,BY] - U[i,j,m-1,BY]) + dyi*(U[i,j,m,BZC] - U[i,j-1,m,BZC])
          Jz =   0.5*dxi*(U[i+1,j,m,BY] - U[i-1,j,m,BY]) - dyi*(U[i,j,m,BXC] - U[i,j-1,m,BXC])

          Sp[i,j,m,1] = ( - Jx * 0.5*(U[i,j-1,m,BZC] + U[i,j,m,BZC])
                          + Jz * 0.5*(U[i,j-1,m,BXC] + U[i,j,m,BXC]) )

          Jx =   0.5*dyi*(U[i,j+1,m,BZ] - U[i,j-1,m,BZ]) - dzi*(U[i,j,m,BYC] - U[i,j,m-1,BYC])
          Jy = - 0.5*dxi*(U[i+1,j,m,BZ] - U[i-1,j,m,BZ]) + dzi*(U[i,j,m,BXC] - U[i,j,m-1,BXC])

          Sp[i,j,m,2] = ( Jx * 0.5*(U[i,j,m-1,BYC] + U[i,j,m,BYC])
                        - Jy * 0.5*(U[i,j,m-1,BXC] + U[i,j,m,BXC]) )

  ELIF D2D:

    for i in range(i1, i2+2):
      for j in range(j1, j2+2):
        for m in range(m1, m2+1):

          Jy = - dxi*(U[i,j,m,BZC] - U[i-1,j,m,BZC])
          Jz =   dxi*(U[i,j,m,BYC] - U[i-1,j,m,BYC]) - 0.5*dyi*(U[i,j+1,m,BX] - U[i,j-1,m,BX])

          Sp[i,j,m,0] = ( Jy * 0.5*(U[i-1,j,m,BZC] + U[i,j,m,BZC])
                        - Jz * 0.5*(U[i-1,j,m,BYC] + U[i,j,m,BYC]) )

          Jx =   dyi*(U[i,j,m,BZC] - U[i,j-1,m,BZC])
          Jz = - dyi*(U[i,j,m,BXC] - U[i,j-1,m,BXC]) + 0.5*dxi*(U[i+1,j,m,BY] - U[i-1,j,m,BY])

          Sp[i,j,m,1] = ( - Jx * 0.5*(U[i,j-1,m,BZC] + U[i,j,m,BZC])
                          + Jz * 0.5*(U[i,j-1,m,BXC] + U[i,j,m,BXC]) )

  ELSE:

    for i in range(i1, i2+2):
      for j in range(j1, j2+1):
        for m in range(m1, m2+1):

          Jy = - dxi*(U[i,j,m,BZC] - U[i-1,j,m,BZC])
          Jz =   dxi*(U[i,j,m,BYC] - U[i-1,j,m,BYC])

          Sp[i,j,m,0] = ( Jy * 0.5*(U[i-1,j,m,BZC] + U[i,j,m,BZC])
                        - Jz * 0.5*(U[i-1,j,m,BYC] + U[i,j,m,BYC]) )


  # !!!!!!!!!!!!!
  for i in prange(i1, i2+1, nogil=True, num_threads=OMP_NT, schedule='dynamic'):
    for j in range(j1, j2+1):
      for m in range(m1, m2+1):

        de = dxi*(Sp[i+1,j,m,0] - Sp[i,j,m,0])
        IF D2D: de = de + dyi*(Sp[i,j+1,m,1] - Sp[i,j,m,1])
        IF D3D: de = de + dzi*(Sp[i,j,m+1,2] - Sp[i,j,m,2])
        de = dteta * de

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



# ==============================================================

# Calculate diffusive time step.

cdef double get_dt_res(BnzGrid grid, BnzPhysics phys, double cour_diff):

  cdef:
    ints i,j,m, k,k1
    int id
    double dBx=0, dBy=0, dBz=0, dBi=0
    double D, D_dl2, D_dl2_max = 0.
    double dxi2=grid.dli[0]**2, dyi2=grid.dli[1]**2, dzi2=grid.dli[2]**2

  cdef:
    real4d U = grid.U
    double *D_dl2_max_loc = <double *>calloc(OMP_NT, sizeof(double))

  IF MPI:
    cdef:
      double[::1] var     = np.empty(1, dtype='f8')
      double[::1] var_max = np.empty(1, dtype='f8')


  for i in prange(grid.i1, grid.i2+1, nogil=True, num_threads=OMP_NT, schedule='dynamic'):
    id = threadid()
    for j in range(grid.j1, grid.j2+1):
      for m in range(grid.m1, grid.m2+1):
        for k in range(3):

          D = phys.eta

          k1 = BXC+k

          dBx = U[i+1,j,m,k1] - U[i-1,j,m,k1]
          IF D2D: dBy = U[i,j+1,m,k1] - U[i,j-1,m,k1]
          IF D3D: dBz = U[i,j,m+1,k1] - U[i,j,m-1,k1]

          dBi = 1./sqrt(dBx**2 + dBy**2 + dBz**2 + 1e-20)

          D_dl2 = dxi2 * D * fabs(dBx*dBi)
          IF D2D: D_dl2 = D_dl2 + dyi2 * D * fabs(dBy*dBi)
          IF D3D: D_dl2 = D_dl2 + dzi2 * D * fabs(dBz*dBi)

          if D_dl2 > D_dl2_max_loc[id]: D_dl2_max_loc[id] = D_dl2


  for i in range(OMP_NT):
    if D_dl2_max_loc[i] > D_dl2_max: D_dl2_max = D_dl2_max_loc[i]

  free(D_dl2_max_loc)

  IF MPI:
    var[0] = D_dl2_max
    mpi.COMM_WORLD.Allreduce(var, var_max, op=mpi.MAX)
    D_dl2_max = var_max[0]

  return cour_diff / D_dl2_max



# =========================================================

# Semi-implicit directionally-split viscosity integrator.

cdef void diffuse_implicit_res(BnzSim sim):

  cdef:
    ints[::1] bc_vars
    real4d B0 = sim.grid.tmp.V0

  bc_vars = np.asarray([BX,BY,BZ], dtype=np.intp)
  bc_grid.apply_bc_grid(sim, bc_vars)

  # implicit update of magnetic field (requires one layer of ghost cells)
  implicit_update_res(sim.grid, B0, sim.phys, sim.dt, sim.bc.bc_flags)

  # set BC for energy update (need one layer of ghost cells)
  bc_grid.apply_bc_grid(sim, bc_vars)

  # calculate cell-centered magnetic field
  interp_b_field(sim.grid.U, sim.grid)

  update_nrg_res(sim.grid, B0, sim.phys, sim.dt)



# =======================================================

# Semi-implicit directionally-split magnetic field update
# due to resistivity.

cdef void implicit_update_res(BnzGrid grid, real4d B0, BnzPhysics phys,
                              double dt, int bc_flags[3][2]) nogil:

  cdef:
    ints i,j,m,k,n, ii,jj,mm
    ints Ntot_max, Nact_max

    double dtetai

    real *a
    real *b
    real *c
    real *d
    real *X1
    real *X2
    real *d2
    real *B1d

  cdef:
    real4d U = grid.U

  cdef:

    ints Nx=grid.Nact[0], Ny=grid.Nact[1], Nz=grid.Nact[2]

    ints i1=grid.i1, i2=grid.i2
    ints j1=grid.j1, j2=grid.j2
    ints m1=grid.m1, m2=grid.m2

    double dxi2=grid.dli[0]**2, dyi2=grid.dli[1]**2, dzi2=grid.dli[2]**2


  dtetai = 1./ (dt * phys.eta)

  Ntot_max = maxi(maxi(grid.Ntot[0],grid.Ntot[1]),grid.Ntot[2])
  Nact_max = maxi(maxi(grid.Nact[0],grid.Nact[1]),grid.Nact[2])

  with nogil, parallel(num_threads=OMP_NT):

    for i in prange(i1, i2+1, schedule='dynamic'):
      for j in range(j1, j2+1):
        for m in range(m1, m2+1):
          for k in range(3):
            B0[i,j,m,k] = U[i,j,m,BX+k]

    a = <real *>calloc(Nact_max+1, sizeof(real))
    b = <real *>calloc(Nact_max+1, sizeof(real))
    c = <real *>calloc(Nact_max+1, sizeof(real))
    d = <real *>calloc(Nact_max+1, sizeof(real))
    B1d = <real *>calloc(Ntot_max, sizeof(real))

    # for periodic bc
    X1 = <real *>calloc(Nact_max, sizeof(real))
    X2 = <real *>calloc(Nact_max, sizeof(real))
    d2 = <real *>calloc(Nact_max, sizeof(real))

    for j in prange(j1,j2+1, schedule='dynamic'):
      for m in range(m1,m2+1):
        for k in range(3):

          for i in range(i1, i2+1):

            ii = i-i1

            d[ii] = dtetai * U[i,j,m,BX+k]
            a[ii] = -dxi2
            c[ii] = dtetai + 2*dxi2
            b[ii] = -dxi2

          for i in range(i1-1, i2+2):
            B1d[i] = U[i,j,m,BX+k]

          # reflection (transverse B component) or outflow
          if bc_flags[0][0]==1 or (bc_flags[0][0]==2 and k!=0):
            c[0] = c[0] + a[0]
          if bc_flags[0][1]==1 or (bc_flags[0][1]==2 and k!=0):
            c[Nx-1] = c[Nx-1] + b[Nx-1]

          # reflection (parallel B component)
          if k==0:
            if bc_flags[0][0]==2:
              b[0] = b[0]-a[0]
            if bc_flags[0][1]==2:
              b[Nx-1] = 0

          #inflow
          if bc_flags[0][0]==3:
            d[0] = d[0] - a[0] * B1d[i1-1]
          if bc_flags[0][1]==3:
            d[Nx-1] = d[Nx-1] - b[Nx-1] * B1d[i2+1]

          if (bc_flags[0][0] != 0 and bc_flags[0][1] != 0):

            utils_diffuse.TDMA(&B1d[i1], a,b,c,d, Nx)
            for i in range(i1, i2+1):
              U[i,j,m,BX+k] = B1d[i]

          #periodic
          if bc_flags[0][0]==0 and bc_flags[0][1]==0:

            utils_diffuse.TDMAper(&B1d[i1], a,b,c,d, d2,X1,X2, Nx)
            for i in range(i1, i2+1):
              U[i,j,m,BX+k] = B1d[i]
            # U[i2+1,j,m,BX+k] = U[i1,j,m,BX+k]


    IF D2D:

      for i in prange(i1,i2+1, schedule='dynamic'):
        for m in range(m1,m2+1):
          for k in range(3):

            for j in range(j1, j2+1):

              jj = j-j1

              d[jj] = dtetai * U[i,j,m,BX+k]
              a[jj] = -dyi2
              c[jj] = dtetai + 2*dyi2
              b[jj] = -dyi2

            for j in range(j1-1, j2+2):
              B1d[j] = U[i,j,m,BX+k]

            #outflow
            if bc_flags[1][0]==1 or (bc_flags[1][0]==2 and k!=1):
              c[0] = c[0] + a[0]
            if bc_flags[1][1]==1 or (bc_flags[1][1]==2 and k!=1):
              c[Ny-1] = c[Ny-1] + b[Ny-1]

            # reflection (parallel B component)
            if k==1:
              if bc_flags[1][0]==2:
                b[0] = b[0]-a[0]
              if bc_flags[1][1]==2:
                b[Ny-1] = 0

            #inflow
            if bc_flags[1][0]==3:
              d[0] = d[0] - a[0] * B1d[j1-1]
            if bc_flags[1][1]==3:
              d[Ny-1] = d[Ny-1] - b[Ny-1] * B1d[j2+1]

            if (bc_flags[1][0] != 0 and bc_flags[1][1] != 0):

              utils_diffuse.TDMA(&B1d[j1], a,b,c,d, Ny)
              for j in range(j1, j2+1):
                U[i,j,m,BX+k] = B1d[j]

            #periodic
            if bc_flags[1][0]==0 and bc_flags[1][1]==0:

              utils_diffuse.TDMAper(&B1d[j1], a,b,c,d, d2,X1,X2, Ny)
              for j in range(j1, j2+1):
                U[i,j,m,BX+k] = B1d[j]
              # U[i, j2+1,m,BX+k] = U[i,j1,m,BX+k]


    IF D3D:

      for i in prange(i1,i2+1, schedule='dynamic'):
        for j in range(j1,j2+1):
          for k in range(3):

            for m in range(m1, m2+1):

              mm = m-m1

              d[mm] = dtetai * U[i,j,m,BX+k]
              a[mm] = -dzi2
              c[mm] = dtetai + 2*dzi2
              b[mm] = -dzi2

            for m in range(m1-1, m2+2):
              B1d[m] = U[i,j,m,BX+k]

            #outflow
            if bc_flags[2][0]==1 or (bc_flags[2][0]==2 and k!=2):
              c[0] = c[0] + a[0]
            if bc_flags[2][1]==1 or (bc_flags[2][1]==2 and k!=2):
              c[Nz-1] = c[Nz-1] + b[Nz-1]

            # reflection (parallel B component)
            if k==2:
              if bc_flags[2][0]==2:
                b[0] = b[0]-a[0]
              if bc_flags[2][1]==2:
                b[Nz-1] = 0

            #inflow
            if bc_flags[2][0]==3:
              d[0] = d[0] - a[0] * B1d[m1-1]
            if bc_flags[2][1]==3:
              d[Nz-1] = d[Nz-1] - b[Nz-1] * B1d[m2+1]

            if bc_flags[2][0] != 0 and bc_flags[2][1] != 0:

              utils_diffuse.TDMA(&B1d[m1], a,b,c,d, Nz)
              for m in range(m1, m2+1):
                U[i,j,m,BX+k] = B1d[m]

            #periodic
            if bc_flags[2][0]==0 and bc_flags[2][1]==0:

              utils_diffuse.TDMAper(&B1d[m1], a,b,c,d, d2,X1,X2, Nz)
              for m in range(m1, m2+1):
                U[i,j,m,BX+k] = B1d[m]
              # U[i,j,m2+1,BX+k] = U[i,j,m1,BX+k]

    free(a)
    free(b)
    free(c)
    free(d)
    free(B1d)

    free(X1)
    free(X2)
    free(d2)

  # END OF PARALEL BLOCK

  for i in prange(i1, i2+1):
    for j in range(j1, j2+1):
      for m in range(m1, m2+1):

        U[i,j,m,BXC] = 0.5*(U[i,j,m,BX] + U[i+1,j,m,BX])

        IF D2D: U[i,j,m,BYC] = 0.5*(U[i,j,m,BY] + U[i,j+1,m,BY])
        ELSE:   U[i,j,m,BYC] = U[i,j,m,BY]

        IF D3D: U[i,j,m,BZC] = 0.5*(U[i,j,m,BZ] + U[i,j,m+1,BZ])
        ELSE:   U[i,j,m,BZC] = U[i,j,m,BZ]
