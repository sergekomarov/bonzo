#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True
# -*- coding: utf-8 -*-

import numpy as np
cimport numpy as np
from cython.parallel import parallel, prange, threadid

from libc.math cimport M_PI, sqrt,floor,log,exp,sin,cos,pow,fabs,fmin,fmax

# from openmp cimport omp_set_lock, omp_unset_lock

cimport bc_exch
cimport bc_prt
from utils_particle cimport clearF,reduceF, calloc_3d_array, free_3d_array
from src.utils cimport mini, maxi, print_root, timediff


# ========================================================================

# Deposit particle charge and current onto grid at time-step n, calculate
# Lorenz force, and apply feedback to fluid variables.

cdef void feedback_predict(Domain dom):

  # cdef:
  #   uint i,j
  #   double Q

  cdef ints rank=0
  IF MPI:
    rank=dom.blocks.id

  cdef timeval tstart, tstart_step, tstop

  print_root(rank, "deposit particle charge and current... ")
  gettimeofday(&tstart, NULL)
  depositJ(dom)
  gettimeofday(&tstop, NULL)
  print_root(rank, "%.1f ms\n", timediff(tstart,tstop))

  # Q=0.
  # for i in range(dom.Nwg[0]):
  #   for j in range(dom.Nwg[1]):
  #     Q = Q + dom.CoupF[0,i,j,0,0]
  # print_root(rank,"total charge before BC = %f\n", Q)

  print_root(rank, "apply exchange BC... ")
  gettimeofday(&tstart, NULL)
  bc_exch.apply_bc_exch(dom)
  gettimeofday(&tstop, NULL)
  print_root(rank, "%.1f ms\n", timediff(tstart,tstop))

  # Q=0.
  # for i in range(dom.ng,dom.N[0]+dom.ng):
  #   for j in range(dom.ng,dom.N[1]+dom.ng):
  #     Q = Q + dom.CoupF[0,i,j,0,0]
  # print_root(rank,"total charge after BC = %f\n", Q)

  # np.save("coup.npy", dom.CoupF)

  print_root(rank, "convert charge/current to Lorenz force... ")
  gettimeofday(&tstart, NULL)
  J2F(dom, dom.U)
  gettimeofday(&tstop, NULL)
  print_root(rank, "%.1f ms\n", timediff(tstart,tstop))

  print_root(rank, "add feedback via Lorenz forces and CR Hall current...")
  gettimeofday(&tstart, NULL)
  applyF(dom, dom.aux.Up, dom.U, 0)  # 0 at predictor step
  gettimeofday(&tstop, NULL)
  print_root(rank, "%.1f ms\n", timediff(tstart,tstop))



# ==============================================================================
#
# 1) Predict particle location at time-step n+1/2.
# 2) Obtain cell-centered electric field corrected for CR Hall current at n+1/2.
# 3) Push particles to step n+1 using the predicted electric field, while
#    extracting their momentum/energy changes and depositing them onto the grid.
# 4) Apply particle feedback to fluid variables using the previously obtained
#    changes in particle momentum/energy.

cdef void feedback_correct(Domain dom):

  cdef:
    ints i,j
    double detot=0., dfxtot=0.

  cdef ints rank=0
  IF MPI:
    rank=dom.blocks.id

  cdef timeval tstart, tstart_step, tstop

  print_root(rank, "predict particle locations at t_n+1/2... ")
  gettimeofday(&tstart, NULL)
  move_prts1(dom)
  gettimeofday(&tstop, NULL)
  print_root(rank, "%.1f ms\n", timediff(tstart,tstop))

  # print_root(rank, "deposit particle charge and current... ")
  # gettimeofday(&tstart, NULL)
  # depositJ(dom)
  # gettimeofday(&tstop, NULL)
  # print_root(rank, "%.1f ms\n", timediff(tstart,tstop))
  #
  # print_root(rank, "apply exchange BC... ")
  # gettimeofday(&tstart, NULL)
  # bc_exch.apply_bc_exch(dom)
  # gettimeofday(&tstop, NULL)
  # print_root(rank, "%.1f ms\n", timediff(tstart,tstop))
  #
  # print_root(rank, "convert charge/current to Lorenz force... ")
  # gettimeofday(&tstart, NULL)
  # J2F(dom, dom.aux.Up)
  # gettimeofday(&tstop, NULL)
  # print_root(rank, "%.1f ms\n", timediff(tstart,tstop))
  #
  # print_root(rank, "correct electric field at t_n+1/2 using CR Hall current... ")
  # gettimeofday(&tstart, NULL)
  # applyF(dom, dom.U, dom.aux.Up, 1)
  # gettimeofday(&tstop, NULL)
  # print_root(rank, "%.1f ms\n", timediff(tstart,tstop))
  #
  print_root(rank, "move particles to t_n+1 and deposit their energy/momentum change... ")
  gettimeofday(&tstart, NULL)
  move_deposit_prts2(dom)
  gettimeofday(&tstop, NULL)
  print_root(rank, "%.1f ms\n", timediff(tstart,tstop))

  print_root(rank, "apply exchange BC... ")
  gettimeofday(&tstart, NULL)
  bc_exch.apply_bc_exch(dom)
  gettimeofday(&tstop, NULL)
  print_root(rank, "%.1f ms\n", timediff(tstart,tstop))

  print_root(rank, "apply particle BC... ")
  gettimeofday(&tstart, NULL)
  bc_prt.apply_bc_prt(dom)
  gettimeofday(&tstop, NULL)
  print_root(rank, "%.1f ms\n", timediff(tstart,tstop))

  print_root(rank, "apply momentum/energy feedback and CR Hall current... ")
  gettimeofday(&tstart, NULL)
  applyF(dom, dom.U, dom.aux.Up, 2)
  gettimeofday(&tstop, NULL)
  print_root(rank, "%.1f ms\n", timediff(tstart,tstop))

  detot=0.
  dfxtot=0.
  for i in range(dom.ng, dom.Nact[0]+dom.ng):
    for j in range(dom.ng, dom.Nact[1]+dom.ng):
      detot = detot + dom.CoupF[0,i,j,0,0]
      dfxtot = dfxtot + dom.CoupF[0,i,j,0,1]

  # print_root(rank, "\ntransferred energy = %e\n", detot*dom.dt/(dom.N[0]*dom.N[1]*dom.N[2]))
  # print_root(rank, "\ntransferred momentum = %e\n", dfxtot*dom.dt/(dom.N[0]*dom.N[1]*dom.N[2]))


# ====================================================================

# Deposit charge/current of all particles and calculate Lorenz forces.

cdef void depositJ(Domain dom) nogil:

  cdef:
    ints n, i,j,m,k
    ints ib,jb,mb
    ints i0,j0,m0
    ints il,iu, jl,ju, ml,mu
    int id
    double q, currx,curry,currz, qsol, qsolog, wtot, a

  cdef:
    int nt = dom.nt
    double ncr = dom.params.ncr
    double qomc = dom.params.qomc
    double sol = dom.params.sol

  cdef:
    real5d CoupF = dom.CoupF
    Particle *p
    real ***W


  # clear charge/current array
  clearF(CoupF, dom.Ntot, nt)

  q = ncr * qomc / dom.ppc
  qsol = sol * q

  # deposit charge and current,
  # store them in the feedback force array temporarily

  with nogil, parallel(num_threads=nt):

    id = threadid()
    W = <real ***>calloc_3d_array(3,3,3, sizeof(real))

    for n in prange(dom.Np, schedule='dynamic'):

      p = &(dom.prts[n])
      qsolog = qsol / p.g
      currx = qsolog * p.u
      curry = qsolog * p.v
      currz = qsolog * p.w

      ib, jb, mb = 0, 0, 0

      dom.getweight(p.x, p.y, p.z, W, &ib, &jb, &mb,
                 dom.dli, dom.Nact, dom.ng)

      il, iu = maxi(0,ib), mini(dom.Ntot[0], ib + dom.ninterp+1)
      jl, ju = maxi(0,jb), mini(dom.Ntot[1], jb + dom.ninterp+1)
      ml, mu = maxi(0,mb), mini(dom.Ntot[2], mb + dom.ninterp+1)

      # wtot=0.

      for i in range(il,iu):
        i0 = i-ib
        for j in range(jl,ju):
          j0 = j-jb
          for m in range(ml,mu):
            m0 = m-mb

            # omp_set_lock(&locks[i*Nyg*Nzg + j*Nzg + m])

            # ! use the force array for charge/current deposits temporarily

            CoupF[id,i,j,m,0] = CoupF[id,i,j,m,0] + W[i0][j0][m0] * q
            CoupF[id,i,j,m,1] = CoupF[id,i,j,m,1] + W[i0][j0][m0] * currx
            CoupF[id,i,j,m,2] = CoupF[id,i,j,m,2] + W[i0][j0][m0] * curry
            CoupF[id,i,j,m,3] = CoupF[id,i,j,m,3] + W[i0][j0][m0] * currz
            # wtot = wtot + W[i0][j0][m0] * q

            # omp_unset_lock(&locks[i*Nyg*Nzg + j*Nzg + m])

      # printf('\nn=%i, wtot=%f\n', n, wtot)

    free_3d_array(W)

  # add up contributions from different OpenMP threads
  reduceF(CoupF, dom.Ntot, nt)



# =================================================

# Convert charge/current deposits to Lorenz forces.

cdef void J2F(Domain dom, real4d U0) nogil:

  # U0 = U at predictor stage, Up at corrector stage

  cdef:
    ints i,j,m
    double R1, a

  cdef:
    real4d CoupF = dom.CoupF[0]
    real4d Ec = dom.aux.Ec

  a = 1./dom.params.qomc

  for i in prange(dom.Ntot[0], nogil=True, num_threads=dom.nt, schedule='dynamic'):
    for j in range(dom.Ntot[1]):
     for m in range(dom.Ntot[2]):

       # ratio of ion to ion+CR densities
       R1 = U0[i,j,m,RHO] / (a*CoupF[i,j,m,0] + U0[i,j,m,RHO])

       # convert charge/current to Lorenz force within the same coupling array
       lorenz_force( &CoupF[i,j,m,0], &CoupF[i,j,m,0],
                        &Ec[i,j,m,0],    &U0[i,j,m,BX], R1)



# =============================================

# Calculate Lorenz force from charge, current,
# electric and magnetic fields at cell centers.

cdef inline void lorenz_force(real *F, real *J, real *E, real *B,
                              double R1) nogil:

  cdef double Fx,Fy,Fz

  Fx = R1 * (J[0]*E[0] + J[2]*B[2] - J[3]*B[1])
  Fy = R1 * (J[0]*E[1] + J[3]*B[0] - J[1]*B[2])
  Fz = R1 * (J[0]*E[2] + J[1]*B[1] - J[2]*B[0])
  F[1],F[2],F[3] = Fx,Fy,Fz


# ===============================================================

# Apply particle feedback via Lorenz forces and CR Hall currents.

cdef void applyF(Domain dom, real4d U1, real4d U0, int a) nogil:

  # a=0: predictor stage of fluid update
  #   Lorenz force has been calculated from deposited particle charge and current.
  #   1) add energy/momentum feedback as source terms
  #   2) update electric field and Pointing flux due to CR Hall current
  # a=1: predictor stage of particle update
  #   Only need electric field corrected for CR Hall current at time-step n+1/2 at cell centers.
  #   It is then used by particle mover.
  # a=2: corrector
  #   Forces and energy change have been extracted from particle mover as
  #   changes in particle momentum/energy.

  cdef:
    ints i,j,m,k
    double ek,ex,ey,ez, sp, rhoi
    ints Nxtot=dom.Ntot[0], Nytot=dom.Ntot[1], Nztot=dom.Ntot[2]
    double dt = dom.dt
    double dth = 0.5*dom.dt

  cdef:
    real4d CoupF = dom.CoupF[0]
    real4d Ec = dom.aux.Ec
    real4d Fx = dom.aux.Fx
    real4d Fy = dom.aux.Fy
    real4d Fz = dom.aux.Fz
    real4d U = dom.U

  # Correct electric field at cell centers due to CR Hall current.

  # for i in prange(Nxtot, nogil=True, num_threads=dom.nt, schedule='dynamic'):
  #   for j in range(Nytot):
  #     for m in range(Nztot):
  #
  #       rhoi = 1./U0[i,j,m,RHO]
  #
  #       for k in range(3):
  #         # CR Hall current contribution to electric field at cell centers
  #         Ec[i,j,m,k] = Ec[i,j,m,k] - CoupF[i,j,m,k] * rhoi

  if a==0:

    # At predictor stage, add momentum and energy feedback derived from
    # deposited charge and current.

    for i in prange(Nxtot, nogil=True, num_threads=dom.nt, schedule='dynamic'):
      for j in range(Nytot):
        for m in range(Nztot):

          rhoi = 1./U0[i,j,m,RHO]

          # energy feedback
          U1[i,j,m,EN] = ( U[i,j,m,EN]
                      - dth * (CoupF[i,j,m,1] * U0[i,j,m,MX]
                             + CoupF[i,j,m,2] * U0[i,j,m,MY]
                             + CoupF[i,j,m,3] * U0[i,j,m,MZ]) * rhoi)

          for k in range(3):
            # momentum feedback
            U1[i,j,m,MX+k] = U[i,j,m,MX+k] - dth * CoupF[i,j,m,k+1]

  elif a==2:

    # At corrector stage, add momentum/energy feedback using the change
    # in particles' momentum/energy as they are pushed by EM fields.

    for i in prange(Nxtot, nogil=True, num_threads=dom.nt, schedule='dynamic'):
      for j in range(Nytot):
        for m in range(Nztot):

          # energy feedback
          U1[i,j,m,EN] = U[i,j,m,EN] - dt * CoupF[i,j,m,0]

          for k in range(3):
            # momentum feedback
            U1[i,j,m,MX+k] = U[i,j,m,MX+k] - dt * CoupF[i,j,m,k+1]

  # If not particle predictor stage, use CR-generated electric field to
  # correct fluxes that will be later used by the MHD solver to calculate
  # edge-centered electric field, update magnetic field via CT and fluid
  # energy density via the Pointing flux.

  # if a != 1:
  #
  #   for i in prange(1,Nxtot, nogil=True, num_threads=dom.nt, schedule='dynamic'):
  #     for j in range(Nytot):
  #       for m in range(Nztot):
  #
  #         rhoi = 1./U0[i,j,m,RHO]
  #
  #         if Fx[i,j,m,0] > 0:
  #           ez = CoupF[i-1,j,m,3] * rhoi
  #           ey = CoupF[i-1,j,m,2] * rhoi
  #           sp = ey * U0[i-1,j,m,BZ] - ez * U0[i-1,j,m,BY]
  #         elif Fx[i,j,m,0] < 0:
  #           ez = CoupF[i,j,m,3] * rhoi
  #           ey = CoupF[i,j,m,2] * rhoi
  #           sp = ey * U0[i,j,m,BZ] - ez * U0[i,j,m,BY]
  #         else:
  #           ez = 0.5 * rhoi * (CoupF[i-1,j,m,3] + CoupF[i,j,m,3])
  #           ey = 0.5 * rhoi * (CoupF[i-1,j,m,2] + CoupF[i,j,m,2])
  #           sp = 0.5 * (ey * (U0[i-1,j,m,BZ] + U0[i,j,m,BZ])
  #                     - ez * (U0[i-1,j,m,BY] + U0[i,j,m,BY]))
  #
  #         # Correct the inductive flux used in the CT update of magnetic field
  #         Fx[i,j,m,BY] = Fx[i,j,m,BY] + ez
  #         Fx[i,j,m,BZ] = Fx[i,j,m,BZ] - ey
  #
  #         # Correct the Pointing flux used in the hydro update of energy density
  #         Fx[i,j,m,EN] = Fx[i,j,m,EN] - sp
  #
  #   if dom.Nact[1]>1:
  #
  #     for i in prange(Nxtot, nogil=True, num_threads=dom.nt, schedule='dynamic'):
  #       for j in range(1,Nytot):
  #         for m in range(Nztot):
  #
  #           rhoi = 1./U0[i,j,m,RHO]
  #
  #           if Fy[i,j,m,0] > 0:
  #             ex = CoupF[i,j-1,m,1] * rhoi
  #             ez = CoupF[i,j-1,m,3] * rhoi
  #             sp = ez * U0[i,j-1,m,BX] - ex * U0[i,j-1,m,BZ]
  #           elif Fy[i,j,m,0] < 0:
  #             ex = CoupF[i,j,m,1] * rhoi
  #             ez = CoupF[i,j,m,3] * rhoi
  #             sp = ez * U0[i,j,m,BX] - ex * U0[i,j,m,BZ]
  #           else:
  #             ex = 0.5 * rhoi * (CoupF[i,j-1,m,1] + CoupF[i,j,m,1])
  #             ez = 0.5 * rhoi * (CoupF[i,j-1,m,3] + CoupF[i,j,m,3])
  #             sp = 0.5 * (ez * (U0[i,j-1,m,BX] + U0[i,j,m,BX])
  #                       - ex * (U0[i,j-1,m,BZ] + U0[i,j,m,BZ]))
  #
  #           Fy[i,j,m,BX] = Fy[i,j,m,BX] - ez
  #           Fy[i,j,m,BZ] = Fy[i,j,m,BZ] + ex
  #           Fy[i,j,m,EN] = Fy[i,j,m,EN] - sp
  #
  #   if dom.Nact[2]>1:
  #
  #     for i in prange(Nxtot, nogil=True, num_threads=dom.nt, schedule='dynamic'):
  #       for j in range(Nytot):
  #         for m in range(1,Nztot):
  #
  #           rhoi = 1./U0[i,j,m,RHO]
  #
  #           if Fz[i,j,m,0] > 0:
  #             ex = CoupF[i,j,m-1,1] * rhoi
  #             ey = CoupF[i,j,m-1,2] * rhoi
  #             sp = ex * U0[i,j,m-1,BY] - ey * U0[i,j,m-1,BX]
  #           elif Fz[i,j,m,0] < 0:
  #             ex = CoupF[i,j,m,1] * rhoi
  #             ey = CoupF[i,j,m,2] * rhoi
  #             sp = ex * U0[i,j,m,BY] - ey * U0[i,j,m,BX]
  #           else:
  #             ex = 0.5 * rhoi * (CoupF[i,j,m-1,1] + CoupF[i,j,m,1])
  #             ey = 0.5 * rhoi * (CoupF[i,j,m-1,2] + CoupF[i,j,m,2])
  #             sp = 0.5 * (ex * (U0[i,j,m-1,BY] + U0[i,j,m,BY])
  #                       - ey * (U0[i,j,m-1,BX] + U0[i,j,m,BX]))
  #
  #           Fz[i,j,m,BX] = Fz[i,j,m,BX] + ey
  #           Fz[i,j,m,BY] = Fz[i,j,m,BY] - ex
  #           Fz[i,j,m,EN] = Fz[i,j,m,EN] - sp



# ===========================================================================

# Move particles by the first half-step (predictor) with constant velocities.

cdef void move_prts1(Domain dom) nogil:

  cdef:
    ints n
    double a
    double dth_sol = 0.5*dom.dt*dom.params.sol
    Particle *p

  for n in prange(dom.Np, nogil=True, num_threads=dom.nt, schedule='dynamic'):

    p = &(dom.prts[n])

    a = dth_sol / p.g

    # predict particle location at half-step
    p.x = p.x + a*p.u
    p.y = p.y + a*p.v if dom.Nact[1]>1 else p.y
    p.z = p.z + a*p.w if dom.Nact[2]>1 else p.z



# ===================================================================

# Move particles by the second half-step and deposit energy/momentum
# changes of every particle separately.

cdef void move_deposit_prts2(Domain dom) nogil:

  cdef:
    ints n
    real xh,yh,zh
    real fx,fy,fz, dedt
    real ***W
    Particle *p

  cdef double detot=0.
  cdef double dfxtot=0.

  clearF(dom.CoupF, dom.Ntot, dom.nt)

  with nogil, parallel(num_threads=dom.nt):

    W = <real ***>calloc_3d_array(3,3,3, sizeof(real))
    fx,fy,fz,dedt = 0,0,0,0

    for n in prange(dom.Np, schedule='dynamic'):

      p = &(dom.prts[n])

      # save particle coordinates at time n+1/2
      xh, yh, zh = p.x, p.y, p.z

      # move particle from n+1/2 to n+1,
      # while keeping track of its momentum/energy change
      move_single2(dom, p, &fx,&fy,&fz, &dedt, W)
      # deposit particle momentum/energy change into feedback array
      depositF_single(dom, xh,yh,zh, fx,fy,fz, dedt, W)

      detot += dedt
      dfxtot += fx

    free_3d_array(W)

  reduceF(dom.CoupF, dom.Ntot, dom.nt)

  # print_root(0, '\nparticle energy loss = %e\n', detot*dom.params.ncr * dom.dt/dom.Np )
  # print_root(0, '\nparticle x momentum loss = %e\n', dfxtot*dom.params.ncr * dom.dt/dom.Np)



# ==============================================================

# Move a single particle by the second half-step (n+1/2 to n+1).

cdef void move_single2(Domain dom, Particle *p,
          real *fx, real *fy, real *fz, real *dedt,
          real ***W) nogil:

  cdef:
    real uhe,vhe,whe, u_,v_,w_, du,dv,dw
    real ex=0,ey=0,ez=0, bx=0,by=0,bz=0, b2
    double rx,ry,rz, r1
    double g0,ghinv_sol, a,  dth_qomc_osol

  cdef:
    double sol = dom.params.sol
    double qomc = dom.params.qomc
    double dt = dom.dt

  # interpolate electric and magnetic fields at time n+1/2

  getEB(dom, &ex,&ey,&ez, &bx,&by,&bz,
        p.x, p.y, p.z, W)

  dth_qomc_osol = 0.5*dt*qomc/sol

  ex = dth_qomc_osol * ex
  ey = dth_qomc_osol * ey
  ez = dth_qomc_osol * ez
  bx = dth_qomc_osol * bx
  by = dth_qomc_osol * by
  bz = dth_qomc_osol * bz

  # half-kick by electric field (1/2 included in the prefactor)
  uhe = p.u + ex
  vhe = p.v + ey
  whe = p.w + ez

  # first rotation
  ghinv_sol = sol / sqrt(1. + uhe**2 + vhe**2 + whe**2)
  rx = bx * ghinv_sol
  ry = by * ghinv_sol
  rz = bz * ghinv_sol
  r1 = 2. / (1. + rx**2 + ry**2 + rz**2)
  u_ = (vhe * rz - whe * ry + uhe) * r1
  v_ = (whe * rx - uhe * rz + vhe) * r1
  w_ = (uhe * ry - vhe * rx + whe) * r1

  # second rotation and full kick by electric field
  du = v_ * rz - w_ * ry + 2*ex
  dv = w_ * rx - u_ * rz + 2*ey
  dw = u_ * ry - v_ * rx + 2*ez

  # assign new 4-velocities and gamma
  p.u = p.u + du
  p.v = p.v + dv
  p.w = p.w + dw
  g0 = p.g
  p.g = sqrt(1. + (p.u)**2 + (p.v)**2 + (p.w)**2)

  a = 0.5*dt*sol / p.g

  # move particle from step n+1/2 to n
  p.x = p.x + a*p.u
  p.y = p.y + a*p.v if dom.Nact[1]>1 else p.y
  p.z = p.z + a*p.w if dom.Nact[2]>1 else p.z

  a = sol/dt

  # momentum and energy change per unit of time (per unit of mass)
  fx[0] = a*du
  fy[0] = a*dv
  fz[0] = a*dw
  dedt[0] = a*sol * (p.g - g0)



# ==================================================================

# Deposit momentum/energy change of a single particle at step n+1/2.

cdef void depositF_single(Domain dom,
                real xh, real yh, real zh,
                real fx, real fy, real fz,
                real dedt, real ***W) nogil:

  cdef:
    ints i,j,m
    ints ib,jb,mb, i0,j0,m0
    ints il,iu,jl,ju,ml,mu
    int id

  cdef:
    double a = dom.params.ncr / dom.ppc

  ib, jb, mp = 0, 0, 0
  dom.getweight(xh, yh, zh, W, &ib, &jb, &mb,
             dom.dli, dom.Nact, dom.ng)

  # printf("kernel: %f %f %f %f %f %f %f %f %f\n", W[0][0][0],W[1][0][0],W[2][0][0],W[0][1][0],W[1][1][0],W[2][1][0],W[0][2][0],W[1][2][0],W[2][2][0])
  # printf("coord: %f %f %f %f\n", xh,yh,zh, dom.dx)
  # printf("kernel: %i %i %i\n", ib,jb,mb)
  # printf("dim: %i %i %i\n", dom.Nact)
  # printf("dfx=%f, dfy=%f, dfz=%f, dek=%f\n", dfx,dfy,dfz,dek)

  # normalize particle density to physical density
  fx = a * fx
  fy = a * fy
  fz = a * fz
  dedt = a * dedt

  il, iu = maxi(0,ib), mini(dom.Ntot[0], ib+dom.ninterp+1)
  jl, ju = maxi(0,jb), mini(dom.Ntot[1], jb+dom.ninterp+1)
  ml, mu = maxi(0,mb), mini(dom.Ntot[2], mb+dom.ninterp+1)

  id = threadid()

  for i in range(il,iu):
    i0 = i-ib
    for j in range(jl,ju):
      j0 = j-jb
      for m in range(ml,mu):
        m0 = m-mb

        # omp_set_lock(&locks[i*dom.Nyg*dom.Nzg + j*dom.Nzg + m])

        # energy change
        dom.CoupF[id,i,j,m,0] = dom.CoupF[id,i,j,m,0] + W[i0][j0][m0]*dedt

        # force
        dom.CoupF[id,i,j,m,1] = dom.CoupF[id,i,j,m,1] + W[i0][j0][m0]*fx
        dom.CoupF[id,i,j,m,2] = dom.CoupF[id,i,j,m,2] + W[i0][j0][m0]*fy
        dom.CoupF[id,i,j,m,3] = dom.CoupF[id,i,j,m,3] + W[i0][j0][m0]*fz

        # omp_unset_lock(&locks[i*dom.Nyg*dom.Nzg + j*dom.Nzg + m])



# ==========================================================================

# Interpolate electric and magnetic field at a given location at time n+1/2.
# Use predictor stage cell-centered magnetic and electric field.
# Electric field has been corrected for CR Hall current calculated via
# particle predictor step.

cdef inline void getEB(Domain dom,
            real *ex, real *ey, real *ez,
            real *bx, real *by, real *bz,
            real x, real y, real z,
            real ***W) nogil:

  cdef:
    ints i,j,m
    ints i0,j0,m0, ib,jb,mb
    ints il,iu, jl,ju, ml,mu
    real bsq, edotb

  ib, jb, mb = 0, 0, 0
  dom.getweight(x,y,z, W, &ib,&jb,&mb, dom.dli, dom.Nact, dom.ng)

  il, iu = maxi(0,ib), mini(dom.Ntot[0], ib+dom.ninterp+1)
  jl, ju = maxi(0,jb), mini(dom.Ntot[1], jb+dom.ninterp+1)
  ml, mu = maxi(0,mb), mini(dom.Ntot[2], mb+dom.ninterp+1)

  ex[0], ey[0], ez[0] = 0, 0, 0
  bx[0], by[0], bz[0] = 0, 0, 0

  for i in range(il,iu):
    i0 = i-ib
    for j in range(jl,ju):
      j0 = j-jb
      for m in range(ml,mu):
        m0 = m-mb

        # omp_set_lock(&locks[i*dom.Nyg*dom.Nzg + j*dom.Nzg + m])

        ex[0] = ex[0] + W[i0][j0][m0] * dom.aux.Ec[i,j,m,0]
        ey[0] = ey[0] + W[i0][j0][m0] * dom.aux.Ec[i,j,m,1]
        ez[0] = ez[0] + W[i0][j0][m0] * dom.aux.Ec[i,j,m,2]
        bx[0] = bx[0] + W[i0][j0][m0] * dom.aux.Up[i,j,m,BX]
        by[0] = by[0] + W[i0][j0][m0] * dom.aux.Up[i,j,m,BY]
        bz[0] = bz[0] + W[i0][j0][m0] * dom.aux.Up[i,j,m,BZ]

        # omp_unset_lock(&locks[i*dom.Nyg*dom.Nzg + j*dom.Nzg + m])

  # printf("bx=%f, by=%f, bz=%f\n", bx[0],by[0],bz[0])

  # ensure EdotB=0 after interpolation

  bsq = bx[0]**2 + by[0]**2 + bz[0]**2 + 1e-25
  edotb = ex[0]*bx[0] + ey[0]*by[0] + ez[0]*bz[0]
  ex[0] = ex[0] - edotb * bx[0]/bsq
  ey[0] = ey[0] - edotb * by[0]/bsq
  ez[0] = ez[0] - edotb * bz[0]/bsq
