#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True
# -*- coding: utf-8 -*-

IF MPI:
  from mpi4py import MPI as mpi
  from mpi4py cimport MPI as mpi

import numpy as np
cimport numpy as np

from libc.math cimport sqrt,floor,ceil,log,exp,sin,cos,pow,fabs,fmin,fmax
from libc.stdlib cimport rand, RAND_MAX, srand

from bnz.coord cimport lind2gcrd, lind2gind
from bnz.utils cimport gen_sol2d, gen_sol3d
from bnz.utils import gen_fld3d
from bnz.read_config import read_user_param


IF SPREC:
  np_real = np.float32
ELSE:
  np_real = np.float64


# density profile
cdef double rho_r(double r):
  return (1+(r/1.2)**2)**(-1.5)


cdef void set_problem(BnzSim sim):

  # dom: Domain object
  # W: array of primitive hydrodynamic variables [RHO,VX,VY,VZ,P,(PPD),(PE),PSC]
  # B: array of staggered magnetic field

  cdef:
    BnzGrid grid = sim.grid
    GridParams gp = grid.params
    GridData gd = grid.data
    real4d W = gd.W
    real4d B = gd.B
    BnzPhysics phys = sim.phys

  cdef np.ndarray[double, ndim=4] A0 = np.zeros(
                (3, gp.Ntot[2]+1, gp.Ntot[1]+1, gp.Ntot[0]+1),
                dtype=np.float64)

  cdef:
    ints i,j,k,n, igp,jgp,kgp
    ints ig=0, jg=0, kg=0
    real x=0,y=0,z=0, xa,ya,za
    real rc,rs, r0, rho0, p0,T0, Brms0, lami0, vthi0,vthe0

  cdef ints rank=0
  IF MPI:
    cdef mpi.Comm comm = mpi.COMM_WORLD
    rank = comm.Get_rank()


  #-----------------------------------------

  cdef:
    double beta = read_user_param('beta', 'f', sim.output.usr_dir)
    double lmin = read_user_param('lmin', 'f', sim.output.usr_dir)
    double lmax = read_user_param('lmax', 'f', sim.output.usr_dir)

  # set main parameters (overloads configuration file)

  phys.gam = 5./3

  # box size = 30 kpc

  # central density
  # 0.22 cm^(-3) (M87)
  # 0. cm^(-3) (Perseus)
  rho0 = 1.

  # r0
  # 0.93 kpc (M87)
  # 26 kpc (Perseus)
  r0 = 26./40.

  # central total pressure
  p0 = 1.
  T0 = p0/rho0
  # gives hydro speed of sound = sqrt(gam)

  # central magnetic field
  Brms0 = sqrt(2*p0 / beta)

  # lam0 = 0.05 kpc
  lami0 = 3.1e-3
  # ion pressure is p0/2, vthi=sqrt(2*Ti/mi)
  vthi0 = sqrt(p0/rho0)
  vthe0 = 42*vthi0

  phys.nuiic0 = 0.#vthi0 / lami0
  phys.kappa0 = 0.93 * lami0 * vthe0 * rho0 #* 5.7 # compensate for separate electron pressure


  #---------------------------------------------------

  # generate tangled magnetic field with B^2(r)~rho(r)

  # first generate vector potential

  IF not MFIELD:

    if rank==0:
      # A0 = gen_fld3d(gp.Nact_glob[2], gp.Nact_glob[1], gp.Nact_glob[0],
      #             Linj_cells=<ints>(lmax*gp.dli[0]),
      #             Lmin_cells=<ints>(lmin*gp.dli[0]))
      # np.save('weak_shock/A0.npy',A0)
      A0 = np.load('weak_shock/A0.npy')

    IF MPI:
      # ia1, ia2 = gp.pos[0]*gp.Nact[0], (gp.pos[0]+1)*gp.Nact[0]+1
      # ja1, ja2 = gp.pos[1]*gp.Nact[1], (gp.pos[1]+1)*gp.Nact[1]+1
      # ka1, ka2 = gp.pos[2]*gp.Nact[2], (gp.pos[2]+1)*gp.Nact[2]+1
      # Aloc = comm.bcast(Aglob[:,ia1:ia2,ja1:ja2,ka1:ka2], root=0)
      A0 = comm.bcast(A0, root=0)
      comm.barrier()


    # scale vector potential as sqrt(rho)

    # for k in range(gp.Nact[2]):
    #   for j in range(gp.Nact[1]):
    #     for i in range(gp.Nact[0]):
    #
    #       lind2gind(&ig,&jg,&kg, i,j,k, gp)
    #       x = (ig+0.5)*gp.dl[0]
    #       y = (jg+0.5)*gp.dl[1]
    #       z = (kg+0.5)*gp.dl[2]
    #
    #       # if ig==dom.N_glob[0]: ig=0
    #       # if jg==dom.N_glob[1]: jg=0
    #       # if kg==dom.N_glob[2]: kg=0
    #
    #       # vector potential is defined at cell edges
    #       xa = x
    #       ya = y-0.5*gp.dl[1]
    #       za = z-0.5*gp.dl[2]
    #
    #       r = sqrt((xa-0.5*gp.Lglob[0])**2
    #              + (ya-0.5*gp.Lglob[1])**2
    #              + (za-0.5*gp.Lglob[2])**2)
    #
    #       # if r<0.2: A0[ig,jg,mg,0] = sqrt(rho_r(r,r0,rho0))
    #       A0[0,kg,jg,ig] *= sqrt(rho_r(r,r0,rho0))
    #
    #       xa = x-0.5*gp.dl[0]
    #       ya = y
    #       za = z-0.5*gp.dl[2]
    #
    #       r = sqrt((xa-0.5*gp.Lglob[0])**2
    #              + (ya-0.5*gp.Lglob[1])**2
    #              + (za-0.5*gp.Lglob[2])**2)
    #
    #       A0[1,kg,jg,ig] *= sqrt(rho_r(r,r0,rho0))
    #
    #       xa = x-0.5*gp.dl[0]
    #       ya = y-0.5*gp.dl[1]
    #       za = z
    #
    #       r = sqrt((xa-0.5*gp.Lglob[0])**2
    #              + (ya-0.5*gp.Lglob[1])**2
    #              + (za-0.5*gp.Lglob[2])**2)
    #
    #       A0[2,kg,jg,ig] *= sqrt(rho_r(r,r0,rho0))


    # np.save('data/A_ii.npy',A0)

    # obtain magnetic field from vector potential

    for k in range(gp.k1, gp.k2+1):
      for j in range(gp.j1, gp.j2+1):
        for i in range(gp.i1, gp.i2+1):

          lind2gind(&ig,&jg,&kg, i,j,k, gp)

          ig -= gp.i1
          jg -= gp.j1
          kg -= gp.k1

          igp = ig+1 if ig<gp.Nact_glob[0]-1 else 0
          jgp = jg+1 if jg<gp.Nact_glob[1]-1 else 0
          kgp = kg+1 if kg<gp.Nact_glob[2]-1 else 0

          B[0,k,j,i] =  ((A0[2,kg,jgp,ig] - A0[2,kg,jg,ig]) * gp.dli[1]
                        -(A0[1,kgp,jg,ig] - A0[1,kg,jg,ig]) * gp.dli[2])
          B[1,k,j,i] = -((A0[2,kg,jg,igp] - A0[2,kg,jg,ig]) * gp.dli[0]
                        -(A0[0,kgp,jg,ig] - A0[0,kg,jg,ig]) * gp.dli[2])
          B[2,k,j,i] =  ((A0[1,kg,jg,igp] - A0[1,kg,jg,ig]) * gp.dli[0]
                        -(A0[0,kg,jgp,ig] - A0[0,kg,jg,ig]) * gp.dli[1])

    # divB = 1./dom.dx*(B[1:,:-1,:-1,0]-B[:-1,:-1,:-1,0]+
    #                   B[:-1,1:,:-1,1]-B[:-1,:-1,:-1,1]+
    #                   B[:-1,:-1,1:,2]-B[:-1,:-1,:-1,2])
    # np.save('data/divB.npy',divB)


    # normalize magnetic field

    # cdef:
    #   real norm=0.
    #   ints norm_cells=0
    #   real r_norm = 0.48
    #   real dr = 0.02
    #   real Bxc,Byc,Bzc
    #
    # for k in range(gp.k1, gp.k2+1):
    #   for j in range(gp.j1, gp.j2+1):
    #     for i in range(gp.i1, gp.i2+1):
    #
    #       lind2gcrd(&x,&y,&z, i,j,k, gp)
    #
    #       Bxc = 0.5*(B[0,k,j,i] + B[0,k,j,i+1])
    #       Byc = 0.5*(B[1,k,j,i] + B[1,k,j+1,i])
    #       Bzc = 0.5*(B[2,k,j,i] + B[2,k+1,j,i])
    #
    #       r = sqrt((x - 0.5*gp.Lglob[0])**2
    #              + (y - 0.5*gp.Lglob[1])**2
    #              + (z - 0.5*gp.Lglob[2])**2)
    #
    #       if r > r_norm-dr and r < r_norm+dr:
    #         norm = norm + Bxc**2 + Byc**2 + Bzc**2
    #         norm_cells += 1
    #
    # IF MPI:
    #   cdef:
    #     double[::1] norm_loc = np.array([norm], dtype='f8')
    #     double[::1] norm_sum = np.empty(1, dtype='f8')
    #     ints[::1] norm_cells_loc = np.array([norm_cells], dtype='i8')
    #     ints[::1] norm_cells_sum = np.empty(1, dtype='i8')
    #   comm.Allreduce(norm_loc, norm_sum, op=mpi.SUM)
    #   comm.Allreduce(norm_cells_loc, norm_cells_sum, op=mpi.SUM)
    #   norm  = norm_sum[0]
    #   norm_cells = norm_cells_sum[0]
    #
    #
    #
    # for k in range(gp.Nact[2]):
    #   for j in range(gp.Nact[1]):
    #     for i in range(gp.Nact[0]):
    #
    #       for n in range(3):
    #         B[n,k,j,i] *=  sqrt(2 * rho_r(r_norm, r0,rho0) * T0 * norm_cells / (phys.beta * norm))
    #
    #       B[0,k,j,i] += 1e-3


    cdef:
      double norm=0.
      real Bxc,Byc,Bzc

    for k in range(gp.k1, gp.k2):
      for j in range(gp.j1, gp.j2):
        for i in range(gp.i1, gp.i2):

          Bxc = 0.5*(B[0,k,j,i] + B[0,k,j,i+1])
          Byc = 0.5*(B[1,k,j,i] + B[1,k,j+1,i])
          Bzc = 0.5*(B[2,k,j,i] + B[2,k+1,j,i])

          norm = norm + Bxc**2 + Byc**2 + Bzc**2

    IF MPI:
      cdef:
        double[::1] norm_loc = np.array([norm], dtype='f8')
        double[::1] norm_sum = np.empty(1, dtype='f8')
      comm.Allreduce(norm_loc, norm_sum, op=mpi.SUM)
      norm = norm_sum[0] / (gp.Nact_glob[0] * gp.Nact_glob[1] * gp.Nact_glob[2])

    for n in range(3):
      for k in range(gp.k1, gp.k2+1):
        for j in range(gp.j1, gp.j2+1):
          for i in range(gp.i1, gp.i2+1):

            B[n,k,j,i] *=  sqrt(2 * rho0 * T0 / (beta * norm))
            if n==0: B[0,k,j,i] += 1e-4

    if rank==0:
      np.save('weak_shock/B0.npy',B)

# end of IF MFIELD


  #-----------------------------------------------------------------------------

  for k in range(gp.k1, gp.k2+1):
    for j in range(gp.j1, gp.j2+1):
      for i in range(gp.i1, gp.i2+1):

        lind2gcrd(&x,&y,&z, i,j,k, gp)

        rs = sqrt((x-0.5*gp.Lglob[0])**2
                + (y-0.5*gp.Lglob[1])**2
                + (z-0.5*gp.Lglob[2])**2)

        rc = sqrt((x-(0.5*gp.Lglob[0] + 0*0.147))**2
                + (y-(0.5*gp.Lglob[1] - 0*0.204))**2
                + (z-0.5*gp.Lglob[2])**2)

        # density
        W[RHO,k,j,i] = rho0
        # if rc**2<0.22:
        #   W[RHO,k,j,i] = rho0 * rho_r(rc)
        # else:
        #   W[RHO,k,j,i] = rho0 * rho_r(sqrt(0.22))

        # total pressure
        W[PR,k,j,i] = T0 * W[RHO,k,j,i]

        # explosion
        if rs < 0.04: W[PR,k,j,i] *= 110.
        #134: 1.27
        #122: 1.26

        IF TWOTEMP:
          # ion pressure equals electron pressure equals half total pressure
          W[PR,k,j,i] *= 0.5
          W[PE,k,j,i] = W[PR,k,j,i]

        IF CGL:
          W[PPD,k,j,i] = W[PR,k,j,i]

        # passive scalar
        W[PSC,k,j,i] = 1. + 0.1 * cos(2*M_PI / gp.Lglob[1] * y)

        B[0,k,j,i] = 1e-8


# ===============================================================

# =====================================================================

cdef void do_user_work_cons(real4d U1, real4d B1, real4d U0, real4d B0,
                       ints lims[6], BnzSim sim, double dt):
  return


cdef void set_bc_grid_ptrs_user(BnzBC bc):
    # bc.bc_grid_funcs[0][0] = x1_bc_grid
  return

IF PIC or MHDPIC:
  cdef void set_bc_prt_ptrs_user(BnzBC bc):
    # bc.bc_prt_funcs[0][0] = x1_bc_prt
    return


cdef void set_phys_ptrs_user(BnzPhysics phys):
  # phys.grav_pot_func = grav_pot
  # phys.thcond_elec_func = thcond_elec
  return

cdef real grav_pot(real x, real y, real z, double g, double Lglob[3]) nogil:

  cdef:
    real rx = x - (0.5*Lglob[0] + 0*0.147)
    real ry = y - (0.5*Lglob[1] - 0*0.204)
    real rz = z -  0.5*Lglob[2]
    real r2 = rx**2 + ry**2 + rz**2
    real rc2 = 1.44
    real rb2 = 0.22

  #-T0*log(rho(r)) = 0.5*T0*log(1+(r/r0)**2), where T0=1

  # return log(1+r2/rc**2)

  if r2<rb2:
    return 1.5 * log(1+r2/rc2)
  else:
    return 1.5 * log(1+rb2/rc2)

# cdef real thcond_elec(real, real, real, real4d W, real4d B) nogil:
#   return

# Set user history variables and particle selection function.

cdef void set_output_user(BnzOutput output):
  # set up to NHST_U (8 by default) function pointers
  # output.hst_funcs_u[0] = hst_var1
  # output.hst_names_u[0] = "B2h"
  # IF PIC:
  #   output.prt_sel_func = select_particle
  return
