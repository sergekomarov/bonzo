#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True
#cython: language_level=2
# -*- coding: utf-8 -*-

IF MPI:
  from mpi4py import MPI as mpi
  from mpi4py cimport MPI as mpi

import numpy as np
cimport numpy as np

from libc.math cimport M_PI, sqrt,floor,ceil,log,exp,sin,cos,pow,fabs,fmin,fmax
from libc.stdlib cimport rand, RAND_MAX, srand

from src.coord cimport loc2glob_ind
from src.utils import gen_scal_fld_3d
import h5py

cdef void set_user_problem(Domain dom,
    np.ndarray[real, ndim=3] rho,
    np.ndarray[real, ndim=4] v,
    np.ndarray[real, ndim=3] p,
    np.ndarray[real, ndim=3] ppd,
    np.ndarray[real, ndim=3] pe,
    np.ndarray[real, ndim=3] pscal,
    np.ndarray[real, ndim=4] B):

  cdef:

    ints Nx = dom.N[0]
    ints Ny = dom.N[1]
    ints Nz = dom.N[2]
    double Lx = dom.L[0]
    double Ly = dom.L[1]
    double Lz = dom.L[2]
    double dx = dom.dx
    Parameters *pm = &(dom.params)

    ints i,j,m,k
    double x=0,y=0,z=0
    ints ig=0, jg=0, mg=0

  cdef ints rank=0
  IF MPI:
    rank = dom.blocks.id
    cdef:
      mpi.Comm comm = dom.blocks.comm
      double[::1] var     = np.empty(1, dtype='f8')
      double[::1] var_sum = np.empty(1, dtype='f8')

  #-----------------------------------------------


  # read initial conditions from dynamo simulation

  # f = h5py.File('out_mhd192_pr1000_m013/box_009.hdf5', 'r',
  #                driver='mpio', comm=comm)

  # for i in range(Nx):
  #
  #   if rank==0: print i
  #   for j in range(Ny):
  #   for m in range(Nz):
  #
  #   loc2glob_ind(dom, i,j,m, &ig,&jg,&mg)
  #
  #   ig = ig + dom.i1
  #   jg = jg + dom.j1
  #   mg = mg + dom.m1
  #
  #   rho[i,j,m] = f['rho'][ig,jg,mg]
  #   p[i,j,m] =   f['p'][ig,jg,mg]
  #   v[i,j,m,0] =  f['vx'][ig,jg,mg]
  #   v[i,j,m,1] =  f['vy'][ig,jg,mg]
  #   v[i,j,m,2] =  f['vz'][ig,jg,mg]
  #   p[i,j,m] =   f['p'][ig,jg,mg]
  #   B[i,j,m,0] = f['Bxf'][ig,jg,mg]
  #   B[i,j,m,1] = f['Byf'][ig,jg,mg]
  #   B[i,j,m,2] = f['Bzf'][ig,jg,mg]
  #
  # f.close()

  sh = (dom.Nwg_glob[0], dom.Nwg_glob[1], dom.Nwg_glob[2])

  cdef:
    np.ndarray[real, ndim=3] rho_ = np.zeros(sh)
    np.ndarray[real, ndim=3] p_ = np.zeros(sh)
    np.ndarray[real, ndim=3] vx_ = np.zeros(sh)
    np.ndarray[real, ndim=3] vy_ = np.zeros(sh)
    np.ndarray[real, ndim=3] vz_ = np.zeros(sh)
    np.ndarray[real, ndim=3] Bx_ = np.zeros(sh)
    np.ndarray[real, ndim=3] By_ = np.zeros(sh)
    np.ndarray[real, ndim=3] Bz_ = np.zeros(sh)

  if rank==0:
    with h5py.File('out_mhd128_pr1000_m02/box_016.hdf5', 'r') as f:
      rho_ = np.asarray(f['rho'])
      p_ = np.asarray(f['p'])
      vx_ = np.asarray(f['vx'])
      vy_ = np.asarray(f['vy'])
      vz_ = np.asarray(f['vz'])
      Bx_ = np.asarray(f['Bxf'])
      By_ = np.asarray(f['Byf'])
      Bz_ = np.asarray(f['Bzf'])

  IF MPI:
    rho_ = comm.bcast(rho_, root=0)
    p_ = comm.bcast(p_, root=0)
    vx_ = comm.bcast(vx_, root=0)
    vy_ = comm.bcast(vy_, root=0)
    vz_ = comm.bcast(vz_, root=0)
    Bx_ = comm.bcast(Bx_, root=0)
    By_ = comm.bcast(By_, root=0)
    Bz_ = comm.bcast(Bz_, root=0)
    comm.barrier()

  Tbg = 0.

  for i in range(Nx):
    for j in range(Ny):
      for m in range(Nz):

        loc2glob_ind(dom, i,j,m, &ig,&jg,&mg)

        ig = ig + dom.i1
        jg = jg + dom.j1
        mg = mg + dom.m1

        rho[i,j,m] = rho_[ig,jg,mg]
        p[i,j,m] =   p_[ig,jg,mg]
        v[i,j,m,0] = 0#vx_[ig,jg,mg]
        v[i,j,m,1] = 0#vy_[ig,jg,mg]
        v[i,j,m,2] = 0#vz_[ig,jg,mg]

        Tbg = Tbg + p[i,j,m]/rho[i,j,m]

  IF MPI:
    var[0] = Tbg
    comm.Allreduce(var, var_sum, op=mpi.SUM)
    Tbg = var_sum[0] / (dom.N_glob[0]*dom.N_glob[1]*dom.N_glob[2])

  for i in range(Nx+1):
    for j in range(Ny+1):
      for m in range(Nz+1):

        loc2glob_ind(dom, i,j,m, &ig,&jg,&mg)

        ig = ig + dom.i1
        jg = jg + dom.j1
        mg = mg + dom.m1

        B[i,j,m,0] = Bx_[ig,jg,mg]
        B[i,j,m,1] = By_[ig,jg,mg]
        B[i,j,m,2] = Bz_[ig,jg,mg]

  # ------------------------------------------

  # generate random temperature distribution

  pm.lmax=1.
  pm.lmin=0.1

  T0 = None
  if rank==0:
  #   T0 = gen_scal_fld_3d(dom.N_glob[0], dom.N_glob[1], dom.N_glob[2],
  #               rms=0.17, p=5./3,
  #               Linj_cells=<ints>(pm.lmax/dom.dx),
  #               Lmin_cells=<ints>(pm.lmin/dom.dx), nt=dom.nt)
  #   np.save('data/T0.npy',T0)
    T0 = np.load('data/T0.npy')

  IF MPI:
    T0 = comm.bcast(T0, root=0)
    comm.barrier()

  # -------------------------------------------------------------

  # set initial temperature distribution keeping pressure balance

  for i in range(Nx):
    for j in range(Ny):
      for m in range(Nz):

        loc2glob_ind(dom, i,j,m, &ig,&jg,&mg)

        # p[i,j,m] = p[i,j,m]
        # modulate density to keep pressure balance

        # Bxc = 0.5*(B[i,j,m,0]+B[i+1,j,m,0])
        # Byc = 0.5*(B[i,j,m,1]+B[i,j+1,m,1])
        # Bzc = 0.5*(B[i,j,m,2]+B[i,j,m+1,2])
        #
        # ptot0 = p[i,j,m] + 0.5*(Bxc**2 + Byc**2 + Bzc**2)

        rho[i,j,m] = p[i,j,m] / (Tbg * (1.+T0[ig,jg,mg]))

        # passive scalar
        pscal[i,j,m] = Tbg * (1.+T0[ig,jg,mg])

  # ---------------------------------------------

  # transport coefficients

  # viscosity

  lami = 0.015
  vthi = sqrt(Tbg)
  # pm.mu = 0.33 * lami * vthi

  # thermal conductivity

  lame = 0.001 / 3#0.005
  vthe = 42*sqrt(Tbg)
  pm.kappa0 = 0.93 * lame * vthe



#========================================

cdef void set_user_output(Domain dom):
  return


cdef void x1_bc_mhd_user(Domain dom) nogil:
  return
cdef void x2_bc_mhd_user(Domain dom) nogil:
  return

cdef void y1_bc_mhd_user(Domain dom) nogil:
  return
cdef void y2_bc_mhd_user(Domain dom) nogil:
  return

cdef void z1_bc_mhd_user(Domain dom) nogil:
  return
cdef void z2_bc_mhd_user(Domain dom) nogil:
  return


IF PIC:

  cdef void x1_bc_ex_user(Domain dom) nogil:
    return
  cdef void x2_bc_ex_user(Domain dom) nogil:
    return

  cdef void y1_bc_ex_user(Domain dom) nogil:
    return
  cdef void y2_bc_ex_user(Domain dom) nogil:
    return

  cdef void z1_bc_ex_user(Domain dom) nogil:
    return
  cdef void z2_bc_ex_user(Domain dom) nogil:
    return


  cdef void x1_bc_prt_user(Domain dom) nogil:
    return
  cdef void x2_bc_prt_user(Domain dom) nogil:
    return

  cdef void y1_bc_prt_user(Domain dom) nogil:
    return
  cdef void y2_bc_prt_user(Domain dom) nogil:
    return

  cdef void z1_bc_prt_user(Domain dom) nogil:
    return
  cdef void z2_bc_prt_user(Domain dom) nogil:
    return


cdef double grav_pot(double g, real x, real y, real z,
                     double[::1] L) nogil:

  cdef double Rx = x-0.5*L[0]
  cdef double Ry = y-0.5*L[1]
  cdef double Rz = z-0.5*L[2]
  cdef double R = sqrt(Rx**2+Ry**2+Rz**2)
  cdef double rc=0.12*0.4
  cdef double gam=5./3

  # if R < rc*sqrt(3):
  #     return 1./gam * log(1+(R/rc)**2)
  # else:
  #     return 1./gam * log(4)
  #     #        return (-1.5*sqrt(3)*rc/R + log(4)+1.5) / gam

  return g*y
