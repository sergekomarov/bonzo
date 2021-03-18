# -*- coding: utf-8 -*-

import numpy as np
cimport numpy as np

from libc.math cimport sqrt,floor,ceil,log,exp,sin,cos,pow,fabs,fmin,fmax, isnan
from libc.stdio cimport printf

from bnz.utils cimport print_root, timediff

from integr_mhd_jobs cimport *
cimport bnz.output as output
cimport bnz.problem.problem as problem


# =============================================================

# MHD van Leer integrator.

cdef void integrate(BnzSim sim):

  cdef:

    BnzGrid grid = sim.grid
    GridParams gp = grid.params
    GridData gd = grid.data
    BnzPhysics phys = sim.phys
    BnzMethod method = sim.method

    real4d Ws = gd.W
    real4d Wss = gd.W

    ints order
    double dt
    ints lims[6]

    # C structs to measure timings
    timeval tstart, tstart_step, tstop

  cdef ints rank=0
  IF MPI: rank = mpi.COMM_WORLD.Get_rank()

  cdef int use_diffusion = (phys.kappa0 != 0. or phys.mu != 0. or
                            phys.eta != 0 or phys.mu4 != 0 or
                            phys.eta4 != 0 or phys.nuiic0 != 0 or phys.kL != 0)

  cdef ints lims_tot[6]
  lims_tot[:] = [0,gp.Ntot[0]-1, 0,gp.Ntot[1]-1, 0,gp.Ntot[2]-1]


  #=============================
  # start main integration loop
  #=============================

  while sim.t < sim.tmax:

    print_root(rank,
        "\n==========step %i, time = %.5f==========\n\n",
        sim.step+1, sim.t)

    gettimeofday(&tstart_step, NULL)

    IF not FIXDT:
      sim.dt = new_dt_diag(gd.W, gp, phys, method.cour)
      if sim.t+sim.dt > sim.tmax: sim.dt = sim.tmax-sim.t

    update_phys(grid, lims, phys, sim.dt)


    # --------------------------------------------------------------------------
    # STEP I

    # Us = U_n + dt * L(U_n)

    #-------------------------------------------
    order=method.reconstr_order
    dt = sim.dt
    lims = &(get_integr_lims(order, gp.Ntot)[0])
    #-------------------------------------------

    # Us = U_n + dt * L(W_n,B_n)
    integrate_hydro(gd.Us, gd.U, gd.W, gd.B, grid, method, phys, order, dt)

    IF MFIELD:
      # Bs = B_n + dt * L(W_n,B_n)
      integrate_field(gd.Us, gd.Bs, gd.W, gd.B, grid, method, phys, order, dt)

    # Us = Us + dt * S(U_n)
    add_source_terms(gd.Us, gd.Bs, gd.U, gd.B, grid, phys, order, dt)

    problem.do_user_work_cons(gd.Us,gd.Bs, gd.U,gd.B, lims, sim, dt)

    cons2prim_diag(Ws, gd.Us, lims, phys.gam)


    # --------------------------------------------------------------------------
    # STEP II

    # Uss = 1/4 * (3*U_n + Us + dt * L(Us)))

    #-------------------------------------------
    order=method.reconstr_order
    dt = sim.dt
    lims = &(get_integr_lims(order, gp.Ntot)[0])
    #-------------------------------------------

    # Uss = Us + dt * L(Ws,Bs)
    integrate_hydro(gd.Uss, gd.Us, Ws, gd.Bs, grid, method, phys, order, dt)

    IF MFIELD:
      # Bss = Bs + dt * L(Ws,Bs)
      integrate_field(gd.Uss, gd.Bss, Ws, gd.Bs, grid, method, phys, order, dt)

    # Uss = Uss + dt * S(Us)
    add_source_terms(gd.Uss, gd.Bss, gd.Us, gd.Bs, grid, phys, order, dt)

    # Uss = 3/4 * U_n + 1/4 * Uss
    combine(gd.Uss, gd.U, gd.Uss, 0.75, 0.25, lims,NWAVES)

    IF MFIELD:
      combine(gd.Bss, gd.B, gd.Bss, 0.75,0.25, lims,3)

    problem.do_user_work_cons(gd.Uss,gd.Bss, gd.Us,gd.Bs, lims, sim, dt)

    cons2prim_diag(Wss, gd.Uss, lims, phys.gam)


    # --------------------------------------------------------------------------
    # STEP III

    # U_n+1 = 1/3 * (U_n + 2 * (Uss + dt * L(Uss)))

    #-------------------------------------------
    order=method.reconstr_order
    dt = sim.dt
    lims = &(get_integr_lims(order, gp.Ntot)[0])
    #-------------------------------------------

    # use Us as Usss = Uss + dt * L(Uss))

    # Usss = Uss + dt * L(Wss,Bss)
    integrate_hydro(gd.Us, gd.Uss, Wss, gd.Bss, grid, method, phys, order, dt)

    IF MFIELD:
      # Bsss = Bss + dt * L(Wss,Bss)
      integrate_field(gd.Us, gd.Bs, Wss, gd.Bss, grid, method, phys, order, dt)

    # Usss = Usss + dt * S(Uss)
    add_source_terms(gd.Us, gd.Bs, gd.Uss, gd.Bss, grid, phys, order, dt)

    # U_n+1 = 1/3 * U_n + 2/3 * Usss
    combine(gd.U, gd.U, gd.Us, 1./3, 1.-1./3, lims,NWAVES)
    IF MFIELD:
      combine(gd.B, gd.B, gd.Bs, 1./3, 1.-1./3, lims,3)

    if sim.t+sim.dt > 0.01:
      sim.t += sim.dt
      problem.do_user_work_cons(gd.U,gd.B, gd.U,gd.B, lims, sim, dt)
      sim.t -= sim.dt

    cons2prim_diag(gd.W, gd.U, lims, phys.gam)


    # --------------------------------------------------------------------------

    if use_diffusion:
      diffuse_diag(gd.W, gd.B, sim)

    apply_bc_grid_diag(sim, np.arange(NVARS))

    prim2cons_diag(gd.U, gd.W, lims_tot, phys.gam)

    # update time and step
    sim.t = sim.t + sim.dt
    sim.step = sim.step + 1

    # write output and restart files
    output.write_output(sim)
    output.write_restart(sim)

    #---------------------------------------------------------------------------

    # print mean energy densities
    print_nrg_diag(sim)

    # print timings for complete timestep
    gettimeofday(&tstop, NULL)
    print_root(rank, "\nstep %i completed in %.1f ms\n",
            sim.step, timediff(tstart_step,tstop))



# ===============================================================

cdef void combine(real4d A3, real4d A1, real4d A2,
                  real c1, real c2, ints lims[6], ints Nvar) nogil:

  cdef ints i,j,k,n

  for n in range(Nvar):
    for k in range(lims[4], lims[5]+1):
      for j in range(lims[2], lims[3]+1):
        for i in range(lims[0], lims[1]+1):
          A3[n,k,j,i] = c1 * A1[n,k,j,i] + c2 * A2[n,k,j,i]
