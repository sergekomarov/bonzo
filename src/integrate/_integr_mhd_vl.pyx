# -*- coding: utf-8 -*-

IF MPI:
  from mpi4py import MPI as mpi
  from mpi4py cimport MPI as mpi

import numpy as np
cimport numpy as np

from libc.math cimport M_PI, sqrt,floor,ceil,log,exp,sin,cos,pow,fabs,fmin,fmax, isnan
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

    ints order
    double dt
    ints lims[6]

    real4d Ws = gd.W

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
    # start measuring timings
    gettimeofday(&tstart_step, NULL)


    IF not FIXDT:
      sim.dt = new_dt_diag(gd.W, gp, phys, method.cour)
      if sim.t+sim.dt > sim.tmax: sim.dt = sim.tmax-sim.t

    update_phys(grid, lims, phys, sim.dt)


    # --------------------------------------------------------------------------
    # PREDICTOR STEP

    #----------------------------------------
    order=0
    dt = 0.5*sim.dt
    lims = &(get_integr_lims(order, gp.Ntot)[0])
    #----------------------------------------

    # Us = U_n + dt/2 * L(W_n,B_n)
    integrate_hydro(gd.Us, gd.U, gd.W, gd.B, grid, method, phys, order, dt)

    IF MFIELD:
      # Bs = B_n + dt/2 * L(W_n,B_n); interpolate Bs to cell-centers in Us
      integrate_field(gd.Us, gd.Bs, gd.W, gd.B, grid, method, phys, order, dt)

    # Us = Us + dt/2 * S(U_n)
    add_source_terms(gd.Us, gd.Bs, gd.U, gd.B, grid, phys, order, dt)

    problem.do_user_work_cons(gd.Us,gd.Bs, gd.U,gd.B, lims, sim, 0.5*dt)

    cons2prim_diag(Ws, gd.Us, lims, phys.gam)


    # --------------------------------------------------------------------------
    # CORRECTOR STEP

    #-------------------------------------------
    order=method.reconstr_order
    dt = sim.dt
    lims = &(get_integr_lims(order, gp.Ntot)[0])
    #-------------------------------------------

    # U_n+1 = U_n + dt * L(Ws)
    integrate_hydro(gd.U, gd.U, Ws, gd.Bs, grid, method, phys, order, dt)

    IF MFIELD:
      #B_n+1 = B_n + dt * L(Ws); interpolate B to cell-centers in U
      integrate_field(gd.U, gd.B, Ws, gd.B, grid, method, phys, order, dt)

    # U_n+1 = U_n+1 + dt * S(Us)
    add_source_terms(gd.U, gd.B, gd.Us, gd.Bs, grid, phys, order, dt)

    if sim.t+sim.dt > 0.01:
      sim.t += sim.dt
      problem.do_user_work_cons(gd.U,gd.B, gd.Us,gd.Bs, lims, sim, dt)
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
