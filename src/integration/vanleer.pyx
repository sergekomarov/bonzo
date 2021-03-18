# -*- coding: utf-8 -*-

import numpy as np
cimport numpy as np

cimport bnz.mhd.eos as eos
cimport ct
cimport problem.problem as problem
cimport bnz.diagnostics as diagnostics
cimport bnz.srcterms.gravity as gravity
cimport bnz.srcterms.turb_driv as turb_driv

from integrator cimport get_integr_lims


cdef void init_sim(BnzGrid grid, BnzIntegr integr, BnzIO io, str usr_dir):

  print_root('initialize grid...\n')
  grid = BnzGrid(usr_dir)

  print_root('initialize integrator...\n')
  integr = BnzIntegr(grid.coord, usr_dir)

  print_root('initialize input/output...\n')
  io = BnzIO(usr_dir)

  if io.restart:
    print_root('restart...\n')
    io.set_restart(grid, integr)
  else:
    print_root('set problem...\n')
    problem.set_problem(grid, integr, usr_dir)

  cdef:
    GridCoord *gc = grid.coord
    int lims_act[6]
    int lims_tot[6]
  lims_act[:] = [gc.i1,gc.i2, gc.j1,gc.j2, gc.k1,gc.k2]
  lims_tot[:] = [0,gc.Ntot[0]-1, 0,gc.Ntot[1]-1, 0,gc.Ntot[2]-1]

  print_root('apply boundary conditions...\n')
  grid.apply_grid_bc(integr, np.arange(NVAR))
  IF MFIELD:
    ct.interp_bc(grid.prim, grid.bfld, grid.coord, lims_act)
    grid.apply_grid_bc(integr, np.array([BXC,BYC,BZC]))

  print_root('convert to conserved variables...\n')
  grid.prim2cons(lims_tot, integr.gam)

  if not io.restart:
    print_root('write initial conditions...\n')
    io.write_output(grid,integr)


# ---------------------------------------------------------------

# MHD van Leer integrator.

cdef void advance(BnzGrid grid, BnzIntegr integr, BnzIO io, real tmax):

  cdef:

    GridCoord *gc = grid.coord
    GridData gd  = grid.data
    int order
    real dt
    int lims[6]

    real4d w = grid.data.prim
    real4d u = grid.data.cons
    real4d b = grid.data.bfld
    real4d ws = grid.data.prim
    real4d us = integr.data.cons_s
    real4d bs = integr.data.bfld_s

    # C structs to measure timings
    timeval tstart, tstart_step, tstop

  cdef int lims_tot[6]
  lims_tot[:] = [0,gc.Ntot[0]-1, 0,gc.Ntot[1]-1, 0,gc.Ntot[2]-1]


  #=============================
  # start main integration loop
  #=============================

  while integr.time < tmax:

    print_root( "\n==========step %i, time = %.5f==========\n\n",
                integr.step+1, integr.time)
    # start measuring timings
    gettimeofday(&tstart_step, NULL)

    IF not FIXDT:
      integr.new_dt(gd.prim, gc)
      if integr.time+integr.dt > tmax: integr.dt = tmax-integr.time

    integr.update_physics(gc, lims, integr.dt)


    # --------------------------------------------------------------------------
    # PREDICTOR STEP

    #-------------------------------------------
    order=0
    dt = 0.5*integr.dt
    lims = &(get_integr_lims(order, gc.Ntot)[0])
    #-------------------------------------------

    # Us = U_n + dt/2 * L(W_n,B_n)
    integr.integrate_hydro(us, u,w,b, gc, order,dt)

    IF MFIELD:
      # Bs = B_n + dt/2 * L(W_n,B_n); interpolate Bs to cell-centers in Us
      integr.integrate_field(us,bs, w,b, gc, order,dt)

    # Us = Us + dt/2 * S(W_n)
    integr.add_source_terms(us,bs, w,b, gc, order,dt)

    eos.cons2prim(ws,us, lims, integr.gam)

    IF CGL or TWOTEMP:
      integr.diffusion.collide(ws, lims, dt)

    #problem.do_user_work_cons(grid, integr, us,bs, u,b, lims, sim, 0.5*dt)

    # --------------------------------------------------------------------------
    # CORRECTOR STEP

    #-------------------------------------------
    order=integr.reconstr_order
    dt = integr.dt
    lims = &(get_integr_lims(order, gc.Ntot)[0])
    #-------------------------------------------

    # U_n+1 = U_n + dt * L(Ws,Bs)
    integr.integrate_hydro(u, u,ws,bs, gc, order,dt)

    IF MFIELD:
      #B_n+1 = B_n + dt * L(Ws); interpolate B to cell-centers in U
      integr.integrate_field(u,b, ws,b, gc, order,dt)

    # U_n+1 = U_n+1 + dt * S(Ws)
    integr.add_source_terms(u,b, ws,bs, gc, order,dt)

    # if integr.time+integr.dt > 0.01:
    #   integr.time += integr.dt
    #   problem.do_user_work_cons(u,b, us,bs, lims, sim, dt)
    #   integr.time -= integr.dt

    grid.cons2prim(lims, integr.gam)

    IF CGL or TWOTEMP:
      integr.diffusion.collide(grid.data.prim, lims, dt)


    # --------------------------------------------------------------------------

    # integrate diffusion terms if needed (does nothing when all coefficients = 0)
    integr.diffusion.diffuse(grid, dt)

    grid.apply_grid_bc(integr, np.arange(NVAR))
    grid.prim2cons(lims_tot, integr.gam)

    # update time and step
    integr.time += integr.dt
    integr.step += 1

    # write output and restart files
    io.write_output(grid,integr)
    io.write_restart(grid,integr)

    # print mean energy densities
    diagnostics.print_nrg(grid,integr)

    # print timings for complete timestep
    gettimeofday(&tstop, NULL)
    print_root("\nstep %i completed in %.1f ms\n",
            sim.step, timediff(tstart_step,tstop))
