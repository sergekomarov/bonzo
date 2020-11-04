# -*- coding: utf-8 -*-

from mpi4py import MPI as mpi
from mpi4py cimport MPI as mpi

import numpy as np
cimport numpy as np

import sys

from scipy.special import factorial

from libc.math cimport sqrt,floor,ceil,log,exp,sin,cos,pow,fabs,fmin,fmax
from utils cimport calloc_2d_array, calloc_3d_array, maxi, mini


IF SPREC:
  np_real = np.float32
ELSE:
  np_real = np.float64



# =========================================================================

cdef void set_interp_coeff(GridCoord *gc, int p):

  cdef:
    ints *Ntot = gc.Ntot
    ints *Nact = gc.Nact
    ints *pos  = gc.pos
    ints ng    = gc.ng
    real **lf  = gc.lf
    real ***cm = gc.cm
    real ***cp = gc.cp

  cdef ints Ntot_max = maxi(maxi(Ntot[0],Ntot[1]),Ntot[2])

  cm = <real***>calloc_3d_array(3,p, Ntot_max, sizeof(real))
  # need both c+/c- if p is odd, i.e. interpolation polynomial is cell-centered
  # this is true for WENO
  cp = <real***>calloc_3d_array(3,p, Ntot_max, sizeof(real))

  cdef ints imin_glob,imax_glob, jmin_glob,jmax_glob, ax

  # X coordinate

  imin_glob = pos[0]*Nact[0]-ng if pos[0]!=0 else 0
  imax_glob = imin_glob + Ntot[0]
  ax=0

  if gc.coord_scale[0] == CS_UNI:
    # uniform x-coordinate
    if gc.coord_geom   == CG_CAR:
      set_car_uni( cm[0],cp[0], Ntot[0], p)
    elif gc.coord_geom == CG_CYL:
      set_cyl_unir(cm[0],cp[0], imin_glob,imax_glob, ng, p)
    elif gc.coord_geom == CG_SPH:
      set_sph_unir(cm[0],cp[0], imin_glob,imax_glob, ng, p)

  else:
    if gc.coord_geom == CG_CAR:
      # nonuniform cartesian x-coordinate
      set_car_nonuni(   cm[0],cp[0], lf[0], Ntot[0], p)
    else:
      set_noncar_nonuni(cm[0],cp[0], lf[0], ax, gc.coord_geom,
                        imin_glob,imax_glob, p)

  # Y coordinate

  jmin_glob = pos[1]*Nact[1]-ng if pos[1]!=0 else 0
  jmax_glob = jmin_glob + Ntot[1]
  ax=1

  if gc.coord_scale[1] == CS_UNI and gc.coord_geom != CG_SPH:
    set_car_uni(cm[1],cp[1], Ntot[1], p)
  else:
    if gc.coord_geom == CG_CAR or gc.coord_geom == CG_CYL:
      # nonuniform cartesian x-coordinate
      set_car_nonuni(cm[1],cp[1], lf[1], Ntot[1], p)
    else:
      # coefficients for uniform spherical meridional coordinate
      # are calculated via the same method as for nonuniform coordinates
      set_noncar_nonuni(cm[1],cp[1], lf[1], ax, gc.coord_geom,
                        jmin_glob,jmax_glob, p)

  # Z coordinate

  if gc.coord_scale[2] == CS_UNI:
    set_car_uni(cm[2],cp[2], Ntot[2], p)
  else:
    set_car_nonuni(cm[2],cp[2], lf[2], Ntot[2], p)




# ========================================================================

cdef void set_car_uni(real **cm, real **cp, ints N, ints p):

  cdef ints i, n

  if p==3:
    for i in range(N):
      # WENO
      cm[0][i] = -1./6
      cm[1][i] = 5./6
      cm[2][i] = 1./3

      cp[0][i] = 1./3
      cp[1][i] = 5./6
      cp[2][i] = -1./6

  elif p==4:
    for i in range(N):
      # PPM
      cm[0][i] = -1./12
      cm[1][i] = 7./12
      cm[2][i] = 7./12
      cm[3][i] = -1./12

    for n in range(4):
      for i in range(N-1):
        cp[n][i] = cm[n][i+1]


# =============================================================================

cdef void set_cyl_unir(real **cm, real **cp,
                  ints imin_glob, ints imax_glob, ints ng, int p):

  cdef:
    ints i, i_, n
    double dinv, id
    ints ng = (imax_glob-imin_glob)/2

  if p==3:

    for i in range(imin_glob,imax_glob+1):

      if i>=ng: i = i-ng+1
      else: i = ng-i
      i_ = i-imin_glob

      id = <double>i

      dinv = 1. / ( 12 * (id**2 - id - 1) * (2*id - 1) )

      cm[0][i_] =  (2*id - 3) * ( 4*id**2 +    id - 1 ) * dinv
      cm[1][i_] =  (2*id - 1) * (10*id**2 - 11*id - 10) * dinv
      cm[2][i_] = -(2*id + 1) * ( 2*id**2 -  4*id + 1 ) * dinv

      cp[0][i_] = -(2*id - 3) * ( 2*id**2        - 1 ) * dinv
      cp[1][i_] =  (2*id - 1) * (10*id**2 - 9*id - 11) * dinv
      cp[2][i_] =  (2*id + 1) * ( 4*id**2 - 9*id + 4 ) * dinv

  if p==4:

    for i in range(imin_glob,imax_glob+1):

      if i>=ng: i = i-ng+1
      else: i = ng-i
      i_ = i-imin_glob

      id = <double>i

      dinv = 1. / ( 120*id**4 - 360*id**2 + 96 )

      cm[0][i_] = -(2*id - 3) * ( 5*id**3 +  8*id**2  - 3*id  - 4) * dinv
      cm[1][i_] =  (2*id - 1) * (35*id**3 + 24*id**2 - 93*id - 60) * dinv
      cm[2][i_] =  (2*id + 1) * (35*id**3 - 24*id**2 - 93*id + 60) * dinv
      cm[3][i_] = -(2*id + 3) * ( 5*id**3 -  8*id**2  - 3*id  + 4) * dinv

    for n in range(p):
      for i in range(imax_glob-imin_glob):
        cp[n][i] = cm[n][i+1]



# ================================================================================

cdef void set_sph_unir(real **cm, real **cp,
                  ints imin_glob, ints imax_glob, ints ng, int p):

  cdef:
    ints i_, i, n
    double dinv, id

  if p==3:

    for i in range(imin_glob,imax_glob+1):

      if i>=ng: i = i-ng+1
      else: i = ng-i
      i_ = i-imin_glob

      id = <double>i

      dinv = 1. / ( 18 * (10*id**6 - 30*id**5 + 15*id**4 + 20*id**3 - 9*id**2 - 6*id + 4) )

      cm[0][i_] = 2*(3*id**2 - 9*id + 7) * (10*id**4 +   5*id**3 -  3*id**2 -     id + 1)  * dinv
      cm[1][i_] =   (3*id**2 - 3*id + 1) * (50*id**4 - 110*id**3 - 33*id**2 + 100*id + 62) * dinv
      cm[2][i_] =  -(3*id**2 + 3*id + 1) * (10*id**4 -  40*id**3 + 51*id**2 -  22*id + 4)  * dinv

      cp[0][i_] =  -(3*id**2 - 9*id + 7) * (10*id**4            -  9*id**2         + 3) * dinv
      cp[1][i_] =   (3*id**2 - 3*id + 1) * (50*id**4 - 90*id**3 - 63*id**2 + 96*id + 69) * dinv
      cp[2][i_] = 2*(3*id**2 + 3*id + 1) * (10*id**4 - 45*id**3 + 72*id**2 - 48*id + 12) * dinv

  if p==4:

    for i in range(imin_glob,imax_glob+1):

      if i>=ng: i = i-ng+1
      else: i = ng-i
      i_ = i-imin_glob

      id = <double>i

      dinv = 1 ./ (36 * (15*id**8 - 85*id**6 + 150*id**4 - 60*id**2 + 16) )

      cm[0][i_] = -(3*id**2 - 9*id + 7) * ( 15*id**6 +  48*id**5 +  23*id**4 -  48*id**3 -  30*id**2 +   16*id + 12)  * dinv
      cm[1][i_] =  (3*id**2 - 3*id + 1) * (105*id**6 + 144*id**5 - 487*id**4 - 720*id**3 + 510*id**2 + 1008*id + 372) * dinv
      cm[2][i_] =  (3*id**2 + 3*id + 1) * (105*id**6 - 144*id**5 - 487*id**4 + 720*id**3 + 510*id**2 - 1008*id + 372) * dinv
      cm[3][i_] = -(3*id**2 + 9*id + 7) * ( 15*id**6 -  48*id**5 +  23*id**4 +  48*id**3 -  30*id**2 -   16*id + 12)  * dinv

    for n in range(p):
      for i in range(imax_glob-imin_glob):
        cp[n][i] = cm[n][i+1]



# ==============================================================================

# Set interpolation coefficients for nonuniform radial or nonuniform spherical
# meridional coordinates.

cdef void set_noncar_nonuni(real **cm, real **cp, real *xi,
                          ints ax, CoordGeom geom,
                          ints imin_glob, ints imax_glob, ints ng, int p):


  cdef:
    ints i,n,s, ir, nr
    ints is_radial=1, m=1
    real a, cosm,cosp

  if ax==0:
    # radial cylindrical or spherical
    if geom==CG_CYL: m=1
    elif geom==CG_SPH: m=2
    is_radial=1
  elif ax==1:
    # spherical meridional
    is_radial=0

  cdef:
    ints iL = p/2
    ints iR = p-iL-1

  cdef:
    np.ndarray[real,ndim=2] betat = np.zeros((p,p), dtype=np_real)
    np.ndarray[real,ndim=1] dm = np.zeros(p, dtype=np_real)
    np.ndarray[real,ndim=1] dp = np.zeros(p, dtype=np_real)
    np.ndarray[real,ndim=1] cmi = np.zeros(p, dtype=np_real)
    np.ndarray[real,ndim=1] cpi = np.zeros(p, dtype=np_real)

  # check if the bottom boundary is r=0 and exclude ghost cells where r<0

  for i in range(iL, imax_glob-imin_glob-iR):

    for n in range(p):

      # Set up the matrix of the system of linear equations for the
      # interpolation coefficients.

      for s in range(-iL, iR+1):

        # radial cylindrical or spherical coordinate
        if is_radial:

          betat[n,s+iL] = ( (m+1) / (n+m+1)
                * (xi[i+s+1]**(n+m+1) - xi[i+s]**(n+m+1)) /
                  (xi[i+s+1]**(  m+1) - xi[i+s]**(  m+1)) )

        # spherical meridional coordinate
        else:

          a = factorial(n) / (cos(xi[i+s]) - cos(xi[i+s+1]))

          for k in range(n+1):

            cosm = cos(xi[i+s]   + 0.5*M_PI*k)
            cosp = cos(xi[i+s+1] + 0.5*M_PI*k)

            betat[n,s+iL] += (xi[i+s]**(n-k) * cosm - xi[i+s+1]**(n-k) * cosp) / factorial(n-k)

          betat[n,s+iL] = a * betat[n,s+iL]

      # Set up the right-hand side of the linear systems.

      dm[n] = xi[i]**n
      dp[n] = xi[i+1]**n

    # Solve the system.

    cmi = np.linalg.solve(betat, dm)
    for n in range(p): cm[n][i] = cmi[n]

    cpi = np.linalg.solve(betat, dp)
    for n in range(p): cp[n][i] = cpi[n]

  # Set interpolation coefficients in the ghost cells at r<0.

  if imin_glob==0:
    for n in range(p):
      nr = p-n-1
      for i in range(ng):
        ir = 2*ng-i-1
        cm[n][i] = cm[nr][ir]
        cp[n][i] = cp[nr][ir]


# ===================================================================

cdef void set_car_nonuni(real **cm, real **cp,
                         real *xi, ints N, int p):

  cdef:
    ints i
    real dm2,dm1,d0,dp1
    real a1,a2,a3,a4,a5
    real w1,w2,w3,w4,w5,w6, w1_,w2_

  if p==4:

    for i in range(2,N-1):

      # !!! turn into dx

      dm2 = xi[i-1] - xi[i-2]
      dm1 = xi[i]   - xi[i-1]
      d0  = xi[i+1] - xi[i]
      dp1 = xi[i+2] - xi[i+1]

      a1  = d0 / (dm1 + d0 + dp1)

      w1 = a1 * (2*dm1 + d0) / (d0  + dp1)
      w2 = a1 * (d0 + 2*dp1) / (dm1 + d0 )

      a1  = dm1 / (dm2 + dm1 + d0)

      w1_ = a1 * (2*dm2 +  dm1) / (dm1 + d0 )
      w2_ = a1 * (dm1   + 2*d0) / (dm2 + dm1)

      a2 = dm1 / (dm1 + d0)
      a3 = 1./ (dm2 + dm1 + d0 + dp1)
      a4 = (dm2 + dm1) / (2*dm1 + d0 )
      a5 = (dp1 + d0 ) / (2*d0  + dm1)
      a2 = a2 + 2*a2 * a3 * d0 * (a4 - a5)

      w3 = 1 - a2
      w4 = a2
      w5 =   a5 * a3 * d0
      w6 = - a4 * a3 * dm1

      cm[0][i] = - w5*w2_
      cm[1][i] = w3 + w5*(w2_-w1_) + w6*(-w2)
      cm[2][i] = w4 + w5*w1_ + w6*(w2-w1)
      cm[3][i] = w6 * w1

  elif p==3:

  # FINISH THIS!

  # if p==4:
  #   # PPM
  #   cm[0,:] = 0.5
  #   cm[1,:] = 0.5
  #   cm[2,:] = 0.5
  #   cm[3,:] = 0.5
  #   cm[4,:] = 1./6
  #   cm[5,:] = -1./6
  # elif p==3:
  #   # WENO
  #   cm[0,:] = -1./6
  #   cm[1,:] = 5./6
  #   cm[2,:] = 1./3
  #
  #   cp[0,:] = 1./3
  #   cp[1,:] = 5./6
  #   cp[2,:] = -1./6
