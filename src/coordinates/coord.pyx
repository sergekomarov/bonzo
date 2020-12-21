# -*- coding: utf-8 -*-

import numpy as np
cimport numpy as np
import sys
from libc.stdlib cimport malloc, calloc, free

from bnz.util cimport calloc_2d_array, free_2d_array, free_3d_array
from bnz.io.read_config import read_param
from user_grid import set_user_coord_x, set_user_coord_y, set_user_coord_z

# all functions imported below end with _
IF USE_CYLINDRICAL:
  from cylindrical cimport *
ELIF USE_SPHERICAL:
  from spherical cimport *
ELSE:
  from cartesian cimport *

IF SPREC:
  np_real = np.float32
ELSE:
  np_real = np.float64


cdef void init_coord(GridCoord *gc, GridBc gbc, str usr_dir):

  init_coord_data(gc)
  set_spacings(gc, gbc, usr_dir)
  set_geometry_(gc)


# -------------------------------------------------------------------------

cdef void init_coord_data(GridCoord* gc):

  cdef:
    int *Ntot = gc.Ntot
    int Ntot_max = IMAX(IMAX(Ntot[0],Ntot[1]),Ntot[2])

  # cell faces
  gc.lf  = <real**>calloc_2d_array(3, Ntot_max+1, sizeof(real))
  # cell spacings
  gc.dlf = <real**>calloc_2d_array(3, Ntot_max, sizeof(real))
  # inverse cell spacings
  gc.dlf_inv = <real**>calloc_2d_array(3, Ntot_max, sizeof(real))

  # volume coordinates
  gc.lv = <real**>calloc_2d_array(3, Ntot_max, sizeof(real))
  gc.dlv = <real**>calloc_2d_array(3, Ntot_max, sizeof(real))
  gc.dlv_inv = <real**>calloc_2d_array(3, Ntot_max, sizeof(real))

  # interpolation coefficients (Mignone 2014)
  if gc.interp_order==3 or gc.interp_order==4:
    gc.cm = <real***>calloc_3d_array(3, gc.interp_order, Ntot_max, sizeof(real))
    gc.cp = <real***>calloc_3d_array(3, gc.interp_order, Ntot_max, sizeof(real))

  # coefficients used in parabolic reconstruction
  gc.hm_ratio = <real**>calloc_2d_array(3,Ntot_max, sizeof(real))
  gc.hp_ratio = <real**>calloc_2d_array(3,Ntot_max, sizeof(real))

  # inverse scale factors used e.g. in calculation of gradients
  gc.syxf = <real*>calloc(Ntot[0], sizeof(real))
  gc.syxv = <real*>calloc(Ntot[0], sizeof(real))
  gc.szxf = <real*>calloc(Ntot[0], sizeof(real))
  gc.szxv = <real*>calloc(Ntot[0], sizeof(real))
  gc.szyf = <real*>calloc(Ntot[1], sizeof(real))
  gc.szyv = <real*>calloc(Ntot[1], sizeof(real))

  gc.lapl_tmp_xy1 = <real**>calloc_2d_array(IMAX(Ntot[1],OMP_NT), Ntot[0], sizeof(real))
  gc.lapl_tmp_xy2 = <real**>calloc_2d_array(IMAX(Ntot[1],OMP_NT), Ntot[0], sizeof(real))

  # reconstr = read_param("computation", "reconstr", 's',usr_dir)
  # if reconstr=='weno' or reconstr=='parab':
  # else:
  #   gc.lapl_tmp_xy1=NULL
  #   gc.lapl_tmp_xy2=NULL

  init_geom_data_(gc)


# -------------------------------------------------------------------------

cdef void free_coord_data(GridCoord *gc):

  free_2d_array(gc.lf)
  free_2d_array(gc.lv)

  free_2d_array(gc.dlf)
  free_2d_array(gc.dlv)

  free_2d_array(gc.dlf_inv)
  free_2d_array(gc.dlv_inv)

  free_2d_array(gc.hp_ratio)
  free_2d_array(gc.hm_ratio)

  if gc.interp_order==3 or gc.interp_order==4:
    free_3d_array(gc.cm)
    free_3d_array(gc.cp)

  free(gc.syxf)
  free(gc.syxv)
  free(gc.szxf)
  free(gc.szxv)
  free(gc.szyf)
  free(gc.szyv)

  # if gc.lapl_tmp_xy1 != NULL:
  # if gc.lapl_tmp_xy2 != NULL:
  free_2d_array(gc.lapl_tmp_xy1)
  free_2d_array(gc.lapl_tmp_xy2)

  # free data structures associated with curvilinear coordinates only
  free_geom_data_(gc)

  IF MPI: free_3d_array(gc.ranks)


# --------------------------------------------------------------------

cdef void set_spacings(GridCoord *gc, GridBc gbc, str usr_dir):

  cdef:

    int *pos = gc.pos
    int *Ntot = gc.Ntot
    int *Nact = gc.Nact
    int *Ntot_glob = gc.Ntot_glob
    int *Nact_glob = gc.Nact_glob

    real *lmin = gc.lmin
    real *lmax = gc.lmax

  # Set the scale of coordinate axes.

  x_scale = read_param("computation","x_scale",'s',usr_dir)
  if x_scale=='uni':
    gc.scale[0] = CS_UNI
  elif x_scale=='log':
    gc.scale[0] = CS_LOG
  elif x_scale=='usr':
    gc.scale[0] = CS_USR

  IF D2D:
    y_scale = read_param("computation","y_scale",'s',usr_dir)
    if y_scale=='uni':
      gc.scale[1] = CS_UNI
    elif y_scale=='log':
      gc.scale[1] = CS_LOG
    elif y_scale=='usr':
      gc.scale[1] = CS_USR

  IF D3D:
    z_scale =  read_param("computation","z_scale",'s',usr_dir)
    if z_scale=='uni':
      gc.scale[2] = CS_UNI
    elif z_scale=='log':
      gc.scale[2] = CS_LOG
    elif z_scale=='usr':
      gc.scale[2] = CS_USR

  # Set coordinates of cell faces (outflow BC assumed by default).

  cdef:
    **lf = gc.lf
    **dlf = gc.dlf
    **dlf_inv = gc.dlf_inv

  # Set user cell spacings.
  if gc.scale[0]==CS_USR:
    set_user_coord_x(gc)
  if gc.scale[1]==CS_USR:
    set_user_coord_y(gc)
  if gc.scale[2]==CS_USR:
    set_user_coord_z(gc)

  cdef:
    real dx,dxi,a, lmin_log, l1, dlf_
    int n, i,i_,j,k, iglob, ind1=0, ind2=0

  # Set uniform or logarithmic cell spacings.

  for n in range(3):

    if Ntot[n]==1:
      lf[n][0]=lmin[n]
      lf[n][1]=lmax[n]
      continue

    if n==0:
      ind1=gc.i1
      ind2=gc.i2
    elif n==1:
      ind1=gc.j1
      ind2=gc.j2
    elif n==2:
      ind1=gc.k1
      ind2=gc.k2

    # Uniform axis.

    if gc.scale[n]==CS_UNI:

      dx = (lmax[n] - lmin[n]) / Nact_glob[n]
      l1 = pos[n] * Nact[n] * dx

      for i in range(Ntot[n]+1):
        lf[n][i] = l1 + (i-ind1)*dx

    # Logarithmic axis.

    elif gc.scale[n]==CS_LOG:

      if gbc.bc_flags[n][0]==0 or gbc.bc_flags[n][1]==0:
        sys.exit('Error: periodic BC in {}-direction cannot be used with logarithmic cell spacing.'.format(n))
      if gc.lmin[n]<=0:
        sys.exit('Error: lmin[{}]<=0 is incompatible with logarithmic cell spacing'.format(n))

      a = (lmax[n]/lmin[n])**(1./Nact_glob[n])

      for i in range(Ntot[n]+1):
        iglob = i-ind1 + pos[n]*Nact[n]
        lf[n][i] = lmin[n] * a**iglob

    # Reset coordinates in ghost cells as approproate for non-outflow BC.

    if gbc.bc_flags[n][0]==0:
      for i in range(ind1-1,-1,-1):
        i_ = ind2-i+(ind1-1)
        dlf_ = lf[n][i_+1] - lf[n][i_]
        lf[n][i] = lf[n][i+1] - dlf_

    if gbc.bc_flags[n][1]==0:
      for i in range(ind2+1,Ntot[n]):
        i_ = ind1+i-(ind2+1)
        dlf_ = lf[n][i_+1] - lf[n][i_]
        lf[n][i+1] = lf[n][i] + dlf_

    if gbc.bc_flags[n][0]==2:
      for i in range(ind1-1,-1,-1):
        i_ = ind1-i+(ind1-1)
        dlf_ = lf[n][i_+1] - lf[n][i_]
        lf[n][i] = lf[n][i+1] - dlf_

    if gbc.bc_flags[n][1]==2:
      for i in range(ind2+1,Ntot[n]):
        i_ = ind2-i+(ind2+1)
        dlf_ = lf[n][i_+1] - lf[n][i_]
        lf[n][i+1] = lf[n][i] + dlf_

  # Calculate cell spacings and their inverse.

  for n in range(3):
    for i in range(Ntot[n]):
      dlf[n][i] = lf[n][i+1] - lf[n][i]
      dlf_inv[n][i] = 1./dlf[n][i]

  # For user-specified spacings, it is user responsibility to set spacings
  # in ghost cells as appropriate for ouflow (default) or user BC


# --------------------------------------------------------------------------

cdef void add_geom_src_terms(real4d u, real4d w,
                             real4d fx, real4d fy, real4d fz,
                             GridCoord *gc, int *lims, real dt) nogil:

  add_geom_src_terms_(u,w, fx,fy,fz, gc, lims,dt)

  return

# --------------------------------------------------------------------------

cdef inline real get_edge_len_x(GridCoord *gc, int i, int j, int k) nogil:
  return get_edge_len_x_(gc, i,j,k)

cdef inline real get_edge_len_y(GridCoord *gc, int i, int j, int k) nogil:
  return get_edge_len_y_(gc, i,j,k)

cdef inline real get_edge_len_z(GridCoord *gc, int i, int j, int k) nogil:
  return get_edge_len_z_(gc, i,j,k)

# -------------------------------------------------------------------------

cdef inline real get_centr_len_x(GridCoord *gc, int i, int j, int k) nogil:
  return get_centr_len_x_(gc, i,j,k)

cdef inline real get_centr_len_y(GridCoord *gc, int i, int j, int k) nogil:
  return get_centr_len_y_(gc, i,j,k)

cdef inline real get_centr_len_z(GridCoord *gc, int i, int j, int k) nogil:
  return get_centr_len_z_(gc, i,j,k)

# -------------------------------------------------------------------------

cdef inline real get_face_area_x(GridCoord *gc, int i, int j, int k) nogil:
  return get_face_area_x_(gc, i,j,k)

cdef inline real get_face_area_y(GridCoord *gc, int i, int j, int k) nogil:
  return get_face_area_y_(gc, i,j,k)

cdef inline real get_face_area_z(GridCoord *gc, int i, int j, int k) nogil:
  return get_face_area_z_(gc, i,j,k)

# -------------------------------------------------------------------------

cdef inline real get_cell_vol(GridCoord *gc, int i, int j, int k) nogil:
  return get_cell_vol_(gc, i,j,k)


# ---------------------------------------------------------------

cdef void add_laplacian(GridCoord *gc, real3d a, real c) nogil:

  # Subtract/add the Laplacian of an array without copying.

  cdef int i,j,k, id

  cdef:
    int nx = gc.Ntot[0]
    int ny = gc.Ntot[1]
    int nz = gc.Ntot[2]

  cdef:
    real h0 = 1.-2.*c
    real h1 = c

  cdef:
    real2d tmp1 = gc.lapl_tmp_xy1
    real2d tmp2 = gc.lapl_tmp_xy2

  cdef:
    real am1,a0,ap1,ap2
    real _tmp1,_tmp2

  with nogil, parallel(num_threads=OMP_NT):

    id = threadid()

    for k in prange(nz, schedule='dynamic'):

      for j in range(ny):

        _tmp2 = a[k,j,0]

        for i in range(1,nx-2,2):

          _tmp1 = h1*(a[k,j,i-1] + a[k,j,i+1]) + h0*a[k,j,i]
          a[k,j,i-1] = _tmp2

          _tmp2 = h1*(a[k,j,i] + a[k,j,i+2]) + h0*a[k,j,i+1]
          a[k,j,i] = _tmp1

        i = i+2

        if i==nx-2:
          a[k,j,i] = h1*(a[k,j,i-1] + a[k,j,i+1]) + h0*a[k,j,i]

        a[k,j,i-1] = _tmp2

      # -------------------------------------------------------------------

      IF D2D:

        copy1d(&tmp2[id,0], &a[k,0,0], nx)

        for j in range(1,ny-2,2):

          lapl_perp1(&a[k,j-1,0], &a[k,j,0], &a[k,j+1,0], &a[k,j+2,0],
                     &tmp1[id,0], &tmp2[id,0], c, nx)

        j = j+2

        if j==ny-2:
          lapl_perp2(&a[k,j-1,0], &a[k,j,0], &a[k,j+1,0], c, nx)

        copy1d(&a[k,j-1,0], &tmp2[id,0], nx)

  # end of the parallel loop over k

  # ----------------------------------------------------------------------

  IF D3D:

    for j in range(ny):
      copy1d(&tmp2[j,0], &a[0,j,0], nx)

    for k in range(1,nz-2,2):
      for j in range(ny):

        lapl_perp1(&a[k-1,j,0], &a[k,j,0], &a[k+1,j,0], &a[k+2,j,0],
                   &tmp1[j,0], &tmp2[j,0], c, nx)

    k = k+2

    if k==nz-2:

      for j in range(ny):
        lapl_perp2(&a[k-1,j,0], &a[k,j,0], &a[k+1,j,0], c, nx)

    for j in range(ny):
      copy1d(&a[k-1,j,0], &tmp2[j,0], nx)
