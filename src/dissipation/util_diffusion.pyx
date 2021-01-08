# -*- coding: utf-8 -*-

import numpy as np
cimport numpy as np

from libc.stdlib cimport calloc,realloc, free

from bnz.util cimport print_root

IF SPREC:
  np_real = np.float32
ELSE:
  np_real = np.float64


cdef inline real lim(real a, real b) nogil:
  cdef:
    real _a = FABS(a)
    real _b = FABS(b)
  return FMAX(0.,FMIN(FMIN(0.5*(_a+_b),_a),_b))


# =====================================================================================

# Limited and unlimited transverse gradients.

cdef inline real tr_grad_lim_xy(real3d a, int i, int j, int k, GridCoord *gc) nogil:

  cdef:
    real *dy_ = gc.dlv_inv[1]
    real *sy_ = gc.syxv

  return lim(lim((a[k,j,  i-1] - a[k,j-1,i-1]) * dy_[j],
                 (a[k,j+1,i-1] - a[k,j,  i-1]) * dy_[j+1]) * sy_[i-1],
             lim((a[k,j,  i  ] - a[k,j-1,i  ]) * dy_[j],
                 (a[k,j+1,i  ] - a[k,j,  i  ]) * dy_[j+1]) * sy_[i])

cdef inline real tr_grad_lim_xz(real3d a, int i, int j, int k, GridCoord *gc) nogil:

  cdef:
    real *dz_ = gc.dlv_inv[2]
    real *szx_ = gc.szxv
    real *szy_ = gc.szyv

  return lim(lim((a[k,  j,i-1] - a[k-1,j,i-1]) * dz_[k],
                 (a[k+1,j,i-1] - a[k,  j,i-1]) * dz_[k+1]) * szx_[i-1]*szy_[j],
             lim((a[k,  j,i  ] - a[k-1,j,i  ]) * dz_[k],
                 (a[k+1,j,i  ] - a[k,  j,i  ]) * dz_[k+1]) * szx_[i]*szy_[j])


cdef inline real tr_grad_lim_yx(real3d a, int i, int j, int k, GridCoord *gc) nogil:

  cdef real *dx_ = gc.dlv_inv[0]

  return lim(lim((a[k,j-1,i  ] - a[k,j-1,i-1]) * dx_[i],
                 (a[k,j-1,i+1] - a[k,j-1,i  ]) * dx_[i+1]),
             lim((a[k,j,  i  ] - a[k,j,  i-1]) * dx_[i],
                 (a[k,j,  i+1] - a[k,j,  i  ]) * dx_[i+1]))

cdef inline real tr_grad_lim_yz(real3d a, int i, int j, int k, GridCoord *gc) nogil:

  cdef:
    real *dz_ = gc.dlv_inv[2]
    real *szx_ = gc.szxv
    real *szy_ = gc.szyv

  return lim(lim((a[k,  j-1,i] - a[k-1,j-1,i]) * dz_[k],
                 (a[k+1,j-1,i] - a[k,  j-1,i]) * dz_[k+1]) * szx_[i]*szy_[j-1],
             lim((a[k,  j,  i] - a[k-1,j,  i]) * dz_[k],
                 (a[k+1,j,  i] - a[k,  j,  i]) * dz_[k+1]) * szx_[i]*szy_[j])


cdef inline real tr_grad_lim_zx(real3d a, int i, int j, int k, GridCoord *gc) nogil:

  cdef real *dx_ = gc.dlv_inv[0]

  return lim(lim((a[k-1,j,i  ] - a[k-1,j,i-1]) * dx_[i],
                 (a[k-1,j,i+1] - a[k-1,j,i  ]) * dx_[i+1]),
             lim((a[k,  j,i  ] - a[k,  j,i-1]) * dx_[i],
                 (a[k,  j,i+1] - a[k,  j,i  ]) * dx_[i+1]))

cdef inline real tr_grad_lim_zy(real3d a, int i, int j, int k, GridCoord *gc) nogil:

  cdef:
    real *dy_ = gc.dlv_inv[1]
    real *sy_ = gc.syxv

  return lim(lim((a[k-1,j,  i] - a[k-1,j-1,i]) * dy_[j],
                 (a[k-1,j+1,i] - a[k-1,j,  i]) * dy_[j+1]) * sy_[i],
             lim((a[k,  j,  i] - a[k,  j-1,i]) * dy_[j],
                 (a[k,  j+1,i] - a[k,  j,  i]) * dy_[j+1]) * sy_[i])

# ---------------------------------------------------------------------------------

cdef inline real tr_grad_yx(real3d a, int i, int j, int k, GridCoord *gc) nogil:

  cdef:
    real *dy_ = gc.dlv_inv[1]
    real *sy_ = gc.syxv

  return 0.25*( ((a[k,j,  i-1] - a[k,j-1,i-1]) * dy_[j]
               + (a[k,j+1,i-1] - a[k,j,  i-1]) * dy_[j+1]) * sy_[i-1]
               +((a[k,j,  i  ] - a[k,j-1,i  ]) * dy_[j]
               + (a[k,j+1,i  ] - a[k,j,  i  ]) * dy_[j+1]) * sy_[i])

cdef inline real tr_grad_zx(real3d a, int i, int j, int k, GridCoord *gc) nogil:

  cdef:
    real *dz_ = gc.dlv_inv[2]
    real *szx_ = gc.szxv
    real *szy_ = gc.szyv

  return 0.25*(((a[k,  j-1,i] - a[k-1,j-1,i]) * dz_[k]
               +(a[k+1,j-1,i] - a[k,  j-1,i]) * dz_[k+1]) * szx_[i]*szy_[j-1]
              +((a[k,  j,  i] - a[k-1,j,  i]) * dz_[k]
               +(a[k+1,j,  i] - a[k,  j,  i]) * dz_[k+1]) * szx_[i]*szy_[j])


cdef inline real tr_grad_xy(real3d a, int i, int j, int k, GridCoord *gc) nogil:

  cdef real *dx_ = gc.dlv_inv[0]

  return 0.25*(((a[k,j-1,i  ] - a[k,j-1,i-1]) * dx_[i]
               +(a[k,j-1,i+1] - a[k,j-1,i  ]) * dx_[i+1])
              +((a[k,j,  i  ] - a[k,j,  i-1]) * dx_[i]
               +(a[k,j,  i+1] - a[k,j,  i  ]) * dx_[i+1]))

cdef inline real tr_grad_zy(real3d a, int i, int j, int k, GridCoord *gc) nogil:

  cdef:
    real *dz_ = gc.dlv_inv[2]
    real *szx_ = gc.szxv
    real *szy_ = gc.szyv

  return 0.25*(((a[k,  j-1,i] - a[k-1,j-1,i]) * dz_[k]
               +(a[k+1,j-1,i] - a[k,  j-1,i]) * dz_[k+1]) * szx_[i]*szy_[j-1]
              +((a[k,  j,  i] - a[k-1,j,  i]) * dz_[k]
              + (a[k+1,j,  i] - a[k,  j,  i]) * dz_[k+1]) * szx_[i]*szy_[j])


cdef inline real tr_grad_xz(real3d a, int i, int j, int k, GridCoord *gc) nogil:

  cdef real *dx_ = gc.dlv_inv[0]

  return 0.25*(((a[k-1,j,i  ] - a[k-1,j,i-1]) * dx_[i]
               +(a[k-1,j,i+1] - a[k-1,j,i  ]) * dx_[i+1])
              +((a[k,  j,i  ] - a[k,  j,i-1]) * dx_[i]
               +(a[k,  j,i+1] - a[k,  j,i  ]) * dx_[i+1]))

cdef inline real tr_grad_yz(real3d a, int i, int j, int k, GridCoord *gc) nogil:

  cdef:
    real *dy_ = gc.dlv_inv[1]
    real *sy_ = gc.syxv

  return 0.25*(((a[k-1,j,  i] - a[k-1,j-1,i]) * dy_[j]
               +(a[k-1,j+1,i] - a[k-1,j,  i]) * dy_[j+1]) * sy_[i]
              +((a[k,  j,  i] - a[k,  j-1,i]) * dy_[j]
               +(a[k,  j+1,i] - a[k,  j,  i]) * dy_[j+1]) * sy_[i])
