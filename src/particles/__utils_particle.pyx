# -*- coding: utf-8 -*-

import numpy as np
cimport numpy as np
from cython.parallel import parallel, prange, threadid

from libc.stdlib cimport malloc, calloc, free
from libc.math cimport sqrt,floor,log,exp,sin,cos,pow,fabs,fmin,fmax
from libc.stdio cimport printf

from bnz.utils cimport rand01

IF SPREC:
  np_real = np.float32
ELSE:
  np_real = np.float64



# ====================================================

# Generate tabulated cumulative Maxwell distribution.

cdef void init_maxw_table(real[::1] gamma_table,
                          real[::1] pdf_table,
                          double delgam):

  cdef:
    ints i, j
    ints pdf_sz = pdf_table.shape[0]

  cdef double maxg = fmax(delgam, 1e-8)*20. + 1.

  cdef real[::1] func = np.zeros(pdf_sz, dtype=np_real)

  for i in range(pdf_sz):
    gamma_table[i] = (maxg-1.)/(pdf_sz-1)*i + 1.
    func[i] = ( (gamma_table[i]) * sqrt((gamma_table[i])**2 - 1)
              * exp(-(gamma_table[i]-1.)/delgam) )

  for i in range(pdf_sz):
    pdf_table[i] = 0.
    for j in range(i+1):
      pdf_table[i] = pdf_table[i] + func[j]

  for i in range(pdf_sz):
    pdf_table[i] /= pdf_table[pdf_sz-1]



# =======================================================

# Generate tabulated cumulative power-law distribution.

cdef void init_powlaw_table(real[::1] gamma_table, real[::1] pdf_table,
              double pind, double ming, double maxg):

  cdef:
    ints i, j
    ints pdf_sz = pdf_table.shape[0]
    real[::1] func = np.zeros(pdf_sz, dtype=np_real)

  for i in range(pdf_sz):
    gamma_table[i] = (maxg-ming)/(pdf_sz-1)*i + ming
    func[i] = gamma_table[i]**(-pind)

  for i in range(pdf_sz):
    pdf_table[i] = 0.
    for j in range(i+1):
      pdf_table[i] = pdf_table[i] + func[j]

  for i in range(pdf_sz):
    pdf_table[i] /= pdf_table[pdf_sz-1]



# =========================

# Draw a random 4-velocity.

cdef void distr_prt(real *u1, real *u2, real *u3, real *gam,
            real[::1] gamma_table, real[::1] pdf_table,
            double gamd, double c):

  cdef:
    ints i
    int flag
    double rannum, gam1, gaminv
    double pcosth, psinth, pphi
    double betad, p0t
    double pt1,pt2,pt3

  cdef ints pdf_sz = pdf_table.shape[0]

  betad = sqrt(1.-1./gamd**2)

  rannum = rand01()
  if rannum == 1.: rannum = rand01()

  i=0
  flag=1
  gam1=0.

  while flag==1:

    if i==pdf_sz-1:
      gam1 = gamma_table[pdf_sz-1]
      flag=0

    if ((rannum >= pdf_table[i]) and (rannum < pdf_table[i+1])):

      gam1 = ( gamma_table[i] + (gamma_table[i+1] - gamma_table[i])
          / (pdf_table[i+1] - pdf_table[i]) * (rannum - pdf_table[i]) )
      flag=0

    i = i + 1

  pcosth = 2*rand01()-1.
  pphi = 2*M_PI*rand01()
  psinth = sqrt(1.-pcosth**2)

  p0t = sqrt((gam1-1.)*(gam1+1.))

  pt1 = p0t*psinth*cos(pphi)
  pt2 = p0t*psinth*sin(pphi)
  pt3 = p0t*pcosth

  # ! using 4-velocities

  u1[0] = (pt1 + betad * gam1) * gamd
  u2[0] = pt2
  u3[0] = pt3
  gam[0] = sqrt(1. + (u1[0]**2 + u2[0]**2 + u3[0]**2))
  u1[0] = c*u1[0]
  u2[0] = c*u2[0]
  u3[0] = c*u3[0]

  # using normal velocities
  # gaminv = c / sqrt(sol**2 + p1**2 + p2**2 + p3**2)
  # v1[0] = p1 * gaminv
  # v2[0] = p2 * gaminv
  # v3[0] = p3 * gaminv



cdef inline void getweight1(ints *ib, ints *jb, ints *mb,
                            real x, real y, real z, real ***W,
                            double dli[3], int ng) nogil:

  cdef:
    ints i,j,m
    double a,d
    real wx[2]
    real wy[2]
    real wz[2]
    double x1,y1,z1

  x1 = x-0.5
  a = x1*dli[0]
  i = <ints>(floor(a))
  d = a - i
  ib[0] = i + ng
  wx[0] = 1.-d
  wx[1] = d

  IF D2D:
    y1 = y-0.5
    a = y1*dli[1]
    j = <ints>(floor(a))
    d = a - j
    jb[0] = j + ng
    wy[0] = 1.-d
    wy[1] = d
  ELSE:
    jb[0]=0
    wy[0]=1.
    wy[1]=0.

  IF D3D:
    z1 = z-0.5
    a = z1*dli[2]
    m = <ints>(floor(a))
    d = a - m
    mb[0] = m + ng
    wz[0] = 1.-d
    wz[1] = d
  ELSE:
    mb[0]=0
    wz[0]=1.
    wz[1]=0.

  for i in range(2):
    for j in range(2):
      for m in range(2):
        W[i][j][m] = wx[i] * wy[j] * wz[m]



# ============================================================

cdef inline void getweight2(ints *ib, ints *jb, ints *mb,
                            real x, real y, real z, real ***W,
                            double dli[3], int ng) nogil:

  cdef:
    ints i,j,m
    double a,d
    real wx[3]
    real wy[3]
    real wz[3]

  a = x*dli[0]
  i = <ints>(floor(a))
  d = a - i
  ib[0] = i-1+ng
  wx[0] = 0.5*(1.-d)**2
  wx[1] = 0.75-(d-0.5)**2
  wx[2] = 0.5*d**2

  IF D2D:
    a = y*dli[1]
    j = <ints>(floor(a))
    d = a - j
    jb[0] = j-1+ng
    wy[0] = 0.5*(1.-d)**2
    wy[1] = 0.75-(d-0.5)**2
    wy[2] = 0.5*d**2
  ELSE:
    jb[0]=0
    wy[0]=1.
    wy[1]=0.
    wy[2]=0.

  IF D3D:
    a = z*dli[2]
    m = <ints>(floor(a))
    d = a - m
    mb[0] = m-1+ng
    wz[0] = 0.5*(1.-d)**2
    wz[1] = 0.75-(d-0.5)**2
    wz[2] = 0.5*d**2
  ELSE:
    mb[0]=0
    wz[0]=1.
    wz[1]=0.
    wz[2]=0.

  for i in range(3):
    for j in range(3):
      for m in range(3):
        W[i][j][m] = wx[i] * wy[j] * wz[m]



# ============================================================

cdef void clearF(real4d CoupF, ints Ntot[3]) nogil:

  cdef ints i,j,k,n

  for n in range(4):
    for k in range(Ntot[2]):
      for j in range(Ntot[1]):
        for i in range(Ntot[0]):

            CoupF[n,k,j,i] = 0.


# cdef void reduceF(real5d CoupF, ints Ntot[3], int nt) nogil:
#
#   cdef ints i,j,m,k,n
#
#   for i in prange(Ntot[0], nogil=True, num_threads=nt, schedule='dynamic'):
#     for j in range(Ntot[1]):
#       for m in range(Ntot[2]):
#         for k in range(4):
#           for n in range(1,nt):
#             CoupF[0,i,j,m,k] = CoupF[0,i,j,m,k] + CoupF[n,i,j,m,k]



# cdef void sort_prts(Domain dom) nogil:
#
#   cdef int i,n
#   cdef double dx = dom.dx
#   cdef uint ng = dom.ng
#   cdef double lg = ng*dx
#   cdef double Lxg = dom.Lx + 2*lg
#   cdef double Lxgt = Lxg/dom.nt
#
#   cdef int[:,::1] ind = np.zeros((dom.nt,dom.Np))
#
#   for n in range(dom.Np):
#     i = <int>((dom.prts[n].x + lg) / Lxgt)
#     ind[n]
