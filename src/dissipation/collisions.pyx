# -*- coding: utf-8 -*-

import numpy as np
cimport numpy as np
from cython.parallel import prange, parallel

from libc.stdio cimport printf
from bnz.util cimport print_root


# cdef void set_nuiic(real3d nuiic, real4d u, int *lims,
#                     real nuiic0, real gam) nogil:
#
#   # Set ion collision rate array.
#
#   cdef:
#     int i,j,k
#     real pi,Ti, b2,m2, m0, rhoi
#
#   for k in prange(lims[4],lims[5]+1, nogil=True, num_threads=OMP_NT, schedule='dynamic'):
#     for j in range(lims[2],lims[3]+1):
#       for i in range(lims[0],lims[1]+1):
#
#         # rhoi = 1./u[RHO,k,j,i]
#         #
#         # b2 = SQR(u[BX,k,j,i]) + SQR(u[BY,k,j,i]) + SQR(u[BZ,k,j,i])
#         # m2 = SQR(u[MX,k,j,i]) + SQR(u[MY,k,j,i]) + SQR(u[MZ,k,j,i])
#         #
#         # pi = (gam-1.) * (u[EN,k,j,i] - 0.5*(b2 + m2 * rhoi))
#         # IF TWOTEMP:
#         #   pi = pi - EXP(u[SE,k,j,i] * rhoi) * POW(u[RHO,k,j,i], gam)
#         #
#         # Ti = pi / u[RHO,k,j,i]
#         nuiic[k,j,i] = nuiic0 #* u[RHO,k,j,i] #/ (Ti * SQRT(Ti))



# ----------------------------------------------------------------------------------

cdef void collide(BnzDiffusion diff, real4d w, int *lims, real dt) nogil:

  # Add ion collisions to isotropize pressure
  # and electron-ion collisions to equilibrate temperature.
  # Use an implicit update for stability.

  cdef:
    int i,j,k
    real b2
    real pi, pi0, pe, pipe, pipd,pipl, pipdf,pipdm, pipd0, dpi
    real beta0, beta023
    real nuiic_dt, nuiec_dt, pipd_pipl

  cdef:
    real nuiic
    real two3rd = 2./3
    real one3rd = 1./3
    real nuip_dt = 1e20
    real nuip = nuip_dt / dt
    real nuip_dt_1i = 1./(1.+nuip_dt)

  for k in prange(lims[4],lims[5]+1, nogil=True, num_threads=OMP_NT, schedule='dynamic'):
    for j in range(lims[2],lims[3]+1):
      for i in range(lims[0],lims[1]+1):

        nuiic = diff.nuiic0
        pi = w[PR,k,j,i]

        IF CGL:

          pipd = w[PPD,k,j,i]
          pipl = 3.*pi-2.*pipd
          b2 = SQR(w[BX,k,j,i]) + SQR(w[BY,k,j,i]) + SQR(w[BZ,k,j,i])

          # ion pitch-angle scattering by micro-instabilities

          beta0 = 2.*pi/b2
          beta023 = beta0 - two3rd + SMALL_NUM
          pipdf = 0.5 * beta023 * b2
          pipdm = 0.5 * (pipdf + FABS(pipdf)*SQRT(1. + 4.*beta0/SQR(beta023)))

          pipd0 = pipd

          # firehose
          if pipd < pipdf:
            pipd = (pipd + nuip_dt * pipdf) * nuip_dt_1i
            IF IONTC:
              # set ion collision rate for heat flux calculation
              diff.nuii_eff[k,j,i] = FMAX(nuiic,
                          - (pipd - pipdf) / (pipd - pi + SMALL_NUM) * nuip)
          else:
            IF IONTC: nuii_eff[k,j,i] = nuiic
            ELSE: pass

          # mirror
          if pipd > pipdm:
            pipd = (pipd + nuip_dt * pipdm) * nuip_dt_1i
            IF IONTC:
              # set ion collision rate for heat flux calculation
              diff.nuii_eff[k,j,i] = FMAX(nuiic,
                         - (pipd - pipdm) / (pipd - pi + SMALL_NUM) * nuip)
          else:
            IF IONTC: nuii_eff[k,j,i] = nuiic
            ELSE: pass

          # effect of ion Coulomb collisions
          nuiic_dt = dt * nuiic
          pipd = (pipd + nuiic_dt * pi) / (1. + nuiic_dt)

        # electron-ion collisions

        IF TWOTEMP:

          pe = w[PE,k,j,i]
          pi0 = pi

          # calculate ie equilibration rate from ii collision rate
          # pipe = pi/pe
          nuiec_dt = 0.07 * nuiic * dt #* pipe * sqrt(pipe)

          # change in mean ion pressure
          pi = (pi + nuiec_dt * pe) / (1. + nuiec_dt)
          dpi = pi-pi0

          # change in electron pressure
          pe = pe - dpi

          w[PR,k,j,i] = pi
          w[PE,k,j,i] = pe

          # u[SE,k,j,i] = w[RHO,k,j,i] * (LOG(pe) - diff.gam * LOG(w[RHO,k,j,i]))

          # change in ion pressure components
          IF CGL: pipd = pipd + dpi

        IF CGL:
          w[PPD,k,j,i] = pipd
          # pipd_pipl = pipd / (3.*pi - 2.*pipd)
          # u[LA,k,j,i] = w[RHO,k,j,i] * LOG(pipd_pipl * SQR(w[RHO,k,j,i]) / (b2*SQRT(b2)))



# ----------------------------------------------------------------------------

# cdef void collide_cons(real4d u, int *lims, BnzDiffusion diff, real dt) nogil:
#
#   # Add ion collisions to isotropize pressure
#   # and electron-ion collisions to equilibrate temperature.
#   # Use an implicit update for stability.
#
#   cdef:
#     int i,j,k
#     real b2, m2, rhoi
#     real pi, pi0, pe, pipe, pipd,pipl, pipdf,pipdm, pipd0, dpi
#     real beta0, beta023
#     real nuiic_dt, nuiec_dt, pipd_pipl
#
#   cdef:
#     real nuiic
#     real two3rd = 2./3
#     real one3rd = 1./3
#     real nuip_dt = 1e20
#     real nuip = nuip_dt / dt
#     real nuip_dt_1i = 1./(1.+nuip_dt)
#
#   for k in prange(lims[4],lims[5]+1, nogil=True, num_threads=OMP_NT, schedule='dynamic'):
#     for j in range(lims[2],lims[3]+1):
#       for i in range(lims[0],lims[1]+1):
#
#         nuiic = diff.nuiic0
#
#         b2 = 0.
#         IF MFIELD: b2 = SQR(u[BX,k,j,i]) + SQR(u[BY,k,j,i]) + SQR(u[BZ,k,j,i])
#         m2 = SQR(u[MX,k,j,i]) + SQR(u[MY,k,j,i]) + SQR(u[MZ,k,j,i])
#
#         rhoi = 1./u[RHO,k,j,i]
#
#         # calculate electron and ion pressures
#
#         pe = 0.
#         IF TWOTEMP:
#           pe = EXP(u[SE,k,j,i] * rhoi) * POW(u[RHO,k,j,i], diff.gam)
#
#         pi = (diff.gam-1.) * (u[EN,k,j,i] - 0.5*(b2 + m2*rhoi)) - pe #* 0.5
#
#         IF CGL:
#
#           # calculate components of ion pressure
#           pipd_pipl = EXP(u[LA,k,j,i] * rhoi) * SQR(rhoi) * b2*SQRT(b2)
#           pipd = 3.*pipd_pipl / (1. + 2.*pipd_pipl) * pi
#
#           # ion pitch-angle scattering by micro-instabilities
#
#           beta0 = 2.*pi/b2
#           beta023 = beta0 - two3rd + SMALL_NUM
#           pipdf = 0.5 * beta023 * b2
#           pipdm = 0.5 * (pipdf + FABS(pipdf)*SQRT(1. + 4.*beta0/SQR(beta023)))
#
#           pipd0 = pipd
#
#           # firehose
#           if pipd < pipdf:
#             pipd = (pipd + nuip_dt * pipdf) * nuip_dt_1i
#             IF IONTC:
#               # set ion collision rate for heat flux calculation
#               diff.nuii_eff[k,j,i] = FMAX(nuiic,
#                           - (pipd - pipdf) / (pipd - pi + SMALL_NUM) * nuip)
#           else:
#             IF IONTC: nuii_eff[k,j,i] = nuiic
#             ELSE: pass
#
#           # mirror
#           if pipd > pipdm:
#             pipd = (pipd + nuip_dt * pipdm) * nuip_dt_1i
#             IF IONTC:
#               # set ion collision rate for heat flux calculation
#               diff.nuii_eff[k,j,i] = FMAX(nuiic,
#                          - (pipd - pipdm) / (pipd - pi + SMALL_NUM) * nuip)
#           else:
#             IF IONTC: nuii_eff[k,j,i] = nuiic
#             ELSE: pass
#
#           # effect of ion Coulomb collisions
#           nuiic_dt = dt * nuiic
#           pipd = (pipd + nuiic_dt * pi) / (1. + nuiic_dt)
#
#         # electron-ion collisions
#
#         IF TWOTEMP:
#
#           pi0 = pi
#
#           # calculate ie equilibration rate from ii collision rate
#           # pipe = pi/pe
#           nuiec_dt = 0.07 * nuiic * dt #* pipe * sqrt(pipe)
#
#           # change in mean ion pressure
#           pi = (pi + nuiec_dt * pe) / (1. + nuiec_dt)
#           dpi = pi-pi0
#
#           # change in electron pressure
#           pe = pe - dpi
#
#           u[SE,k,j,i] = u[RHO,k,j,i] * (LOG(pe) - diff.gam * LOG(u[RHO,k,j,i]))
#
#           # change in ion pressure components
#           IF CGL: pipd = pipd + dpi
#
#         IF CGL:
#           # write ion pressure components into the conservative variable
#           pipd_pipl = pipd / (3.*pi - 2.*pipd)
#           u[LA,k,j,i] = u[RHO,k,j,i] * LOG(pipd_pipl * SQR(u[RHO,k,j,i]) / (b2*SQRT(b2)))
#
#
