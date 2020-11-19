import numpy as np
cimport numpy as np
from libc.math cimport sqrt,floor,ceil,log,exp,sin,cos,pow,fabs,fmin,fmax
import h5py


# background profiles

cdef inline double rho_prof(double r):

  return  0.0575 * (1 + (r/29)**2)**(-1.5*0.53)
    # ( (4.6e-2 / (1+(r/55.)**2)**1.8 + 4.8e-3 / (1+(r/200)**2)**0.87)
    #         * (1-0.06*exp(-(r-30)**2/81))
    #         * (1+0.04*exp(-(r-15)**2/64.)) )


cdef inline double t_prof(double r):

  # return (1. + (r/rct1)**3.5) / (2.45 + (r/rct1)**3.6) * 2.45
  return 8.1 *  ( (1. + (r/58.)**3.5) / (2.45 + (r/58.)**3.6 ) )
                #* (1.55 + (r/20.)**2.04  ) / (1.   + (r/20.)**2.) )

# bremsstrahlung emission integrated over observed energy range

cdef inline double brm_pow(double _n, double _T, double nu1, double nu2):

  return _n**2 #* sqrt(_T) * (exp(-nu1/_T) - exp(-nu2/_T))


cpdef surf_bright(fname, fname_out, int two_fluid):

  cdef:

    double dl, Lbox

    double[:,:,::1] rho, T
    np.ndarray[double, ndim=2] sb, sb_bg

    double n0,T0,rho0,r0
    double nu1,nu2
    double rho_bg, T_bg, rho_sim, T_sim

    int i,j,k, Nx,Ny,Nz

    double x0s,y0s,z0s, x0c,y0c,z0c, r


    # slc = s_[3:-3,3:-3,3:-3]

  with h5py.File(fname,'r') as f:

    if two_fluid:
      p_ = 2*np.asarray(f['pe'], dtype=np.float64)#[slc]
    else:
      p_ = np.asarray(f['p'], dtype=np.float64)#[slc]

    rho_ = np.asarray(f['rho'], dtype=np.float64)#[slc]

    dl = f.attrs['dl'][0]
    Lbox = f.attrs['size'][0]

  T = p_ / rho_
  rho = rho_

  # normalizations
  rho0 = 0.0575 #cm^(-3)
  T0 = 3.1 #KeV
  r0 = 40. #kpc

  nu1 = 0.7/T0
  nu2 = 2./T0

  # integrate emission along line of sight of length Lint

  # coordinates of shock center
  # in kpc
  x0s = 20.
  y0s = 20. #+ 8.17
  z0s = 20.

  # coordinates of cluster center
  x0c = x0s + 5.89
  y0c = y0s - 8.17
  z0c = z0s

  # convert cell size to kpc
  dl *= r0

  (Nz,Ny,Nx) = np.shape(rho)

  sb    = np.zeros((Ny,Nx))
  sb_bg = np.zeros((Ny,Nx))

  Tpj = np.zeros((Ny,Nx))
  Tpj_bg = np.zeros((Ny,Nx))

  ppj = np.zeros((Ny,Nx))
  ppj_bg = np.zeros((Ny,Nx))

  for j in range(Ny):
    for i in range(Nx):
      for k in range(4*Nz):

        r = sqrt((i*dl-x0c)**2 + (j*dl-y0c)**2 + (k*dl-z0c)**2)

        rho_bg = rho_prof(r) / rho_prof(0)
        T_bg   =   t_prof(r) / t_prof(0)

        sb_bg[j,i] += brm_pow(rho_bg, 0,0,0) * dl
        Tpj_bg[j,i] += T_bg * brm_pow(rho_bg,0,0,0) * dl
        ppj_bg[j,i] += rho_bg*T_bg * dl

        if k<Nz:
          rho_sim = rho[k,j,i] * rho_bg
          T_sim   =   T[k,j,i] * T_bg
          sb[j,i]  += brm_pow(rho_sim, 0,0,0) * dl
          Tpj[j,i] += T_sim * brm_pow(rho_sim, 0,0,0) * dl
          ppj[j,i] += rho_sim * T_sim * dl
        else:
          sb[j,i]  += brm_pow(rho_bg, 0,0,0) * dl
          Tpj[j,i] += T_bg * brm_pow(rho_bg, 0,0,0) * dl
          ppj[j,i] += rho_bg * T_bg * dl


  np.save('sb_bg.npy', sb_bg)
  np.save('sb_'+fname_out+'.npy', sb/sb_bg)

  np.save('T_bg.npy', Tpj_bg / sb_bg)
  np.save('T_'+fname_out+'.npy', Tpj/Tpj_bg / (sb/sb_bg))

  np.save('p_bg.npy', ppj_bg)
  np.save('p_'+fname_out+'.npy', ppj/ppj_bg)
