from numpy import *
import h5py

def surf_bright(fname, two_fluid):

    # 1) load data

    # slc = s_[3:-3,3:-3,3:-3]

    with h5py.File(fname,'r') as f:

        if two_fluid:
            p = 2*asarray(f['pe'])#[slc]
        else:
            p = asarray(f['p'])#[slc]

        rho = asarray(f['rho'])#[slc]
        dl = f.attrs['dl'][0]
        Lbox = f.attrs['size'][0]

    T = p / rho

    # normalizations
    n0 = 0.05 #cm^(-3)
    T0 = 3.1 #KeV
    r0 = 40. #kpc

    # 2) background profiles

    # beta=0.53
    # rcn = 26./r0
    rct1 = 58./r0
    rcn2 = 20./r0

    def n_prof(r):
        return  ( (4.6e-2 / (1+(r/55.)**2)**1.8 + 4.8e-3 / (1+(r/200)**2)**0.87) *
                    (1-0.06*exp(-(r-30)**2/81))
                  * (1+0.04*exp(-(r-15)**2/64.)) )
    def t_prof(r):
        return (1. + (r/rct1)**3.5) / (2.45 + (r/rct1)**3.6) * 2.45
        return 8.1 *  ( (1.   + (r/rct1)**3.5) / (2.45 + (r/rct1)**3.6 ) )
                      #* (1.55 + (r/rct2)**2.04  ) / (1.   + (r/rct2)**2.) )


    # 3) bremsstrahlung emission integrated over observed energy range

    nu1 = 0.7/T0
    nu2 = 2./T0

    def brm_pow(_n, _T):
        return _n**2 #* sqrt(_T) * (exp(-nu1/_T) - exp(-nu2/_T))


    # 4) integrate emission along line of sight of length Lint

    Lint = 4.

    # total surface brightness from simulation domain
    sb = brm_pow(rho, T).sum(axis=0) * dl

    # add contribution outside the box
    for l in arange(Lbox,Lint,dl):
        sb += brm_pow(n_prof(l), t_prof(l)) * dl

    return sb
