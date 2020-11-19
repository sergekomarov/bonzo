import h5pyim

g = lambda x,t,sig,a: 1./sqrt(1+4*a*t/sig**2) * exp(-(x-1)**2/(sig**2*(1+4*a*t/sig**2))) + 1e-4

a = 0.02
sig=0.15

fnames = ['grid_300.hdf5']
errs = []

for fname in fnames:

    f=h5py.File(fname,'r')
    p = f['p'][0,0,:]
    t= f.attrs['t']
    dx = f.attrs['dl'][0]
    L = f.attrs['size'][0]
    N = shape(p)[0]
    f.close()

    x = arange(0,L,dx)+0.5*dx

    err = array([abs(pi - g(xi,t,sig,a)) for xi,pi in zip(x,p)]).sum() / N
    errs.append(err)

    plot(x,p)
    plot(x, g(x,t,sig,a))
