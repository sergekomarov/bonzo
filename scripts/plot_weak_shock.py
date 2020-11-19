import numpy as np

N = 256
sd = 8
k = 128

axu=0
axv=1
axw=2

dat_fname = 'box_016_c.hdf5'
b0_fname = 'B0_256.npy'


# load data

slc = np.s_[:,:,k]

with h5py.File(dat_fname,'r') as f:
    p = asarray(f['p'])[slc]
    rho = asarray(f['rho'])[slc]
    bx = asarray(f['Bxc'])[slc]
    by = asarray(f['Byc'])[slc]
    bz = asarray(f['Bzc'])[slc]
    ppd = asarray(f['ppd'])[slc]

B0 = load(b0_fname)


# make plot

figure()

imshow(rho.T[::-1], extent=[0,1,0,1],interpolation='nearest', vmin=0.9)
colorbar()

slc_u = np.s_[::sd,::sd,k,axu]
slc_v = np.s_[::sd,::sd,k,axv]
slc_w = np.s_[::sd,::sd,k,axw]

X,Y = mgrid[0:1:float(sd)/N, 0:1:float(sd)/N]
Btot = sqrt(B0[slc_u]**2 + B0[slc_v]**2+ B0[slc_w]**2)
Bm = sqrt(B0[slc_u]**2 + B0[slc_v]**2)

quiver(X,Y, B0[slc_u]/Bm, B0[slc_v]/Bm,
        color='w', scale = 40, width=2e-3, headwidth=2.5)
