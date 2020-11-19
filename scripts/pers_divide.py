from numpy import *
from matplotlib.pyplot import *
import h5py
from astropy.io import fits
from scipy.ndimage.filters import gaussian_filter as gf


rc = 29.
beta0=0.53

pers_fname = '../weak_shock/merged_image_07-2.fits'

def beta_mod(r):
    return (1+(r/rc)**2)**(0.5-3*beta0)

# select part of image containing shock
f = fits.open(pers_fname)
i1=4007 #3964
i2=4113
j1=3821
j2=3927 #3958

pers = asarray(f[0].data, dtype=double)
f.close()

(M,N) = shape(pers)

# cluster center is ofset relative to shock
i0c = 3957
j0c = 3965

# x0c = x0s + 5.89
# y0c = y0s - 8.17

dl = 0.5 * 0.38
beta = zeros(shape(pers))

for i in range(M):
    for j in range(N):
        r = sqrt((j - j0c)**2 + (i - i0c)**2)
        beta[i,j] = beta_mod(r/dl)
        pers[i,j] /= beta[i,j]

save('pers_div0.npy',pers)
perss = gf(pers,1.5)*1. / gf(pers,20)
save('pers_div1.npy',perss)

figure()
imshow(log(perss[::-1]), extent = [0,N*dl,0,M*dl],
       interpolation='nearest', cmap='cubehelix')
show()

# choose the region containing the shock
perss = perss[i1:i2,j1:j2]
beta  =  beta[i1:i2,j1:j2]
save('pers_div2.npy',perss)
save('beta_pers1.npy',beta)

# shock center is in bottom right corner
x0s = 20.
y0s = 0. #+ 8.17
R = 14.63

x = linspace(x0s-R,x0s,200)
y = sqrt(R**2 - (x-x0s)**2) + y0s

(M,N) = shape(perss)

figure()
imshow(perss[::-1], extent = [0,N*dl,0,M*dl],
       interpolation='nearest', cmap='cubehelix')
plot(x,y,'w--', alpha=0.6, linewidth=2.)
show()
