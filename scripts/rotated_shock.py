import h5py
import numpy as np

slc = np.s_[0,3:-3,3:-3]

f = h5py.File('../shock_test/grid_100.hdf5','r')

rho = np.asarray(f['rho'])[slc]
vx = np.asarray(f['vx'])[slc]
vy = np.asarray(f['vy'])[slc]
vz = np.asarray(f['vz'])[slc]
p = np.asarray(f['p'])[slc]
bx = np.asarray(f['bxc'])[slc]
by = np.asarray(f['byc'])[slc]
bz = np.asarray(f['bzc'])[slc]
Lbox = f.attrs['size']
dl = f.attrs['dl']

f.close()

Nbox = np.shape(rho)[::-1]
print Nbox

a = []
s = []

def transform_inv(ax, ay, az):

  ax1 = (ax  + 2*ay)/np.sqrt(5)
  ay1 = (-2*ax + ay)/np.sqrt(5)
  az1 = az

  return (ax1,ay1,az1)

for i in range(Nbox[0]):
    j = (i-Nbox[0]/2)/2 - Nbox[1]/2
    # k = (i-Nbox[0]/2)/2 + Nbox[2]/2
    if j>=0 and j<Nbox[1]:
        a.append(rho[j,i])
        s.append(transform_inv(vx[j,i], vy[j,i], vz[j,i]))


np.save('a.npy', a)
np.save('s.npy', s)
