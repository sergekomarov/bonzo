import os,glob,sys
import h5py
import numpy as np
import matplotlib.pyplot as plt

out_folder = sys.argv[1]

flist = glob.glob(os.path.join(out_folder,'slc_*'))
N = len(flist)
DT = np.zeros(N)
t = np.zeros(N)

i=0

for fname in flist:
    with h5py.File(fname, 'r') as f:
        t[i] = f.attrs['t']
        rho = np.asarray(f['rho'])
        p = np.asarray(f['p'])

    T = p/rho
    DT[i] = np.sqrt((T**2).mean() - T.mean()**2) / T.mean()
    i+=1

np.save('temp_damp.npy',[t,DT])
# plt.figure()
# plt.semilogy(t, DT)
# plt.show()
