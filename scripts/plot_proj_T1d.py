from numpy import *
from matplotlib.pyplot import *
import h5py
from astropy.io import fits
from scipy.ndimage.filters import gaussian_filter as gf

L0 = 20.
Dr = 5.

dat_fname1 = 'T_h.npy'
dat_fname2 = 'T_hc.npy'
dat_fname3 = 'T_hec.npy'

dat_fname4 = 'p_h.npy'
dat_fname5 = 'p_hc.npy'
dat_fname6 = 'p_hec.npy'

# pers_fname = 'pers_div.npy'

figsize=(5.,3.8)

ymin_t=0.99
ymax_t=1.045

ymin_p=0.985
ymax_p=1.07


# =============================================================

fig, (ax1,ax2) = subplots(2, 1, figsize=figsize)
fig.subplots_adjust(hspace=0.02, left=0.125, right=0.97,
                    bottom=0.13, top=0.97)

T1 = load(dat_fname1)
T2 = load(dat_fname2)
T3 = load(dat_fname3)

p1 = load(dat_fname4)
p2 = load(dat_fname5)
p3 = load(dat_fname6)

# select upper left quarter
(M,N) = shape(T1)
ic = M/2
T1 = T1[ic:,:ic]
T2 = T2[ic:,:ic]
T3 = T3[ic:,:ic]
p1 = p1[ic:,:ic]
p2 = p2[ic:,:ic]
p3 = p3[ic:,:ic]

dx = L0 / ic

xs = 5.4
ish = xs/dx
j1 = int(ish - Dr/dx - 1)
j2 = int(ish + Dr/dx - 1)

props = dict(boxstyle='square', facecolor='w', alpha=0.7)

#-----------------------------------------------------------------

# simulation
l = arange(0,j2-j1)*dx

if j1<0:
    l = l[-j1:]
    j1 = 0

l += 10.07

ax1.plot(l, T1[0,j1:j2][::-1], linewidth=1.5, label='HD')
ax1.plot(l, T2[0,j1+22:j2+22][::-1], linewidth=1.5, linestyle='--', label='HD+TC')
ax1.plot(l, T3[0,j1+8:j2+8][::-1], linewidth=1.5, linestyle='-.', label='HD+TC+TT')

ax2.plot(l, p1[0,j1:j2][::-1], linewidth=1.5)
ax2.plot(l, p2[0,j1+22:j2+22][::-1], linewidth=1.5, linestyle='--')
ax2.plot(l, p3[0,j1+8:j2+8][::-1], linewidth=1.5, linestyle='-.')

ax1.set_xticklabels([])
ax2.set_xlabel(r'$r$ [kpc]')

ax1.set_ylim(ymin=ymin_t, ymax=ymax_t)
ax2.set_ylim(ymin=ymin_p, ymax=ymax_p)

ax1.set_ylabel(r'$T_{e,{\rm proj}}$')
ax2.set_ylabel(r'$p_{e,{\rm proj}}$')
ax2.set_yticks([1,1.02,1.04,1.06])

ax1.tick_params(direction='in', top=True, right=True)
ax2.tick_params(direction='in', top=True, right=True)

ax1.legend(loc='upper right',fancybox=True, fontsize=11,
           borderpad=0.25,labelspacing=0.1,handletextpad=0.1, frameon=False)

show()
