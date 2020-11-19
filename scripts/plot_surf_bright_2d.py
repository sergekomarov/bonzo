from numpy import *
from matplotlib.pyplot import *
import h5py
from astropy.io import fits
from scipy.ndimage.filters import gaussian_filter as gf
from surf_bright import surf_bright
from mpl_toolkits.axes_grid1 import make_axes_locatable


L0 = 20.
cmap='cubehelix'
vmin=0.94
vmax=1.27

dat_fname1 = 'sb_h.npy'
dat_fname2 = 'sb_mc.npy'
dat_fname3 = 'sb_mec.npy'
pers_fname = 'pers_div.npy'

# figsize=(8.34,4.)
figsize=(10.,3.4)

# =============================================================

sb1 = load(dat_fname1)
sb2 = load(dat_fname2)
sb3 = load(dat_fname3)

(M,N) = shape(sb1)
ic = M/2
sb1 = sb1[ic:,:ic]
sb2 = sb2[ic:,:ic]
sb3 = sb3[ic:,:ic]

fig, (ax1,ax2,ax3,ax4) = subplots(1, 4, figsize=figsize, sharey=True)
fig.subplots_adjust(wspace=0.01,left=0.05, right=0.95,
                    bottom=0.22, top=0.97)

props = dict(boxstyle='square', facecolor='w', alpha=0.5)

name = 'HD'

im1 = ax1.imshow(sb1[::-1], extent=[0,L0,0,L0], interpolation='nearest',
                vmin=vmin, vmax=vmax, cmap=cmap)
ax1.set_xticks([0,5,10,15,20])
# ax1.set_xticklabels([])
ax1.set_yticks([0,5,10,15,20])
ax1.set_xlabel(r'$x$ [kpc]')
ax1.set_ylabel(r'$y$ [kpc]')
ax1.text(0.06,0.89, name, fontsize=14, bbox=props, transform=ax1.transAxes)

# --------------------------------------------------------------

name = 'ATC'

im2 = ax2.imshow(sb2[::-1], extent=[0,L0,0,L0], interpolation='nearest',
                vmin=vmin, vmax=vmax, cmap=cmap)
ax2.set_xticks([0,5,10,15,20])
# ax1.set_xticklabels([])
ax2.set_yticks([0,5,10,15,20])
ax2.set_xlabel(r'$x$ [kpc]')
# ax2.set_ylabel(r'$y$ [kpc]')
ax2.text(0.06,0.89, name, fontsize=14, bbox=props, transform=ax2.transAxes)


# --------------------------------------------------------------

name = 'ATC+TT'

im3 = ax3.imshow(sb3[::-1], extent=[0,L0,0,L0], interpolation='nearest',
                vmin=vmin, vmax=vmax, cmap=cmap)
ax3.set_xticks([0,5,10,15,20])
ax3.set_yticks([0,5,10,15,20])
# ax3.set_yticklabels([])
ax3.set_xlabel(r'$x$ [kpc]')
ax3.text(0.06,0.89, name, fontsize=14, bbox=props, transform=ax3.transAxes)


# --------------------------------------------------------------

name = 'Perseus'

pers=load(pers_fname)
perss = gf(pers,2) / 282.

im4 = imshow(perss[::-1], interpolation='nearest',
        extent=[0,L0,0,L0], cmap=cmap, vmin=vmin,vmax=vmax)
#plot_perseus(ax4, pers_fname, vmin,vmax)
ax4.set_xlabel(r'$x$ [kpc]')
ax4.set_xticks([0,5,10,15,20])
ax4.set_yticks([0,5,10,15,20])
# ax4.set_yticklabels([])
ax4.text(0.06,0.89, name, fontsize=14, bbox=props, transform=ax4.transAxes)

divider = make_axes_locatable(ax4)
cax = divider.append_axes("right", size="4.5%", pad=0.15)
cbar=fig.colorbar(im4,ax=ax4,cax=cax)
# cbar.ax.tick_params(labelsize=18)

show()
