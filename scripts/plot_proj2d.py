from numpy import *
from matplotlib.pyplot import *
import h5py
from astropy.io import fits
from scipy.ndimage.filters import gaussian_filter as gf
from surf_bright import surf_bright
from mpl_toolkits.axes_grid1 import make_axes_locatable


L0 = 20.
cmap='jet'
vmin_t=0.9804
vmax_t=1.043

vmin_p=0.971
vmax_p=1.063

dat_fname1 = 'T_h.npy'
dat_fname2 = 'T_mc.npy'
dat_fname3 = 'T_mec.npy'

dat_fname4 = 'p_h.npy'
dat_fname5 = 'p_mc.npy'
dat_fname6 = 'p_mec.npy'

# figsize=(7.,4.1)
figsize=(10,8)

# =============================================================


T1 = load(dat_fname1)
T2 = load(dat_fname2)
T3 = load(dat_fname3)

p1 = load(dat_fname4)
p2 = load(dat_fname5)
p3 = load(dat_fname6)

(M,N) = shape(T1)
ic = M/2

T1 = T1[ic:,:ic]
T2 = T2[ic:,:ic]
T3 = T3[ic:,:ic]

p1 = p1[ic:,:ic]
p2 = p2[ic:,:ic]
p3 = p3[ic:,:ic]

fig, axes = subplots(2, 3, figsize=figsize, sharex=True, sharey=True)
fig.subplots_adjust(wspace=0.0, hspace=0.08, left=0.06, right=0.945,
                    bottom=0.13, top=0.97)

ax1=axes[0,0]
ax2=axes[0,1]
ax3=axes[0,2]

ax4=axes[1,0]
ax5=axes[1,1]
ax6=axes[1,2]

props = dict(boxstyle='square', facecolor='w', alpha=0.5)

name = r'HD, $T_{e,{\rm proj}}$'

im1 = ax1.imshow(T1[::-1], extent=[0,L0,0,L0], interpolation='nearest',
                vmin=vmin_t, vmax=vmax_t, cmap=cmap)
ax1.set_xticks([0,5,10,15,20])
ax1.set_yticks([0,5,10,15,20])
# ax1.set_xlabel(r'$x$ [kpc]')
ax1.set_ylabel(r'$y$ [kpc]')
ax1.text(0.06,0.88, name, fontsize=14, bbox=props, transform=ax1.transAxes)
# fig.colorbar(im1, ax=ax1)


# --------------------------------------------------------------

name = r'ATC, $T_{e,{\rm proj}}$'

im2 = ax2.imshow(T2[::-1], extent=[0,L0,0,L0], interpolation='nearest',
                vmin=vmin_t, vmax=vmax_t, cmap=cmap)
ax2.set_xticks([0,5,10,15,20])
ax2.set_yticks([0,5,10,15,20])
# ax2.set_xlabel(r'$x$ [kpc]')
# ax2.set_ylabel(r'$y$ [kpc]')
ax2.text(0.06,0.88, name, fontsize=14, bbox=props, transform=ax2.transAxes)
# fig.colorbar(im2, ax=ax2)


# --------------------------------------------------------------

name = r'ATC+TT, $T_{e,{\rm proj}}$'

im3 = ax3.imshow(T3[::-1], extent=[0,L0,0,L0], interpolation='nearest',
                vmin=vmin_t, vmax=vmax_t, cmap=cmap)
ax3.set_xticks([0,5,10,15,20])
ax3.set_yticks([0,5,10,15,20])
# ax3.set_xlabel(r'$x$ [kpc]')
ax3.text(0.06,0.88, name, fontsize=14, bbox=props, transform=ax3.transAxes)

divider = make_axes_locatable(ax3)
cax = divider.append_axes("right", size="4.5%", pad=0.15)
cbar=fig.colorbar(im3,ax=ax3,cax=cax)

# ------------------------------------------------------------------

name = r'HD, $p_{e,{\rm proj}}$'

im4 = ax4.imshow(p1[::-1], extent=[0,L0,0,L0], interpolation='nearest',
                vmin=vmin_p, vmax=vmax_p, cmap=cmap)
ax4.set_xticks([0,5,10,15,20])
# ax4.set_xticklabels([])
ax4.set_yticks([0,5,10,15,20])
ax4.set_xlabel(r'$x$ [kpc]')
ax4.set_ylabel(r'$y$ [kpc]')
ax4.text(0.06,0.88, name, fontsize=14, bbox=props, transform=ax4.transAxes)
# fig.colorbar(im4, ax=ax4)


# --------------------------------------------------------------

name = r'ATC, $p_{e,{\rm proj}}$'

im5 = ax5.imshow(p2[::-1], extent=[0,L0,0,L0], interpolation='nearest',
                vmin=vmin_p, vmax=vmax_p, cmap=cmap)
ax5.set_xticks([0,5,10,15,20])
ax5.set_yticks([0,5,10,15,20])
ax5.set_xlabel(r'$x$ [kpc]')
# ax5.set_ylabel(r'$y$ [kpc]')
ax5.text(0.06,0.88, name, fontsize=14, bbox=props, transform=ax5.transAxes)
# fig.colorbar(im5, ax=ax5)


# --------------------------------------------------------------

name = r'ATC+TT, $p_{e,{\rm proj}}$'

im6 = ax6.imshow(p3[::-1], extent=[0,L0,0,L0], interpolation='nearest',
                vmin=vmin_p, vmax=vmax_p, cmap=cmap)
ax6.set_xticks([0,5,10,15,20])
ax6.set_yticks([0,5,10,15,20])
ax6.set_xlabel(r'$x$ [kpc]')
ax6.text(0.06,0.88, name, fontsize=14, bbox=props, transform=ax6.transAxes)

divider = make_axes_locatable(ax6)
cax = divider.append_axes("right", size="4.5%", pad=0.15)
cbar=fig.colorbar(im6,ax=ax6,cax=cax)

show()
