from numpy import *
from matplotlib.pyplot import *
import h5py

sd = 8
N=384
k=N/2

L0 = 20.
cmap='jet'

vmin_rho=0.85
vmax_rho=1.33

vmin_t=0.9
vmax_t = 1.23

vmin_p=0.75
vmax_p = 1.54

dat_fname1 = 'grid_mc.hdf5'
dat_fname2 = 'grid_mec.hdf5'

#figsize=(7.34,6.5)
figsize=(10,8)


# =============================================================


slc = np.s_[k,k:,:k]

fig, axes = subplots(2, 4, sharey=True, sharex=True, figsize=figsize)
fig.subplots_adjust(wspace=0.0, hspace=0.07, left=0.03, right=0.98,
                    bottom=0.115, top=0.97)

ax1=axes[0,0]
ax2=axes[0,1]
ax3=axes[0,2]
ax4=axes[0,3]

ax5=axes[1,0]
ax6=axes[1,1]
ax7=axes[1,2]
ax8=axes[1,3]


props = dict(boxstyle='square', facecolor='w', alpha=0.8)
slc_b = np.s_[::sd,::sd]


with h5py.File(dat_fname1,'r') as f:
    p = asarray(f['p'])[slc]
    rho = asarray(f['rho'])[slc]
    bx = asarray(f['bxc'])[slc]
    by = asarray(f['byc'])[slc]
    bz = asarray(f['bzc'])[slc]

name = '$n_e$, ATC'

im1 = ax1.imshow(rho[::-1], extent=[0,L0,0,L0], interpolation='nearest',
                vmin=vmin_rho, vmax=vmax_rho, cmap=cmap)
ax1.set_xticks([0,5,10,15,20])
# ax1.set_xticklabels([])
ax1.set_yticks([0,5,10,15,20])
# ax1.set_xlabel(r'$x$ [kpc]')
ax1.set_ylabel(r'$y$ [kpc]')
ax1.text(0.06,0.88, name, fontsize=13, bbox=props, transform=ax1.transAxes)
fig.colorbar(im1, ax=ax1)

dx = L0 * float(sd)/k#/90
X,Y = meshgrid(arange(0,L0,dx), arange(0,L0,dx))
Btot = sqrt(bx[slc_b]**2 + by[slc_b]**2 + bz[slc_b]**2)
Bm   = sqrt(bx[slc_b]**2 + by[slc_b]**2)

ax1.quiver(X,Y, bx[slc_b]/Bm, by[slc_b]/Bm,
        color='k', scale = 30, width=3e-3, headwidth=2.8)


# --------------------------------------------------------------

T=p/rho

name = '$T_e$, ATC'

im2 = ax2.imshow(T[::-1], extent=[0,L0,0,L0], interpolation='nearest',
                vmin=vmin_t, vmax=vmax_t, cmap=cmap)
ax2.set_xticks([0,5,10,15,20])
ax2.set_yticks([0,5,10,15,20])
# ax2.set_xlabel(r'$x$ [kpc]')
# ax2.set_ylabel(r'$y$ [kpc]')
ax2.text(0.06,0.88, name, fontsize=13, bbox=props, transform=ax2.transAxes)
fig.colorbar(im2, ax=ax2, ticks=[0.9,1,1.1,1.2])

ax2.quiver(X,Y, bx[slc_b]/Bm, by[slc_b]/Bm,
        color='k', scale = 30, width=3e-3, headwidth=2.8)

# --------------------------------------------------------------

name = '$T_i$, ATC'

im3 = ax3.imshow(T[::-1], extent=[0,L0,0,L0], interpolation='nearest',
                vmin=vmin_t, vmax=vmax_t, cmap=cmap)
ax3.set_xticks([0,5,10,15,20])
ax3.set_yticks([0,5,10,15,20])
# ax3.set_xlabel(r'$x$ [kpc]')
# ax3.set_ylabel(r'$y$ [kpc]')
ax3.text(0.06,0.88, name, fontsize=13, bbox=props, transform=ax3.transAxes)
fig.colorbar(im3, ax=ax3, ticks=[0.9,1,1.1,1.2])

ax3.quiver(X,Y, bx[slc_b]/Bm, by[slc_b]/Bm,
        color='k', scale = 30, width=3e-3, headwidth=2.8)

# --------------------------------------------------------------

name = '$p_e$, ATC'

im4 = ax4.imshow(p[::-1], extent=[0,L0,0,L0], interpolation='nearest',
                vmin=vmin_p, vmax=vmax_p, cmap=cmap)
ax4.set_xticks([0,5,10,15,20])
ax4.set_yticks([0,5,10,15,20])
# ax4.set_xlabel(r'$x$ [kpc]')
# ax4.set_ylabel(r'$y$ [kpc]')
ax4.text(0.06,0.88, name, fontsize=13, bbox=props, transform=ax4.transAxes)
fig.colorbar(im4, ax=ax4, ticks=[0.8,1,1.2,1.4])

ax4.quiver(X,Y, bx[slc_b]/Bm, by[slc_b]/Bm,
        color='k', scale = 30, width=3e-3, headwidth=2.8)


# =============================================================


with h5py.File(dat_fname2,'r') as f:
    pi = asarray(f['p'])[slc]
    pe = asarray(f['pe'])[slc]
    rho = asarray(f['rho'])[slc]
    bx = asarray(f['bxc'])[slc]
    by = asarray(f['byc'])[slc]
    bz = asarray(f['bzc'])[slc]


name = '$n_e$, ATC+TT'

im5 = ax5.imshow(rho[::-1], extent=[0,L0,0,L0], interpolation='nearest',
                vmin=vmin_rho, vmax=vmax_rho, cmap=cmap)
ax5.set_xticks([0,5,10,15,20])
# ax5.set_xticklabels([])
ax5.set_yticks([0,5,10,15,20])
# ax5.set_yticklabels([])
ax5.set_xlabel(r'$x$ [kpc]')
ax5.set_ylabel(r'$y$ [kpc]')
ax5.text(0.06,0.88, name, fontsize=13, bbox=props, transform=ax5.transAxes)
fig.colorbar(im5, ax=ax5)

dx = L0 * float(sd)/k
X,Y = meshgrid(arange(0,L0,dx), arange(0,L0,dx))
Btot = sqrt(bx[slc_b]**2 + by[slc_b]**2 + bz[slc_b]**2)
Bm   = sqrt(bx[slc_b]**2 + by[slc_b]**2)

ax5.quiver(X,Y, bx[slc_b]/Bm, by[slc_b]/Bm,
        color='k', scale = 30, width=3e-3, headwidth=2.8)


# --------------------------------------------------------------

T=2*pe/rho

name = '$T_e$, ATC+TT'

im6 = ax6.imshow(T[::-1], extent=[0,L0,0,L0], interpolation='nearest',
                vmin=vmin_t, vmax=vmax_t, cmap=cmap)
ax6.set_xticks([0,5,10,15,20])
ax6.set_yticks([0,5,10,15,20])
# ax6.set_yticklabels([])
ax6.set_xlabel(r'$x$ [kpc]')
# ax6.set_ylabel(r'$y$ [kpc]')

ax6.text(0.06,0.88, name, fontsize=13, bbox=props, transform=ax6.transAxes)
fig.colorbar(im6, ax=ax6, ticks=[0.9,1,1.1,1.2])

ax6.quiver(X,Y, bx[slc_b]/Bm, by[slc_b]/Bm,
        color='k', scale = 30, width=3e-3, headwidth=2.8)

# --------------------------------------------------------------

T=2*pi/rho

name = '$T_i$, ATC+TT'

im7 = ax7.imshow(T[::-1], extent=[0,L0,0,L0], interpolation='nearest',
                vmin=vmin_t, vmax=vmax_t, cmap=cmap)
ax7.set_xticks([0,5,10,15,20])
ax7.set_yticks([0,5,10,15,20])
# ax7.set_yticklabels([])
ax7.set_xlabel(r'$x$ [kpc]')
# ax7.set_ylabel(r'$y$ [kpc]')

ax7.text(0.06,0.88, name, fontsize=13, bbox=props, transform=ax7.transAxes)
fig.colorbar(im7, ax=ax7, ticks=[0.9,1,1.1,1.2])

ax7.quiver(X,Y, bx[slc_b]/Bm, by[slc_b]/Bm,
        color='k', scale = 30, width=3e-3, headwidth=2.8)

# --------------------------------------------------------------

name = '$p_e$, ATC+TT'

im8 = ax8.imshow(2*pe[::-1], extent=[0,L0,0,L0], interpolation='nearest',
                vmin=vmin_p, vmax=vmax_p, cmap=cmap)
ax8.set_xticks([0,5,10,15,20])
ax8.set_yticks([0,5,10,15,20])
# ax8.set_yticklabels([])
ax8.set_xlabel(r'$x$ [kpc]')
# ax8.set_ylabel(r'$y$ [kpc]')

ax8.text(0.06,0.88, name, fontsize=13, bbox=props, transform=ax8.transAxes)
fig.colorbar(im8, ax=ax8, ticks=[0.8,1,1.2,1.4])

ax8.quiver(X,Y, bx[slc_b]/Bm, by[slc_b]/Bm,
        color='k', scale = 30, width=3e-3, headwidth=2.8)

show()
