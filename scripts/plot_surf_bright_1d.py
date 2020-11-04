from numpy import *
from matplotlib.pyplot import *
import h5py
from astropy.io import fits
from scipy.ndimage.filters import gaussian_filter as gf


L0 = 20.
cmap='cubehelix'
vmin=0.93
vmax=1.3

dat_fname1 = 'sb_h.npy'
dat_fname2 = 'sb_hc.npy'
dat_fname3 = 'sb_hec.npy'
pers_fname = 'pers_div.npy'

Dr = 4.
Dphi = pi/28.
phis = [0.35*pi, 0.25*pi, 0.15*pi]
Nbin=25

figsize1=(4.7,4.)
figsize2=(4.7,5.5)

ymin=0.96
ymax=1.18


def plot_perseus(ax):

    # plot X-ray image

    pers = load(pers_fname)
    (M,N) = shape(pers)

    dl = 0.5 * 0.38

    perss = gf(pers,1.5)*1. / 284.

    im=ax.imshow(perss[::-1], extent = [0,N*dl,0,M*dl],
               interpolation='nearest', cmap=cmap, vmin=vmin, vmax=vmax)

    # draw shock interface

    R = 15.
    x0 = 20.5
    y0 = -0.2

    x = linspace(x0-R,x0,200)
    y = sqrt(R**2 - (x-x0)**2) + y0

    ax.plot(x,y,'w--', alpha=0.6, linewidth=2.)
    ax.set_ylim(ymin=0,ymax=20)
    ax.set_xlim(xmin=0,xmax=20)

    # draw sectors for averageing

    def plot_seg(phi):

        ax.plot([x0-(R-Dr)*cos(phi-Dphi), x0-(R+Dr)*cos(phi-Dphi),
              x0-(R+Dr)*cos(phi+Dphi), x0-(R-Dr)*cos(phi+Dphi),
              x0-(R-Dr)*cos(phi-Dphi)],
             [y0+(R-Dr)*sin(phi-Dphi), y0+(R+Dr)*sin(phi-Dphi),
              y0+(R+Dr)*sin(phi+Dphi), y0+(R-Dr)*sin(phi+Dphi),
              y0+(R-Dr)*sin(phi-Dphi)],
             'w', alpha=0.7, linewidth=1.6)

    for phi0 in phis:
        plot_seg(phi0)


    # produce averaged radial profiles

    sb=[]
    rms=[]

    dr = 2*Dr/Nbin

    for phi0 in phis:

        sb1 = zeros(Nbin)
        rms1 = zeros(Nbin)
        cnt1 = zeros(Nbin)

        for i in range(M):
            for j in range(N):

                x=i*dl
                y=j*dl

                r = sqrt((x-x0)**2 + (y-y0)**2)
                phi = arccos((x0-x) / r)

                for k in arange(Nbin):
                    r1 = R-Dr + k*dr
                    r2 = r1 + dr
                    phi1 = phi0-Dphi
                    phi2 = phi0+Dphi
                    if phi1<phi and phi<phi2 and r1<r and r<r2:
                        sb1[k]  += perss[j,i]
                        rms1[k] += perss[j,i]**2
                        cnt1[k] += 1
                        # ax.plot([x],[y],'w.',ms=0.5, alpha=0.7)

        sb1  = sb1/cnt1
        rms1 = rms1/cnt1
        rms1 = sqrt(rms1 - sb1**2)

        sb.append(sb1)
        rms.append(rms1)
        l = arange(Nbin)*dr + 0.5*dr

    return (l, sb, rms, im)


# =============================================================

fig, ax = subplots(1, 1, figsize=figsize1)
fig.subplots_adjust(left=0.12, right=0.98, bottom=0.12, top=0.97)

l_pers, sb_pers, rms_pers, im = plot_perseus(ax)
fig.colorbar(im, ax=ax, cmap=cmap)

ax.set_xticks([0,5,10,15,20])
ax.set_yticks([0,5,10,15,20])
ax.set_xlabel(r'$x$ [kpc]')
ax.set_ylabel(r'$y$ [kpc]')

ax.text(12.5,14.6, 'I', color='w', fontsize=18)
ax.text(8.,11.3, 'II', color='w', fontsize=18)
ax.text(4.7,7., 'III', color='w', fontsize=18)

show()


# =============================================================

sb1 = load(dat_fname1)
sb2 = load(dat_fname2)
sb3 = load(dat_fname3)

# select upper left quarter
(M,N) = shape(sb1)
ic = M/2
sb1 = sb1[ic:,:ic]
sb2 = sb2[ic:,:ic]
sb3 = sb3[ic:,:ic]

dx = L0 / ic

xs = 5.4
ish = xs/dx
j1 = int(ish - Dr/dx - 1)
j2 = int(ish + Dr/dx - 1)


fig, (ax1,ax2,ax3) = subplots(3, 1, figsize=figsize2)
fig.subplots_adjust(left=0.12, hspace=0.02,right=0.98, bottom=0.11, top=0.97)

props = dict(boxstyle='square', facecolor='w', alpha=0.7)

#-----------------------------------------------------------------

name = 'I'

# simulation
l = arange(0,j2-j1)*dx

if j1<0:
    l = l[-j1:]
    j1 = 0

l += 11

ax1.plot(l, sb1[0,j1:j2][::-1], linewidth=1.5, label='HD')
ax1.plot(l, sb2[0,j1+24:j2+24][::-1], linewidth=1.5, linestyle='--', label='HD+TC')
ax1.plot(l, sb3[0,j1+10:j2+10][::-1], linewidth=1.5, linestyle='-.', label='HD+TC+TT')

# data
l_pers += 11
ax1.plot(l_pers, sb_pers[0], color='C3', label='Perseus')
ax1.plot(l_pers, sb_pers[0]-rms_pers[0], color='C3', alpha=0.3)
ax1.plot(l_pers, sb_pers[0]+rms_pers[0], color='C3', alpha=0.3)
ax1.fill_between(l_pers, sb_pers[0]-rms_pers[0], sb_pers[0]+rms_pers[0],
                alpha=0.05, color='C3')

# ax1.errorbar(l_pers, sb_pers[0], yerr=rms_pers[0],
#              capsize=3, ecolor='#1f77b4', fmt='none')

ax1.text(0.033,0.1, name, fontsize=14, bbox=props, transform=ax1.transAxes)

ax1.set_ylim(ymin=ymin, ymax=ymax)

ax1.set_ylabel(r'SB')
# ax1.set_xticks([])
ax1.set_xticklabels([])


#---------------------------------------------------------------


name = 'II'

# simulation
ax2.plot(l, sb1[0,j1:j2][::-1], linewidth=1.5, label='HD')
ax2.plot(l, sb2[0,j1+24:j2+24][::-1], linewidth=1.5, linestyle='--', label='HD+TC')
ax2.plot(l, sb3[0,j1+10:j2+10][::-1], linewidth=1.5, linestyle='-.', label='HD+TC+TT')

# data
ax2.plot(l_pers, sb_pers[1], color='C3', label='Perseus')
ax2.plot(l_pers, sb_pers[1]-rms_pers[1], color='C3', alpha=0.3)
ax2.plot(l_pers, sb_pers[1]+rms_pers[1], color='C3', alpha=0.3)
ax2.fill_between(l_pers, sb_pers[1]-rms_pers[1], sb_pers[1]+rms_pers[1],
                alpha=0.05, color='C3')

# ax2.errorbar(l_pers, sb_pers[1], yerr=rms_pers[1],
#              capsize=3, ecolor='#1f77b4', fmt='none')

ax2.text(0.033,0.1, name, fontsize=14, bbox=props, transform=ax2.transAxes)

ax2.set_ylim(ymin=ymin, ymax=ymax)

ax2.set_ylabel(r'SB')
# ax2.set_xticks([])
ax2.set_xticklabels([])

#---------------------------------------------------------------


name = 'III'

# simulation
ax3.plot(l, sb1[0,j1:j2][::-1], linewidth=1.5, label='HD')
ax3.plot(l, sb2[0,j1+24:j2+24][::-1], linewidth=1.5, linestyle='--', label='HD+TC')
ax3.plot(l, sb3[0,j1+10:j2+10][::-1], linewidth=1.5, linestyle='-.', label='HD+TC+TT')

# data
ax3.plot(l_pers, sb_pers[2], color='C3', label='Perseus')
ax3.plot(l_pers, sb_pers[2]-rms_pers[2], color='C3', alpha=0.3)
ax3.plot(l_pers, sb_pers[2]+rms_pers[2], color='C3', alpha=0.3)
ax3.fill_between(l_pers, sb_pers[2]-rms_pers[2], sb_pers[2]+rms_pers[2],
                alpha=0.05, color='C3')


# ax3.errorbar(l_pers, sb_pers[2], yerr=rms_pers[2],
#              capsize=3, ecolor='#1f77b4', fmt='none')

ax3.text(0.033,0.1, name, fontsize=14, bbox=props, transform=ax3.transAxes)

ax3.set_ylim(ymin=ymin, ymax=ymax)

ax3.set_ylabel(r'SB')
ax3.set_xlabel(r'$r$ [kpc]')

ax1.tick_params(direction='in', top=True, right=True)
ax2.tick_params(direction='in', top=True, right=True)
ax3.tick_params(direction='in', top=True, right=True)

ax1.legend(loc='upper right',fancybox=True, fontsize=11,
           borderpad=0.25,labelspacing=0.1,handletextpad=0.1, frameon=False)


show()
