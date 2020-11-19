from numpy import *
from matplotlib.pyplot import *
import h5py


L0 = 20.

xmin = 8.
xmax = 20.

ymin_rho=0.85
ymax_rho=1.33
ymin_t=0.902
ymax_t = 1.23
ymin_p=0.802
ymax_p = 1.57

N=768

dat_fname1 = 'slc_h.hdf5'
dat_fname2 = 'slc_hc.hdf5'
dat_fname3 = 'slc_hec.hdf5'

figsize=(5.5,5.4)


# =============================================================

fig, (ax1,ax2,ax3) = subplots(3, 1, figsize=figsize)
fig.subplots_adjust(hspace=0.02, left=0.12, right=0.96,
                    bottom=0.1, top=0.97)

ic=N/2
slc = np.s_[0,:,:]

# -------------------------------------------------------------

with h5py.File(dat_fname1,'r') as f:
    p2 = asarray(f['p'])[slc]
    rho2 = asarray(f['rho'])[slc]

T2=p2/rho2

dl = 2*L0/N

# lmin=6.7
# imin=int(lmin/dl)
# l = arange(imin, ic, 1)*dl + 0.5*dl

rho=[]
T=[]
p=[]
l=[]
lng = 0.

for i in range(ic-1,-1,-1):
    j = ic#+ic-i-1
    rho.append(rho2[i,j])
    T.append(T2[i,j])
    p.append(p2[i,j])
    l.append(lng)
    lng += dl

ax1.plot(l, rho, linewidth=1.5, label='HD')
ax2.plot(l, T, linewidth=1.5)
ax3.plot(l, p, linewidth=1.5)

#---------------------------------------------------------------

with h5py.File(dat_fname2,'r') as f:
    p2 = asarray(f['p'])[slc]
    rho2 = asarray(f['rho'])[slc]

T2=p2/rho2

rho=[]
T=[]
p=[]

for i in range(ic-1,-1,-1):
    j = ic#+ic-i-1
    rho.append(rho2[i,j])
    T.append(T2[i,j])
    p.append(p2[i,j])

ax1.plot(l, rho, linestyle='--', color='C1', linewidth=1.5, label='HD+TC')
ax2.plot(l, T, linestyle='--', color='C1', linewidth=1.5)
ax3.plot(l, p, linestyle='--', color='C1', linewidth=1.5)

#---------------------------------------------------------------


with h5py.File(dat_fname3,'r') as f:
    pe2 = 2*asarray(f['pe'])[slc]
    pi2 = 2*asarray(f['p'])[slc]
    rho2 = asarray(f['rho'])[slc]

Te2=pe2/rho2
Ti2=pi2/rho2

rho=[]
Te=[]
Ti=[]
pe=[]

for i in range(ic-1,-1,-1):
    j = ic#+ic-i-1
    rho.append(rho2[i,j])
    Ti.append(Ti2[i,j])
    Te.append(Te2[i,j])
    pe.append(pe2[i,j])

ax1.plot(l, rho, linestyle='-.', color='C2', linewidth=1.5, label='HD+TC+TT')
ax2.plot(l, Te, linestyle='-.',  color='C2', linewidth=1.5)
ax2.plot(l, Ti, linestyle=':', color='C2', linewidth=1.5)
ax3.plot(l, pe, linestyle='-.',  color='C2', linewidth=1.5)


#---------------------------------------------------------------


ax1.set_ylabel(r'$n_e/n_0$', fontsize=15)
ax2.set_ylabel(r'$T_e/T_0$', fontsize=15)
ax3.set_ylabel(r'$p_e/p_0$', fontsize=15)

ax3.set_xlabel(r'$r$ [kpc]')
ax1.set_xticklabels([])
ax2.set_xticklabels([])

ax1.set_xlim(xmin=xmin,xmax=xmax)
ax2.set_xlim(xmin=xmin,xmax=xmax)
ax3.set_xlim(xmin=xmin,xmax=xmax)

ax1.set_ylim(ymin=ymin_rho, ymax=ymax_rho)
ax2.set_ylim(ymin=ymin_t, ymax=ymax_t)
ax3.set_ylim(ymin=ymin_p, ymax=ymax_p)

ax1.legend(loc='upper right',fancybox=True, fontsize=12,
           borderpad=0.25,labelspacing=0.1,handletextpad=0.1, frameon=False)

ax1.tick_params(direction='in', top=True, right=True)
ax2.tick_params(direction='in', top=True, right=True)
ax3.tick_params(direction='in', top=True, right=True)

show()
