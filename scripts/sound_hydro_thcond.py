from numpy import *
from matplotlib.pyplot import *

beta=100.
gam=1.67
k0 = 4*3.14

lam = 0.05 * array([0.0033,0.01,0.033,0.1,0.33, 0.6, 1.]) * k0 / 0.6
D = array([2., 5.8, 17.8, 28, 11., 6.2, 3.8])

lam2 = 0.025 * array([0.0033,0.01,0.033,0.1,0.33, 0.6, 1.]) * k0 / 0.6
D2 = array([1, 2.96, 9.44, 24., 21., 12., 7.4])

cf = sqrt(0.5*beta*gam)
om = k0*cf

L = 1./ (D/om * lam) * 2
L2 = 1./ (D2/om * lam2) * 2

f = lambda x: 46*(1+8.54e-4 * x**(-3))**(2./3)


figure()
loglog(lam, L)
loglog(lam2, L2)

lam1 = linspace(lam[0],lam[-1],100)
y = array([f(xi) for xi in lam1])
loglog(lam1, y, 'k--')

xlabel(r'$k \lambda_e$')
ylabel(r'$L_d$')

subplots_adjust(bottom=0.15)
