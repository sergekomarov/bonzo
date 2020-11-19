beta=100.
gam=1.67
va2 = 1.
k0 = 4*pi

th = pi/12*arange(7)
D1 = array([70.7, 58.3, 46.2, 24.2, 30.2, 21.1, 11.3])
D2 = array([60. , 55. , 39. , 14. ,  6. ,  9.3, 11.3])
D3 = array([46.7, 42.2, 25.2,  4.9,  1.5,  8.1, 11.3])

alpha = 0.5*beta*gam
A = sqrt((1+alpha)**2 - 4*alpha*cos(th)**2)
cf = sqrt(0.5*va2 * (1+alpha+A))
om = k0*cf / (2*pi)

D1om = D1/om
D2om = D2/om
D3om = D3/om

figure(figsize=(11,5))
x = D1om*cos(th)
y = D1om*sin(th)
plot(x,y, 'x-')
x = D2om*cos(th)
y = D2om*sin(th)
plot(x,y, 'x-')
x = D3om*cos(th)
y = D3om*sin(th)
plot(x,y, 'x-')

for th1 in th:
    r1 = linspace(0,D1om[0],10)
    x1 = r1*cos(th1)
    y1 = r1*sin(th1)
    plot(x1,y1, 'k--', linewidth=0.5)

xlabel(r'$D_{\parallel}/\omega$')
ylabel(r'$D_{\perp}/\omega$')
xlim(xmin=0,xmax=4.)
ylim(ymin=0,ymax=1.5)
subplots_adjust(bottom=0.15)

text(1.6,0.67, r'$\lambda_e = 0.01 \lambda_{e,{\rm Sp}}$', fontsize=16)
text(2.1,1., r'$\lambda_e = 0.1 \lambda_{e,{\rm Sp}}$', fontsize=16)
text(2.4,1.17, r'$\lambda_e = \lambda_{e,{\rm Sp}}$', fontsize=16)

# plot sound speed for anisotropic plasma
beta=100.
p = 0.5*beta
dp = 0.01*p
ppd = p-2./3*dp
ppl = p+1./3*dp

C1 = lambda th, ppd,ppl: 1+2*ppd+(2*ppl-ppd)*cos(th)**2
C2 = lambda th, ppd,ppl: cos(th)**2 * (3*ppl*(3*ppl*cos(th)**2-C1(th,ppd,ppl)) + ppd**2 * (1-cos(th)**2))
Cf = lambda th, ppd,ppl: sqrt(0.5*(C1(th,ppd,ppl) + sqrt(C1(th,ppd,ppl)**2 + 4*C2(th,ppd,ppl))))
D = lambda th,ppd,ppl: C1(th,ppd,ppl)**2 + 4*C2(th,ppd,ppl)

cf = array([Cf(th1,ppd,ppl) for th1 in th])
d = array([D(th1,ppd,ppl) for th1 in th])

x = cf*cos(th)
y = cf*sin(th)
plot(x,y, 'x-')
