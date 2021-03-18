#include "../defs_c.h"

extern void cons2prim_1(real **w1, real **u1, int i1, int i2, real gam);
extern void prim2cons_1(real **u1, real **w1, int i1, int i2, real gam);
extern void prim2char_1(real **vc1, real **w1, int i1, int i2, real gam);
extern void char2prim_1(real **vc1, real **w1, int i1, int i2, real gam);

extern void cons2prim(real *w,  real *u, real gam) __attribute__((always_inline));
extern void prim2cons(real *u,  real *w, real gam) __attribute__((always_inline));

extern void prim2char(real *vc, real *w, real gam) __attribute__((always_inline));
extern void char2prim(real *vc, real *w, real gam) __attribute__((always_inline));

extern void prim2flux(real *f,  real *w, real gam) __attribute__((always_inline));
extern void primcons2flux(real *f, real *w, real *u, real gam) __attribute__((always_inline));

extern real fms(real *w, real bx, real gam) __attribute__((always_inline));


void inline cons2prim(real *w, real *u, real gam) {

  real rhoi = 1./u[RHO];

  w[RHO] = u[RHO];
  w[VX] = u[MX] * rhoi;
  w[VY] = u[MY] * rhoi;
  w[VZ] = u[MZ] * rhoi;

  real v2 = SQR(w[VX]) + SQR(w[VY]) + SQR(w[VZ]);
  real b2 = 0.;
#if MFIELD
  b2 = SQR(u[BX]) + SQR(u[BY]) + SQR(u[BZ]);
#endif

  w[PR] = (gam-1.) * (u[EN] - 0.5*(b2 + v2*u[RHO]));

#if TWOTEMP
  w[PE] = POW(u[RHO], gam) * EXP(u[SE] * rhoi);
  w[PR] -= w[PE];
#endif

#if CGL
  real ppd_ppl = EXP(u[LA] * rhoi) * b2*SQRT(b2) * SQR(rhoi);
  w[PPD] = 3.*ppd_ppl / (1. + 2.*ppd_ppl) * w[PR];
#endif

  w[PSC] = u[PSC];

#if MFIELD
  w[BX] = u[BX];
  w[BY] = u[BY];
  w[BZ] = u[BZ];
#endif

}


// ----------------------------------------------------------------

void inline prim2cons(real *u, real *w, real gam) {

  real ppd_ppl, v2, b2;
  real gamm1i = 1./(gam-1.);

  u[RHO] = w[RHO];
  u[MX]  = w[RHO]*w[VX];
  u[MY]  = w[RHO]*w[VY];
  u[MZ]  = w[RHO]*w[VZ];

  v2 = SQR(w[VX]) + SQR(w[VY]) + SQR(w[VZ]);
  b2 = 0.;
#if MFIELD
  b2 = SQR(w[BX]) + SQR(w[BY]) + SQR(w[BZ]);
#endif
  u[EN] = 0.5 * (w[RHO] * v2 + b2) + w[PR] * gamm1i;

#if TWOTEMP
  u[SE] = w[RHO] * (LOG(w[PE]) - gam*LOG(w[RHO]));
  u[EN] += w[PE] * gamm1i;
#endif

#if CGL
  ppd_ppl = w[PPD] / (3.*w[PR] - 2.*w[PPD]);
  u[LA] = u[RHO] * LOG(ppd_ppl * SQR(u[RHO]) / (b2*SQRT(b2)));
#endif

  u[PSC] = w[PSC];

#if MFIELD
  u[BX] = w[BX];
  u[BY] = w[BY];
  u[BZ] = w[BZ];
#endif

}


// ----------------------------------------------------------------

void inline prim2char(real *vc, real *w, real gam) {

#if MFIELD

  real vc_rho = vc[RHO];
  real vc_vx  = vc[VX];
  real vc_vy  = vc[VY];
  real vc_vz  = vc[VZ];
  real vc_p   = vc[PR];
  real vc_by  = vc[BY];
  real vc_bz  = vc[BZ];

  real w_rho = w[RHO];
  real w_p   = w[PR];
  real w_bx  = w[BX];
  real w_by  = w[BY];
  real w_bz  = w[BZ];

  real rhoi = 1./w_rho;

  real s = SIGN(w_bx);

  real a2rho = gam * w_p;
  real a = SQRT(a2rho * rhoi);

  real bx2    = SQR(w_bx);
  real bperp2 = SQR(w_by) + SQR(w_bz);
  real b2 = bx2 + bperp2;

  real x = a2rho + b2;
  real y = SQRT(FABS( x * x - 4. * a2rho * bx2 ));
  real cf2rho = 0.5 * (x + y);
  real cs2rho = 0.5 * (x - y);
  real cf = SQRT(rhoi *      cf2rho);
  real cs = SQRT(rhoi * FABS(cs2rho));

  real z = 1. / (cf2rho - cs2rho);

  real alphaf,alphas;
  if (cf <= cs) {
    alphaf = 1.;
    alphas = 0.;
  }
  else if (a <= cs) {
    alphaf = 0.;
    alphas = 1.;
  }
  else if (cf <= a) {
    alphaf = 1.;
    alphas = 0.;
  }
  else {
    alphaf = SQRT(z * FABS(a2rho - cs2rho));
    alphas = SQRT(z * FABS(cf2rho - a2rho));
  }

  real nf = 0.5/(a2rho*rhoi);
  real qf_ = nf * cf * alphaf * s;
  real qs_ = nf * cs * alphas * s;

  real alphaf_ = nf * alphaf;
  real alphas_ = nf * alphas;

  real rhoisr = SQRT(rhoi);
  real arhoisr = a * rhoisr;
  real Af_ = alphaf_ * arhoisr;
  real As_ = alphas_ * arhoisr;

  x = s * rhoisr;

  real betay=0.;
  real betaz=0.;
  if (bperp2 != 0.) {
    real bperpi = 1./SQRT(bperp2);
    betay = bperpi * w_by;
    betaz = bperpi * w_bz;
  }


  vc[RHO] = ( alphaf_ * (rhoi * vc_p - cf * vc_vx)
          + qs_ * (betay * vc_vy + betaz * vc_vz)
          + As_ * (betay * vc_by + betaz * vc_bz) );

  vc[VX] = 0.5 * ( (-betaz * vc_vy + betay * vc_vz)
                + x * (-betaz * vc_by + betay * vc_bz) );

  vc[VY] =  ( alphas_ * (rhoi * vc_p - cs * vc_vx)
          - qf_ * (betay * vc_vy + betaz * vc_vz)
          - Af_ * (betay * vc_by + betaz * vc_bz) );   //!!! check signs

  vc[VZ] = vc_rho - 2.*nf * vc_p;

  vc[PR] = ( alphas_ * (rhoi * vc_p + cs * vc_vx)
          + qf_ * (betay * vc_vy + betaz * vc_vz)
          - Af_ * (betay * vc_by + betaz * vc_bz) );

  vc[BY] =  0.5 * ( (betaz * vc_vy - betay * vc_vz)
                + x * (-betaz * vc_by + betay * vc_bz) );

  vc[BZ] =  ( alphaf_ * (rhoi * vc_p + cf * vc_vx)
          - qs_ * (betay * vc_vy + betaz * vc_vz)
          + As_ * (betay * vc_by + betaz * vc_bz) );

#else

  real w_rho = w[RHO];
  real w_p = w[PR];

  real vc_rho = vc[RHO];
  real vc_vx  = vc[VX];
  real vc_vy  = vc[VY];
  real vc_vz  = vc[VZ];
  real vc_p   = vc[PR];

  real a2i = w_rho / (gam * w_p);
  real ai = SQRT(a2i);

  vc[RHO] = 0.5 * (a2i * vc_p - ai * w_rho * vc_vx);
  vc[VX]  = vc_rho - a2i * vc_p;
  vc[VY]  = vc_vy;
  vc[VZ]  = vc_vz;
  vc[PR]  = 0.5 * (a2i * vc_p + ai * w_rho * vc_vx);

#endif

}


// -----------------------------------------------------------------

void inline char2prim(real *vc, real *w, real gam) {

#if MFIELD

  real vc_rho = vc[RHO];
  real vc_vx  = vc[VX];
  real vc_vy  = vc[VY];
  real vc_vz  = vc[VZ];
  real vc_p   = vc[PR];
  real vc_by  = vc[BY];
  real vc_bz  = vc[BZ];

  real w_rho = w[RHO];
  real w_p   = w[PR];
  real w_bx  = w[BX];
  real w_by  = w[BY];
  real w_bz  = w[BZ];

  real rhoi = 1./w_rho;

  real s = SIGN(w_bx);

  real a2rho = gam * w_p;
  real a = SQRT(a2rho * rhoi);

  real bx2    = SQR(w_bx);
  real bperp2 = SQR(w_by) + SQR(w_bz);
  real b2 = bx2 + bperp2;

  real x = a2rho + b2;
  real y = SQRT(FABS( x * x - 4. * a2rho * bx2 ));
  real cf2rho = 0.5 * (x + y);
  real cs2rho = 0.5 * (x - y);
  real cf = SQRT(rhoi *      cf2rho);
  real cs = SQRT(rhoi * FABS(cs2rho));

  real z = 1. / (cf2rho - cs2rho);

  real alphaf,alphas;
  if (cf <= cs) {
    alphaf = 1.;
    alphas = 0.;
  }
  else if (a <= cs) {
    alphaf = 0.;
    alphas = 1.;
  }
  else if (cf <= a) {
    alphaf = 1.;
    alphas = 0.;
  }
  else {
    alphaf = SQRT(z * FABS(a2rho - cs2rho));
    alphas = SQRT(z * FABS(cf2rho - a2rho));
  }

  real qf = cf * alphaf * s;
  real qs = cs * alphas * s;

  real rhosr = SQRT(w_rho);
  real arhosr = a * rhosr;
  real Af = alphaf * arhosr;
  real As = alphas * arhosr;

  real betay=0.;
  real betaz=0.;
  if (bperp2 != 0.) {
    real bperpi = 1./SQRT(bperp2);
    betay = bperpi * w_by;
    betaz = bperpi * w_bz;
  }

  vc[RHO] = w_rho * (alphaf * (vc_rho + vc_bz) + alphas * (vc_vy + vc_p)) + vc_vz;

  vc[VX] = cf*alphaf * (vc_bz - vc_rho) + cs*alphas * (vc_p - vc_vy);

  vc[VY] = betay * (qs * (vc_rho - vc_bz) - qf * (vc_vy - vc_p)) - betaz * (vc_vx - vc_by);
  vc[VZ] = betaz * (qs * (vc_rho - vc_bz) - qf * (vc_vy - vc_p)) + betay * (vc_vx - vc_by);

  vc[PR] = a2rho * (alphaf * (vc_rho + vc_bz) + alphas * (vc_vy + vc_p));

  vc[BY] = betay * (As * (vc_rho + vc_bz) - Af * (vc_vy + vc_p)) - betaz*s*rhosr * (vc_vx + vc_by);
  vc[BZ] = betaz * (As * (vc_rho + vc_bz) - Af * (vc_vy + vc_p)) + betay*s*rhosr * (vc_vx + vc_by);

#else

  real vc_rho = vc[RHO];
  real vc_vx  = vc[VX];
  real vc_vy  = vc[VY];
  real vc_vz  = vc[VZ];
  real vc_p   = vc[PR];

  real rhoi = 1./w[RHO];

  real a2 = gam * rhoi * w[PR];
  real a = SQRT(a2);

  vc[RHO] = vc_rho + vc_vx + vc_p;
  vc[VX] = a * rhoi * (vc_p - vc_rho);
  vc[VY] = vc_vy;
  vc[VZ] = vc_vz;
  vc[PR] = a2 * (vc_rho + vc_p);

#endif

}


// ---------------------------------------------------

void inline prim2flux(real *f, real *w, real gam) {

  real v2, mx, en, h, se, loga;
  real bx2,bxby,bxbz;
  real ptot, pgas,pipd,pipl, ppd_ppl;

  real b2 = 0.;
#if MFIELD
  bx2  = SQR(w[BX]);
  b2   = bx2 + SQR(w[BY]) + SQR(w[BZ]);
  bxby = w[BX]*w[BY];
  bxbz = w[BX]*w[BZ];
#endif

  pgas = w[PR];
  ptot = 0.5*b2;

  real a = 1.;
#if !CGL
  ptot += pgas;
#else
  pipd = w[PPD];
  pipl = 3.*pgas - 2.*pipd;
  a += (pipd - pipl) / b2;
  ppd_ppl = w[PPD] / (3.*w[PR] - 2.*w[PPD]);

  ptot += pipd;
  loga = w[RHO] * LOG(ppd_ppl * SQR(w[RHO]) / (b2*SQRT(b2)));
#endif

#if TWOTEMP
  pgas += w[PE];
  ptot += w[PE];
  se = w[RHO] * (LOG(w[PE]) - gam * LOG(w[RHO]));
#endif

  v2 = SQR(w[VX]) + SQR(w[VY]) + SQR(w[VZ]);

  //total energy density
  en = 0.5 * (w[RHO] * v2 + b2) + pgas/(gam-1.);
  // enthalpy density
  h = en + ptot;

  mx = w[RHO] * w[VX];

  f[RHO] = mx;
  f[MX]  = mx * w[VX] + ptot;
  f[MY]  = mx * w[VY];
  f[MZ]  = mx * w[VZ];
  f[EN]  = w[VX] * h;
  f[PSC] = w[VX] * w[PSC];
#if TWOTEMP
  f[SE] = w[VX] * se;
#endif
#if CGL
  f[LA] = w[VX] * loga;
#endif

#if MFIELD

  f[MX] -= a * bx2;
  f[MY] -= a * bxby;
  f[MZ] -= a * bxbz;
  f[EN] -= a * (w[VX]*bx2 + w[VY]*bxby + w[VZ]*bxbz);

  f[BX] = 0.;
  f[BY] = w[VX] * w[BY] - w[VY] * w[BX];
  f[BZ] = w[VX] * w[BZ] - w[VZ] * w[BX];

#endif

}


// -------------------------------------------------------------

void inline primcons2flux(real *f, real *w,real *u, real gam) {

  real bx2,bxby,bxbz;
  real pipd,pipl;

  real b2 = 0.;
  real ptot = 0.;
#if MFIELD
  bx2 = SQR(w[BX]);
  b2 = bx2 + SQR(w[BY]) + SQR(w[BZ]);
  bxby = w[BX] * w[BY];
  bxbz = w[BX] * w[BZ];
  ptot += 0.5*b2;
#endif

  real a = 1.;
#if CGL
  pipd = w[PPD];
  pipl = 3.*w[PR] - 2.*pipd;
  a += (pipd - pipl) / b2;
  ptot += pipd;
# else
  ptot += w[PR];
#endif

#if TWOTEMP
  ptot += w[PE];
#endif

  for (int n=0; n<BX; ++n)
    f[n] = u[n]*w[VX];

  f[MX] += ptot;
  f[EN] += ptot*w[VX];

#if MFIELD
  f[MX] -= a * bx2;
  f[MY] -= a * bxby;
  f[MZ] -= a * bxbz;
  f[EN] -= a * (w[VX]*bx2 + w[VY]*bxby + w[VZ]*bxbz);
  f[BX] = 0.;
  f[BY] = w[VX] * w[BY] - w[VY] * w[BX];
  f[BZ] = w[VX] * w[BZ] - w[VZ] * w[BX];
#endif

}


// ---------------------------------------------------

// Fast magnetosonic speed.

real inline fms(real *w, real bx, real gam) {

  real b2, bx2;
  real p, gamp;
  real bux2, pipd,pipl, gampe, c1,c2, apl2,apd2;

#if MFIELD
  bx2 = SQR(w[BX]);
  b2 = bx2 + SQR(w[BY]) + SQR(w[BZ]);
#endif

// total gas pressure
#if !CGL

p = w[PR];
#if TWOTEMP
p += w[PE];
#endif
gamp = gam*p;

#endif

//----------------------------------------

#if CGL

  bux2 = bx2 / b2;
  pipd = w[PPD];
  pipl = 3.*w[PR] - 2.*pipd;

  c1 = b2 + 2.*pipd + (2.*pipl-pipd) * bux2;
  apl2 = 3.*pipl;
  apd2 = pipd;

#if TWOTEMP
  gampe = gam * w[PE];
  c1 += gampe;
  apl2 += gampe;
  apd2 += gampe;
#endif

  c2 = bux2 * (apl2 * (apl2 * bux2 - c1) + apd2 * apd2 * (1.-bux2));

  return SQRT(FABS( 0.5/w[RHO] * (c1 + SQRT(FABS( c1*c1 + 4.*c2 )) ) ));

//---------------------------------------

#elif MFIELD

  c1 = gamp + b2;
  c2 = SQRT(FABS( c1*c1 - 4. * gamp * bx2 ));

  return SQRT(0.5/w[RHO] * (c1 + c2));

//---------------------------------------

#else

  return SQRT(gamp / w[RHO]);

#endif

}
