#include "../defs.h"

extern void prim2cons(real *u, real *w, real gam) __attribute__((always_inline));
extern void prim2flux(real *f, real *w, real gam) __attribute__((always_inline));
extern void primcons2flux(real *f, real *w, real *u, real gam) __attribute__((always_inline));
extern real fms(real *w, real Bx, real gam) __attribute__((always_inline));

extern void cons2prim_1(real **W1, real **U1, int i1, int i2, real gam);
extern void prim2cons_1(real **U1, real **W1, int i1, int i2, real gam);
extern void prim2char_1(real **vc, real **W1, int i1, int i2, real gam);
extern void char2prim_1(real **vc, real **W1, int i1, int i2, real gam);


// ======================================================

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


// =============================================================

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



// ================================================================

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
#if !CGL
  ptot += w[PR];
#else
  pipd = w[PPD];
  pipl = 3.*w[PR] - 2.*pipd;
  a += (pipd - pipl) / b2;
  ptot += pipd;
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


// ============================================================

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

//------------------------------------------------------------------------
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

//-----------------------------------------------------------------------
#elif MFIELD

  c1 = gamp + b2;
  c2 = SQRT(FABS( c1*c1 - 4. * gamp * bx2 ));

  return SQRT(0.5/w[RHO] * (c1 + c2));

//-----------------------------------------------------------------------
#else

  return SQRT(gamp / w[RHO]);

#endif

}
