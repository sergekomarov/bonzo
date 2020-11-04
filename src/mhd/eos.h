#include "../defs.h"

extern void prim2flux(real *f, real *w, real gam) __attribute__((always_inline));
extern void primcons2flux(real *f, real *w, real *u, real gam) __attribute__((always_inline));
extern real fms(real *w, real Bx, real gam) __attribute__((always_inline));

extern void cons2prim_1(real **W1, real **U1, ints i1, ints i2, real gam);
extern void prim2cons_1(real **U1, real **W1, ints i1, ints i2, real gam);
extern void prim2char_1(real **vc, real **W1, ints i1, ints i2, real gam);
extern void char2prim_1(real **vc, real **W1, ints i1, ints i2, real gam);


// =============================================================

void inline prim2flux(real *f, real *w, real gam) {

  real v2h,h;
  real b2=0., b2h=0., vb=0.;
  real a, ptot, pgas,pipd,pipl;
  real se, mx, ppd_ppl, loga;


  // calculate v2h, b2h, and h (enthalpy density) to use later

  v2h = 0.5*(SQR(w[VX]) + SQR(w[VY]) + SQR(w[VZ]));
#if MFIELD
  b2h = 0.5*(SQR(w[BX]) + SQR(w[BY]) + SQR(w[BZ]));
#endif

  // contribution of anisotropic pressure

#if CGL

  pgas = w[PR];
  pipd = w[PPD];
  pipl = 3.*pgas - 2.*pipd;

  b2 = 2.*b2h;
  a = 1. + (pipd - pipl) / b2;
  ppd_ppl = w[PPD] / (3.*w[PR] - 2.*w[PPD]);
  loga = w[RHO] * LOG(ppd_ppl * SQR(w[RHO]) / (b2*SQRT(b2)));
  ptot = pipd + b2h;

#else

  a = 1.;
  pgas = w[PR];
  ptot = pgas + b2h;

#endif

  //-----------------------
  // A = 1.
  //-----------------------

#if TWOTEMP
  ptot = ptot + w[PE];
  pgas = pgas + w[PE];
  se = w[RHO] * (LOG(w[PE]) - gam * LOG(w[RHO]));
#endif

  h = v2h + (b2h + pgas/(gam-1) + ptot) / w[RHO];
  mx = w[RHO] * w[VX];
#if MFIELD
  vb = w[VX]*w[BX] + w[VY]*w[BY] + w[VZ]*w[BZ];
#endif

  // calculate fluxes

  f[RHO] = mx;

  f[MX]  = mx * w[VX] + ptot;
#if MFIELD
  f[MX] = f[MX] - a * SQR(w[BX]);
#endif

  f[MY]  = mx * w[VY];
#if MFIELD
  f[MY] = f[MY] - a * w[BX] * w[BY];
#endif

  f[MZ]  = mx * w[VZ];
#if MFIELD
  f[MZ] = f[MZ] - a * w[BX] * w[BZ];
#endif

  f[EN] = mx * h;
#if MFIELD
  f[EN] = f[EN] - a * w[BX] * vb;
#endif

#if CGL
  f[LA] = loga * w[VX];
#endif

#if TWOTEMP
  f[SE] = se * w[VX];
#endif

  f[PSC] = w[PSC] * w[VX];

#if MFIELD
  f[BX] = 0.;
  f[BY] = w[VX] * w[BY] - w[VY] * w[BX];
  f[BZ] = w[VX] * w[BZ] - w[VZ] * w[BX];
#endif

}



// ================================================================

void inline primcons2flux(real *f, real *w,real *u, real gam) {

  real v2h,h;
  real b2=0., b2h=0., vb=0.;
  real a, ptot, pgas,pipd,pipl;
  real se, mx, ppd_ppl, loga;


  // calculate v2h, b2h, and h (enthalpy density) to use later

  v2h = 0.5*(SQR(w[VX]) + SQR(w[VY]) + SQR(w[VZ]));
#if MFIELD
  b2h = 0.5*(SQR(w[BX]) + SQR(w[BY]) + SQR(w[BZ]));
#endif

  // contribution of anisotropic pressure

#if CGL

  pgas = w[PR];
  pipd = w[PPD];
  pipl = 3.*pgas - 2.*pipd;

  b2 = 2.*b2h;
  a = 1. + (pipd - pipl) / b2;
  ptot = pipd + b2h;

#else

  a = 1.;
  pgas = w[PR];
  ptot = pgas + b2h;

#endif

  //-----------------------
  // A = 1.
  //-----------------------

#if TWOTEMP
  ptot = ptot + w[PE];
  pgas = pgas + w[PE];
#endif

  h = v2h + (b2h + pgas/(gam-1) + ptot) / w[RHO];
  mx = w[RHO] * w[VX];
#if MFIELD
  vb = w[VX]*w[BX] + w[VY]*w[BY] + w[VZ]*w[BZ];
#endif

  // calculate fluxes

  f[RHO] = mx;

  f[MX]  = mx * w[VX] + ptot;
#if MFIELD
  f[MX] = f[MX] - a * SQR(w[BX]);
#endif

  f[MY]  = mx * w[VY];
#if MFIELD
  f[MY] = f[MY] - a * w[BX] * w[BY];
#endif

  f[MZ]  = mx * w[VZ];
#if MFIELD
  f[MZ] = f[MZ] - a * w[BX] * w[BZ];
#endif

  f[EN] = mx * h;
#if MFIELD
  f[EN] = f[EN] - a * w[BX] * vb;
#endif

#if CGL
  f[LA] = u[LA] * w[VX];
#endif

#if TWOTEMP
  f[SE] = u[SE] * w[VX];
#endif

  f[PSC] = w[PSC] * w[VX];

#if MFIELD
  f[BX] = 0.;
  f[BY] = w[VX] * w[BY] - w[VY] * w[BX];
  f[BZ] = w[VX] * w[BZ] - w[VZ] * w[BX];
#endif

}


// ============================================================

// Fast magnetosonic speed.

real inline fms(real *w, real Bx, real gam) {

  real B2=0., bx2, pipd,pipl, gampe, C1,C2, apl2,apd2;
  real p, gamp, x,y;

#if MFIELD
  B2 = SQR(w[BX]) + SQR(w[BY]) + SQR(w[BZ]);
#endif

#if CGL

  bx2 = Bx*Bx / B2;
  pipd = w[PPD];
  pipl = 3.*w[PR] - 2.*pipd;

  C1 = B2 + 2.*pipd + (2.*pipl-pipd) * bx2;
  apl2 = 3.*pipl;
  apd2 = pipd;

#if TWOTEMP
  gampe = gam * w[PE];
  C1 = C1 + gampe;
  apl2 = apl2 + gampe;
  apd2 = apd2 + gampe;
#endif

  C2 = bx2 * (apl2 * (apl2 * bx2 - C1) + apd2 * apd2 * (1.-bx2));

  return SQRT(FABS( 0.5/w[RHO] * (C1 + SQRT(FABS( C1*C1 + 4.*C2 )) ) ));


#elif MFIELD

  p = w[PR];
#if TWOTEMP
  p = p + w[PE];
#endif

  gamp = gam * p;
  x = gamp + B2;
  y = SQRT(FABS( x * x - 4. * gamp * Bx*Bx ));

  return SQRT(0.5/w[RHO] * (x + y));


#else

  p = w[PR];
#if TWOTEMP
  p = p + w[PE];
#endif

  return SQRT(gam * p / w[RHO]);

#endif

}
