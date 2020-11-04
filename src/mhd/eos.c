#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "eos.h"

// =============================================================================

void cons2prim_1(real **W1, real **U1, ints i1, ints i2, real gam) {

  real w[NWAVES];
  real u[NWAVES];
  real rhoi, ppd, ppd_ppl, v2, b2=0.;

#pragma omp simd private(u, w)
  for (ints i=i1; i<=i2; ++i) {

    for (ints n=0; n<NWAVES; ++n)
      u[n] = U1[n][i];

    //-----------------------

    w[RHO] = u[RHO];
    rhoi = 1./u[RHO];
    w[VX] = u[MX] * rhoi;
    w[VY] = u[MY] * rhoi;
    w[VZ] = u[MZ] * rhoi;

    v2 = SQR(w[VX]) + SQR(w[VY]) + SQR(w[VZ]);
#if MFIELD
    b2 = SQR(u[BX]) + SQR(u[BY]) + SQR(u[BZ]);
#endif

    w[PR] = (gam-1.) * (u[EN] - 0.5*(b2 + v2*u[RHO]));

#if TWOTEMP
    w[PE] = POW(u[RHO], gam) * EXP(u[SE] * rhoi);
    w[PR] = w[PR] - w[PE];
#endif

#if CGL
    ppd_ppl = EXP(u[LA] * rhoi) * b2*SQRT(b2) * SQR(rhoi);
    w[PPD] = 3.*ppd_ppl / (1. + 2.*ppd_ppl) * w[PR];
#endif

    w[PSC] = u[PSC];   // passive scalar

#if MFIELD
    w[BX] = u[BX];
    w[BY] = u[BY];
    w[BZ] = u[BZ];
#endif

    //-----------------------

    for (ints n=0; n<NWAVES; ++n)
      W1[n][i] = w[n];

  }

}


// ================================================================================

void prim2cons_1(real **U1, real **W1, ints i1, ints i2, real gam) {

  real w[NWAVES];
  real u[NWAVES];
  real ppd_ppl, v2, b2=0.;
  real gamm1i = 1./(gam-1);

#pragma omp simd private(u, w)
  for (ints i=i1; i<=i2; ++i) {

    for (ints n=0; n<NWAVES; ++n)
      w[n] = W1[n][i];

    //-----------------------

    u[RHO] = w[RHO];
    u[MX]  = w[RHO]*w[VX];
    u[MY]  = w[RHO]*w[VY];
    u[MZ]  = w[RHO]*w[VZ];

    v2 = SQR(w[VX]) + SQR(w[VY]) + SQR(w[VZ]);
#if MFIELD
    b2 = SQR(w[BX]) + SQR(w[BY]) + SQR(w[BZ]);
#endif
    u[EN] = 0.5 * (w[RHO] * v2 + b2) + w[PR] * gamm1i;

#if TWOTEMP
    u[SE] = w[RHO] * (LOG(w[PE]) - gam*LOG(w[RHO]));
    u[EN] = u[EN] + w[PE] * gamm1i;
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

    //-----------------------

    for (ints n=0; n<NWAVES; ++n)
      U1[n][i] = u[n];

  }

}


// ==============================================================================

void prim2char_1(real **vc, real **W1,
                 ints i1, ints i2, real gam) {

#if MFIELD

  real rhoi,rhoisr, s, a2rho,a, arhoisr, bx2, bperp2,bperpi, b2;
  real x,y,z, cf2rho,cs2rho, cf,cs;
  real alphaf, alphas, alphaf_,alphas_, qf_,qs_, Af_,As_, betay,betaz, nf;

  real vc_rho, vc_vx, vc_vy, vc_vz, vc_p, vc_by, vc_bz;
  real w_rho,  w_p,  w_bx,  w_by, w_bz;

#pragma omp simd simdlen(SIMD_WIDTH)
  for (ints i=i1; i<=i2; ++i) {

    vc_rho = vc[RHO][i];
    vc_vx  = vc[VX][i];
    vc_vy  = vc[VY][i];
    vc_vz  = vc[VZ][i];
    vc_p   = vc[PR][i];
    vc_by  = vc[BY][i];
    vc_bz  = vc[BZ][i];

    w_rho = W1[RHO][i];
    w_p   = W1[PR][i];
    w_bx  = W1[BX][i];
    w_by  = W1[BY][i];
    w_bz  = W1[BZ][i];

    rhoi = 1./w_rho;

    s = FSIGN(w_bx);

    a2rho = gam * w_p;
    a = SQRT(a2rho * rhoi);

    bx2    = SQR(w_bx);
    bperp2 = SQR(w_by) + SQR(w_bz);
    b2 = bx2 + bperp2;

    x = a2rho + b2;
    y = SQRT(FABS( x * x - 4. * a2rho * bx2 ));
    cf2rho = 0.5 * (x + y);
    cs2rho = 0.5 * (x - y);
    cf = SQRT(rhoi *      cf2rho);
    cs = SQRT(rhoi * FABS(cs2rho));

    z = 1. / (cf2rho - cs2rho);

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

    nf = 0.5/(a2rho*rhoi);
    qf_ = nf * cf * alphaf * s;
    qs_ = nf * cs * alphas * s;

    alphaf_ = nf * alphaf;
    alphas_ = nf * alphas;

    rhoisr = SQRT(rhoi);
    arhoisr = a * rhoisr;
    Af_ = alphaf_ * arhoisr;
    As_ = alphas_ * arhoisr;

    x = s * rhoisr;

    betay=0.;
    betaz=0.;
    if (bperp2 != 0.) {
      bperpi = 1./SQRT(bperp2);
      betay = bperpi * w_by;
      betaz = bperpi * w_bz;
    }


    vc[RHO][i] = ( alphaf_ * (rhoi * vc_p - cf * vc_vx)
            + qs_ * (betay * vc_vy + betaz * vc_vz)
            + As_ * (betay * vc_by + betaz * vc_bz) );

    vc[VX][i] = 0.5 * ( (-betaz * vc_vy + betay * vc_vz)
                  + x * (-betaz * vc_by + betay * vc_bz) );

    vc[VY][i] =  ( alphas_ * (rhoi * vc_p - cs * vc_vx)
            - qf_ * (betay * vc_vy + betaz * vc_vz)
            - Af_ * (betay * vc_by + betaz * vc_bz) );   //!!! check signs

    vc[VZ][i] = vc_rho - 2.*nf * vc_p;

    vc[PR][i] = ( alphas_ * (rhoi * vc_p + cs * vc_vx)
            + qf_ * (betay * vc_vy + betaz * vc_vz)
            - Af_ * (betay * vc_by + betaz * vc_bz) );

    vc[BY][i] =  0.5 * ( (betaz * vc_vy - betay * vc_vz)
                  + x * (-betaz * vc_by + betay * vc_bz) );

    vc[BZ][i] =  ( alphaf_ * (rhoi * vc_p + cf * vc_vx)
            - qs_ * (betay * vc_vy + betaz * vc_vz)
            + As_ * (betay * vc_by + betaz * vc_bz) );

    }


#else

  real a2i, ai;
  real vc_rho, vc_vx,vc_vy,vc_vz, vc_p;
  real w_rho, w_p;

#pragma omp simd simdlen(SIMD_WIDTH)
  for (ints i=i1; i<=i2; ++i) {

    w_rho = W1[RHO][i];
    w_p = W1[PR][i];

    vc_rho = vc[RHO][i];
    vc_vx  = vc[VX][i];
    vc_vy  = vc[VY][i];
    vc_vz  = vc[VZ][i];
    vc_p   = vc[PR][i];

    a2i = w_rho / (gam * w_p);
    ai = SQRT(a2i);

    vc[RHO][i] = 0.5 * (a2i * vc_p - ai * w_rho * vc_vx);
    vc[VX][i]  = vc_rho - a2i * vc_p;
    vc[VY][i]  = vc_vy;
    vc[VZ][i]  = vc_vz;
    vc[PR][i]  = 0.5 * (a2i * vc_p + ai * w_rho * vc_vx);

  }

#endif

}


// ============================================================================

void char2prim_1(real **vc, real **W1,
                 ints i1, ints i2, real gam) {

#if MFIELD

  real rhoi,rhosr, s, a2rho,a, arhosr, bx2, bperp2,bperpi, b2;
  real x,y,z, cf2rho,cs2rho, cf,cs, alphaf,alphas, qf,qs, Af,As, betay,betaz;

  real vc_rho, vc_vx, vc_vy, vc_vz, vc_p, vc_by, vc_bz;
  real w_rho,  w_p,  w_bx,  w_by, w_bz;

#pragma omp simd simdlen(SIMD_WIDTH)
  for (ints i=i1; i<=i2; ++i) {

    vc_rho = vc[RHO][i];
    vc_vx  = vc[VX][i];
    vc_vy  = vc[VY][i];
    vc_vz  = vc[VZ][i];
    vc_p   = vc[PR][i];
    vc_by  = vc[BY][i];
    vc_bz  = vc[BZ][i];

    w_rho = W1[RHO][i];
    w_p   = W1[PR][i];
    w_bx  = W1[BX][i];
    w_by  = W1[BY][i];
    w_bz  = W1[BZ][i];

    rhoi = 1./w_rho;

    s = FSIGN(w_bx);

    a2rho = gam * w_p;
    a = SQRT(a2rho * rhoi);

    bx2    = SQR(w_bx);
    bperp2 = SQR(w_by) + SQR(w_bz);
    b2 = bx2 + bperp2;

    x = a2rho + b2;
    y = SQRT(FABS( x * x - 4. * a2rho * bx2 ));
    cf2rho = 0.5 * (x + y);
    cs2rho = 0.5 * (x - y);
    cf = SQRT(rhoi *      cf2rho);
    cs = SQRT(rhoi * FABS(cs2rho));

    z = 1. / (cf2rho - cs2rho);

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

    qf = cf * alphaf * s;
    qs = cs * alphas * s;

    rhosr = SQRT(w_rho);
    arhosr = a * rhosr;
    Af = alphaf * arhosr;
    As = alphas * arhosr;

    betay=0.;
    betaz=0.;
    if (bperp2 != 0.) {
      bperpi = 1./SQRT(bperp2);
      betay = bperpi * w_by;
      betaz = bperpi * w_bz;
    }


    vc[RHO][i] = w_rho * (alphaf * (vc_rho + vc_bz) + alphas * (vc_vy + vc_p)) + vc_vz;

    vc[VX][i]  = cf*alphaf * (vc_bz - vc_rho) + cs*alphas * (vc_p - vc_vy);

    vc[VY][i]  = betay * (qs * (vc_rho - vc_bz) - qf * (vc_vy - vc_p)) - betaz * (vc_vx - vc_by);
    vc[VZ][i]  = betaz * (qs * (vc_rho - vc_bz) - qf * (vc_vy - vc_p)) + betay * (vc_vx - vc_by);

    vc[PR][i] = a2rho * (alphaf * (vc_rho + vc_bz) + alphas * (vc_vy + vc_p));

    vc[BY][i]  = betay * (As * (vc_rho + vc_bz) - Af * (vc_vy + vc_p)) - betaz*s*rhosr * (vc_vx + vc_by);
    vc[BZ][i]  = betaz * (As * (vc_rho + vc_bz) - Af * (vc_vy + vc_p)) + betay*s*rhosr * (vc_vx + vc_by);

  }


#else

  real a, a2, rhoi;
  real vc_rho, vc_vx,vc_vy,vc_vz, vc_p;

#pragma omp simd simdlen(SIMD_WIDTH)
  for (ints i=i1; i<=i2; ++i) {

    vc_rho = vc[RHO][i];
    vc_vx  = vc[VX][i];
    vc_vy  = vc[VY][i];
    vc_vz  = vc[VZ][i];
    vc_p   = vc[PR][i];

    rhoi = 1./W1[RHO][i];

    a2 = gam * rhoi * W1[PR][i];
    a = SQRT(a2);

    vc[RHO][i] = vc_rho + vc_vx + vc_p;
    vc[VX][i] = a * rhoi * (vc_p - vc_rho);
    vc[VY][i] = vc_vy;
    vc[VZ][i] = vc_vz;
    vc[PR][i] = a2 * (vc_rho + vc_p);

  }

#endif

}
