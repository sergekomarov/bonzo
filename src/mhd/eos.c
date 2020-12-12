#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "eos.h"

// =============================================================================

void cons2prim_1(real **w1, real **u1, int i1, int i2, real gam) {

  real w[NMODE];
  real u[NMODE];

#pragma omp simd private(u, w)
  for (int i=i1; i<=i2; ++i) {

    for (int n=0; n<NMODE; ++n)
      u[n] = u1[n][i];

    //-----------------------

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

    w[PSC] = u[PSC];   // passive scalar

#if MFIELD
    w[BX] = u[BX];
    w[BY] = u[BY];
    w[BZ] = u[BZ];
#endif

    //-----------------------

    for (int n=0; n<NMODE; ++n)
      w1[n][i] = w[n];

  }

}


// ================================================================================

void prim2cons_1(real **u1, real **w1, int i1, int i2, real gam) {

  real w[NMODE];
  real u[NMODE];
  real gamm1i = 1./(gam-1.);

#pragma omp simd private(u, w)
  for (int i=i1; i<=i2; ++i) {

    for (int n=0; n<NMODE; ++n)
      w[n] = w1[n][i];

    //-----------------------

    u[RHO] = w[RHO];
    u[MX]  = w[RHO]*w[VX];
    u[MY]  = w[RHO]*w[VY];
    u[MZ]  = w[RHO]*w[VZ];

    real v2 = SQR(w[VX]) + SQR(w[VY]) + SQR(w[VZ]);
    real b2 = 0.;
#if MFIELD
    b2 = SQR(w[BX]) + SQR(w[BY]) + SQR(w[BZ]);
#endif
    u[EN] = 0.5 * (w[RHO] * v2 + b2) + w[PR] * gamm1i;

#if TWOTEMP
    u[SE] = w[RHO] * (LOG(w[PE]) - gam*LOG(w[RHO]));
    u[EN] += w[PE] * gamm1i;
#endif

#if CGL
    real ppd_ppl = w[PPD] / (3.*w[PR] - 2.*w[PPD]);
    u[LA] = u[RHO] * LOG(ppd_ppl * SQR(u[RHO]) / (b2*SQRT(b2)));
#endif

    u[PSC] = w[PSC];

#if MFIELD
    u[BX] = w[BX];
    u[BY] = w[BY];
    u[BZ] = w[BZ];
#endif

    //-----------------------

    for (int n=0; n<NMODE; ++n)
      u1[n][i] = u[n];

  }

}


// ==============================================================================

void prim2char_1(real **vc, real **w1,
                 int i1, int i2, real gam) {

#if MFIELD

#pragma omp simd simdlen(SIMD_WIDTH)
  for (int i=i1; i<=i2; ++i) {

    real vc_rho = vc[RHO][i];
    real vc_vx  = vc[VX][i];
    real vc_vy  = vc[VY][i];
    real vc_vz  = vc[VZ][i];
    real vc_p   = vc[PR][i];
    real vc_by  = vc[BY][i];
    real vc_bz  = vc[BZ][i];

    real w_rho = w1[RHO][i];
    real w_p   = w1[PR][i];
    real w_bx  = w1[BX][i];
    real w_by  = w1[BY][i];
    real w_bz  = w1[BZ][i];

    real rhoi = 1./w_rho;

    real s = FSIGN(w_bx);

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

#pragma omp simd simdlen(SIMD_WIDTH)
  for (int i=i1; i<=i2; ++i) {

    real w_rho = w1[RHO][i];
    real w_p = w1[PR][i];

    real vc_rho = vc[RHO][i];
    real vc_vx  = vc[VX][i];
    real vc_vy  = vc[VY][i];
    real vc_vz  = vc[VZ][i];
    real vc_p   = vc[PR][i];

    real a2i = w_rho / (gam * w_p);
    real ai = SQRT(a2i);

    vc[RHO][i] = 0.5 * (a2i * vc_p - ai * w_rho * vc_vx);
    vc[VX][i]  = vc_rho - a2i * vc_p;
    vc[VY][i]  = vc_vy;
    vc[VZ][i]  = vc_vz;
    vc[PR][i]  = 0.5 * (a2i * vc_p + ai * w_rho * vc_vx);

  }

#endif

}


// ============================================================================

void char2prim_1(real **vc, real **w1,
                 int i1, int i2, real gam) {

#if MFIELD

#pragma omp simd simdlen(SIMD_WIDTH)
  for (int i=i1; i<=i2; ++i) {

    real vc_rho = vc[RHO][i];
    real vc_vx  = vc[VX][i];
    real vc_vy  = vc[VY][i];
    real vc_vz  = vc[VZ][i];
    real vc_p   = vc[PR][i];
    real vc_by  = vc[BY][i];
    real vc_bz  = vc[BZ][i];

    real w_rho = w1[RHO][i];
    real w_p   = w1[PR][i];
    real w_bx  = w1[BX][i];
    real w_by  = w1[BY][i];
    real w_bz  = w1[BZ][i];

    real rhoi = 1./w_rho;

    real s = FSIGN(w_bx);

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

    vc[RHO][i] = w_rho * (alphaf * (vc_rho + vc_bz) + alphas * (vc_vy + vc_p)) + vc_vz;

    vc[VX][i]  = cf*alphaf * (vc_bz - vc_rho) + cs*alphas * (vc_p - vc_vy);

    vc[VY][i]  = betay * (qs * (vc_rho - vc_bz) - qf * (vc_vy - vc_p)) - betaz * (vc_vx - vc_by);
    vc[VZ][i]  = betaz * (qs * (vc_rho - vc_bz) - qf * (vc_vy - vc_p)) + betay * (vc_vx - vc_by);

    vc[PR][i] = a2rho * (alphaf * (vc_rho + vc_bz) + alphas * (vc_vy + vc_p));

    vc[BY][i]  = betay * (As * (vc_rho + vc_bz) - Af * (vc_vy + vc_p)) - betaz*s*rhosr * (vc_vx + vc_by);
    vc[BZ][i]  = betaz * (As * (vc_rho + vc_bz) - Af * (vc_vy + vc_p)) + betay*s*rhosr * (vc_vx + vc_by);

  }


#else

#pragma omp simd simdlen(SIMD_WIDTH)
  for (int i=i1; i<=i2; ++i) {

    real vc_rho = vc[RHO][i];
    real vc_vx  = vc[VX][i];
    real vc_vy  = vc[VY][i];
    real vc_vz  = vc[VZ][i];
    real vc_p   = vc[PR][i];

    real rhoi = 1./w1[RHO][i];

    real a2 = gam * rhoi * w1[PR][i];
    real a = SQRT(a2);

    vc[RHO][i] = vc_rho + vc_vx + vc_p;
    vc[VX][i] = a * rhoi * (vc_p - vc_rho);
    vc[VY][i] = vc_vy;
    vc[VZ][i] = vc_vz;
    vc[PR][i] = a2 * (vc_rho + vc_p);

  }

#endif

}
