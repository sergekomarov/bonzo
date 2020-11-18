#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "eos.h"
#include "fluxes.h"


// private functions

static void sLsR_simple(real *sL, real *sR,
                        real *wL, real *wR,
                        real gam) __attribute__((always_inline));

static void sLsR_einfeldt(real *sL, real *sR,
                          real *wL, real *wR,
                          real gam) __attribute__((always_inline));

static void sLsR_pressure(real *sL, real *sR,
                          real *wL, real *wR,
                          real gam) __attribute__((always_inline));


// =================================================

// Simple estimate of left/right maximum wave speed.

void inline sLsR_simple(real *sL, real *sR,
                        real *wL, real *wR,
                        real gam)
{

  real cfL, cfR, cf_max, bx=0.;

#if MFIELD
  bx = wL[BX];
#endif

  cfL = fms(wL, bx, gam);
  cfR = fms(wR, bx, gam);
  cf_max = FMAX(cfL, cfR);
  *sL = FMIN(wL[VX], wR[VX]) - cf_max;
  *sR = FMAX(wL[VX], wR[VX]) + cf_max;

}


// ====================================================

// Einfeldt's estimate of maximum propagation speed.

void inline sLsR_einfeldt(real *sL, real *sR,
                          real *wL, real *wR,
                          real gam)
{

  real cfL, cfR;
  real rhoL,rhoR, pL,pR;
  real vxL,vxR, vyL,vyR, vzL,vzR;
  real byL,byR, bzL,bzR;
  real rhoLsr, rhoRsr, rhoLRi, rL,rR, rhot;
  real vxt,vyt,vzt, vt2, byt,bzt,ht;
  real caxt2, cat2, x, cst2, csat2, cft;
  real hL,hR, v2hL,v2hR;
  real bx=0., b2L=0., b2R=0.;
  real gam_gamm1 = gam/(gam-1);


  rhoL = wL[RHO]; rhoR = wR[RHO];
  vxL  = wL[VX];   vxR = wR[VX];
  vyL  = wL[VY];   vyR = wR[VY];
  vzL  = wL[VZ];   vzR = wR[VZ];
  pL   = wL[PR];    pR = wR[PR];
#if MFIELD
  bx  = wL[BX];
  byL = wL[BY]; byR = wR[BY];
  bzL = wL[BZ]; bzR = wR[BZ];
#endif

  rhoLsr = SQRT(rhoL);
  rhoRsr = SQRT(rhoR);
  rhoLRi = 1./(rhoLsr + rhoRsr);
  rL = rhoLsr * rhoLRi;
  rR = rhoRsr * rhoLRi;

  v2hL = 0.5*(SQR(vxL) + SQR(vyL) + SQR(vzL));
  v2hR = 0.5*(SQR(vxR) + SQR(vyR) + SQR(vzR));

#if MFIELD
  b2L = SQR(bx) + SQR(byL) + SQR(bzL);
  b2R = SQR(bx) + SQR(byR) + SQR(bzR);
#endif

  hL = v2hL + (b2L + pL * gam_gamm1) / rhoL;
  hR = v2hR + (b2R + pR * gam_gamm1) / rhoR;

  rhot = rhoLsr * rhoRsr;
  vxt = rL * vxL + rR * vxR;
  vyt = rL * vyL + rR * vyR;
  vzt = rL * vzL + rR * vzR;
  vt2 = SQR(vxt) + SQR(vyt) + SQR(vzt);
  ht = rL * hL + rR * hR;

#if MFIELD

  byt = rR * byL + rL * byR;
  bzt = rR * bzL + rL * bzR;

  caxt2 = SQR(bx) / rhot;
  cat2 = (SQR(bx) + SQR(byt) + SQR(bzt)) / rhot;
  x = 0.5 * (SQR(bzR-bzL) + SQR(byR-byL)) * SQR(rhoLRi);

  cst2 = (gam-1.) * (ht - 0.5*vt2 - cat2) - (gam-2.)*x;
  csat2 = cst2 + cat2;
  cft = SQRT(0.5 * (csat2 + SQRT(SQR(csat2) - 4.*SQR(caxt2) * SQR(cst2))));

#else
  cft = SQRT((gam-1) * (ht - 0.5*vt2));
#endif

  cfL = fms(wL, bx, gam);
  cfR = fms(wR, bx, gam);

  *sL = FMIN(vxL - cfL, vxt - cft);
  *sR = FMAX(vxR + cfR, vxt + cft);


  // eta = 0.5*rhoLsr*rhoRsr * SQR(rhoLRi);
  // d = SQRT( (rhoLsr * SQR(aL) + rhoRsr * SQR(aR)) * rhoLRi + eta * SQR(vxR-vxL) );
  //
  // *sL = vxt-d;
  // *sR = vxt+d;

}


// =================================================

// Toro's estimate of left/right maximum wave speed.

void inline sLsR_pressure(real *sL, real *sR,
                          real *wL, real *wR,
                          real gam)
{

  real rhoL,rhoR, vxL,vxR;
  real aL, aR, ptL,ptR, qL, qR, pvrs;
  real g1gh = 0.5*(gam+1.)/gam;


  rhoL = wL[RHO]; rhoR = wR[RHO];
  vxL  = wL[VX];   vxR = wR[VX];
  ptL  = wL[PR];   ptR = wR[PR];
#if TWOTEMP
  ptL = ptL + wL[PE]; ptR = ptR + wR[PE];
#endif

  aL = fms(wL, 0., gam);
  aR = fms(wR, 0., gam);

  pvrs = FMAX(0., 0.5*(ptL+ptR) - 0.125*(vxR-vxL) * (rhoL+rhoR) * (aL+aR));

  if (pvrs < ptL) qL = 1.;
  else qL = SQRT(1. + g1gh * (pvrs/ptL - 1.));
  if (pvrs < ptR) qR = 1.;
  else qR = SQRT(1. + g1gh * (pvrs/ptR - 1.));

  *sL = vxL - aL*qL;
  *sR = vxR + aR*qR;

}


// ==============================================================

// HLL(E) flux.

void hll_flux(real **_flux, real **_wL, real **_wR, real *_bx,
             ints i1, ints i2, real gam)
{

  real uL[NMODES];
  real uR[NMODES];
  real wL[NMODES];
  real wR[NMODES];
  real fL[NMODES];
  real fR[NMODES];

#pragma omp simd simdlen(SIMD_WIDTH) private(uL,uR,wL,wR,fL,fR)
  for (ints i=i1; i<=i2; ++i) {

    for (ints n=0; n<NMODES; ++n) {
      wL[n] = _wL[n][i];
      wR[n] = _wR[n][i];
    }
#if MFIELD
    wL[BX] = _bx[i];
    wR[BX] = _bx[i];
#endif

    prim2cons(uL, wL, gam);
    prim2cons(uR, wR, gam);

    primcons2flux(fL, wL,uL, gam);
    primcons2flux(fR, wR,uR, gam);

    // estimate max and min propagation speed
    real sL = 0., sR = 0.;
    sLsR_simple(&sL, &sR, wL, wR, gam);

    real aL = FMIN(sL, 0.);
    real aR = FMAX(sR, 0.);

    real aLaR = aL*aR;
    real aRmaLi = 1./(aR-aL);

    // calculate HLL flux
    for (ints n=0; n<NMODES; ++n)
      _flux[n][i] = (aR * fL[n] - aL * fR[n] + aLaR * (uR[n] - uL[n])) * aRmaLi;

#if MFIELD
    _flux[BX][i] = 0.;
#endif

  }
}


// ===================================================

// HLLC flux for pure hydro.

#if !MFIELD

void hllc_flux(real **_flux, real **_wL, real **_wR, real *_bx,
              ints i1, ints i2, real gam)
{

  real wL[NMODES];
  real wR[NMODES];

  real uL[NMODES];
  real uR[NMODES];

  real fL[NMODES];
  real fR[NMODES];

  // real d[NMODES];

#pragma omp simd simdlen(SIMD_WIDTH) private(uL,uR, wL,wR, fL,fR)
  for (ints i=i1; i<=i2; ++i) {

    for (ints n=0; n<NMODES; ++n) {
      wL[n] = _wL[n][i];
      wR[n] = _wR[n][i];
    }

    real sL = 0., sR = 0.;

    prim2cons(uL, wL, gam);
    prim2cons(uR, wR, gam);
    sLsR_pressure(&sL,&sR, wL,wR, gam);

    real ptL = wL[PR];
    real ptR = wR[PR];
#if TWOTEMP
    ptL = ptL + wL[PE];
    ptR = ptR + wR[PE];
#endif

    real vLsL = wL[VX] - sL;
    real vRsR = wR[VX] - sR;

    real xL = uL[RHO] * vLsL;
    real xR = uR[RHO] * vRsR;
    real xRxLi = 1./(xR-xL);

    real yL = ptL + uL[MX] * vLsL;
    real yR = ptR + uR[MX] * vRsR;

    real sM =  xRxLi * (yR - yL);
    real pts = xRxLi * (xR*yL - xL*yR);

    sL = FMIN(sL,0.);
    sR = FMAX(sR,0.);
    vLsL = wL[VX] - sL;
    vRsR = wR[VX] - sR;

    // calculate F_K-U_K*S_K from Toro' book

    for (ints n=0; n<NMODES; ++n) {
      fL[n] = uL[n] * vLsL;
      fR[n] = uR[n] * vRsR;
      // d[n] = 0.;
    }
    fL[MX] += ptL;
    fR[MX] += ptR;
    fL[EN] += ptL*wL[VX];
    fR[EN] += ptR*wR[VX];

    // d[MX] = sLR*pts;
    // d[EN] = sLR*pts*sM;

    real sLsMi = 1./(sL-sM);
    real sRsMi = 1./(sR-sM);
    real zL,zR,sLR;

    if (sM>0.) {
      zL = - sM * sLsMi;
      zR = 0.;
      // d[MX] = sL * sLsMi * pts;
      sLR =  sL * sLsMi;
    } else {
      zL = 0.;
      zR = - sM * sRsMi;
      // d[MX] = sR * sRsMi * pts;
      sLR =  sR * sRsMi;
    }

    // d[EN] = d[MX]*sM;

    for (ints n=0; n<NMODES; n++)
      _flux[n][i] = zL*fL[n] + zR*fR[n]; // + d[n];

    _flux[MX][i] += sLR*pts;
    _flux[EN][i] += sLR*pts*sM;

  }

}

#endif



// HLLC flux for pure hydro.

// #if !MFIELD
//
// void hllc_flux(real **_flux, real **_wL, real **_wR, real *_bx,
//               ints i1, ints i2, real gam)
// {
//
//   real ptL, ptR;
//   real pts;
//   real sM, sL,sR;
//   real sRsLi, sLsMi,sRsMi;
//   real xL,xR;
//   ints i, n;
//
//   real wL[NMODES];
//   real wR[NMODES];
//
//   real uL[NMODES];
//   real uR[NMODES];
//
//   real uLs[NMODES];
//   real uRs[NMODES];
//
//   real fL[NMODES];
//   real fR[NMODES];
//
//
// #pragma omp simd simdlen(SIMD_WIDTH) private(uL,uR, uLs,uRs, wL,wR, fL,fR)
//   for (i=i1; i<=i2; ++i) {
//
//     for (n=0; n<NMODES; ++n) {
//       wL[n] = _wL[n][i];
//       wR[n] = _wR[n][i];
//     }
//
//     prim2cons(uL, wL, gam);
//     prim2cons(uR, wR, gam);
//
//     primcons2flux(fL, wL,uL, gam);
//     primcons2flux(fR, wR,uR, gam);
//
//     sL = 0.; sR = 0.;
//     sLsR_simple(&sL,&sR, wL, wR, gam);
//
//     ptL = wL[PR]; ptR = wR[PR];
//
// #if TWOTEMP
//     ptL = ptL + wL[PE];
//     ptR = ptR + wR[PE];
// #endif
//
//     sM = ( (sR * uR[MX] -  sL * uL[MX]  - (fR[MX]  - fL[MX]))
//          / (sR * uR[RHO] - sL * uL[RHO] - (fR[RHO] - fL[RHO])) );
//
//     sLsMi = 1./(sL-sM);
//     sRsMi = 1./(sR-sM);
//
//     xL = (sL - wL[VX]) * sLsMi;
//     xR = (sR - wR[VX]) * sRsMi;
//
//     uLs[RHO] = wL[RHO] * xL;
//     uRs[RHO] = wR[RHO] * xR;
//
//     uLs[PSC] = wL[PSC] * xL;
//     uRs[PSC] = wR[PSC] * xR;
//
//     // left * state
//
//     uLs[MX]  = uLs[RHO] * sM;
//     uLs[MY]  = uLs[RHO] * wL[VY];
//     uLs[MZ]  = uLs[RHO] * wL[VZ];
//
//     uLs[EN] = ( uL[EN] * (sL-wL[VX]) + (pts*sM - ptL*wL[VX]) ) * sLsMi;
//
//     // right * state
//
//     uRs[MX]  = uRs[RHO] * sM;
//     uRs[MY]  = uRs[RHO] * (wR[VY]);
//     uRs[MZ]  = uRs[RHO] * (wR[VZ]);
//
//     uRs[EN] = ( uR[EN] * (sR-wR[VX]) + (pts*sM - ptR*wR[VX]) ) * sRsMi;
//
// #if TWOTEMP
//     uLs[SE] = uL[SE]*xL; uRs[SE] = uR[SE]*xR;
// #endif
//
//     if (sL>0.)
//       for (n=0; n<NMODES; ++n)
//         _flux[n][i] = fL[n];
//
//     if (sL<=0. && 0.<sM)
//       for (n=0; n<NMODES; ++n)
//         _flux[n][i] = fL[n] + sL * (uLs[n] - uL[n]);
//
//     if (sM<=0. && 0.<sR)
//       for (n=0; n<NMODES; ++n)
//         _flux[n][i] = fR[n] + sR * (uRs[n] - uR[n]);
//
//     if (sR <= 0.)
//       for (n=0; n<NMODES; ++n)
//         _flux[n][i] = fR[n];
//
//   }  // end of loop over i
//
// }
//
// #endif


// =======================================================

// Isothermal Riemann solver based on Mignone 2012.

void hllt_flux(real **_flux, real **_wL, real **_wR, real *_bx,
              ints i1, ints i2, real gam)
{

  real uL[NMODES];
  real uR[NMODES];

  real uLs[NMODES];
  real uRs[NMODES];
  real uCs[NMODES];

  real wL[NMODES];
  real wR[NMODES];

  real fL[NMODES];
  real fR[NMODES];


#pragma omp simd simdlen(SIMD_WIDTH) private(uL,uR, uLs,uRs, uCs, wL,wR, fL,fR)
  for (ints i=i1; i<=i2; ++i) {

    for (ints n=0; n<NMODES; ++n) {
      wL[n] = _wL[n][i];
      wR[n] = _wR[n][i];
    }
#if MFIELD
    real bx = _bx[i];
    wL[BX] = bx;
    wR[BX] = bx;
#endif

    prim2cons(uL, wL, gam);
    prim2cons(uR, wR, gam);

    primcons2flux(fL, wL,uL, gam);
    primcons2flux(fR, wR,uR, gam);

    real sL=0., sR=0.;
    sLsR_simple(&sL, &sR, wL, wR, gam);

    real sRsLi = 1./(sR-sL);

    // HLL density
    real rhos  = (sR * uR[RHO] - sL * uL[RHO] - (fR[RHO] - fL[RHO])) * sRsLi;
    // HLL x-momentum
    real mxs   = (sR * uR[MX] -  sL * uL[MX]  - (fR[MX] -  fL[MX])) * sRsLi;
    // HLL density flux
    real frhos = (sR * fL[RHO] - sL * fR[RHO] + sR*sL * (uR[RHO] - uL[RHO])) * sRsLi;
    // normal velocity from HLL density flux and state
    real us = frhos/rhos;
    // HLL passive scalar
    real pscs  = (sR*uR[PSC] - sL*uL[PSC] - (fR[PSC] - fL[PSC])) * sRsLi;
    // HLL electron entropy
#if TWOTEMP
    real ses = (sR*uR[SE] - sL*uL[SE] - (fR[SE] - fL[SE])) * sRsLi;
#endif

    uLs[RHO] = rhos;
    uRs[RHO] = rhos;
    uLs[MX] = mxs;
    uRs[MX] = mxs;
    uLs[PSC] = pscs;
    uRs[PSC] = pscs;
#if TWOTEMP
    uLs[SE] = ses;
    uRs[SE] = ses;
#endif

    // vy*, vz*

    uLs[MY] = rhos * wL[VY];
    uRs[MY] = rhos * wR[VY];

    uLs[MZ] = rhos * wL[VZ];
    uRs[MZ] = rhos * wR[VZ];

    real sLs=us, sRs=us;

#if MFIELD

    real rhosi = 1./rhos;
    real sqrt_rhosi = SQRT(rhosi);

    sLs -= FABS(bx) * sqrt_rhosi;
    sRs += FABS(bx) * sqrt_rhosi;

    real xL = 1./((sL-sLs)*(sL-sRs));
    real xR = 1./((sR-sLs)*(sR-sRs));

    real yL = bx * xL * (us - wL[VX]);
    real yR = bx * xR * (us - wR[VX]);

    uLs[MY] -= yL*wL[BY];
    uRs[MY] -= yR*wR[BY];

    uLs[MZ] -= yL*wL[BZ];
    uRs[MZ] -= yR*wR[BZ];

    // By*, Bz*

    yL = (wL[RHO] * SQR(sL-wL[VX]) - SQR(bx)) * rhosi * xL;
    yR = (wR[RHO] * SQR(sR-wR[VX]) - SQR(bx)) * rhosi * xR;

    uLs[BY] = yL * wL[BY];
    uRs[BY] = yR * wR[BY];

    uLs[BZ] = yL * wL[BZ];
    uRs[BZ] = yR * wR[BZ];


    // central state from consistency condition

    uCs[RHO] = rhos;
    uCs[MX] = mxs;
    uCs[PSC] = pscs;
#if TWOTEMP
    uCs[SE] = ses;
#endif

    real sgnb = SIGN(bx);
    real x = 0.5 * sgnb / sqrt_rhosi;

    uCs[MY] = 0.5 * (uLs[MY] + uRs[MY]) + x*(uRs[BY] - uLs[BY]);
    uCs[MZ] = 0.5 * (uLs[MZ] + uRs[MZ]) + x*(uRs[BZ] - uLs[BZ]);

    uCs[BY] = 0.5 * (uLs[BY] + uRs[BY]) + x*rhosi * (uRs[MY] - uLs[MY]);
    uCs[BZ] = 0.5 * (uLs[BZ] + uRs[BZ]) + x*rhosi * (uRs[MZ] - uLs[MZ]);

#endif


    if (sL>0.)
      for (ints n=0; n<NMODES; ++n)
        _flux[n][i] = fL[n];

    if (sL<=0. && 0.<sLs)
      for (ints n=0; n<NMODES; ++n)
        _flux[n][i] = fL[n] + sL * (uLs[n] - uL[n]);

#if MFIELD
    if (sLs<=0. && 0.<sRs)
      for (ints n=0; n<NMODES; ++n)
        _flux[n][i] = fL[n] + sL * (uLs[n] - uL[n]) + sLs * (uCs[n] - uLs[n]);
#endif

    if (sRs<=0. && 0.<sR)
      for (ints n=0; n<NMODES; ++n)
        _flux[n][i] = fR[n] + sR * (uRs[n] - uR[n]);

    if (sR<=0.)
      for (ints n=0; n<NMODES; ++n)
        _flux[n][i] = fR[n];

#if MFIELD
    _flux[BX][i] = 0.;
#endif


  }  // end of loop over i

}



#if MFIELD

// ===================================================

// HLLD flux.

void hlld_flux(real **_flux, real **_wL, real **_wR, real *_bx,
              ints i1, ints i2, real gam)
{

  real uL[NMODES];
  real uR[NMODES];

  real uLs[NMODES];
  real uRs[NMODES];

  real uLss[NMODES];
  real uRss[NMODES];

  real wL[NMODES];
  real wR[NMODES];

  real fL[NMODES];
  real fR[NMODES];


#pragma omp simd simdlen(SIMD_WIDTH) private(uL,uR, uLs,uRs, uLss,uRss, wL,wR, fL,fR)
  for (ints i=i1; i<=i2; ++i) {

    for (ints n=0; n<NMODES; ++n) {
      wL[n] = _wL[n][i];
      wR[n] = _wR[n][i];
    }
    real bx = _bx[i];
    wL[BX] = bx;
    wR[BX] = bx;

    prim2cons(uL, wL, gam);
    prim2cons(uR, wR, gam);

    primcons2flux(fL, wL,uL, gam);
    primcons2flux(fR, wR,uR, gam);

    real sL=0., sR=0.;
    sLsR_simple(&sL, &sR, wL, wR, gam);

    real b2hL = 0.5 * (SQR(bx) + SQR(wL[BY]) + SQR(wL[BZ]));
    real b2hR = 0.5 * (SQR(bx) + SQR(wR[BY]) + SQR(wR[BZ]));

    real ptL = wL[PR] + b2hL;
    real ptR = wR[PR] + b2hR;

#if TWOTEMP
    ptL += wL[PE];
    ptR += wR[PE];
#endif

    // normal velocity from HLL state
    real deni = 1./(sR * uR[RHO] - sL * uL[RHO] - (fR[RHO] - fL[RHO]));
    real sM =  (sR * uR[MX] - sL * uL[MX] - (fR[MX] - fL[MX])) * deni;

    real sLvxL = sL - wL[VX];
    real sRvxR = sR - wR[VX];
    real sLsMi = 1./(sL - sM);
    real sRsMi = 1./(sR - sM);

    // *-states
    // p*L=p*R, vx*L=vx*R=sM

    real xL = sLvxL * sLsMi;
    real xR = sRvxR * sRsMi;

    // density, normal momentum, and passive scalar

    uLs[RHO] = uL[RHO]*xL;
    uRs[RHO] = uR[RHO]*xR;

    uLs[PSC] = uL[PSC]*xL;
    uRs[PSC] = uR[PSC]*xR;

    // electron entropy
#if TWOTEMP
    uLs[SE] = uL[SE]*xL;
    uRs[SE] = uR[SE]*xR;
#endif

    // pressure
    real pts = wL[RHO] * sLvxL * (sM-wL[VX]) + ptL;

    // tangential velocities

    real rhosLvxL = wL[RHO] * sLvxL;
    real rhosRvxR = wR[RHO] * sRvxR;
    real bx2 = SQR(bx);

    real wvbL = 1./(rhosLvxL * (sL-sM) - bx2);
    real wvbR = 1./(rhosRvxR * (sR-sM) - bx2);

    real bxsMvxL = bx * (sM-wL[VX]);
    real bxsMvxR = bx * (sM-wR[VX]);

    real wvL = bxsMvxL * wvbL;
    real wvR = bxsMvxR * wvbR;

    real vyLs = wL[VY] - wL[BY] * wvL;
    real vyRs = wR[VY] - wR[BY] * wvR;

    real vzLs = wL[VZ] - wL[BZ] * wvL;
    real vzRs = wR[VZ] - wR[BZ] * wvR;

    uLs[MX] = uLs[RHO] * sM;
    uRs[MX] = uRs[RHO] * sM;

    uLs[MY] = uLs[RHO] * vyLs;
    uRs[MY] = uRs[RHO] * vyRs;

    uLs[MZ] = uLs[RHO] * vzLs;
    uRs[MZ] = uRs[RHO] * vzRs;

    // tangential magnetic field

    real rhosLvxL2_bx2 = rhosLvxL * sLvxL - bx2;
    real rhosRvxR2_bx2 = rhosRvxR * sRvxR - bx2;

    real wbL = rhosLvxL2_bx2 * wvbL;
    real wbR = rhosRvxR2_bx2 * wvbR;

    uLs[BY] = wL[BY] * wbL;
    uRs[BY] = wR[BY] * wbR;

    uLs[BZ] = wL[BZ] * wbL;
    uRs[BZ] = wR[BZ] * wbR;

    // energy density

    real vsbsL = sM*bx + vyLs*uLs[BY] + vzLs*uLs[BZ];
    real vsbsR = sM*bx + vyRs*uRs[BY] + vzRs*uRs[BZ];

    uLs[EN] = (sLvxL * uL[EN] - ptL*wL[VX] + pts*sM
          + bx * (wL[VX]*bx + wL[VY]*wL[BY] + wL[VZ]*wL[BZ] - vsbsL)) * sLsMi;
    uRs[EN] = (sRvxR * uR[EN] - ptR*wR[VX] + pts*sM
          + bx * (wR[VX]*bx + wR[VY]*wR[BY] + wR[VZ]*wR[BZ] - vsbsR)) * sRsMi;

    // **-states
    // rho**=rho*
    // p**=p*

    uLss[RHO] = uLs[RHO];
    uRss[RHO] = uRs[RHO];

    uLss[MX] = uLs[MX];
    uRss[MX] = uRs[MX];

    uLss[PSC] = uLs[PSC];
    uRss[PSC] = uRs[PSC];

    // electron entropy
#if TWOTEMP
    uLss[SE] = uLs[SE];
    uRss[SE] = uRs[SE];
#endif

    real sqrt_rhoLs = SQRT(uLs[RHO]);
    real sqrt_rhoRs = SQRT(uRs[RHO]);

    real sLs = sM - FABS(bx)/sqrt_rhoLs;
    real sRs = sM + FABS(bx)/sqrt_rhoRs;

    real rhoLRi = 1./(sqrt_rhoLs + sqrt_rhoRs);
    real sgnb = SIGN(bx);

    real vyss = (sqrt_rhoLs*vyLs + sqrt_rhoRs*vyRs + (uRs[BY]-uLs[BY])*sgnb) * rhoLRi;
    real vzss = (sqrt_rhoLs*vzLs + sqrt_rhoRs*vzRs + (uRs[BZ]-uLs[BZ])*sgnb) * rhoLRi;

    uLss[MY] = uLs[RHO]*vyss;
    uRss[MY] = uRs[RHO]*vyss;

    uLss[MZ] = uLs[RHO]*vzss;
    uRss[MZ] = uRs[RHO]*vzss;

    real byss = (sqrt_rhoLs * uRs[BY] + sqrt_rhoRs * uLs[BY]
            + sqrt_rhoLs * sqrt_rhoRs * (vyRs-vyLs)*sgnb) * rhoLRi;
    real bzss = (sqrt_rhoLs * uRs[BZ] + sqrt_rhoRs * uLs[BZ]
            + sqrt_rhoLs * sqrt_rhoRs * (vzRs-vzLs)*sgnb) * rhoLRi;

    uLss[BY] = byss;
    uRss[BY] = byss;
    uLss[BZ] = bzss;
    uRss[BZ] = bzss;

    real vssbss = sM*bx + vyss*byss + vzss*bzss;

    uLss[EN] = uLs[EN] - sqrt_rhoLs * (vsbsL - vssbss)*sgnb;
    uRss[EN] = uRs[EN] + sqrt_rhoRs * (vsbsR - vssbss)*sgnb;


    if (sL>0.)
      for (ints n=0; n<NMODES; ++n)
        _flux[n][i] = fL[n];

    if (sL<=0. && 0.<sLs)
      for (ints n=0; n<NMODES; ++n)
        _flux[n][i] = fL[n] + sL * (uLs[n] - uL[n]);

    if (sLs<=0. && 0.<sM)
      for (ints n=0; n<NMODES; ++n)
        _flux[n][i] = fL[n] + sL * (uLs[n] - uL[n]) + sLs * (uLss[n] - uLs[n]);

    if (sM<=0. && 0.<sRs)
      for (ints n=0; n<NMODES; ++n)
        _flux[n][i] = fR[n] + sR * (uRs[n] - uR[n]) + sRs * (uRss[n] - uRs[n]);

    if (sRs<=0. && 0.<sR)
      for (ints n=0; n<NMODES; ++n)
        _flux[n][i] = fR[n]  + sR * (uRs[n] - uR[n]);

    if (sR<=0.)
      for (ints n=0; n<NMODES; ++n)
        _flux[n][i] = fR[n];

    _flux[BX][i] = 0.;

  }  // end of loop over i

}

#endif


// =====================================================================

// Flux resolving Alfven and FMS discontinuities for anisotropic plasma.

#if CGL

void hlla_flux(real **_flux, real **_wL, real **_wR, real *_bx,
              ints i1, ints i2, real gam)
{

  real wL[NMODES];
  real wR[NMODES];
  real uL[NMODES];
  real uR[NMODES];
  real fL[NMODES];
  real fR[NMODES];
  real uhll[NMODES];
  real fhll[NMODES];

  // for i in range(i1,i2+1):
  //
  //   b2hL = WL[BX][i]**2 + WL[BY][i]**2 + WL[BZ][i]**2
  //   b2hR = WR[BX][i]**2 + WR[BY][i]**2 + WR[BZ][i]**2
  //   F[i][NMODES] = 3*((WR[i][PPD]-WR[i][P])/b2hR - (WL[i][PPD]-WL[i][P])/b2hL)

#pragma omp simd simdlen(SIMD_WIDTH) private(uL,uR, uhll,fhll, wL,wR, fL,fR)
  for (ints i=i1; i<=i2; ++i) {

    for(ints n=0; n<NMODES; ++n) {
      wL[n] = _wL[n][i];
      wR[n] = _wR[n][i];
    }
    real bx = _bx[i];
    wL[BX] = bx;
    wR[BX] = bx;

    prim2cons(uL, wL, gam);
    prim2cons(uR, wR, gam);

    primcons2flux(fL, wL,uL, gam);
    primcons2flux(fR, wR,uR, gam);

    real sL=0., sR=0.;
    sLsR_simple(&sL, &sR, wL, wR, gam);
    real cL = FMIN(sL, 0.);
    real cR = FMAX(sR, 0.);

    real b2hL = 0.5 * (SQR(bx) + SQR(wL[BY]) + SQR(wL[BZ]));
    real b2hR = 0.5 * (SQR(bx) + SQR(wR[BY]) + SQR(wR[BZ]));

    // pressure anisotropies outside the Riemann fan

    real aL1 = 1.5*(wL[PPD] - wL[PR]) / b2hL + 1.;
    real aR1 = 1.5*(wR[PPD] - wR[PR]) / b2hR + 1.;

    //calculate HLL average state and flux

    real cLcR = cL*cR;
    real cRmcLi = 1./(cR-cL);

    for (ints k=0; k<NMODES; k++) {
      fhll[k] = (cR * fL[k] - cL * fR[k] + cLcR * (uR[k] - uL[k])) * cRmcLi;
      uhll[k] = (cR * uR[k] - cL * uL[k] - fR[k] + fL[k]) * cRmcLi;
    }

    real bx2 = SQR(bx);
    real rhosi = 1./uhll[RHO];

    // pressure anisotropy across the fan

    real b2hll = bx2 + SQR(uhll[BY]) + SQR(uhll[BZ]);
    real m2hll = SQR(uhll[MX]) + SQR(uhll[MY]) + SQR(uhll[MZ]);
    real p = (gam-1.) * (uhll[EN] - 0.5*(b2hll + m2hll * rhosi));

#ifdef TWOTEMP
    real pe = POW(uhll[RHO], gam) * EXP(uhll[SE] * rhosi);
    p -= pe;
#endif

    real ppd_ppl = EXP(uhll[LA] * rhosi) * b2hll*SQRT(b2hll) * SQR(rhosi);
    real ppd = 3.*ppd_ppl / (1. + 2.*ppd_ppl) * p;

    real as1 = 3.*(ppd-p) / b2hll + 1.;
    real as1 = FMAX(as1, 0.);

    // velocities of tangential discontinuities sLa and sRa

    real us = fhll[RHO] * rhosi;
    real cs = FABS(bx) * SQRT(as1 * rhosi);
    real sLs = us - cs;
    real sRs = us + cs;

    // tangential components in L,R star regions

    real xL = 1. / ((sL - sLs) * (sL - sRs));
    real xR = 1. / ((sR - sLs) * (sR - sRs));

    // By*, Bz*

    real yL = rhosi * xL * (wL[RHO] * SQR(sL-wL[VX]) - SQR(bx) * aL1);
    real yR = rhosi * xR * (wR[RHO] * SQR(sR-wR[VX]) - SQR(bx) * aR1);

    real byLs = yL * wL[BY];
    real byRs = yR * wR[BY];

    real bzLs = yL * wL[BZ];
    real bzRs = yR * wR[BZ];

    // vy*, vz*

    real yL = bx * xL * (aL1*(sL-us) - as1*(sL-wL[VX]));
    real yR = bx * xR * (aR1*(sR-us) - as1*(sR-wR[VX]));

    real myLs = uhll[RHO] * wL[VY] + yL * wL[BY];
    real myRs = uhll[RHO] * wR[VY] + yR * wR[BY];

    real mzLs = uhll[RHO] * wL[VZ] + yL * wL[BZ];
    real mzRs = uhll[RHO] * wR[VZ] + yR * wR[BZ];

    // fluxes in L,R star regions, including antidiffusion

    real a1min = FMIN(FMIN(aL1,aR1),as1);
    real h = 0.;
    if (a1min > 0.03) h=1.;
    // da_max = FMAX(FMAX(F[i-1][WAVES],F[i][WAVES]),F[mini(i+1,i2)][WAVES])
    real om_ad = h ;//* exp(-10.*da_max);

    // if (om_ad != 1.)
    //   printf("%f %f\n", da_max, om_ad);

    // om_ad=1.;

    real omsL = om_ad * sL;
    real omsR = om_ad * sR;

    real fLs_my = fhll[MY] + omsL * (myLs - uhll[MY]);
    real fLs_mz = fhll[MZ] + omsL * (mzLs - uhll[MZ]);

    real fLs_by = fhll[BY] + omsL * (byLs - uhll[BY]);
    real fLs_bz = fhll[BZ] + omsL * (bzLs - uhll[BZ]);

    real fRs_my = fhll[MY] + omsR * (myRs - uhll[MY]);
    real fRs_mz = fhll[MZ] + omsR * (mzRs - uhll[MZ]);

    real fRs_by = fhll[BY] + omsR * (byRs - uhll[BY]);
    real fRs_bz = fhll[BZ] + omsR * (bzRs - uhll[BZ]);

    // fluxes in C star region from consistency condition

    // if (cs != 0.) {

    real x = sLs/(sRs-sLs);

    real fCs_my = fLs_my + x * (sRs * (myRs - myLs) - fRs_my + fLs_my);
    real fCs_mz = fLs_mz + x * (sRs * (mzRs - mzLs) - fRs_mz + fLs_mz);
    real fCs_by = fLs_by + x * (sRs * (byRs - byLs) - fRs_by + fLs_by);
    real fCs_bz = fLs_bz + x * (sRs * (bzRs - bzLs) - fRs_bz + fLs_bz);

    // }

    // set Riemann fluxes

    if (aL1<0. || aR1<0. || as1<0.)

      for (ints n=0; n<NMODES; n++)
        _flux[n][i]=fhll[n];

    else {

      for (ints n=0; n<BX; n++)
        if (n != MY && n != MZ)
          _flux[n][i] = fhll[n];


      if (sL<=0. && 0.<sLs) {
        _flux[MY][i] = fLs_my;
        _flux[MZ][i] = fLs_mz;
        _flux[BY][i] = fLs_by;
        _flux[BZ][i] = fLs_bz;
      }

      if (sLs<=0. && 0.<sRs) {
        _flux[MY][i] = fCs_my;
        _flux[MZ][i] = fCs_mz;
        _flux[BY][i] = fCs_by;
        _flux[BZ][i] = fCs_bz;
      }

      if (sRs<=0. && 0.<sR) {
        _flux[MY][i] = fRs_my;
        _flux[MZ][i] = fRs_mz;
        _flux[BY][i] = fRs_by;
        _flux[BZ][i] = fRs_bz;
      }

    }

    _flux[BX][i] = 0.;

    // F[PSC]=0.

  // free(DA)

  }

}
#endif
