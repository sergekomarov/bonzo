#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "eos.h"
#include "reconstr.h"

// ==========================================================

// Donor cell reconstruction.

void reconstr_const(real **wl, real **wr, real ***w, real ***scr,
                     GridCoord *gc, int ax,
                     ints i1, ints i2, ints j, ints k,
                     int char_proj, real gam) {

  ints ip = (ax==0) ? 1 : 0;

  for (ints n=0; n<NMODES; ++n)
#pragma omp simd simdlen(SIMD_WIDTH)
    for (ints i=i1; i<=i2; ++i) {
      wr[n][i]    = w[0][n][i];
      wl[n][i+ip] = w[0][n][i];
    }

}


// ==========================================================

// Linear reconstruction.

void reconstr_linear(real **wl, real **wr, real ***w, real ***scr,
                     GridCoord *gc, int ax,
                     ints i1, ints i2, ints j, ints k,
                     int char_proj, real gam) {

  real **dm = scr[0];
  real **dp = scr[1];
  real **d  = scr[0];
  real **wc = scr[2];

  real *xif  = gc.lf[ax];
  real *xiv  = gc.lv[ax];
  real *dxiv = gc.dlv[ax];
  real *dxiv_inv = gc.dlv_inv[ax];

  int is_flat_uni = ( (gc.coord_space[ax]==CS_UNI) &&
                     ((gc.coord_geom[ax]==CG_CAR) ||
                      (gc.coord_geom[ax]==CG_CYL && ax>0) ||
                      (gc.coord_geom[ax]==CG_SPH && ax==2)) );

  ints ip = (ax==0) ? 1 : 0;
  ints jk=0;
  if (ax==1) jk=j;
  else if (ax==2) jk=k;

  for (ints n=0; n<NMODES; n++) {
#pragma omp simd
    for (ints i=i1; i<=i2; i++) {
      wc[n][i] = w[1][n][i];
      dm[n][i] = w[1][n][i] - w[0][n][i];
      dp[n][i] = w[2][n][i] - w[1][n][i];
    }
  }

  if (char_proj) {
    prim2char_1(dm, wc, i1,i2, gam);
    prim2char_1(dp, wc, i1,i2, gam);
  }

  // any flat uniform coordinate

  if (is_flat_uni) {

    for (ints n=0; n<NMODES; n++)
#pragma omp simd simdlen(SIMD_WIDTH)
      for (ints i=i1; i<=i2; i++)
        d[n][i] = 0.5*vl_lim(dm[n][i], dp[n][i]);

    if (char_proj)
      char2prim_1(d, wc, i1,i2, gam);

    for (ints n=0; n<NMODES; n++) {
#pragma omp simd
      for (ints i=i1; i<=i2; i++) {

        wr[n][i]    = wc[n][i] - d[n][i];
        wl[n][i+ip] = wc[n][i] + d[n][i];
      }
    }

  }

  // nonuniform or curvilinear coordinate

  else {

    // x-coordinate (in the direction of vectorization)

    if (ax==0) {

      for (ints n=0; n<NMODES; n++) {
#pragma omp simd simdlen(SIMD_WIDTH)
        for (ints i=i1; i<=i2; i++) {

          // skip multiplication by dxif[i] to avoid division by the same number in the end
          real dm_ = dm[n][i] * dxiv_inv[i];
          real dp_ = dp[n][i] * dxiv_inv[i+1];

          real dmdp = dm_*dp_;
          real fm = dxiv[i-1] / (xiv[i]   - xif[i]);
          real fp = dxiv[i  ] / (xif[i+1] - xiv[i]);

          // modified van Leer
          d[n][i] = dmdp * ( (fp * dm_ + fm * dp_) /
            ( dm_*dm_ + (fp+fm-2.) * dmdp + dp_*dp_ ) );

          if (dmdp<=0.) d[n][i]=0.;

          // monotonized central
          // if (dp1 != 0.) {
          //   v = dm_/dp_;
          //   d[n][i] = MAX(0., MIN(0.5*(1+v), MIN(fp, fm*v))) * dp_;
          // }
          // else d[n][i] = fp;

        }
      }

      if (char_proj)
        char2prim_1(d, wc, i1,i2, gam);

      for (ints n=0; n<NMODES; n++) {
#pragma omp simd simdlen(SIMD_WIDTH)
        for (ints i=i1; i<=i2; i++) {
          wr[n][i]   = wc[n][i] - d[n][i] * (xiv[i]   - xif[i]);
          wl[n][i+1] = wc[n][i] + d[n][i] * (xif[i+1] - xiv[i]);
        }
      }

    }

    // y- or z-coordinate

    else {

      real fm = dxiv[jk  ] / (xiv[jk]   - xif[jk]);
      real fp = dxiv[jk+1] / (xif[jk+1] - xiv[jk]);

      for (ints n=0; n<NMODES; n++) {
#pragma omp simd simdlen(SIMD_WIDTH)
        for (ints i=i1; i<=i2; i++) {

          real dm_ = dm[n][i] * dxiv_inv[jk];
          real dp_ = dp[n][i] * dxiv_inv[jk+1];
          real dmdp = dm_*dp_;

          // modified van Leer
          d[n][i] = dmdp * ( (fp * dm_ + fm * dp_) /
            ( dm_*dm_ + (fp+fm-2.) * dmdp + dp_*dp_ ) );

          if (dmdp<=0.) d[n][i]=0.;

        }
      }

      if (char_proj)
        char2prim_1(d, wc, i1,i2, gam);

      real dximh = xiv[jk]   - xif[jk];
      real dxiph = xif[jk+1] - xiv[jk];

      for (ints n=0; n<NMODES; n++) {
#pragma omp simd
        for (ints i=i1; i<=i2; i++) {
          wr[n][i] = wc[n][i] - d[n][i] * dximh;
          wl[n][i] = wc[n][i] + d[n][i] * dxiph;
        }
      }

    }

  }

}



// ==============================================================

// WENO reconstruction.

void reconstr_weno(real **wl, real **wr, real ***w, real ***scr,
                     GridCoord *gc, int ax,
                     ints i1, ints i2, ints j, ints k,
                     int char_proj, real gam) {

  int is_flat_uni = ( (gc.coord_space[ax]==CS_UNI) &&
                     ((gc.coord_geom[ax]==CG_CAR) ||
                      (gc.coord_geom[ax]==CG_CYL && ax>0) ||
                      (gc.coord_geom[ax]==CG_SPH && ax==2)) );

  ints ip = (ax==0) ? 1 : 0;
  ints jk=0;
  if (ax==1) jk=j;
  else if (ax==2) jk=k;

  real **dm = scr[0];
  real **dp = scr[1];
  real **wc = scr[2];

  real *dxiv     = gc.dlv[ax];
  real *dxiv_inv = gc.dlv_inv[ax];

  real *cm = gc.cm[ax];
  real *cp = gc.cp[ax];

  real Cref;
  if (is_flat_uni) Cref = 20. / gc.Nact_glob[ax];
  else Cref = 20. / (gc.lmax[ax]-gc.lmin[ax]);


  for (ints n=0; n<NMODES; ++n) {
#pragma omp simd
    for (ints i=i1; i<=i2; ++i) {
      wc[n][i] = w[1][n][i];
      dm[n][i] = w[1][n][i] - w[0][n][i];
      dp[n][i] = w[2][n][i] - w[1][n][i];
    }
  }

  if (char_proj) {
    prim2char_1(dm, wc, i1,i2, gam);
    prim2char_1(dp, wc, i1,i2, gam);
  }

  // uniform flat cordinate.

  if (is_flat_uni) {

    for (ints n=0; n<NMODES; ++n) {
#pragma omp simd simdlen(SIMD_WIDTH)
      for (ints i=i1; i<=i2; ++i) {

        real dp_ = dp[n][i];
        real dm_ = dm[n][i];

        real x = (dp_ - dm_) * (dp_ - dm_);
        real qref2 = SQR(Cref);// * MAX(W[0][n][i],MAX(W[1][n][i],W[2][n][i]));

        real am0 = 1. + x / (dm_*dm_ + qref2);
        real ap0 = 1. + x / (dp_*dp_ + qref2);

        dm[n][i] = (am0 * dm_ + 0.5 * ap0 * dp_) / (2*am0 + ap0);
        dp[n][i] = (ap0 * dp_ + 0.5 * am0 * dm_) / (2*ap0 + am0);

      }
    }

    if (char_proj) {
      char2prim_1(dm, wc, i1,i2, gam);
      char2prim_1(dp, wc, i1,i2, gam);
    }

    for (ints n=0; n<NMODES; ++n) {
#pragma omp simd
      for (ints i=i1; i<=i2; ++i) {
        wr[n][i]    = wc[n][i] - dm[n][i];
        wl[n][i+ip] = wc[n][i] + dp[n][i];
      }
    }

  }

  // nonuniform or curvilinear coordinate

  else {

    if (ax==0) {

      for (ints n=0; n<NMODES; ++n) {
#pragma omp simd simdlen(SIMD_WIDTH)
        for (ints i=i1; i<=i2; ++i) {

          real dm_ = dm[n][i] * dxiv_inv[i];
          real dp_ = dp[n][i] * dxiv_inv[i+1];

          real x = (dp_ - dm_) * (dp_ - dm_);
          real qref2 = SQR(Cref);// * MAX(W[0][n][i],MAX(W[1][n][i],W[2][n][i]));

          real y0 = (1. + x / (dm_*dm_ + qref2)) * dxiv[i];
          real y1 = (1. + x / (dp_*dp_ + qref2)) * dxiv[i+1];

          real am0 = - cm[0][i] * y0;
          real am1 =   cm[2][i] * y1;

          real ap0 = - cp[0][i] * y0;
          real ap1 =   cp[2][i] * y1;

          dm[n][i] = (am0 * dm_ + am1 * dp_) / (am0 + am1);
          dp[n][i] = (ap0 * dm_ + ap1 * dp_) / (ap0 + ap1);

        }
      }

      if (char_proj) {
        char2prim_1(dm, wc, i1,i2, gam);
        char2prim_1(dp, wc, i1,i2, gam);
      }

      for (ints n=0; n<NMODES; ++n) {
#pragma omp simd
        for (ints i=i1; i<=i2; ++i) {
          wr[n][i]   = wc[n][i] - dm[n][i] * (xiv[i]   - xif[i]);
          wl[n][i+1] = wc[n][i] + dp[n][i] * (xif[i+1] - xiv[i]);
        }
      }

    }

    // y- or z-coordinate

    else {

      for (ints n=0; n<NMODES; ++n) {
#pragma omp simd simdlen(SIMD_WIDTH)
        for (ints i=i1; i<=i2; ++i) {

          real dm_ = dm[n][i] * dxiv_inv[jk];
          real dp_ = dp[n][i] * dxiv_inv[jk+1];

          real x = (dp_ - dm_) * (dp_ - dm_);
          real qref2 = SQR(Cref);// * MAX(W[0][n][i],MAX(W[1][n][i],W[2][n][i]));

          real y0 = (1. + x / (dm_*dm_ + qref2)) * dxiv[jk];
          real y1 = (1. + x / (dp_*dp_ + qref2)) * dxiv[jk+1];

          real am0 = - cm[0][jk] * y0;
          real am1 =   cm[2][jk] * y1;

          real ap0 = - cp[0][jk] * y0;
          real ap1 =   cp[2][jk] * y1;

          dm[n][i] = (am0 * dm_ + am1 * dp_) / (am0 + am1);
          dp[n][i] = (ap0 * dm_ + ap1 * dp_) / (ap0 + ap1);

        }
      }

      if (char_proj) {
        char2prim_1(dm, wc, i1,i2, gam);
        char2prim_1(dp, wc, i1,i2, gam);
      }

      real dximh = xiv[jk]   - xif[jk];
      real dxiph = xif[jk+1] - xiv[jk];

      for (ints n=0; n<NMODES; ++n) {
#pragma omp simd
        for (ints i=i1; i<=i2; ++i) {
          wr[n][i] = wc[n][i] - dm[n][i] * dximh;
          wl[n][i] = wc[n][i] + dp[n][i] * dxiph;
        }
      }

    } // end of if x or y coordinate

  } // end of if is_flat_uni

}


// =========================================================

// Parabolic reconstruction.

void reconstr_parab(real **_wl, real **_wr, real ***_w, real ***scr,
                     GridCoord *gc, int ax,
                     ints i1, ints i2, ints j, ints k,
                     int char_proj, real gam) {

  real wm2,wm1,wc,wp1,wp2, wmh,wph, wmh_,wph_;
  real ddm1,ddc,ddp1;
  real d2m1,d2c,d2p1;
  real d2mh, d2mh_lim, d2ph, d2ph_lim, d2hc, d2hc_lim;
  real wmax1,wmax2,wmax, x;
  real dmhc, dphc;
  real dm,dp, dm_abs,dp_abs;

  real one6th = 1./6;
  real one12th = 1./12;
  real C=1.25;

  int is_flat_uni = ( (gc.coord_space[ax]==CS_UNI) &&
                     ((gc.coord_geom[ax]==CG_CAR) ||
                      (gc.coord_geom[ax]==CG_CYL && ax>0) ||
                      (gc.coord_geom[ax]==CG_SPH && ax==2)) );

  ints ip = (ax==0) ? 1 : 0;
  ints jk=0;
  if (ax==1) jk=j;
  else if (ax==2) jk=k;

  real *cm = gc.cm[ax];
  real *cp = gc.cp[ax];
  real *hm_ratio = gc.hm_ratio[ax];
  real *hp_ratio = gc.hp_ratio[ax];

  real **_wc  = scr[0];
  real **_wc0 = scr[1];
  real **_wm2 = scr[2];
  real **_wm1 = scr[3];
  real **_wp1 = scr[4];
  real **_wp2 = scr[5];

  // reuse some scratch for temporary arrays
  real **_wl_tmp = scr[2];
  real **_wr_tmp = scr[3];


  for (ints n=0; n<NMODES; ++n) {
#pragma omp simd
    for (ints i=i1; i<=i2; ++i) {
      _wm2[n][i] = _w[0][n][i];
      _wm1[n][i] = _w[1][n][i];
      _wc[n][i]  = _w[2][n][i];
      _wc0[n][i] = _w[2][n][i];
      _wp1[n][i] = _w[3][n][i];
      _wp2[n][i] = _w[4][n][i];
    }
  }

  if (char_proj) {
    prim2char_1(_wm2, _wc0, i1,i2, gam);
    prim2char_1(_wm1, _wc0, i1,i2, gam);
    prim2char_1(_wc,  _wc0, i1,i2, gam);
    prim2char_1(_wp1, _wc0, i1,i2, gam);
    prim2char_1(_wp2, _wc0, i1,i2, gam);
  }

  // uniform flat coordinate

  if (is_flat_uni) {

    for (ints n=0; n<NMODES; ++n) {

#pragma omp simd simdlen(SIMD_WIDTH)
      for (ints i=i1; i<=i2; ++i) {

        wm2 = _wm2[n][i]; wm1 = _wm1[n][i]; wc = _wc[n][i];
        wp1 = _wp1[n][i]; wp2 = _wp2[n][i];

        // approximate left and right interfaces by parabolic interpolation

        ddm1 = wc  - wm2;
        ddc  = wp1 - wm1;
        ddp1 = wp2 - wc;

        wmh = 0.5*(wm1 + wc ) - one12th * (ddc  - ddm1);
        wph = 0.5*(wc  + wp1) - one12th * (ddp1 - ddc);

        d2m1 = wm2 + wc  - 2.*wm1;
        d2c  = wm1 + wp1 - 2.*wc;
        d2p1 = wc  + wp2 - 2.*wp1;

        // correct left interface using second-order derivatives to avoid extrema

        d2mh = 3.*(wm1 + wc - 2.*wmh);
        d2mh_lim = 0.;

        if (d2m1 * d2mh > 0. && d2mh * d2c > 0.)
          d2mh_lim = FSIGN(d2mh) * MIN(C * MIN(FABS(d2m1), FABS(d2c)), FABS(d2mh));

        if ((wmh - wm1) * (wc  - wmh) < 0.)
          wmh = 0.5*(wm1 + wc) - one6th * d2mh_lim;

        // correct right interface

        d2ph = 3.*(wc + wp1 - 2.*wph);
        d2ph_lim = 0.;

        if (d2c * d2ph > 0. && d2ph * d2p1 > 0.)
          d2ph_lim = FSIGN(d2ph) * MIN(C * MIN(FABS(d2c), FABS(d2p1)), FABS(d2ph));

        if ((wph - wc) * (wp1 - wph) < 0.)
          wph = 0.5*(wc + wp1) - one6th * d2ph_lim;

        // construct limited parabolic interpolant

        d2hc = 6.*(wmh + wph - 2.*wc);
        d2hc_lim = 0.;

        if (d2m1 * d2c > 0. && d2c * d2p1 > 0. && d2p1 * d2hc > 0.)
          d2hc_lim = FSIGN(d2hc) * MIN(C * MIN(FABS(d2m1), FABS(d2c)),
                                       MIN(C * FABS(d2p1), FABS(d2hc)));

        // check if correction is larger than roundoff error

        wmax1 = MAX(FABS(wm1), FABS(wm2));
        wmax2 = MAX(FABS(wc),  MAX(FABS(wp1), FABS(wp2)));
        wmax  = MAX(wmax1, wmax2);

        x=0.;
        if (FABS(d2hc) > 1e-12*wmax)
          x = d2hc_lim / d2hc;

        // apply correction if nonmonotonicities are detected

        dmhc = wc  - wmh;
        dphc = wph - wc;

        if (dmhc * dphc <= 0. || (wp1 - wc)*(wc - wm1) <= 0.) {
          if (x <= (1.-1e-12)) {
            wmh = wc - x * dmhc;
            wph = wc + x * dphc;
          }
        } else {
          if (FABS(dmhc) >= 2.*FABS(dphc))
            wmh = wc - 2.*dphc;
          else if (FABS(dphc) >= 2.*FABS(dmhc))
            wph = wc + 2.*dmhc;
        }

        _wl_tmp[n][i] = wph;
        _wr_tmp[n][i] = wmh;

      }
    } // end of loops over i and n

  }

  // nonuniform or curvilinear coordinate

  else {

    if (ax==0) {

      for (ints n=0; n<NMODES; ++n) {
#pragma omp simd simdlen(SIMD_WIDTH)
        for (ints i=i1; i<=i2; ++i) {

          real wm2 = _wm2[n][i];
          real wm1 = _wm1[n][i];
          real wc  =  _wc[n][i];
          real wp1 = _wp1[n][i];
          real wp2 = _wp2[n][i];

          real hpr = hp_ratio[n][i];
          real hmr = hm_ratio[n][i];

          real wmh = ( (cm[1][i] * wm1 + cm[2][i] * wc )
                     + (cm[0][i] * wm2 + cm[3][i] * wp1) );

          real wph = ( (cp[1][i] * wc  + cp[2][i] * wp1)
                     + (cp[0][i] * wm1 + cp[3][i] * wp2) );

          wmh = MIN(wmh, MAX(wc,wm1));
          wmh = MAX(wmh, MIN(wc,wm1));

          wph = MIN(wph, MAX(wc,wp1));
          wph = MAX(wph, MIN(wc,wp1));

          real dp = wph-wc;
          real dm = wc-wmh;

          real dp_abs = FABS(dp);
          real dm_abs = FABS(dm);

          if (dp*dm<=0.) {
            wph=wc;
            wmh=wc;
          }
          else {
            if (dm_abs >= hpr*dp_abs)
              wmh = wc - hpr*dp;
            if (dp_abs >= hmr*dm_abs)
              wph = wc + hmr*dm;
          }

          _wr_tmp[n][i] = wmh;
          _wl_tmp[n][i] = wph;

          // dm2 = wm1 - wm2;
          // dm1 = wc  - wm1;
          // dp1 = wp1 -  wc;
          // dp2 = wp2 - wp1;

          // dm1_lim = MClim(dm2, dm1);
          // dc_lim  = MClim(dm1, dp1);
          // dp1_lim = MClim(dp1, dp2);

          // dm = - 0.5*dm1 - (dc_lim  - dm1_lim) * one6th;
          // dp =   0.5*dp1 - (dp1_lim - dc_lim ) * one6th;

          // // dm = - 0.5*dm1 - (dp1 - dm2) * one12th;
          // // dp =   0.5*dp1 - (dp2 - dm1) * one12th;

          // if (dm * dp > 0) {
          //
          //   _wr_tmp[n][i] = _wc[n][i];
          //   _wl_tmp[n][i] = _wc[n][i];
          //
          // } else {
          //
          //   dm_abs = ABS(dm);
          //   dp_abs = ABS(dp);
          //   if (dm_abs >= 2*dp_abs) dm = -2*dp;
          //   else if (dp_abs >= 2*dm_abs) dp = -2*dm;
          //
          //   _wr_tmp[n][i] = w[2][n][i] + dm;
          //   _wl_tmp[n][i] = w[2][n][i] + dp;
          // }

        }
      } // end of loops over i and n

    }

    // y- or z-coordinate

    else {

      for (ints n=0; n<NMODES; ++n) {
#pragma omp simd simdlen(SIMD_WIDTH)
        for (ints i=i1; i<=i2; ++i) {

          real wm2 = _wm2[n][i];
          real wm1 = _wm1[n][i];
          real wc  =  _wc[n][i];
          real wp1 = _wp1[n][i];
          real wp2 = _wp2[n][i];

          real hpr = hp_ratio[n][jk];
          real hmr = hm_ratio[n][jk];

          real wmh = ( (cm[1][jk] * wm1 + cm[2][jk] * wc)
                     + (cm[0][jk] * wm2 + cm[3][jk] * wp1) );

          real wph = ( (cp[1][jk] *  wc + cp[2][jk] * wp1)
                     + (cp[0][jk] * wm1 + cp[3][jk] * wp2) );

          wmh = MIN(wmh, MAX(wc,wm1));
          wmh = MAX(wmh, MIN(wc,wm1));

          wph = MIN(wph, MAX(wc,wp1));
          wph = MAX(wph, MIN(wc,wp1));

          real dp = wph-wc;
          real dm = wc-wmh;

          real dp_abs = FABS(dp);
          real dm_abs = FABS(dm);

          if (dp*dm<=0.) {
            wph=wc;
            wmh=wc;
          }
          else {
            if (dm_abs >= hpr*dp_abs)
              wmh = wc - hpr*dp;
            if (dp_abs >= hmr*dm_abs)
              wph = wc + hmr*dm;
          }

          _wr_tmp[n][i] = wmh;
          _wl_tmp[n][i] = wph;

        }
      } // end of loops over i and n

    } // end of if x- or y-coordinate

  } // end of is_flat_uni


  if (char_proj) {
    char2prim_1(_wl_tmp, _wc0, i1,i2, gam);
    char2prim_1(_wr_tmp, _wc0, i1,i2, gam);
  }

  for (ints n=0; n<NMODES; ++n) {
#pragma omp simd
    for (ints i=i1; i<=i2; ++i) {
      _wr[n][i]    = _wr_tmp[n][i];
      _wl[n][i+ip] = _wl_tmp[n][i];
    }
  }

}
