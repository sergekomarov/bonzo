#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void move(PrtData *pd, real ****fcoup,
          PrtProp *pp, GridCoord *gc,
          real dt, real sol, real qomc, real rhocr, int ppc) {

  real dth_sol = 0.5*dt*sol;
  real eta = 0.5*dt*qomc/sol;

  // real sol_dt = sol/dt;
  real cf = sol/dt * rhocr/ppc;

  int const NINTERP=2;
  int const DIW=NINTERP+1;

#if D2D
  int const DJW=NINTERP+1;
#else
  int const DJW=1;
#endif

#if D3D
  int const DKW=NINTERP+1;
#else
  int const DKW=1;
#endif

#if D3D
  int const NW=27;
  int const NW2=24;
#elif D2D
  int const NW=9;
  int const NW2=8;
#else
  int const NW=4;
  int const NW2=4;
#endif

  int ngridx=gc.Ntot[0], ngridy=gc.Ntot[1], ngridz=gc.Ntot[2], ng=gc.ng;
  real dl_inv[3];
  dl_inv[0] = gc.dlf_inv[0];
  dl_inv[1] = gc.dlf_inv[1];
  dl_inv[2] = gc.dlf_inv[2];

  ints nprt = pp.Np;

  real wgt[3][3][3];

  real fxv[SIMD_WIDTH] __attribute__((aligned(CACHELINE_BYTES)));
  real fyv[SIMD_WIDTH] __attribute__((aligned(CACHELINE_BYTES)));
  real fzv[SIMD_WIDTH] __attribute__((aligned(CACHELINE_BYTES)));
  real dedtv[SIMD_WIDTH] __attribute__((aligned(CACHELINE_BYTES)));

  int wgtv[NW][SIMD_WIDTH] __attribute__((aligned(CACHELINE_BYTES)));
  int ibv[SIMD_WIDTH] __attribute__((aligned(CACHELINE_BYTES)));
  int jbv[SIMD_WIDTH] __attribute__((aligned(CACHELINE_BYTES)));
  int kbv[SIMD_WIDTH] __attribute__((aligned(CACHELINE_BYTES)));

  real *fcoupv = (real*)calloc(4*NW2*ngridx*ngridy*ngridz, sizeof(real));


  for (ints n=0; n<nprt; n+=SIMD_WIDTH) {

    int rem = nprt - n;
    real *restrict xn = pd.x + n;
    real *restrict yn = pd.y + n;
    real *restrict zn = pd.z + n;
    real *restrict un = pd.u + n;
    real *restrict vn = pd.v + n;
    real *restrict wn = pd.w + n;
    real *restrict gn = pd.g + n;

#pragma omp simd aligned(xn,yn,zn,un,vn,wn,gn:64) simdlen(SIMD_WIDTH) private(wgt)
    for (int nn=0; nn<IMIN(SIMD_WIDTH,rem); nn++) {

      int ib=0, jb=0, kb=0;

      getweight2(wgt, &ib,&jb,&kb, xhv[nn],yhv[nn],zhv[nn], dl_inv, ng);

      for (int i=0; i<NW; i++)
        wgtv[i][nn] = wgt+i;

      ibv[nn] = ib;
      jbv[nn] = jb;
      kbv[nn] = kb;

    }

    // add deposits to temporary vectorized 1D array fcoupv

    for (int nn=0; nn<IMIN(SIMD_WIDTH,rem); nn++) {

      real dedt = dedtv[nn];
      real fx = fxv[n];
      real fy = fyv[n];
      real fz = fzv[n];

      int ind = ibv[nn] + jbv[nn]*ngridx + kbv[nn]*ngridx*ngridy;
      real *fcoupvn = fcoupv + 4*NW2*ind;

#pragma omp simd aligned(fcoupvn) simdlen(IMIN(SIMD_WIDTH,NW2))
      for (int i=0; i<NW2; i++) {

        real wgti = wgtv[i][nn];
        fcoupvn[i] += wgti*dedt;
        fcoupvn[i+  NW2] += wgti*fx;
        fcoupvn[i+2*NW2] += wgti*fy;
        fcoupvn[i+3*NW2] += wgti*fz;

      }

#if D3D
      real wgt24 = wgtv[24][nn];
      real wgt25 = wgtv[25][nn];
      real wgt26 = wgtv[26][nn];

      int ind24 = 96*(ind +     2*ngridx + 2*ngridx*ngridy);
      int ind25 = 96*(ind + 1 + 2*ngridx + 2*ngridx*ngridy);
      int ind26 = 96*(ind + 2 + 2*ngridx + 2*ngridx*ngridy);

      fcoupv[ind24] += wgt24*dedt;
      fcoupv[ind24+24] += wgt24*fx;
      fcoupv[ind24+48] += wgt24*fy;
      fcoupv[ind24+72] += wgt24*fz;

      fcoupv[ind25] += wgt25*dedt;
      fcoupv[ind25+24] += wgt25*fx;
      fcoupv[ind25+48] += wgt25*fy;
      fcoupv[ind25+72] += wgt25*fz;

      fcoupv[ind26] += wgt26*dedt;
      fcoupv[ind26+24] += wgt26*fx;
      fcoupv[ind26+48] += wgt26*fy;
      fcoupv[ind26+72] += wgt26*fz;
#elif D2D
      real wgt8 = wgtv[8][nn];
      int ind8 = 32*(ind + 2 + 2*ngridx);
      fcoupv[ind8] += wgt8*dedt;
      fcoupv[ind8+ 8] += wgt8*fx;
      fcoupv[ind8+16] += wgt8*fy;
      fcoupv[ind8+32] += wgt8*fz;
#endif

    }

  }

  // add deposits from temporary array fcoupv to the main particle feedback 3D array

  for (int k=0; k<ngridz-2; ++k) {
    for (int j=0; j<ngridy-2; ++j) {
#pragma omp simd simdlen(SIMD_WIDTH)
      for (int i=0; i<ngridx-2; ++i) {

        int ind = NW2 * (i + j*ngridx + k*ngridx*ngridy);

        for (int n=0; n<4; ++n) {

          for (int k0=0; k0<DKW; ++k0) {
            for (int j0=0; j0<DJW; ++j0) {
              for (int i0=0; i0<DIW; ++i0) {

                int m = i0 + j0*DIW + k0*DIW*DJW;

                if (m<NW2) fcoup[n][k+k0][j+j0][i+i0] += fcoupv[ind+m];
              }
            }
          }

          ind += NW2;

        }

      }
    }
  }

}



// ===================================================================

inline void getweight2(real ***wgt, ints *ib, ints *jb, ints *kb,
                       real x, real y, real z, real *dl_inv, int ng) {

  real ir,jr,kr;
  real a,d;
  real wx[3];
  real wy[3];
  real wz[3];

  real x1 = x-0.5;
  a = x1 * dl_inv[0];
  ir = FLOOR(a);
  d = a - ir;
  ib[0] = (int)ir - 1 + ng;
  wx[0] = 0.5*SQR(1.-d);
  wx[1] = 0.75 - SQR(d-0.5);
  wx[2] = 0.5*SQR(d);

#if D2D
  real y1 = y-0.5;
  a = y1 * dl_inv[1];
  jr = FLOOR(a);
  d = a - jr;
  jb[0] = (int)jr - 1 + ng;
  wy[0] = 0.5*SQR(1.-d);
  wy[1] = 0.75 - SQR(d-0.5);
  wy[2] = 0.5*SQR(d);
#else
  jb[0]=0;
  wy[0]=1.;
  wy[1]=0.;
  wy[2]=0.;
#endif

#if D3D
  real z1 = z-0.5;
  a = z1 * dl_inv[2];
  kr = FLOOR(a);
  d = a - kr;
  kb[0] = (int)kr - 1 + ng;
  wz[0] = 0.5*SQR(1.-d);
  wz[1] = 0.75 - SQR(d-0.5);
  wz[2] = 0.5*SQR(d);
#else
  kb[0]=0;
  wz[0]=1.;
  wz[1]=0.;
  wz[2]=0.;
#endif

  for (int k=0; k<=3; ++k)
    for (int j=0; j<=3; ++j)
      for (int i=0; i<=3; ++i)
        wgt[k][j][i] = wx[i] * wy[j] * wz[k];

}
