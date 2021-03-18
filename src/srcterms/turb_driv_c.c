#include <stdio.h>
#include <stdlib.h>
#include "turb_driv_c.h"

void advance_driv_force_i(real *fdrivx, real *fdrivy, real *fdrivz,
                          real *c1, real *c2,
                          real *xi, real y, real z, int is, int ie,
                          real kx0, real ky0, real kz0,
                          real dt_tau, real f1, int nmod) {

  int nmod2 = 2*nmod+1;
  real nmodf = (real)nmod;

#pragma omp simd simdlen(SIMD_WIDTH)
  for (int i=is; i<=ie; ++i) {

    real x = xi[i];

    real th = -nmodf * (kx0 * x + ky0 * y + kz0 * z);
    real cn = COS(th);
    real sn = SIN(th);

    real cnx = COS(kx0*x);
    real cny = COS(ky0*y);
    real cnz = COS(kz0*z);

    real snx = SIN(kx0*x);
    real sny = SIN(ky0*y);
    real snz = SIN(kz0*z);

    real gx=0.;
    real gy=0.;
    real gz=0.;

    real tz = -nmodf*kz0;

    for (int p=0; p<nmod2; ++p) {

      real ty = -nmodf*ky0;

      real cn00 = cn;
      real sn00 = sn;

      for (int q=0; q<nmod2; ++q) {

        real tx = -nmodf*kx0;

        real cn0 = cn;
        real sn0 = sn;

        for (int s=0; s<nmod2; ++s) {

          int m = 3*(s + nmod2*q + nmod2*nmod2*p);

          real ax = - c1[m]  *sn + c2[m]  *cn;
          real ay = - c1[m+1]*sn + c2[m+1]*cn;
          real az = - c1[m+2]*sn + c2[m+2]*cn;

          gx +=   ty * az - tz * ay;
          gy += - tx * az + tz * ax;
          gz +=   tx * ay - ty * ax;

          tx += kx0;

          real cn_ = cn;
          cn  = cn_ * cnx - sn  * snx;
          sn  = sn  * cnx + cn_ * snx;

        }

        ty += ky0;

        cn = cn0;
        sn = sn0;

        real cn_ = cn;
        cn  = cn_ * cny - sn  * sny;
        sn  = sn  * cny + cn_ * sny;

      }

      tz += kz0;

      cn = cn00;
      sn = sn00;

      real cn_ = cn;
      cn  = cn_ * cnz - sn  * snz;
      sn  = sn  * cnz + cn_ * snz;
    }

    fdrivx[i] = (1.-dt_tau) * fdrivx[i] + f1 * gx;
    fdrivy[i] = (1.-dt_tau) * fdrivy[i] + f1 * gy;
    fdrivz[i] = (1.-dt_tau) * fdrivz[i] + f1 * gz;

  }

}



// void advance_driv_force_c(real ****fdriv, GridCoord*, int *lims,
//                           real f0, real tau, int nmod, real dt) {
//
//   // Set random mode amplitudes c1 and c2.
//
//   int nmod2 = 2*nmod+1;
//   int nc = 3*nmod2*nmod2*nmod2;
//   double *c1 = (double*)calloc(nc, sizeof(double));
//   double *c2 = (double*)calloc(nc, sizeof(double));
//
//   double w1,w2, norm;
//   int m;
//
//   if (gc.rank==0) {
//
//     // srand((unsigned)(gc.size_tot + gc.rank));
//
//     for (int p=0; p<nmod2; ++p) {
//       for (int q=0; q<nmod2; ++q) {
//         for (int s=0; s<nmod2; ++s) {
//           for (int n=0; n<3; ++n) {
//
//             if (p==nmod || q==nmod || s==nmod)
//               norm = 0.;
//             else
//               norm = 1./(SQR(p-nmod) + SQR(q-nmod) + SQR(s-nmod));
//
//             w1 = (double)rand()/RAND_MAX;
//             w2 = (double)rand()/RAND_MAX;
//
//             if (w1==0.) w1 = 1e-20;
//             m = n + 3*(s + nmod2*q + nmod2*nmod2*p);
//
//             c1[m1] = norm * SQRT(-2 * LOG(w1)) * COS(2*M_PI * w2);
//             c2[m1] = norm * SQRT(-2 * LOG(w1)) * SIN(2*M_PI * w2);
//
//             //printf("%f %f\n", c1[m1],c2[m1]);
//
//           }
//         }
//       }
//     }
//   }
//
// #ifdef MPI
//   MPI_Bcast(c1, nc, MPI_DOUBLE, 0, MPI_COMM_WORLD);
//   MPI_Bcast(c2, nc, MPI_DOUBLE, 0, MPI_COMM_WORLD);
// #endif
//
//   // Advance the driving force by a solenoidal increment.
//
//   int m;
//   real x,y,z;
//   real th, cn00,sn00, cn0,sn0, cn,sn, cn_,sn_, cnx,cny,cnz, snx,sny,snz;
//   real ax,ay,az, gx,gy,gz, tx,ty,tz;
//
//   real two_pi_lx = 2.*M_PI / (gc.lmax[0]-gc.lmin[0]);
//   real two_pi_ly = 2.*M_PI / (gc.lmax[1]-gc.lmin[1]);
//   real two_pi_lz = 2.*M_PI / (gc.lmax[2]-gc.lmin[2]);
//
//   real dt_tau = dt/tau;
//   real f1 = SQRT(dt_tau)/(2.*M_PI)*f0;
//
//   //printf("f1: %f\n", f1);
//
//   // FIX TYPE CASTS, PARALLELIZE over k?
//
//   for (int k=lims[4]; k<=lims[5]; ++k) {
//     z = gc.lv[2][k];
//
//     for (int j=lims[2]; j<=lims[3]; ++j) {
//       y = gc.lv[1][j];
//
// #pragma omp simd simdlen(SIMD_WIDTH)
//       for (int i=lims[0]; i<=lims[1]; ++i) {
//         x = gc.lv[0][i];
//
//         th = -nmod * (two_pi_lx * x + two_pi_ly * y + two_pi_lz * z);
//         cn = COS(th);
//         sn = SIN(th);
//
//         cnx = COS(two_pi_lx*x);
//         cny = COS(two_pi_ly*y);
//         cnz = COS(two_pi_lz*z);
//
//         snx = SIN(two_pi_lx*x);
//         sny = SIN(two_pi_ly*y);
//         snz = SIN(two_pi_lz*z);
//
//         gx=0.;
//         gy=0.;
//         gz=0.;
//
//         tz = -nmod*two_pi_lz;
//
//         for (int p=0; p<nmod2; ++p) {
//
//           ty = -nmod*two_pi_ly;
//
//           cn00 = cn;
//           sn00 = sn;
//
//           for (int q=0; q<nmod2; ++q) {
//
//             tx = -nmod*two_pi_lx;
//
//             cn0 = cn;
//             sn0 = sn;
//
//             for (int s=0; s<nmod2; ++s) {
//
//               m = 3*(s + nmod2*q + nmod2*nmod2*p);
//
//               ax = - c1[m]  *sn + c2[m]  *cn;
//               ay = - c1[m+1]*sn + c2[m+1]*cn;
//               az = - c1[m+2]*sn + c2[m+2]*cn;
//
//               gx = gx + ty * az - tz * ay;
//               gy = gy - tx * az + tz * ax;
//               gz = gz + tx * ay - ty * ax;
//
//               tx = tx + two_pi_Lx;
//
//               cn_ = cn;
//               cn  = cn_ * cnx - sn  * snx;
//               sn  = sn  * cnx + cn_ * snx;
//
//             }
//
//             ty = ty + two_pi_Ly;
//
//             cn = cn0;
//             sn = sn0;
//
//             cn_ = cn;
//             cn  = cn_ * cny - sn  * sny;
//             sn  = sn  * cny + cn_ * sny;
//
//           }
//
//           tz = tz + two_pi_Lz;
//
//           cn = cn00;
//           sn = sn00;
//
//           cn_ = cn;
//           cn  = cn_ * cnz - sn  * snz;
//           sn  = sn  * cnz + cn_ * snz;
//         }
//
//         fdriv[0][k][j][i] = (1.-dt_tau) * fdriv[0][k][j][i] + f1 * gx;
//         fdriv[1][k][j][i] = (1.-dt_tau) * fdriv[1][k][j][i] + f1 * gy;
//         fdriv[2][k][j][i] = (1.-dt_tau) * fdriv[2][k][j][i] + f1 * gz;
//
//       }
//     }
//   }
//
//   free(c1);
//   free(c2);
//
// }
