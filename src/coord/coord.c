#include <stdio.h>
#include <stdlib.h>
#include "coord.h"

void add_laplacian(real ***a, real c, GridCoord *gc) {

  // Subtract/add the Laplacian of an array without copying.

  int nx = (gc->Ntot)[0];
  int ny = (gc->Ntot)[1];
  int nz = (gc->Ntot)[2];

  real h0 = 1.-2.*c;
  real h1 = c;

  real **tmp1 = gc->lapl_tmp_xy1;
  real **tmp2 = gc->lapl_tmp_xy2;

  for (int k=0; k<nz; ++k) {
    for (int j=0; j<ny; ++j) {

      tmp2[0][0] = a[k][j][0];

#pragma omp simd
      for (int i=1; i<nx-2; i+=2) {

        tmp1[0][0] = h1*(a[k][j][i-1] + a[k][j][i+1]) + h0*a[k][j][i];
        a[k][j][i-1] = tmp2[0][0];

        tmp2[0][0] = h1*(a[k][j][i] + a[k][j][i+2]) + h0*a[k][j][i+1];
        a[k][j][i] = tmp1[0][0];

      }

      i+=2;

      if (i==nx-2)
        a[k][j][i] = h1*(a[k][j][i-1] + a[k][j][i+1]) + h0*a[k][j][i];

      a[k][j][i-1] = tmp2[0][0];

    }
  }

#if D2D

  for (int k=0; k<nz; ++k) {

#pragma omp simd
    for (int i=0; i<nx; ++i)
      tmp2[0][i] = a[k][0][i];

    for (int j=1; j<ny-2; j+=2) {
#pragma omp simd
      for (int i=0; i<nx; ++i) {

        real am1 = a[k][j-1][i];
        real a0  = a[k][j  ][i];
        real ap1 = a[k][j+1][i];
        real ap2 = a[k][j+2][i];

        tmp1[0][i] = h1 * (am1 + ap1) + h0 * a0;
        a[k][j-1][i] = tmp2[0][i];

        tmp2[0][i] = h1 * (a0  + ap2) + h0 * ap1;
        a[k][j][i] = tmp1[0][i];

      }
    }

    j+=2;

    if (j==ny-2)
#pragma omp simd
      for (int i=0; i<nx; ++i)
        a[k][j][i] = h1*(a[k][j-1][i] + a[k][j+1][i]) + h0*a[k][j][i];

#pragma omp simd
    for (int i=0; i<nx; ++i)
      a[k][j-1][i] = tmp2[0][i];

  }


#endif

#if D3D

  for (int j=0; j<ny; ++j)
    for (int i=0; i<nx; ++i)
      tmp2[j][i] = a[0][j][i];

  for (int k=1; k<nz-2; k+=2) {
    for (int j=0; j<ny; ++j) {
#pragma omp simd
      for (int i=0; i<nx; ++i) {

        real am1 = a[k-1][j][i];
        real a0  = a[k  ][j][i];
        real ap1 = a[k+1][j][i];
        real ap2 = a[k+2][j][i];

        tmp1[j][i] = h1*(am1 + ap1) + h0*a0;
        a[k-1][j][i] = tmp2[j][i];

        tmp2[j][i] = h0*(a0  + ap2) + h0*ap1;
        a[k][j][i] = tmp1[j][i];

      }
    }
  }

  k+=2;

  if (k==nz-2)
    for (int j=0; j<ny; ++j)
#pragma omp simd
      for (int i=0; i<nx; ++i)
        a[k][j][i] = h1*(a[k-1][j][i] + a[k+1][j][i]) + h0*a[k][j][i];

  for (int j=0; j<ny; ++j)
#pragma omp simd
    for (int i=0; i<nx; ++i)
      a[k-1][j][i] = tmp2[j][i];

#endif

}
