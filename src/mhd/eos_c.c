#include <stdio.h>
#include <stdlib.h>
#include "eos_c.h"

// Variable transformations done on vectors.

// inline functions are defined in eos_c.h

void cons2prim_1(real **w1, real **u1, int i1, int i2, real gam) {

  real w[NMODE];
  real u[NMODE];

#pragma omp simd private(u, w)
  for (int i=i1; i<=i2; ++i) {

    for (int n=0; n<NMODE; ++n)
      u[n] = u1[n][i];

    cons2prim(w,u,gam);

    for (int n=0; n<NMODE; ++n)
      w1[n][i] = w[n];

  }
}


// ------------------------------------------------------------------

void prim2cons_1(real **u1, real **w1, int i1, int i2, real gam) {

  real w[NMODE];
  real u[NMODE];

#pragma omp simd private(u, w)
  for (int i=i1; i<=i2; ++i) {

    for (int n=0; n<NMODE; ++n)
      w[n] = w1[n][i];

    prim2cons(u,w,gam);

    for (int n=0; n<NMODE; ++n)
      u1[n][i] = u[n];

  }
}


// ----------------------------------------------------------------

void prim2char_1(real **vc1, real **w1, int i1, int i2, real gam) {

  real w[NMODE];
  real vc[NMODE];

#pragma omp simd simdlen(SIMD_WIDTH) private(w,vc)
  for (int i=i1; i<=i2; ++i) {

    for (int n=0; n<NMODE; ++n)
      w[n] = w1[n][i];
      vc[n] = vc1[n][i];

    prim2char(vc,w,gam);

    for (int n=0; n<NMODE; ++n)
      vc1[n][i] = vc[n];

  }
}


// -----------------------------------------------------------------

void char2prim_1(real **vc1, real **w1, int i1, int i2, real gam) {

  real w[NMODE];
  real vc[NMODE];

#pragma omp simd simdlen(SIMD_WIDTH) private(w,vc)
  for (int i=i1; i<=i2; ++i) {

    for (int n=0; n<NMODE; ++n)
      w[n] = w1[n][i];
      vc[n] = vc1[n][i];

    char2prim(vc,w,gam);

    for (int n=0; n<NMODE; ++n)
      vc1[n][i] = vc[n];

  }
}
