#include "../defs_c.h"
#include "coord_c.h"

extern void reconstr_const(real **wl, real **wr, real ***w, real ***scr,
                     GridCoord *gc,  int ax,
                     int i1, int i2, int j, int k,
                     int char_proj, real gam);

extern void reconstr_linear(real **wl, real **wr, real ***w, real ***scr,
                     GridCoord *gc, int ax,
                     int i1, int i2, int j, int k,
                     int char_proj, real gam);


// extern void reconstr_parab(real **wl, real **wr, real ***w, real ***scr,
//                      GridCoord *gc, int ax,
//                      int i1, int i2, int j, int k,
//                      int char_proj, real gam);
//
// extern void reconstr_weno(real **wl, real **wr, real ***w, real ***scr,
//                      GridCoord *gc, int ax,
//                      int i1, int i2, int j, int k,
//                      int char_proj, real gam);


// Limiters.

real inline mm_lim(real a, real b) {

  return (a*b <= 0.) ? 0. : ((a>0.) ? MIN(a,b) : MAX(a,b));

}

real inline mc_lim(real a, real b) {

  return mm_lim(2.*mm_lim(a,b), 0.5*(a+b));
}

real inline vl_lim(real a, real b) {

  real ab = a*b;
  return (ab > 0.) ? ( 2.*ab / (a+b) ) : 0.;

}

// real inline l2_lim(real a, real b) {
//
//   real mn = 0.75*a;
//   real mx = 1.333333333*a;
//   real c = 0.5*(a+b);
//
//   return (c <= mn) ? mn : ( (c >= mx) ? mx : c );
//
// }
