#include <stddef.h>
#include "../defs.h"

extern real MMlim(real a, real b) __attribute__((always_inline));
extern real MClim(real a, real b) __attribute__((always_inline));
extern real VLlim(real a, real b) __attribute__((always_inline));
extern real L2lim(real a, real b) __attribute__((always_inline));

extern void reconstr_const(real **wl, real **wr, real ***w, real ***scr,
                     GridCoord *gc,  int ax,
                     ints i1, ints i2, ints j, ints k,
                     int char_proj, real gam);

extern void reconstr_linear(real **wl, real **wr, real ***w, real ***scr,
                     GridCoord *gc, int ax,
                     ints i1, ints i2, ints j, ints k,
                     int char_proj, real gam);

// extern void reconstr_parab0(real **wl, real **wr, real ***w, real ***scr,
//                      GridCoord *gc, int ax,
//                      ints i1, ints i2, ints j, ints k,
//                      int char_proj, real gam);

extern void reconstr_parab(real **wl, real **wr, real ***w, real ***scr,
                     GridCoord *gc, int ax,
                     ints i1, ints i2, ints j, ints k,
                     int char_proj, real gam);

extern void reconstr_weno(real **wl, real **wr, real ***w, real ***scr,
                     GridCoord *gc, int ax,
                     ints i1, ints i2, ints j, ints k,
                     int char_proj, real gam);


// ========================================================

// Various limiters.


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


real inline l2_lim(real a, real b) {

  real mn = 0.75*a;
  real mx = 1.333333333*a;
  real c = 0.5*(a+b);

  return (c <= mn) ? mn : ( (c >= mx) ? mx : c );

}
