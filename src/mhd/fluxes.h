#include "../defs.h"

extern void hll_flux(real **flux, real **wl, real **wr, real *bx,
             int i1, int i2, real gam);

#if !MFIELD
extern void hllc_flux(real **flux, real **wl, real **wr, real *bx,
              int i1, int i2, real gam);
#endif

extern void hllt_flux(real **flux, real **wl, real **wr, real *bx,
              int i1, int i2, real gam);

#if MFIELD
extern void hlld_flux(real **flux, real **wl, real **wr, real *bx,
              int i1, int i2, real gam);
#endif

#if CGL
extern void hlla_flux(real **flux, real **wl, real **wr, real *bx,
              int i1, int i2, real gam);
#endif
