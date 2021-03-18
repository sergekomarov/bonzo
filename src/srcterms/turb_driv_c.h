#include <stddef.h>
#include "../defs_c.h"
#include "../coordinates/coord_c.h"

extern void advance_driv_force_i(real *fdrivx, real *fdrivy, real *fdrivz,
                                 real *c1, real *c2,
                                 real *xi, real y, real z, int is, int ie,
                                 real kx0, real ky0, real kz0,
                                 real dt_tau, real f1, int nmod)
// extern void advance_driv_force_c(real****, GridCoord*, int*, real,real,int, real)
