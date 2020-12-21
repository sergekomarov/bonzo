#ifndef COORDINATES_H
#define COORDINATES_H

#include <stdint.h>
#include "../defs.h"

typedef enum {
  CS_UNI,
  CS_LOG,
  CS_USR
} CoordScale;

typedef enum {
  CG_CAR,
  CG_CYL,
  CG_SPH
} CoordGeom;


typedef struct {

  // Cells and indices.

  int Nact[3];          // numbers of active cells cells on local grid
  int Ntot[3];          // numbers of cells including ghosts
  int Nact_glob[3];     // active cells in full domain
  int Ntot_glob[3];     // all cells in full domain
  int ng;               // number of ghost cells
  int i1,i2;            // min and max indices of active cells on local grid
  int j1,j2;
  int k1,k2;

  // Coordinates.

  real lmin[3];          // coordinates of left border of global domain
  real lmax[3];          // right border of global domain

  real **lf;             // coordinates of cell faces
  real **lv;             // volume coordinates

  // cell spacings
  real **dlf;            // between cell faces
  real **dlv;            // between cell centers
  real **dlf_inv;        // inverse spacings
  real **dlv_inv;

  CoordGeom  geom;       // coordinate geometry
  CoordScale scale[3];   // scale of coordinate axes

  int interp_order       // order of interpolation at cell faces (Mignone 2014)

  // MPI-related data.

  int rank;             // MPI rank of the grid
  int pos[3];           // 3D index of the grid on the current processor
  int size[3];          // numbers of MPI blocks (grids) in each directions
  int size_tot;         // total number of blocks
  int ***ranks;         // 3D array of grid ranks
  int nbr_ranks[3][2];  // ranks of neighboring grids nbr_ids[axis,L(0)/R(1)]

  // scale factors e.g. to calculate gradients
  real *syxf;
  real *syxv;
  real *szxf;
  real *szxv;
  real *szyf;
  real *szyv;

  // coefficients used in parabolic reconstruction (Mignone paper)
  real **hp_ratio;
  real **hm_ratio;

  // interpolation coefficients (Mignone paper)
  real ***cm;
  real ***cp;

  // Auxilary geometric coefficients to reduce amount of calculations:

  // used to calculate cell volumes, areas, and lengths
  real *rinv_mean;
  real *d2r;
  real *d3r;
  real *sin_thf;
  real *sin_thv;
  real *dcos_thf;

  // used to calculate geometric source terms
  real *src_coeff1;
  real **src_coeff2;

  // temporary arrays to calculate Laplacians of grid arrays
  real **lapl_tmp_xy1;
  real **lapl_tmp_xy2;

  // //scratch arrays used by reconstruction routines
  // real ***rcn_w;
  // real **rcn_wl;
  // real **rcn_wr;

} GridCoord;


extern void lapl_perp1(real *ajm1, real *aj0, real *ajp1, real *ajp2,
                       real *tmp1, real *tmp2, real c, int nx);
extern void lapl_perp2(real *ajm1, real *aj0, real *ajp1, real c, int nx);
extern void copy1d(real *a, real *b, int n);

// void add_laplacian(real ***a, GridCoord *gc, real**,real**, real c)

#endif
