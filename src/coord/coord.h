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

  // Cells and Indices.

  ints Nact[3];          // numbers of active cells cells on local grid
  ints Ntot[3];          // numbers of cells including ghosts
  ints Nact_glob[3];     // active cells in full domain
  ints Ntot_glob[3];     // all cells in full domain
  ints ng;                // number of ghost cells
  ints i1,i2;            // min and max indices of active cells on local grid
  ints j1,j2;
  ints k1,k2;

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

  CoordGeom  geom;      // coordinate geometry
  CoordScale scale[3];  // scale of coordinate axes

  // auxilary coefficients to calculate cell volumes, areas, and lengths
  real *rinv_mean;
  real *d2r;
  real *d3r;
  real *sin_thf;
  real *sin_thc;
  real *dcos_thf;

  // auxilary coefficients to calculate geometric source terms
  real *src_coeff1;
  real **src_coeff2;

  // coefficients used in parabolic reconstruction (Mignone paper)
  real **hp_ratio;
  real **hm_ratio;

  // interpolation coefficients
  real **cm
  real **cp

  // MPI block IDs
  ints rank;             // MPI rank of the grid
  ints pos[3];           // 3D index of the grid on the current processor

  // MPI-related data.

  ints size[3];          // number of MPI blocks (grids) in x,y,z directions
  ints size_tot;         // total number of blocks

  ints ***ranks;         // 3D array of grid ranks

  ints nbr_ranks[3][2];  // ranks of neighboring grids
  // nbr_ids[axis,L(0)/R(1)]

} GridCoord;
