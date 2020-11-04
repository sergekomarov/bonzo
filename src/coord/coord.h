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
  int ng;                // number of ghost cells
  ints i1,i2;            // min and max indices of active cells on local grid
  ints j1,j2;
  ints k1,k2;

  // Coordinates.

  real lmin[3];          // coordinates of left border of global domain
  real lmax[3];          // right border of global domain
  //real l1[3];          // left border of local grid
  //real l2[3];          // right border of local grid

#if !PIC

  real **lf;             // coordinates of cell faces
  real **lv;             // volume coordinates

  // Cell spacings.
  real **dlf;            // between cell faces
  real **dlv;            // between cell centers
  real **dlf_inv;
  real **dlv_inv;        // inverse spacings

  // Auxilary coefficients for non-cartesian coordinates.
  real *rinv_mean;
  real *src_coeff1;
  real **src_coeff2;
  real **hp_ratio;
  real **hm_ratio;

  real ***dv_inv;        // inverse cell volumes
  real ****da;           // areas of cell faces
  real ****ds;           // lengths of cell edges

  CoordGeom  coord_geom;      // coordinate geometry
  CoordScale coord_scale[3];  // scale of coordinate axes

#endif

  // MPI block IDs
  ints rank;             // MPI rank of the grid
  ints pos[3];           // 3D index of the grid on the current processor


#if MPI

  // MPI-related data.

  ints size[3];          // number of MPI blocks (grids) in x,y,z directions
  ints size_tot;         // total number of blocks

  ints ***ranks;         // 3D array of grid ranks

  ints nbr_ranks[3][2];  // ranks of neighboring grids
  // nbr_ids[axis,L(0)/R(1)]

#endif

} GridCoord;
