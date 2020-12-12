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

  // scale factors to use e.g. to calculate gradients
  real *syxf;
  real *syxv;
  real *szxf;
  real *szxv;
  real *szyf;
  real *szyv;

  // coefficients used in parabolic reconstruction (Mignone paper)
  real **hp_ratio;
  real **hm_ratio;

  // interpolation coefficients
  real **cm;
  real **cp;

  // MPI-related data.

  int rank;             // MPI rank of the grid
  int pos[3];           // 3D index of the grid on the current processor
  int size[3];          // number of MPI blocks (grids) in all directions
  int size_tot;         // total number of blocks
  int ***ranks;         // 3D array of grid ranks
  int nbr_ranks[3][2];  // ranks of neighboring grids nbr_ids[axis,L(0)/R(1)]

  // AUXILARY GEOMETRIC COEFFICIENTS:

  // used to calculate cell volumes, areas, and lengths
  real *rinv_mean;
  real *d2r;
  real *d3r;
  real *sin_thf;
  real *sin_thv;
  real *dcos_thf;

  // calculate geometric source terms
  real *src_coeff1;
  real **src_coeff2;

  // temporary arrays to calculate Laplacians of grid arrays
  real **lapl_tmp_xy1;
  real **lapl_tmp_xy2;

} GridCoord;


void add_laplacian(GridCoord *gc, real c, real ***a)
