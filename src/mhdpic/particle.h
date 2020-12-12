#include "../defs.h"

// Properties of a particle specie.

typedef struct {
  real qm;     // charge-to-mass ratio
  long Np;     // number of particles
} SpcProp;

// General particle properties.

typedef struct {

  int ppc;            // number of particles per cell
  int Nprop;          // number of particle properties
  long Npmax;         // size of particle array
  long Np;            // number of active particles of all species
  int Ns;             // number of species
  SpcProp *spc_props; // properties of particle species

//#if MHDPIC
  //real c;           // effective speed of light
  //real q_mc;        // charge-to-mass ratio of CRs relative to thermal ions
  //real rho_cr;      // CR density
//#endif

} PrtProp;


// Arrays of particle properties.

typedef struct {

  // coordinates
  real *x;
  real *y;
  real *z;

  // four-velocities
  real *u;
  real *v;
  real *w;

  real *g;    // relativistic gamma

  real *m;    // mass
  int *spc;   // specie
  long *id;   // particle ID

} PrtData;
