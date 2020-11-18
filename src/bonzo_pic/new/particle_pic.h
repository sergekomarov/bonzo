#include "../defs.h"

// Properties of a particle specie.

typedef struct {
  real qm;     // charge-to-mass ratio
  ints Np;     // number of particles
} SpcProp;

// General particle properties.

typedef struct {

  ints ppc;           // number of particles per cell
  ints Nprop;         // number of particle properties
  ints Npmax;         // length of particle array
  ints Np;            // number of active particles of all species
  ints Ns;            // number of species
  SpcProp *spc_props; // properties of different species

  real c;           // speed of light <-> Courant number
  real me;          // electron mass
  real mime;        // ion-electron mass ratio
  // real c_ompe;   // electron skin depth

} ParticleProp;


// Structure containing arrays of particle properties.

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
  ints *spc;  // specie
  ints *id;   // particle ID

} ParticleData;
