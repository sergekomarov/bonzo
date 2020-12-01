#include <stdint.h>
#include <math.h>

#ifndef DEFINITIONS_H
#define DEFINITIONS_H


// fixed-length integer type
typedef intptr_t ints;
// floating-point type depending on precision
#if SPREC
typedef float real;
#else
typedef double real;
#endif


#if defined(__AVX512F__)
#define SIMD_WIDTH 8
#elif defined(__AVX__)
#define SIMD_WIDTH 4
#elif defined(__AVX2__)
#define SIMD_WIDTH 4
#elif defined(__SSE2__)
#define SIMD_WIDTH 2
#else
#define SIMD_WIDTH 4
#endif


//===========================================================

// number of wave modes in the Riemann problem

#if (!MFIELD && !TWOTEMP)
#define NMODE 6
#endif

#if (!MFIELD && TWOTEMP)
#define NMODE 7
#endif

#if (MFIELD && !CGL && !TWOTEMP)
#define NMODE 9
#endif

#if (MFIELD && ((CGL && !TWOTEMP) || (!CGL && TWOTEMP)))
#define NMODE 10
#endif

#if (MFIELD && CGL && TWOTEMP)
#define NMODE 11
#endif


// indices of fields in the arrays of conserved/primitive variables

/*
RHO: mass density, V: velocities, M: momenta
EN: total energy density
PPD: perpendicular ion pressure
LA: rho * log(p_perp/p_par * rho^2 / B^3)
PR: mean ion pressure = (p_par + 2*p_perp)/3
SE: electron entropy log(pe/rhoe^gamma)
PSC: passive scalar
BC: cell-centered magnetic field
B:  face-centered magnetic field
*/

// conserved variables
enum {RHO=0, MX,MY,MZ, EN};
// primitive variables
enum {VX=1,VY,VZ, PR};

#if MFIELD

#if MHDPIC
#define NFIELD NMODE+7
#define NPRT_PROP 10
#else
#define NFIELD NMODE+3
#endif

enum {PSC=NMODE-4};
enum {BX=NMODE-3, BY=NMODE-2, BZ=NMODE-1};
// BC variable identifiers for cell- and face-centered magnetic field
enum {BXC=BX, BYC=BY, BZC=BZ};
enum {BXF=BZ+1, BYF=BZ+2, BZF=BZ+3};

#else

#define NFIELD NMODE
enum {PSC=NMODE-1};
// hack for boundary conditions to avoid setting BC for magnetic fields
enum {BXC=990, BYC, BZC};
enum {BXF=993, BYF, BZF};

#endif
// if MFIELD

// BC variable identifier for particle feedback force
enum {FC0=BZF+1, FCX=BZF+2, FCY=BZF+3, FCZ=BZF+4};

#if CGL
enum {LA=EN+1};
enum {PPD=PR+1};
#endif

#if TWOTEMP
enum {SE=PSC-1};
enum {PE=PSC-1};
#endif

// indices of STS coefficients
enum {MU=0,NU,MUT,GAMT};

// names of coordinate axes
enum {XAX,YAX,ZAX};

// isotropic/anisotropic electron thermal conduction
enum {TC_ISO, TC_ANISO};


// ====================================================================

#define B_PI 3.14159265
#define SQR(x) ((x) * (x))
#define MIN(a,b) ( ((a) < (b)) ? (a) : (b) )
#define MAX(a,b) ( ((a) > (b)) ? (a) : (b) )
// #define SIGN(a)  ( ((a) < 0.) ? -1. : (((a) > 0.) ? 1. : 0.) )

#if SPREC

//#define FABS(a)  ( fabsf(a) )
//#define FMIN(a) ( fminf(a) )
//#define FMAX(a) ( fmaxf(a) )
#define FABS(a)  ( ((a) < 0.0f) ? -(a) : (a) )
#define POW(a)  ( powf(a) )
#define POW(a)  ( logf(a) )
#define EXP(a)  ( expf(a) )
#define SQRT(a) ( sqrtf(a) )
//#define SIGN(a) ( ( 0.0f < (a) ) - ((a) < 0.0f) )
#define FSIGN(a)  ( ((a) < 0.0f) ? -1.0f : (((a) > 0.0f) ? 1.0f : 0.0f) )

#else

//#define FABS(a)  ( fabs(a) )
//#define FMIN(a) ( fmin(a) )
//#define FMAX(a) ( fmax(a) )
#define FABS(a)  ( ((a) < 0.) ? -(a) : (a) )
#define POW(a)  ( pow(a) )
#define LOG(a)  ( log(a) )
#define EXP(a)  ( exp(a) )
#define SQRT(a) ( sqrt(a) )
//#define SIGN(a) ( ( 0.0 < (a) ) - ((a) < 0.0) )
#define FSIGN(a)  ( ((a) < 0.0) ? -1. : (((a) > 0.0) ? 1. : 0.0) )

#endif


#endif
