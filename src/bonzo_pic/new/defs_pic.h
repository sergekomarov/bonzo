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

#define NVARS 9
#define NPRT_PROP 10
// indices of electromagnetic fields and particle currents
enum {EX=0,EY,EZ, BX,BY,BZ, JX,JY,JZ};

// names of coordinate axes
enum {XAX,YAX,ZAX};


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
