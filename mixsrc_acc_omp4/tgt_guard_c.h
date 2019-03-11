#ifndef _TGT_GUARD_C_
#define _TGT_GUARD_C_

#ifdef _OL_ACC_
#define _ACCTGT_(x) _Pragma(acc #x)
#else
#define _ACCTGT_(x) {}
#endif

#ifdef _OL_OMP_
#define _OMPTGT_(x) _Pragma(omp #x)
#else
#define _OMPTGT_(x) {}
#endif

#endif
