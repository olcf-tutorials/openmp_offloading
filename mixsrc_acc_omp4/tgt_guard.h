#ifdef _OL_OMP_
#define _OMPTGT_(dirstring) $omp dirstring
#define _OMPTGTc(dirstring) $omp_##dirstring
#else
#define _OMPTGT_(dirstring) OpenMP_target_disabled
#define _OMPTGTc(dirstring) OpenMP_target_disabled
#endif

#ifdef _OL_ACC_
#define _ACCTGT_(dirstring) $acc dirstring
#define _ACCTGTc(dirstring) $acc_##dirstring
#else
#define _ACCTGT_(dirstring) OpenACC_target_disabled
#define _ACCTGTc(dirstring) OpenACC_target_disabled
#endif
