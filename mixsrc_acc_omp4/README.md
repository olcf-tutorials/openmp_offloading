# Mixiang OpenACC and OpenMP4.5 offloading directives in the same source files
by Lixiang Luo, IBM Center of Excellence @ ORNL

This tutorial offers a guide to safely write OpenACC and OpenMP4.5 offloading directives in the same source files.

OpenMP and OpenACC are both popular choices for directive-based GPU programming. On CORAL systems, OpenMP target offloading is supported by XL, and OpenACC by PGI. The directives are similar enough to be placed in the same source file. On the other hand, source codes often include host-side OpenMP directives (supported by all compilers on CORAL), regardless of the target offloaing paradigm. A little extra care is necessary so that all the directives can live peacefully with each other, but the effort is minimal.

Note that this tutorial is NOT about using OpenMP and OpenACC target offloading in the same process (compiler and runtime interoperability), which is an interesting but much more complicated subject.

## Fortran
In Fortran, this can be done using macros, preferably inside comments to be more distinguishable from normal codes, like this:
```Fortran
!_OMPTGT_(target teams distribute private(vt))
!_ACCTGT_(parallel loop gang private(vt))
do ie=1,Nelt
  !_OMPTGT_(parallel do collapse(2))
  !_ACCTGT_(loop vector collapse(2))
  do j=1,Np ; do i=1,Np
    vt(i,j) = a(i,j,ie) * v(j,ie)
  end do ; end do
  !_OMPTGT_(parallel do)
  !_ACCTGT_(loop vector)
  do i=1,Np
    v(i) = sum(vt(i,:))
  end do
end do
```
where the guarding macros _OMPTGT_ and _ACCTGT_ are defined as
```
#ifdef _OL_OMP_
#define _OMPTGT_(dirstring) $omp dirstring
#else
#define _OMPTGT_(dirstring) OpenMP_target_disabled
#endif
```
and similarly, _ACCTGT_:
```
#ifdef _OL_ACC_
#define _ACCTGT_(dirstring) $acc dirstring
#else
#define _ACCTGT_(dirstring) OpenACC_target_disabled
#endif
```
To activate _OMPTGT_ during compilation:
```
$ xlf_r -D_OL_OMP_ -qsmp=omp -qoffload foo.F90
```
and similarly, for _ACCTGT_:
```
$ pgfortran -D_OL_ACC_ -ta=tesla:cc70 foo.F90
```
Note the ".F90" extension, which prompts the compiler to execute preprocessor. If your source extension is in lowercase, then "-qpreprocess" (XL) or "-Mpreprocess" (PGI) should be used to force preprocessing.

This approach allows flexible mixing of host-side OpenMP directives with either OpenACC or OpenMP target offloading directives. Host-side OpenMP directives are written plainly, so that they are not affected by the offloading directives at all. However, if situations demand host-side OpenMP directives to be guarded (such as OpenMP/OpenACC API calls for target offloading in multi-threaded codes), the guarding macros (or similarly defined macros) can also be used.

Multi-line directives in free-form Fortran require an ampersand OUTSIDE the macro, as the preprocessor does not parse beyond the ampersand. For example:
```Fortran
!_OMPTGT_(parallel do) &
!_OMPTGT_(collapse(2))
do i=...
  do j=...
```

In fixed-form Fortran, line continuation requires an extra macro:
```
#define _OMPTGT_(dirstring) $omp dirstring
#define _OMPTGTc(dirstring) $omp+dirstring
```

Compiler directives in fixed-form Fortran does not use an ampersand. Instead:
```
!_OMPTGT_(parallel do)
!_OMPTGTc(collapse(2))
      do i=...
        do j=...
```

## C/C++
Compiler directives in C/C++, including OpenACC and OpenMP directives, cannot be generated as a result of macro expansions. Instead, we use C99 _Pragma() preprocesing operator to achieve the same effects. We define the guarding macros as
```
#ifdef _OL_OMP_
#define _OMPTGT_(x) _Pragma(omp #x)
#else
#define _OMPTGT_(x) {}
#endif

#ifdef _OL_ACC_
#define _ACCTGT_(x) _Pragma(acc #x)
#else
#define _ACCTGT_(x) {}
#endif
```

To use the macros:
```C++
_OMPTGT_(target teams distribute private(vt))
_ACCTGT_(parallel loop gang private(vt))
for(ie=0;i<Nelt;i++)
{
_OMPTGT_(parallel do collapse(2))
_ACCTGT_(loop vector collapse(2))
  for(j=0;j<Np;j++) for(i=0;i<Np;i++)
    vt(i+Np*j) = a(i+Np*(j+Np*ie)) * v(j+Np*ie);
_OMPTGT_(parallel do)
_ACCTGT_(loop vector)
  for(i=0;i<Np;i++)
  {
    v(i) = 0.;
    for(j=0;j<Np;j++) v(i)+=vt(i+Np*j));
  }
}
```
To activate _OMPTGT_ during compilation:
```
$ xlc_r -D_OL_OMP_ -qsmp=omp -qoffload foo.c
```
and similarly, for _ACCTGT_:

```
$ pgcc -D_OL_ACC_ -ta=tesla:cc70 foo.c
```

### Bonus for Vim users
To distinguish the C/C++ guarding macros from normal source codes in Vim, write a syntax extension file named "pragmaext.vim", with the content
```Vim
"highlight cComment ctermfg=Green guifg=Green
syn region cPreProc start="^\s*_\(OMP\|ACC\)TGT_" skip="\\$" end="$" keepend contains=ALLBUT,@cPreProcGroup,@Spell
```
and place it in both "$HOME/.vim/after/syntax/c" and "$HOME/.vim/after/syntax/cpp":
```
$ mkdir -p $HOME/.vim/after/syntax/c $HOME/.vim/after/syntax/cpp
$ cp pragmaext.vim $HOME/.vim/after/syntax/c
$ cp pragmaext.vim $HOME/.vim/after/syntax/cpp
```
With this extension, _OMPTGT_ and _ACCTGT_ macros will have the same color as other compiler directives.
