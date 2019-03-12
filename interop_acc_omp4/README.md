# Interoperability of CUDA, OpenACC and OpenMP4.5 runtimes in the same program

The most widely used mechanism of Unified Virtual Address Space (UVAS) is CUDA Managed Memory, which allows (mostly) transparent access of device memory from the host. An excellent review of UVAS and CUDA managed memory can be found [here](http://on-demand.gputechconf.com/gtc/2018/presentation/s8430-everything-you-need-to-know-about-unified-memory.pdf). A related mechanism, while not considered part of UVAS, is the pinned host memory, which allows zero-copy access of host memory from the device. The two mechanisms allow both GPU and CPU to exchange pointers directly without deep copy, making complex data structures much more accessible in heterogeneous programming.

## A proof of concept

Let us start with a simple code in Fortran
<table style="padding:0pt">
<tbody>
<tr>
<td align="center"><b>main-pgi.F90</b></td>
<td align="center"><b>subomp4-xl.F90</b></td>
</tr>
<tr>
<td valign="top">
<pre>program main_pgi<br><br>implicit none<br><br>integer, parameter :: N=9<br>real*8, dimension(N) :: arr<br>integer :: i<br><br>interface<br>subroutine subomp4(arr,n) bind(c,name="subomp4")<br>real*8, dimension(N), device :: arr<br>integer :: n<br>end subroutine subomp4<br>end interface<br><br>!$acc data copy(arr)<br>!$acc parallel loop vector<br>do i=1,N; arr(i)=i; enddo<br>call subomp4(arr,N)<br>!$acc parallel loop vector<br>do i=1,N; arr(i)=arr(i)+10*i; enddo<br>!$acc end data<br><br>write(*,'(9F5.1)') arr(1:N)<br>end<br><span class="PreProc" style="color: rgb(192, 0, 192);"></span></pre>
</td>
<td valign="top">
<pre>subroutine subomp4(a,n)<br>implicit none<br>real*8, dimension(n) :: a<br>integer :: n<br>integer :: i<br><br>!$omp target teams distribute parallel do is_device_ptr(a)<br>do i=1,n; a(i)=a(i)+0.1*i; enddo<br><br>end<br></pre>
</td>
</tr>
</tbody>
</table>

To compile this code on Summit:
```
$ ml xl
$ xlf_r -qsmp=omp -qoffload -qpic -c subomp4-xl.F90
$ xlf_r -qsmp=omp -qoffload -qmkshrobj -o bin-xl.so subomp4-xl.o
$ ml pgi
$ pgfortran -Mcuda -ta=tesla:cc70,cuda9.1 -o run_pgi main-pgi.F90 -L. bin-xl.so
```
Note that "cc70,cuda9.1" should be updated to the corresponding CUDA version currently supported by XL. To run in an interactive session:
```
$ ml cuda
$ export LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH
$ jsrun -n1 -g1 -E LD_LIBRARY_PATH ./run_pgi
```
The expected output is
```
 11.1 22.2 33.3 44.4 55.5 66.6 77.7 88.8 99.9
```

How does this work? This is possible because

* By using a shared object file (bin-xl.so) for OpenMP4.5 target offloading codes and exposing only a host function subomp4(), all device function links therein have resolved at compile time.
* Both OpenACC and OpenMP4.5 runtimes try to attach to any existing CUDA context before initialize a new one. As a result, both runtimes operate in the same CUDA context, so that device pointers can be shared.
* A PGI OpenACC/CUDA Fortran trick ("device" attribute) is used in the interface block to extract the device pointer of the "arr" array.
Note that one cannot use statically-linked binaries (simple object files or a static libraries).

We have two options when using statically-linked binaries, but neither works well:

* PGI is used as the OpenACC compiler and host linker, and XL is used for generating statically-linked OpenMP4.5 binaries - not possible: XL's OpenMP4.5 requires certain supporting binaries to be injected during the linking stage, which PGI has no knowledge of.
* XL is used as the OpenMP4.5 compiler and host linker, and PGI is used for generating statically-linked OpenACC binaries - functional: however, the OpenACC codes must be compiled with the PGI's "nordc" sub-option, severely impacting programmability.

In our solution, XL injects the supporting binaries when it builds the shared object library containing OpenMP4.5 target offloading codes, as if it is linking an executable. It is also possible to apply this technique by using XL as the host linker while using PGI to build an shared object library for the OpenACC codes. Either way, a shared object library must be involved.

## Using managed memory for easier data interoperability

Data interoperability becomes even easier with CUDA managed memory.

main-pgi.F90
```
program main_pgi
implicit none
integer, parameter :: N=9
real*8, dimension(N), managed :: arr
integer :: i
 
do i=1,N; arr(i)=i; enddo
call subomp4(arr,N)
!$acc parallel loop vector
do i=1,N; arr(i)=arr(i)+10*i; enddo

write(*,'(9F5.1)') arr(1:N)
end
```
subomp4-xl.F90
```
subroutine subomp4(a,n)
implicit none
real*8, dimension(n), managed :: a
integer :: n
integer :: i

!$omp target teams distribute parallel do
do i=1,n; a(i)=a(i)+0.1*i; enddo

end
```

Note the differences:

* New "managed" attributes.
* The interface block is removed, as the "device" attribute trick is no longer necessary. Implicit interface is used.
* No data mapping directives in both OpenACC and OpenMP4.5 codes, including the is_device_ptr() clause.
* To illustrate the flexibility of managed memory, the first kernel is executed on CPU instead.

To compile this code on Summit:
```
$ ml xl
$ xlf_r -qpic -qcuda -qsmp=omp -qoffload -qextname -c subomp4-xl.F90
$ xlf_r -qcuda -qsmp=omp -qoffload -qmkshrobj -qextname -o bin-xl.so subomp4-xl.o
$ ml pgi
$ pgfortran -Mcuda -ta=tesla:cc70,cuda9.1 -o run_pgi main-pgi.F90 -L. bin-xl.so
```
CUDA Fortran arguments are added for both compilers. Note that "-qextname" is used for XL Fortran to add the underscore after object names, so they can be recognized by PGI. This is not necessary when explicit interfaces (interface block in the original example) are used. To run the new version with NVPROF:
```
$ jsrun -n1 -g1 -E LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH nvprof --print-gpu-summary ./run_pgi
==68447== NVPROF is profiling process 68447, command: ./run_pgi
 11.1 22.2 33.3 44.4 55.5 66.6 77.7 88.8 99.9
==68447== Profiling application: ./run_pgi
==68447== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   96.63%  156.22us         1  156.22us  156.22us  156.22us  __xl_subomp4__l0_OL_1
                    1.25%  2.0160us         1  2.0160us  2.0160us  2.0160us  [CUDA memcpy HtoD]
                    1.13%  1.8240us         1  1.8240us  1.8240us  1.8240us  [CUDA memcpy DtoH]
                    0.99%  1.6000us         1  1.6000us  1.6000us  1.6000us  main_pgi_11_gpu

==68447== Unified Memory profiling result:
Device "Tesla V100-SXM2-16GB (0)"
   Count  Avg Size  Min Size  Max Size  Total Size  Total Time  Name
       1  64.000KB  64.000KB  64.000KB  64.00000KB  4.928000us  Host To Device
       1  64.000KB  64.000KB  64.000KB  64.00000KB  3.264000us  Device To Host
       1         -         -         -           -  151.9990us  Gpu page fault groups
Total CPU Page faults: 2
```

Results are identical, as expected, although the first kernel no longer runs on GPU. We can also see the two page faults triggered by the second kernel and the stdout print at the end. With managed memory, the data arrays with the "managed" attribute are in a unified memory space, accessible by both CPU codes and GPU codes. Data migration is handled at the CUDA driver level mostly transparent to the programmer.

While still with many restrictions, the combination of dynamically load libraries and CUDA managed memory offers an rather straightforward The current approach to achieve OpenACC/OpenMP4.5 offloading interoperability with minimal effort for additional coding. We will continue to explorer the possibilities on interoperability and post updates on this blog.

## Mixing CUDA with OpenMP4.5

