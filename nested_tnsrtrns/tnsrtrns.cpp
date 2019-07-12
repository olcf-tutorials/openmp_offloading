#include "stdio.h"

#define tile_size_c 5900
// 1100 seems to be the limit for XL OpenMP4.5
//#define tile_size_c 1100

extern "C" void c_tt_( //{{{
  const int tile_size,
  const int dim_a, const int dim_b,
  const int vol_a, const int vol_b,
  const int* __restrict__ shape_a, // output shape
  const int* __restrict__ shape_b, // input shape
  const int* __restrict__ stride_a_l, // local s stride for output shape
  const int* __restrict__ stride_a_g, // global output stride for output shape
  const int* __restrict__ stride_b, // global output stride for input shape
  const double* __restrict__ data_in,
  double* __restrict__ data_out)
{
// Assumed sizes:
//   data_in   [0:vol_a*vol_b]
//   data_out  [0:vol_a*vol_b]
//   shape_a   [0:dim_a]
//   shape_b   [0:dim_b]
//   stride_a_l[0:dim_a]
//   stride_a_g[0:dim_a]
//   stride_b  [0:dim_b]

  #pragma acc parallel loop gang \
    deviceptr(data_in,data_out, \
      shape_a,shape_b,stride_a_l,stride_a_g,stride_b)
  #pragma omp target teams distribute thread_limit(256) \
    is_device_ptr(data_in,data_out, \
      shape_a,shape_b,stride_a_l,stride_a_g,stride_b)
  for (int ib=0;ib<vol_b;ib++)
  {
    double s[tile_size_c];
    float shape_a_inv[10];
    int it_b=ib, id_b=0, im_b;

    for (int i=0;i<dim_a;i++)
    {
      shape_a_inv[i]=1./shape_a[i];
    }

    for (int i=0;i<dim_b;i++)
    {
      im_b = it_b/shape_b[i];
      id_b += stride_b[i]*(it_b-im_b*shape_b[i]);
      it_b = im_b;
    }

    #pragma acc loop vector
    #pragma omp parallel for
    for (int ia=0;ia<vol_a;ia++)
    {
      s[ia] = data_in[ia+ib*vol_a];
    }

    #pragma acc loop vector
    #pragma omp parallel for
    for (int ia=0;ia<vol_a;ia++)
    {
      int it_a=ia, id_a=id_b, is=0, im_a, ic;
      for (int i=0;i<dim_a;i++)
      {
        im_a = it_a*shape_a_inv[i];
        ic = it_a-im_a*shape_a[i];
        id_a += stride_a_g[i]*ic;
        is += stride_a_l[i]*ic;
        it_a = im_a;
      }

      data_out[id_a] = s[is];
    }
  }

} //}}}

extern "C" void c_tt_mapped( //{{{
  const int tile_size,
  const int dim_a, const int dim_b,
  const int vol_a, const int vol_b,
  const int* __restrict__ shape_a, // output shape
  const int* __restrict__ shape_b, // input shape
  const int* __restrict__ stride_a_l, // local s stride for output shape
  const int* __restrict__ stride_a_g, // global output stride for output shape
  const int* __restrict__ stride_b, // global output stride for input shape
  const double* __restrict__ data_in,
  double* __restrict__ data_out)
{
// Assumed sizes:
//   data_in   [0:vol_a*vol_b]
//   data_out  [0:vol_a*vol_b]
//   shape_a   [0:dim_a]
//   shape_b   [0:dim_b]
//   stride_a_l[0:dim_a]
//   stride_a_g[0:dim_a]
//   stride_b  [0:dim_b]

//  #pragma acc data \
//      copyin( data_in[:vol_a*vol_b], shape_a[:dim_a], shape_b[:dim_b], \
//              stride_a_l[:dim_a], stride_a_g[:dim_a], stride_b[:dim_b]) \
//      copyout( data_out[:vol_a*vol_b])
//  {
//  #pragma acc host_data use_device(data_in,data_out)
//  {
//    printf("[c_tt_mapped] %p %p\n",data_in,data_out);
//  }
//  #pragma acc parallel loop gang
  #pragma omp target teams distribute thread_limit(256) \
      map(to: data_in[:vol_a*vol_b], shape_a[:dim_a], shape_b[:dim_b], \
              stride_a_l[:dim_a], stride_a_g[:dim_a], stride_b[:dim_b]) \
      map(from: data_out[:vol_a*vol_b])
  for (int ib=0;ib<vol_b;ib++)
  {
    double s[tile_size_c];
    float shape_a_inv[10];
    int it_b=ib, id_b=0, im_b;

    for (int i=0;i<dim_a;i++)
    {
      shape_a_inv[i]=1./shape_a[i];
    }

    for (int i=0;i<dim_b;i++)
    {
      im_b = it_b/shape_b[i];
      id_b += stride_b[i]*(it_b-im_b*shape_b[i]);
      it_b = im_b;
    }

    //#pragma acc loop vector
    #pragma omp parallel for
    for (int ia=0;ia<vol_a;ia++)
    {
      s[ia] = data_in[ia+ib*vol_a];
    }

    //#pragma acc loop vector
    #pragma omp parallel for
    for (int ia=0;ia<vol_a;ia++)
    {
      int it_a=ia, id_a=id_b, is=0, im_a, ic;
      //#pragma acc loop seq
      for (int i=0;i<dim_a;i++)
      {
        im_a = it_a*shape_a_inv[i];
        ic = it_a-im_a*shape_a[i];
        id_a += stride_a_g[i]*ic;
        is += stride_a_l[i]*ic;
        it_a = im_a;
      }

      data_out[id_a] = s[is];
    }
  }
//  }

} //}}}

extern "C" void c_tt_cpu_( //{{{
  const int tile_size,
  const int dim_a, const int dim_b,
  const int vol_a, const int vol_b,
  const int* __restrict__ shape_a, // output shape
  const int* __restrict__ shape_b, // input shape
  const int* __restrict__ stride_a_l, // local s stride for output shape
  const int* __restrict__ stride_a_g, // global output stride for output shape
  const int* __restrict__ stride_b, // global output stride for input shape
  const double* __restrict__ data_in,
  double* __restrict__ data_out)
{
// Assumed sizes:
//   data_in   [0:vol_a*vol_b]
//   data_out  [0:vol_a*vol_b]
//   shape_a   [0:dim_a]
//   shape_b   [0:dim_b]
//   stride_a_l[0:dim_a]
//   stride_a_g[0:dim_a]
//   stride_b  [0:dim_b]

  #pragma omp taskloop num_tasks(8)
  for (int ib=0;ib<vol_b;ib++)
  {
    double s[tile_size_c];
    int it_b=ib, id_b=0, im_b;

    for (int i=0;i<dim_b;i++)
    {
      im_b = it_b/shape_b[i];
      id_b += stride_b[i]*(it_b-im_b*shape_b[i]);
      it_b = im_b;
    }

    #pragma omp simd
    for (int ia=0;ia<vol_a;ia++)
    {
      s[ia] = data_in[ia+ib*vol_a];
    }

    #pragma omp simd
    for (int ia=0;ia<vol_a;ia++)
    {
      int it_a=ia, id_a=id_b, is=0, im_a, ic;
      for (int i=0;i<dim_a;i++)
      {
        im_a = it_a/shape_a[i];
        ic = it_a-im_a*shape_a[i];
        id_a += stride_a_g[i]*ic;
        is += stride_a_l[i]*ic;
        it_a = im_a;
      }

      data_out[id_a] = s[is];
    }
  }

} //}}}
