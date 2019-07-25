#ifndef TNSRTRNS_H
#define TNSRTRNS_H

#include <cstdlib>
#include <vector>
#include <algorithm>
#include <numeric>

#ifdef _NO_CUDA_MANAGED_MEM_
#define cudaMallocManaged(p,s) posix_memalign((p),128,(s))
#define cudaFree(p) free((p))
#else
#include <cuda_runtime.h>
#endif

extern "C" void c_tt_acc(
  const int dim_a, const int dim_b,
  const int vol_a, const int vol_b,
  const int* __restrict__ shape_a, // output shape
  const int* __restrict__ shape_b, // input shape
  const int* __restrict__ stride_a_l, // local s stride for output shape
  const int* __restrict__ stride_a_g, // global output stride for output shape
  const int* __restrict__ stride_b, // global output stride for input shape
  const double* __restrict__ data_in,
  double* __restrict__ data_out);

extern "C" void c_tt_cuda(
  const int dim_a, const int dim_b,
  const int vol_a, const int vol_b,
  const int* __restrict__ shape_a, // output shape
  const int* __restrict__ shape_b, // input shape
  const int* __restrict__ stride_a_l, // local s stride for output shape
  const int* __restrict__ stride_a_g, // global output stride for output shape
  const int* __restrict__ stride_b, // global output stride for input shape
  const double* __restrict__ data_in,
  double* __restrict__ data_out);

extern "C" void c_tt_cpu(
  const int dim_a, const int dim_b,
  const int vol_a, const int vol_b,
  const int* __restrict__ shape_a, // output shape
  const int* __restrict__ shape_b, // input shape
  const int* __restrict__ stride_a_l, // local s stride for output shape
  const int* __restrict__ stride_a_g, // global output stride for output shape
  const int* __restrict__ stride_b, // global output stride for input shape
  const double* __restrict__ data_in,
  double* __restrict__ data_out);

struct tnsrtrns_task
{
  int allocator_type;
  size_t n_dim;
  std::vector<size_t> dims;
  std::vector<size_t> perm;

// Description of shapes and strides {{{
//  Notes:
//  1) for volume B, no coalescing attempt is made, so 
//     source and target indix order are the same.
//  2) all indices are 0-based
//
//  Member  | Vol | Flat index in |  Index order  | original name
//  --------|-----|---------------|---------------|-----------
//    vb2i  |  B  | source vol B  > src/tgt vol B | shape_b
//    va2i  |  A  | target vol A  > target vol A  | shape_a
//    ib2g  |  B  | target global < source/target | stride_b
//    ia2s  |  A  | source vol A  < target vol A  | stride_a_l
//    ia2g  |  A  | target global < target vol A  | stride_a_g
//
//  perm[] - permutation array (p)
//     j=perm[i] means that the i-th dimension the TRANSPOSED tensor
//     is from the j-th dimension in the original tensor
//  q[] - inverse permutation of p
//
//  Example:
//  0,1,2,3,4,5 -> 1,2,3,5,0,4
//  dims = {41,13,11,9,76,50}
//  perm = {1,2,3,5,0,4}
//  q    = {4,0,1,2,5,3}
//  h    = 3
//  w    = {1,2,0}
//  d[p[:]] = {13,11,9,50,41,76}
//  q[w[:]] = {0,1,4}
//  vol  = d0*d1*d2*d3*d4*d5
//  va2i = {d1,d2,d0}    // va2i[2] is not important
//  vb2i = {d3,d4,d5}    // vb2i[2] is not important
//  ia2s = {d0,d0*d1,1} 
//  ia2g = {1,d1,d1*d2*d3*d5}    // Stride-1 stores
//  ib2g = {d1*d2,d1*d2*d3*d5*d0,d1*d2*d3} 
//
//--- Shape and stride generation ---
//
//  Assume a tile size of T (in the unit of data_dst elements)
//  n is the number of dimensions.
//  p[i] for perm[i], d[i] for dims[i].
//
//  x[] - named array
//
//  x[<i>] - indirect referenced array notation
//      an re-ordered x, with an order indicated by vector <i>,
//      which looks like:
//      x[i[1]], x[i[2]], ...
//
//  *<k>(d[k]) - cumulative product
//      *<k=1:m>(d[k]) = d[1]*d[2]*...*d[m] if m>=1
//                       1                  if m<1
//
//  _*_(d[k0:k1]) - cumulative product
//      d[k0]*d[k0+1]*...*d[k1]
//
//  Sort(p[]) - sorting function
//      returns a sorted vector of all the elements in p.
//
//  Procedure:
//
//  Decide the splitting of vol A and B
//    Find h, such that _*_(d[0:h-1])<=T and _*_(d[0:h])>T.
//
//  vb2i[] := d[ h : n-1 ]
//
//  p[<q>] := Sort(p[])
//      length = n
//      d[<q>] is the transposed dimensions
//
//  qa[<w>] := Sort(q[0:h-1])
//      length = h
//      d[<w>] is the shape for the write loop in each team iteration
//
//  va2i[] := d[<w>]
//      length = h
//
//  ia2s[] := [ *<k=0:w[0]-1>(d[k]]) , *<k=0:w[1]-1>(d[k]), ... ]
//      length = h
//
//  ia2g[] := [ *<k=0:q[w[0]]-1>(d[p[k]]) , *<k=0:q[w[1]]-1>(d[p[k]]) , ... ]
//      length = h
//
//  ib2g[] := [ *<k=0:q[h]-1>(d[p[k]]) , *<k=0:q[h+1]-1>(d[p[k]]) , ... ]
//      lentgh = n-h
//}}}

  static const size_t szTile = 46*1024;  // scratch pad size in bytes
  size_t szElm;

  int *va2i;
  int *vb2i;
  int *ia2s;
  int *ia2g;
  int *ib2g;

  void *data_dst; // only one array for in-place tranpose
  void *data_src;

  size_t dim_a;
  size_t dim_b;
  size_t vol_a;
  size_t vol_b;
  size_t vol_total;

  int myalloc(void **p,size_t s)
  {
    if (allocator_type==0)
    {
      return posix_memalign(p,128,s);
    }
    else
    {
      return cudaMallocManaged(p,s);
    }
  }

  int myfree(void *p)
  {
    if (p)
    {
      if (allocator_type==0)
      {
        free(p);
        p = nullptr;
      }
      else
      {
        void *pt=p;
        p = nullptr;
        cudaFree(pt);
      }
    }
    return 0;
  }

  int alloc_data()
  {
    myalloc((void**)(&data_dst),vol_total*szElm);
    myalloc((void**)(&data_src),vol_total*szElm);
    return ( (data_dst && data_src) ? 0 : -1 );
  }

  int realloc_all(int alloc_type) // this changes the allocator type as well
  {
    myfree(data_dst);
    myfree(data_src);
    myfree(va2i);
    myfree(vb2i);
    myfree(ia2s);
    myfree(ia2g);
    myfree(ib2g);

    allocator_type = alloc_type;
    calc_coef();

    return alloc_data();
  }

  void calc_coef()
  {
    int h, st;

    // Total volume
    vol_total = 1;
    for (int i=0;i<n_dim;i++) vol_total *= dims[i];

    // determine volume A, so that it fits the scratchpad size
    st = 1;
    for (h=0;st<szTile/szElm;h++) st *= dims[h];
    h--;
    dim_a = h;
    dim_b = n_dim - dim_a;

    vol_a = 1;
    for (int i=0;i<h;i++) vol_a *= dims[i];
    vol_b = vol_total / vol_a;

    std::vector<size_t> q(n_dim);
    std::iota(q.begin(),q.end(),0); // must be 0-based, since C array is 0-based
    std::sort(q.begin(),q.end(),[&](const int& a, const int& b) {return (perm[a]<perm[b]);});

    std::vector<size_t> w(dim_a);
    std::iota(w.begin(),w.end(),0); // must be 0-based, since C array is 0-based
    std::sort(w.begin(),w.end(),[&](const int& a, const int& b) {return (q[a]<q[b]);});

    // Generate dimensions and strides
    myalloc((void**)(&va2i),dim_a*sizeof(int));
    myalloc((void**)(&vb2i),dim_b*sizeof(int));
    myalloc((void**)(&ia2s),dim_a*sizeof(int));
    myalloc((void**)(&ia2g),dim_a*sizeof(int));
    myalloc((void**)(&ib2g),dim_b*sizeof(int));

    for (size_t i=0;i<dim_a;i++) va2i[i] = dims[w[i]];

    for (size_t i=dim_a;i<n_dim;i++) vb2i[i-dim_a] = dims[i];

    for (size_t i=0;i<dim_a;i++)
    {
      ia2s[i]=1;
      for (size_t k=0;k<w[i];k++) ia2s[i] *= dims[k];
    }

    for (size_t i=0;i<dim_a;i++)
    {
      ia2g[i]=1;
      for (size_t k=0;k<q[w[i]];k++) ia2g[i] *= dims[perm[k]];
    }

    for (size_t i=dim_a;i<n_dim;i++)
    {
      ib2g[i-dim_a]=1;
      for (size_t k=0;k<q[i];k++) ib2g[i-dim_a] *= dims[perm[k]];
    }
  }

  tnsrtrns_task(size_t sztype,size_t n, int alloc_type)
  : szElm(sztype),n_dim(n),dims(n),perm(n),allocator_type(alloc_type) {}

  tnsrtrns_task(size_t sztype,size_t n):tnsrtrns_task(sztype,n,0) {}

  tnsrtrns_task(size_t sztype,std::vector<int> &ndp_list, int alloc_type)
  : szElm(sztype), allocator_type(0), data_dst(nullptr), data_src(nullptr),
    va2i(nullptr), vb2i(nullptr), ia2s(nullptr), ia2g(nullptr), ib2g(nullptr),
    n_dim(ndp_list[0]),
    dims(ndp_list.begin()+1,ndp_list.begin()+n_dim+1), // +1 for past-the-end
    perm(ndp_list.begin()+n_dim+1,ndp_list.begin()+2*n_dim+1)
  {
    calc_coef();
  }

  // Default to host allocator, if not specified
  tnsrtrns_task(size_t sztype,std::vector<int> &ndp_list):tnsrtrns_task(sztype,ndp_list,0){}

  ~tnsrtrns_task()
  {
    myfree(data_dst);
    myfree(data_src);
    myfree(va2i);
    myfree(vb2i);
    myfree(ia2s);
    myfree(ia2g);
    myfree(ib2g);
  }

};

#endif
