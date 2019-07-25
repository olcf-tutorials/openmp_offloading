#include <iostream>

#define tile_size_c 5900

__global__
void k_tt(
  const int dim_a, const int dim_b,
  const int vol_a, const int vol_b,
  const int* __restrict__ shape_a,
  const int* __restrict__ shape_b,
  const int* __restrict__ stride_a_l,
  const int* __restrict__ stride_a_g,
  const int* __restrict__ stride_b,
  const double* __restrict__ data_in,
  double* __restrict__ data_out)
{

  __shared__ double s[tile_size_c];
  int blk_offset=vol_a*blockIdx.x;

//  for(int ia=threadIdx.x ; ia<vol_a ; ia+=blockDim.x)
//    s[ia]=data_in[ia+blk_offset];

//  int it_b=blockIdx.x, id_b=0, im_b;
//  for (int i=0;i<dim_b;i++)
//  {
//    im_b = it_b/shape_b[i];
//    id_b += stride_b[i]*(it_b-im_b*shape_b[i]);
//    it_b = im_b;
//  }
//  __syncthreads();
//
//  for(int ia=threadIdx.x ; ia<vol_a ; ia+=blockDim.x)
//  {
//    int it_a=ia, id_a=0, is=0, im_a, ic;
//    for (int i=0;i<dim_a;i++)
//    {
//      im_a = it_a/shape_a[i];
//      ic = it_a-im_a*shape_a[i];
//      id_a += stride_a_g[i]*ic;
//      is += stride_a_l[i]*ic;
//      it_a = im_a;
//    }
//
//    data_out[id_a+id_b] = s[is];
//  }
//
}

extern "C" void c_tt_cuda(
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
  //cudaStream_t mystream;
  //std::cout << cudaStreamCreate(&mystream) << std::endl;
  //k_tt<<<vol_b,512,0,mystream>>>
  k_tt<<<vol_b,512>>>
      ( dim_a, dim_b, vol_a, vol_b, shape_a, shape_b, 
        stride_a_l, stride_a_g, stride_b, data_in, data_out );
  cudaStreamSynchronize(cudaStreamPerThread);
  //cudaStreamSynchronize(mystream);
}

