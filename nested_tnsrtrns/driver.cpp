#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <iterator>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <omp.h>
#include "tnsrtrns.h"


int main(int argc, char *argv[])
{

  if (argc<2)
  {
    fprintf(stderr,"Usage: taskgen <file>\n");
    return -1;
  }

  std::ifstream hfTask(argv[1]);

  size_t n_task;
  std::string line;

  hfTask >> n_task;
  getline(hfTask,line); // skip to the next line
  std::cout << "Reading "<<n_task<<" tasks..."<<std::endl;

  std::vector<tnsrtrns_task*> task_list(n_task);
  int i_task=0;

  while (getline(hfTask,line))
  {
    std::stringstream ss_line(line);
    std::vector<int> line_int((std::istream_iterator<int>(ss_line)),
                              (std::istream_iterator<int>()) );
    if (line_int.size()<2) continue; // skip empty lines
    std::cout << std::setw(3) << i_task << " - " << line << std::endl;
    task_list[i_task]=new tnsrtrns_task(sizeof(double),line_int);
    i_task++;
  }
  hfTask.close();

  // Run the bigger tasks first for better load balancing
  std::vector<size_t> q(n_task);
  std::iota(q.begin(),q.end(),0);
  std::sort(q.begin(),q.end(),[&](const int a, const int b)
    {return (task_list[a]->vol_total > task_list[b]->vol_total);});

  std::cout << "Task execution order:  ";
  for (auto &iq : q) std::cout << iq << " ";
  std::cout << std::endl;

  size_t szTotal=0;

  std::cout << "Initializing random arrays... ";
  #pragma omp parallel for schedule(dynamic,1) reduction(+:szTotal) private(i_task)
  for (size_t it=0;it<n_task;it++)
  {
    i_task = q[it];

    if (task_list[i_task]->alloc_data()!=0)
      #pragma omp critical
      std::cout << std::endl << "Failed for i_task=" << i_task << ". Probably out of memory." << std::endl;

    szTotal += task_list[i_task]->vol_total;

    #pragma omp simd
    for (size_t i=0;i<task_list[i_task]->vol_total;i++)
      ((double*)task_list[i_task]->data_src)[i]=i;
  }
  std::cout << "Done! Total size: " << szTotal*sizeof(double)/1024/1024 << " MB" << std::endl ;

  double ts, te;

  ts=omp_get_wtime();
  #pragma omp parallel for schedule(dynamic,1) private(i_task)
  for (size_t it=0;it<n_task;it++)
  {
    i_task = q[it];

    // Aliases for OpenMP data directives
    auto const &      dim_a = task_list[i_task]->dim_a;
    auto const &      dim_b = task_list[i_task]->dim_b;
    auto const &      vol_a = task_list[i_task]->vol_a;
    auto const &      vol_b = task_list[i_task]->vol_b;
    auto const &  vol_total = task_list[i_task]->vol_total;
    auto const &    shape_a = task_list[i_task]->va2i;
    auto const &    shape_b = task_list[i_task]->vb2i;
    auto const & stride_a_l = task_list[i_task]->ia2s;
    auto const & stride_a_g = task_list[i_task]->ia2g;
    auto const &   stride_b = task_list[i_task]->ib2g;
    auto const &    data_in = static_cast<double*>(task_list[i_task]->data_src);
    auto const &   data_out = static_cast<double*>(task_list[i_task]->data_dst);

    #pragma omp target enter data nowait                                    \
      map(to:     data_in[:vol_total], shape_a[:dim_a], shape_b[:dim_b],    \
                  stride_a_l[:dim_a], stride_a_g[:dim_a], stride_b[:dim_b]) \
      map(alloc:  data_out[:vol_total])                                     \
      depend(out: data_in[:vol_total], data_out[:vol_total],                \
                  shape_a[:dim_a], shape_b[:dim_b],                         \
                  stride_a_l[:dim_a], stride_a_g[:dim_a], stride_b[:dim_b])
    c_tt_mapped( dim_a, dim_b, vol_a, vol_b, shape_a, shape_b, 
                 stride_a_l, stride_a_g, stride_b, data_in, data_out);
    #pragma omp target exit data nowait                                      \
      depend(in:   data_out[:vol_total])                                     \
      map(from:    data_out[:vol_total])                                     \
      map(release: data_in[:vol_total], shape_a[:dim_a], shape_b[:dim_b],    \
                   stride_a_l[:dim_a], stride_a_g[:dim_a], stride_b[:dim_b])
  }
  #pragma omp taskwait
  te=omp_get_wtime();
  std::cout<<"OpenMP4.5 offload with mapped data_dst:"<<std::endl;
  std::cout<<te-ts<<std::endl;

}
