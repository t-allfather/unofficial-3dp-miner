#include "cuda_runtime.h"
#include "math_utils.h"
#include <iostream>
#include <fstream>


struct Tuple_select_top_all{
    double val;
    unsigned char hash[32];
};

__device__ void find_top_std_2_cuda(Vec2F64 * cntrs,int * cntrs_len,unsigned int depth,
    unsigned int n_sect, unsigned int grid_size,Vec3Float64 v_min,Vec3Float64 v_max , unsigned char * out_hashes, int &out_hashes_len);

void researchFindAllPathsGPU();