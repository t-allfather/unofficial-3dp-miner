#include "cuda_runtime.h"
#include "math_utils.h"
#include <iostream>
#include <fstream>


struct alignas(32) Tuple_select_top_all{
    alignas(32) double val;
    alignas(32) unsigned char hash[32];
};

void find_top_std_3(Vec2F64 * cntrs,int * cntrs_len,unsigned int depth,
    unsigned int n_sect, unsigned int grid_size,Vec3Float64 v_min,Vec3Float64 v_max , unsigned char * out_hashes, int &out_hashes_len);
