#ifndef BUFFER_GEO
#define BUFFER_GEO
#include <vector>
#include <cstdlib>
#include <iostream>
#include <cstring>
#include <math.h>
#include "math_utils.h"
#include <string>
#include "settings.h"
#include "sphere.h"

using namespace std;

struct BufferGeometry {
    unsigned int * indices;
    Vec3Float64 * positions;
    Vec3Float64 * normals;
    int len;
    int len_indices;
    int run_mode = RUN_ON_CPU;
    Sphere * spherePtr;

    __host__ __device__ BufferGeometry(Vec3Float64 * p,unsigned int * i,Vec3Float64 * n,int l,int l_i,Sphere* sphere=NULL);
    __host__ __device__ ~BufferGeometry();
    __host__ __device__ void fixIndicies();
    __host__ __device__ void fixIndicies_Static(unsigned int * indicies_temp,Vec3Float64 * pos_temp);
    void Clear();
    void SetVertexNormals(Vec3Float64 * n);
    __host__ __device__ void ComputeVertexNormals();
    void roundDecimals();
    string parse();
};


#endif