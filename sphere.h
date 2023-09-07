#ifndef SPHERE_H
#define SPHERE_H
#include "math_utils.h"
#include "cuda_runtime.h"
#include <cstdlib>
#include <iostream>
#include "settings.h"

using namespace std;

// 602
// 1201

#define SPHERE_MAX_SIZE 400
#define SPHERE_MAX_IND_SIZE 1200

struct Sphere
{
    Vec3Float64 vertices[SPHERE_MAX_SIZE];
    Vec3Uint indices[SPHERE_MAX_IND_SIZE];
    Vec3Float64 normals[SPHERE_MAX_SIZE]; 
    int len;
    int len_indicies;
    int run_mode = RUN_ON_CPU;

    __host__ __device__ ~Sphere(){
       
    }
};

#define SPHERE_NORMAL 0
#define SPHERE_RANDOM 1

void InitSphere(Sphere * sphere);
void InitSphereCuda(Sphere * sphere);
__host__ __device__ void CreateSphere(Sphere * sphere,double radius, unsigned int stacks, unsigned int slices,int shape=SPHERE_NORMAL);
#endif