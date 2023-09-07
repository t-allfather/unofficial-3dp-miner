#ifndef MATH_UTILS
#define MATH_UTILS

#include "cuda_runtime.h"
#include <string>

#define VECf64_DOT(a,b) ((a.x * b.x + a.y * b.y + a.z* b.z))
#define VEC_ADD(a,b) ((a.x=a.x+b.x,a.y = a.y + b.y,a.z = a.z + b.z))

struct Vec3Int{
    int x;
    int y;
    int z;
};

struct Vec2F64{
    double x;
    double y;
};

struct Vec2Int{
    int x;
    int y;
};


struct Vec3Uint{
    unsigned int x;
    unsigned int y;
    unsigned int z;

    __host__ __device__ static Vec3Uint create(unsigned int x,unsigned int y,unsigned int z);
};


struct Vec3Float64{
    double x;
    double y;
    double z;

    __host__ __device__ static Vec3Float64 zero();
    __host__ __device__ static Vec3Float64 sub(Vec3Float64 a,Vec3Float64 b);
    __host__ __device__ static Vec3Float64 add(Vec3Float64 a,Vec3Float64 b);
    __host__ __device__ static Vec3Float64 addref(Vec3Float64 * a,Vec3Float64 b);
    __host__ __device__ static Vec3Float64 cross(Vec3Float64 a,Vec3Float64 b);
    __host__ __device__ static Vec3Float64 create(double x,double y,double z);
    __host__ std::string str();
    __host__ __device__ static double dot(Vec3Float64 * a,Vec3Float64 * b);
};

struct Triangle{
    Vec3Float64 p1;
    Vec3Float64 p2;
    Vec3Float64 p3;

    __host__ __device__ Triangle(Vec3Float64 _p1,Vec3Float64 _p2,Vec3Float64 _p3);
    __host__ __device__ void set(Vec3Float64 _p1,Vec3Float64 _p2,Vec3Float64 _p3);
    __host__ __device__ Triangle();
};

struct Eigen{

    unsigned int n=3;
    double a[3][3];
    double v[3][3] = {
        1,0,0,
        0,1,0,
        0,0,1
    };
    double d[3]={0};
    unsigned int n_rot=0;

    __host__ __device__ void Create(double mat[3][3]);
    __host__ __device__ void Solve();
};


struct PolyLine{
    int len=0;
    alignas(32) Vec2Int nodes[40];
    unsigned char allowed[8][8] = {
        {0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0}
    };

    __host__ __device__ void push(int x,int y,bool allowed_mat=true);

    __host__ __device__ void push(Vec2Int a);

    __host__ __device__ void pop();
};

struct PolyLineCompress{
    short len=0;
    unsigned char nodes[40];

    __host__ __device__ static Vec2Int decomp(unsigned char elem);
    __host__ __device__ static unsigned char comp(Vec2Int elem);
};

struct T64_PolyLine{
    double val;
    PolyLine line;
};

struct T64_PolyLine_Compress{
   double val;
   PolyLineCompress line;
};

struct alignas(32) F64_Hash384{
    alignas(32) double val;
    alignas(32) int hash_len;
    alignas(32) unsigned char hash[384];
};

__host__ __device__ int int_to_be(int val);
#endif