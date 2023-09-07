#include "math_utils.h"
#include "buffer_geometry.h"
#include "simple_deque.h"
#include <deque>

//typedef unsigned int UintVec[17];
//typedef UintVec Cell[1806];

struct CellRet
{
    unsigned int * vec;
    int size = SPHERE_MAX_SIZE; //602
    int size2 = 65;
    __host__ __device__ int pos(int pos1,int pos2);
};

struct ContourState{
    Vec2F64 sect[64];
    float mt[64][64];
    short ii[64];
    int sect_len=0;
};


__host__ __device__ void GetNeighbours(unsigned int positions_len,unsigned int len_ind,Vec3Uint * cells,CellRet * adjacentVertices,unsigned char * buffer = NULL);
__host__ __device__ void scrapeMain(
    unsigned int positionIndex,
    Vec3Float64 * positions,
    Vec3Float64 * normals,
    CellRet * adjacentVertices,
    double strength,
    double radius,
    bool * traversed,
    Deque * stack
);

void scrapeMainStd(
    unsigned int positionIndex,
    Vec3Float64 * positions,
    Vec3Float64 * normals,
    CellRet * adjacentVertices,
    double strength,
    double radius,
    bool * traversed,
    std::deque<int> &stack
);


__host__ __device__ void get_contour(BufferGeometry * geo,double z_sect,Vec2F64 * cntr,int &cntr_len);
__host__ __device__ void get_contour2(BufferGeometry * geo,double z_sect,Vec2F64 * cntr,int &cntr_len,float * mt_global);
__host__ __device__ void get_contour_opt(BufferGeometry * geo,double z_sect,Vec2F64 * cntr,int &cntr_len);
__host__ __device__ void get_contour_opt_cuda(BufferGeometry * geo,double z_sect,Vec2F64 * cntr,int &cntr_len,Vec2F64 * sect,short * ii);