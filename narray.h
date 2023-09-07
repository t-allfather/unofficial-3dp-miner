#include <cstdlib>
#include "math_utils.h"

#define ARRAY_TYPE_32 4
#define ARRAY_TYPE_64 8
#define ARRAY_TYPE_VEC64 24

struct Array2
{
    int dim1;
    int dim2;
    unsigned char * buffer;

    void create(int rows,int cols,char type,unsigned char * buffer=NULL);
    int pos(int a,int b);
    double getF64(int a,int b);
    unsigned int getUint(int a,int b);
    Vec3Float64 getVecF64(int a,int b);
};