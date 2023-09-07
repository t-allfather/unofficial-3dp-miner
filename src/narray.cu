#include "narray.h"

void Array2::create(int rows,int cols,char type,unsigned char * bufferInit){
    dim1 = rows;
    dim2 = cols;
    if(bufferInit == NULL){
        if(type == ARRAY_TYPE_32){
            buffer = (unsigned char*)malloc(rows * cols * 4);
        } else if(type == ARRAY_TYPE_64){
            buffer = (unsigned char*)malloc(rows * cols * 8);
        } else if(type == ARRAY_TYPE_VEC64){
            buffer = (unsigned char*)malloc(rows * cols * 24);
        }
    } else {
        buffer = bufferInit;
    }
}

double Array2::getF64(int r,int c){
    double * ptr = (double*)buffer;
    return ptr[r * dim2 + c];
}

Vec3Float64 Array2::getVecF64(int r,int c){
    Vec3Float64 * ptr = (Vec3Float64*)buffer;
    return ptr[r * dim2 + c];
}

unsigned int Array2::getUint(int r,int c){
    unsigned int * ptr = (unsigned int*)buffer;
    return ptr[r * dim2 + c];
}