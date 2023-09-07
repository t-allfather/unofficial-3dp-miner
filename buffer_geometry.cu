#include "buffer_geometry.h"
#include "sphere.h"
#include <unistd.h>

__host__ __device__ BufferGeometry::BufferGeometry(Vec3Float64 * p,unsigned int * i,Vec3Float64 * n,int l,int l_i,Sphere * sphere){
        len = l;
        len_indices = l_i;
        positions = p; // (Vec3Float64*) malloc(len * sizeof(Vec3Float64));
        indices = i; // (unsigned int*) malloc(len_indices * sizeof(unsigned int));
        normals = n; //(Vec3Float64*) malloc(len * sizeof(Vec3Float64));
        spherePtr = sphere;
        /*
        memset(normals,0, len * sizeof(Vec3Float64));
        memcpy(positions,p,len * sizeof(Vec3Float64));
        memcpy(indices,i,len_indices * sizeof(unsigned int));
        */
    }

__host__ __device__ void BufferGeometry::fixIndicies(){
    //printf("%d %d\n",len_indices,len);
    unsigned int * indices_temp = new unsigned int[len_indices];
    Vec3Float64 * pos_temp = new Vec3Float64[len];
    short frv[SPHERE_MAX_SIZE]={0};
    for(int i=0;i<SPHERE_MAX_SIZE;i++)
        frv[i]=-1;
    int l_p = 0;
    int l_i=0;
    for(int i=0;i<len_indices;i++){
        int ind = indices[i+0];
        if(frv[ind] == -1){
            int index = l_p;
            pos_temp[l_p] = positions[ind];
            //if(ind != l_p)
            //printf("pos  %d -> %d\n",ind,l_p);
            l_p++;
            indices_temp[l_i] = index;
            l_i++;
            frv[ind]= index;
        } else {
            //printf("%d -> %d\n",i,l_i);
            indices_temp[l_i] = frv[ind];
            l_i++;
        }
    }
    memcpy(indices,indices_temp,len_indices * 4);
    memcpy(positions,pos_temp,sizeof(Vec3Float64) * len);
    delete[] indices_temp;
    delete[] pos_temp;
}

__host__ __device__ void BufferGeometry::fixIndicies_Static(unsigned int * indices_temp,Vec3Float64 * pos_temp){
    //printf("%d %d\n",len_indices,len);
    short frv[602]={0};
    for(int i=0;i<602;i++)
        frv[i]=-1;
    int l_p = 0;
    int l_i=0;
    for(int i=0;i<len_indices;i++){
        int ind = indices[i+0];
        if(frv[ind] == -1){
            int index = l_p;
            pos_temp[l_p] = positions[ind];
            //if(ind != l_p)
            //printf("pos  %d -> %d\n",ind,l_p);
            l_p++;
            indices_temp[l_i] = index;
            l_i++;
            frv[ind]= index;
        } else {
            //printf("%d -> %d\n",i,l_i);
            indices_temp[l_i] = frv[ind];
            l_i++;
        }
    }
    memcpy(indices,indices_temp,len_indices * 4);
    memcpy(positions,pos_temp,sizeof(Vec3Float64) * len);
}

void BufferGeometry::Clear(){
    
    /*
    if(spherePtr != NULL){
        printf("Free sphere\n");
        free(spherePtr->indices);
        free(spherePtr->normals);
        free(spherePtr->vertices);
    }
    */
}

__host__ __device__ BufferGeometry::~BufferGeometry(){
        /*free(indices);
        free(positions);
        free(normals);
        */
    }

void BufferGeometry::SetVertexNormals(Vec3Float64 * n){
        memcpy(normals,n,len * sizeof(Vec3Float64));
    }

__host__ __device__ void BufferGeometry::ComputeVertexNormals()
    {
        memset(normals,0,sizeof(Vec3Float64) * len);
        // indexed elements
        for(int i=0;i < len_indices/3; i ++){
            unsigned int vA = indices[i*3+0];
            unsigned int vB = indices[i*3+1];
            unsigned int vC = indices[i*3+2];
            Vec3Float64 pA = positions[vA];
            Vec3Float64 pB = positions[vB];
            Vec3Float64 pC = positions[vC];

            Vec3Float64 cb = Vec3Float64::sub(pC,pB);
            Vec3Float64 ab = Vec3Float64::sub(pA, pB);
            Vec3Float64 cr = Vec3Float64::cross(cb,ab);

            VEC_ADD(normals[vA],cr);
            VEC_ADD(normals[vB],cr);
            VEC_ADD(normals[vC],cr);
        }

   
        for(int i=0;i<len;i++){
            Vec3Float64 * normal = &normals[i];
            double n = 1.0 / sqrt(Vec3Float64::dot(normal,normal));
            if(isinf(n)){
                normal->x = 1;
                normal->y = 1;
                normal->z = 1;
            } else {
            normal->x = normal->x * n;
            normal->y = normal->y * n;
            normal->z = normal->z * n;
            }
        }

    }

void BufferGeometry::roundDecimals(){
    for(int i=0;i<len;i++){
        positions[i].x = round(positions[i].x*100)/100;
        positions[i].y = round(positions[i].y*100)/100;
        positions[i].z = round(positions[i].z*100)/100;

        normals[i].x = round(normals[i].x*10000)/10000;
        normals[i].y = round(normals[i].y*10000)/10000;
        normals[i].z = round(normals[i].z*10000)/10000;
    }
}

string BufferGeometry::parse() {
        string s = "o\n";

        for(int i=0;i<len;i++){
            s += "v " + to_string(positions[i].x) +" " + to_string(positions[i].y) +" " + to_string(positions[i].z) +"\n";
        }

        for(int i=0;i<len;i++){
            s += "vn " + to_string(normals[i].x) +" " + to_string(normals[i].y) +" " + to_string(normals[i].z) +"\n";
        }

        for(int i=0;i<len_indices;i+=3){
            s += "f " + to_string(indices[i+0] + 1) +"//" + to_string(indices[i+0] + 1) +" " + 
                        to_string(indices[i+1] + 1) +"//" + to_string(indices[i+1] + 1) +" " + 
                        to_string(indices[i+2] + 1) +"//" + to_string(indices[i+2] + 1) +"\n"; 
        }

        for(int i=0;i<len_indices;i++){
            int a = indices[i];
            if(a >= len){
                cout << a << endl;
            }
        }

        return s;
}