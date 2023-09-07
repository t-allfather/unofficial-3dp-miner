#include "cuda.h"
#include "cuda_runtime.h"
#include "device_functions.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <vector>
#include "timer.h"
#include "sphere.h"
#include "perlin.h"
#include "rock.h"
#include "scrape.h"
#include <fstream>
#include "buffer_geometry.h"
#include <unistd.h>
#include <curand.h>
#include <curand_kernel.h>
#include <float.h>
#include <chrono>
#include "grid_cuda.cuh"
#include "kernel.cuh"
#include "sha3_gpu.cuh"
using namespace std;

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char* const func, const char* const file,
    const int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
            << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
void checkLast(const char* const file, const int line)
{
    cudaError_t err{ cudaGetLastError() };
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << "kernel.cu" << ":" << line
            << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

struct WorkContext{
    Sphere * sphere;
    BufferGeometry * geo;
};

struct GpuContext{
    Sphere * sphere;
    Sphere * sphere_keep;
    BufferGeometry * geo;
};

__device__ double GetRandomValue( curandState& rand_state,double min, double max)
        {
            return (double)curand_uniform(&rand_state)  * (max - min) + min;
        }

__device__ double VaryParameter(curandState& rand_state,double param, double variance, double min, double max)
        {
            param += GetRandomValue(rand_state,-variance * 1.0, +variance * 1.0);
            if (param > max) param = max;
            if (param < min) param = min;
            return param;
        }
__device__ int VaryParameter(curandState& rand_state,int param, int variance, int min, int max)
        {
            param += (int)GetRandomValue(rand_state,-variance * 1.0, +variance * 1.0);
            if (param > max) param = max;
            if (param < min) param = min;
            return param;
        }

__device__ void varyMesh(RockObjParams &rock_obj_params,curandState& rand_state){
     rock_obj_params.meshNoiseScale     = VaryParameter(rand_state,1.0,     0.05,      MESH_NOISE_SCALE_MIN,      1.0);
    rock_obj_params.meshNoiseStrength  = VaryParameter(rand_state,0.5,  0.3,   MESH_NOISE_STRENGTH_MIN,    0.5);
    rock_obj_params.scrapeCount        = VaryParameter(rand_state,7,        3,          SCRAPE_COUNT_MIN,           SCRAPE_COUNT_MAX);
    rock_obj_params.scrapeMinDist      = VaryParameter(rand_state,0.8,      SCRAPE_MIN_DIST_VARY,       SCRAPE_MIN_DIST_MIN,        SCRAPE_MIN_DIST_MAX);
    rock_obj_params.scrapeStrength     = VaryParameter(rand_state,0.05,     0.02,       SCRAPE_STRENGTH_MIN,        SCRAPE_STRENGTH_MAX);
    rock_obj_params.scrapeRadius       = VaryParameter(rand_state,0.1,       SCRAPE_RADIUS_VARY,         SCRAPE_RADIUS_MIN,          0.);
//cout << rock_obj_params.scrapeRadius << endl;
rock_obj_params.scale[0] = VaryParameter(rand_state,1.0, 0.1, SCALE_MIN, SCALE_MAX);
rock_obj_params.scale[1] = VaryParameter(rand_state,1.0, 0.1, SCALE_MIN, SCALE_MAX);
rock_obj_params.scale[2] = VaryParameter(rand_state,1.2, 0.1, SCALE_MIN, 1.2);



}


__device__ void RockCuda(unsigned char * bestHash,Sphere * sphere,Sphere * sphere_keep,curandState * rand_states,int idx,unsigned char * buffer,unsigned char * mt_global,unsigned char * out_hash_arr,int * out_hash_len_arr,RockObjParams * output_params){
    RockObjParams rock_obj_params;
    short grid_size = 8;
    short n_sections = 12;
    bool trans_null = true;
    unsigned char trans[4] = {0,0,0,0};

    if(bestHash!=NULL){
        if(bestHash[0] ==0 && bestHash[1] == 0 && bestHash[2] ==0 && bestHash[3] ==0){

        } else {
        trans[0] = bestHash[0];
        trans[1] = bestHash[1];
        trans[2] = bestHash[2];
        trans[3] = bestHash[3];
        trans_null = false;
        }
    }

    varyMesh(rock_obj_params,rand_states[idx]);

     memcpy(output_params,&rock_obj_params,216);

    CellRet adjacentVertices;
    adjacentVertices.vec = (unsigned int*)buffer;


    unsigned int scrapeIndices[16];
    int scrapeIndicesLen=0;
    for(int i=0 ; i < rock_obj_params.scrapeCount;i++){
        int attempts = 0;

        // find random position which is not too close to the other positions.
        while(true) {
            int randIndex = curand_uniform(&rand_states[idx]) * sphere->len; // (positions.len() as f64 * rand::random::<f64>()).floor() as usize;
            Vec3Float64 p = sphere->vertices[randIndex];
            bool tooClose = false;
            // check that it is not too close to the other vertices.
            for(int j=0; j < scrapeIndicesLen ;j++) {
                Vec3Float64 q = sphere->vertices[scrapeIndices[j]];

                double dist = (p.x - q.x) * (p.x - q.x)  +  (p.y - q.y) * (p.y - q.y)  +  (p.z - q.z) * (p.z - q.z);
                if (dist < rock_obj_params.scrapeMinDist ){
                    tooClose = true;
                    break;
                }
            }
            attempts=attempts+1;

            // if we have done too many attempts, we let it pass regardless.
            // otherwise, we risk an endless loop.
            if(tooClose && attempts < 100) {
                continue;
            } else {
                scrapeIndices[scrapeIndicesLen] = randIndex;
                scrapeIndicesLen++;
                break;
            }
        }
    }

    bool traversed[300];// = (bool*)malloc(sphere->len);
    Deque stack(400);
    stack.front = -1;
    stack.rear=0;

    memcpy(output_params->scrapeIndices,scrapeIndices,4 * scrapeIndicesLen);
    output_params->scrapeIndicesLen = scrapeIndicesLen;

    // now we scrape at all the selected positions.
    for(int i=0;i<scrapeIndicesLen;i++){
        for(int j=0;j<sphere->len;j++){
            traversed[j] = false;
        }
        stack.front = -1;
        stack.rear=0;
         // 100 
        scrapeMain(scrapeIndices[i],  sphere->vertices, sphere->normals, &adjacentVertices, rock_obj_params.scrapeStrength, rock_obj_params.scrapeRadius, traversed, &stack);
    }

   // free(traversed);

   // Perlin perlin;
  //  perlin.Init(curand_uniform(&rand_states[idx]) * INT32_MAX);
    for (int i=0;i<sphere->len;i++){
  //      Vec3Float64 p = sphere->vertices[i];

//        double noise = rock_obj_params.meshNoiseStrength * perlin.Noise(rock_obj_params.meshNoiseScale * p.x, rock_obj_params.meshNoiseScale * p.y, rock_obj_params.meshNoiseScale * p.z);

        Vec3Float64 &pI = sphere->vertices[i];

        pI.x *= rock_obj_params.scale[0];
        pI.y *= rock_obj_params.scale[1];
        pI.z *= rock_obj_params.scale[2];

    }
    
    BufferGeometry geo = BufferGeometry(sphere->vertices,(unsigned int*)sphere->indices,sphere->normals,sphere->len,sphere->len_indicies * 3);
    geo.fixIndicies_Static((unsigned int*)mt_global,(Vec3Float64*) (mt_global + sizeof(Vec2F64) * 1024));

    memcpy(sphere_keep->vertices,geo.positions,geo.len * 24);
    memcpy(sphere_keep->indices,geo.indices,4 * geo.len_indices);
    sphere_keep->len = geo.len;
    sphere_keep->len_indicies = geo.len_indices/3;
    

    Vec3Float64 f1;
    Vec3Float64 f2;
    Vec3Float64 f3;

    Vec3Float64 g0;
    Vec3Float64 g1;
    Vec3Float64 g2;

    Vec3Float64 cross;
    double integral[10];

    double integral_sum[10]={0};
    double coefficients[] =  {1./6., 1./24., 1./24., 1./24., 1./60., 1./60., 1./60., 1./120., 1./120., 1./120.};


    for(int i=0;i<geo.len_indices;i+=3){
        Vec3Float64 tp1 = geo.positions[geo.indices[i+1]];
        Vec3Float64 tp2 = geo.positions[geo.indices[i+2]];
        Vec3Float64 tp3 = geo.positions[geo.indices[i+0]];
        f1.x = tp1.x + tp2.x + tp3.x; 
        f1.y = tp1.y + tp2.y + tp3.y; 
        f1.z = tp1.z + tp2.z + tp3.z;

        f2.x = tp1.x  * tp1.x  +
                  tp2.x  * tp2.x +
                  tp1.x  * tp2.x  +
                  tp2.x  * f1.x; 
        
        f2.y = tp1.y  * tp1.y  +
                  tp2.y  * tp2.y +
                  tp1.y  * tp2.y  +
                  tp2.y  * f1.y;

        f2.z = tp1.z  * tp1.z  +
                  tp2.z  * tp2.z +
                  tp1.z  * tp2.z  +
                  tp2.z  * f1.z; 

        f3.x = tp1.x  * tp1.x * tp1.x  +
                  tp1.x  * tp1.x  * tp2.x +
                  tp1.x  * tp2.x * tp2.x +
                  tp2.x  * tp2.x * tp2.x + 
                  tp3.x  * f2.x ;

        f3.y = tp1.y  * tp1.y * tp1.y  +
                  tp1.y  * tp1.y  * tp2.y +
                  tp1.y  * tp2.y * tp2.y +
                  tp2.y  * tp2.y * tp2.y + 
                  tp3.y  * f2.y ;

        f3.z = tp1.z  * tp1.z * tp1.z  +
                  tp1.z  * tp1.z  * tp2.z +
                  tp1.z  * tp2.z * tp2.z +
                  tp2.z  * tp2.z * tp2.z + 
                  tp3.z  * f2.z ;

        g0.x = f2.x + (tp1.x + f1.x) * tp1.x;
        g0.y = f2.y + (tp1.y + f1.y) * tp1.y;
        g0.z = f2.z + (tp1.z + f1.z) * tp1.z;

        g1.x = f2.x + (tp2.x + f1.x) * tp2.x;
        g1.y = f2.y + (tp2.y + f1.y) * tp2.y;
        g1.z = f2.z + (tp2.z + f1.z) * tp2.z;

        g2.x = f2.x + (tp3.x + f1.x) * tp3.x;
        g2.y = f2.y + (tp3.y + f1.y) * tp3.y;
        g2.z = f2.z + (tp3.z + f1.z) * tp3.z;



        double d1[3];
        d1[0] = tp2.x - tp1.x;
        d1[1] = tp2.y - tp1.y;
        d1[2] = tp2.z - tp1.z;

        double d2[3];
        d2[0] = tp3.x - tp2.x;
        d2[1] = tp3.y - tp2.y;
        d2[2] = tp3.z - tp2.z;

        cross.x = d1[1] * d2[2] - d1[2] * d2[1];
        cross.y = d1[2] * d2[0] - d1[0] * d2[2];
        cross.z = d1[0] * d2[1] - d1[1] * d2[0];

        integral[0] = cross.x * f1.x;

        integral[1] = cross.x * f2.x;
        integral[2] = cross.y * f2.y;
        integral[3] = cross.z * f2.z;

        integral[4] = cross.x * f3.x;
        integral[5] = cross.y * f3.y;
        integral[6] = cross.z * f3.z;

        for(int j=0;j<3;j++){
            int triangle_i = (j + 1) % 3;
            if(j==0){
                integral[7] = cross.x * (
                    tp1.y * g0.x +
                    tp1.y * g1.x +
                    tp1.y * g2.x);
            } else if(j == 1){
                integral[8] = cross.y * (
                    tp1.z * g0.y +
                    tp1.z * g1.y +
                    tp1.z * g2.y);
            } else if(j == 2){
                integral[9] = cross.z * (
                    tp1.x * g0.z +
                    tp1.x * g1.z +
                    tp1.x * g2.z);
            }
            
        }

        for(int j=0;j<10;j++)
            integral_sum[j] += integral[j];

        /*
        cout << triangles[i].p1.str() << endl;
        cout << triangles[i].p2.str() << endl;
        cout << triangles[i].p3.str() << endl;
        cout << d1[0] << " " << d1[1] << " " << d1[2] << endl;
        cout << d2[0] << " " << d2[1] << " " << d2[2] << endl;
        cout << cross[i].str() << endl;
        cout << endl;
        */


    }

    double integrated[10];
    for(int j=0;j<10;j++)
        integrated[j] = integral_sum[j] * coefficients[j];
    double volume = integrated[0];
    double center_mass[3];
    if(volume > 0.0001){
        center_mass[0] = integrated[1] / volume;
        center_mass[1] = integrated[2] / volume;
        center_mass[2] = integrated[3] / volume;
    } else {
        center_mass[0] = 0;
        center_mass[1] = 0;
        center_mass[2] = 0;
    }

    double inertia[3][3];

    inertia[0][0] = integrated[5] + integrated[6]
    -volume * (center_mass[1] * center_mass[1] + center_mass[2] * center_mass[2]);

    inertia[1][1] = integrated[4] + integrated[6] -
        volume * (center_mass[0] * center_mass[0] + center_mass[2] * center_mass[2]);

    inertia[2][2] = integrated[4] + integrated[5] -
        volume * (center_mass[0]*center_mass[0] + center_mass[1] * center_mass[1]);

    inertia[0][1] = integrated[7] -
        volume * center_mass[0] * center_mass[1];

    inertia[1][2]= integrated[8] -
        volume * center_mass[1] * center_mass[2];

    inertia[0][2] = integrated[9] -
        volume * center_mass[0] * center_mass[2];

    inertia[2][0] = inertia[0][2];
    inertia[2][1] = inertia[1][2];
    inertia[1][0] = inertia[0][1];

    
    /*
    cout << "Center mass:";
    cout << center_mass[0] << " " << center_mass[1] << " " << center_mass[2] << endl;
    cout <<"Inertia"<<endl;
    cout << inertia[0][0] << " " << inertia[0][1] << " " << inertia[0][2] << endl;
    cout << inertia[1][0] << " " << inertia[1][1] << " " << inertia[1][2] << endl;
    cout << inertia[2][0] << " " << inertia[2][1] << " " << inertia[2][2] << endl;
    cout << endl;
    */
    

    // fn principal_axis    args: inertia

    double m[3][3] = {{1 * inertia[0][0], -1 * inertia[0][1],-1 * inertia[0][2]},
                                        {-1 * inertia[1][0],1 * inertia[1][1],-1 * inertia[1][2]},
                                        {-1 * inertia[2][0],-1 * inertia[2][1],1 * inertia[2][2]}};

    Eigen eigen;
    eigen.Create(m);
    eigen.Solve();

    double components[3] = {
        eigen.d[1],
        eigen.d[0],
        eigen.d[2]
    };
    double vectors[3][3] = {
        eigen.v[0][0],eigen.v[1][0],eigen.v[2][0],
        eigen.v[0][1],eigen.v[1][1],eigen.v[2][1],
        eigen.v[0][2],eigen.v[1][2],eigen.v[2][2],
    };

    
    /*
    cout << "Components:" << endl;
    cout << components[0] << " " << components[1] << " " << components[2] << endl;
    cout << "Vectors:" << endl;
    for(int i=0;i<3;i++){
        for(int j=0;j<3;j++){
            cout << vectors[i][j] <<" ";
        }
        cout << endl;
    }*/
    

    double pit[4][4] = {
        vectors[0][0],vectors[0][1],vectors[0][2],-center_mass[0],
        vectors[1][0],vectors[1][1],vectors[1][2],-center_mass[1],
        vectors[2][0],vectors[2][1],vectors[2][2],-center_mass[2],
        0,0,0,1
    };


    double b[4][4];  // pit matrix inverse = pit ^ -1
    double bb[4][4];
    for(int i=0;i<3;i++)
        for(int j=0;j<3;j++)
            b[j][i] = pit[i][j];
    
double determinant = pit[0][ 0] * (pit[1][ 1] * pit[2][ 2] - pit[2][ 1] * pit[1][ 2]) -
             pit[0][ 1] * (pit[1][ 0] * pit[2][ 2] - pit[1][ 2] * pit[2][ 0]) +
             pit[0][ 2] * (pit[1][ 0] * pit[2][ 1] - pit[1][ 1] * pit[2][ 0]);

    double invdet = 1 / determinant;

    b[0][ 0] = (pit[1][ 1] * pit[2][ 2] - pit[2][ 1] * pit[1][ 2]) * invdet;
    b[0][ 1] = (pit[0][ 2] * pit[2][ 1] - pit[0][ 1] * pit[2][ 2]) * invdet;
    b[0][ 2] = (pit[0][ 1] * pit[1][ 2] - pit[0][ 2] * pit[1][ 1]) * invdet;
    b[1][ 0] = (pit[1][ 2] * pit[2][ 0] - pit[1][ 0] * pit[2][ 2]) * invdet;
    b[1][ 1] = (pit[0][ 0] * pit[2][ 2] - pit[0][ 2] * pit[2][ 0]) * invdet;
    b[1][ 2] = (pit[1][ 0] * pit[0][ 2] - pit[0][ 0] * pit[1][ 2]) * invdet;
    b[2][ 0] = (pit[1][ 0] * pit[2][ 1] - pit[2][ 0] * pit[1][ 1]) * invdet;
    b[2][ 1] = (pit[2][ 0] * pit[0][ 1] - pit[0][ 0] * pit[2][ 1]) * invdet;
    b[2][ 2] = (pit[0][ 0] * pit[1][ 1] - pit[1][ 0] * pit[0][ 1]) * invdet;


    b[0][3] =0;
    b[1][3]=0;
    b[2][3]=0;
    b[3][3]=1;
    b[3][0]=0;
    b[3][1]=0;
    b[3][2]=0;

    /*
    cout << "Determinant " << determinant << endl;
    for(int i=0;i<4;i++){
        for(int j=0;j<4;j++){
            cout << b[i][j] << " ";
        }
        cout << endl;
    }
    cout << endl;
    */

    if(trans_null == false){
    double v[4];
    for(int i=0;i<4;i++){
        v[i] = (double)trans[i]* 45.0/256.0;
    }
    Vec3Float64 axis;
    axis.x = v[0];
    axis.y = v[1];
    axis.z = v[2];
    double n = 1.0 / sqrt(Vec3Float64::dot(&axis,&axis));
    axis.x = axis.x * n;
    axis.y = axis.y * n;
    axis.z = axis.z * n;
    v[3] = v[3] * 360.0 / 256.0;

    // fn from_axis_angle
    double sn,cs;
    sn = sin(v[3] * M_PI/180.0);
    cs = cos(v[3] * M_PI/180.0);
    double _lsubc = 1.0 - cs;
    bb[0][0] = _lsubc * axis.x * axis.x + cs;
    bb[0][1] = _lsubc * axis.x * axis.y + sn * axis.z;
    bb[0][2] = _lsubc * axis.x * axis.z - sn * axis.y;
    bb[0][3] = 0;

    bb[1][0] = _lsubc * axis.x * axis.y - sn * axis.z;
    bb[1][1] = _lsubc * axis.y * axis.y + cs;
    bb[1][2] = _lsubc * axis.y * axis.z + sn * axis.x;
    bb[1][3] = 0;

    bb[2][0] = _lsubc * axis.x * axis.z + sn * axis.y;
    bb[2][1] = _lsubc * axis.y * axis.z - sn * axis.x;
    bb[2][2] = _lsubc * axis.z * axis.z +cs;
    bb[2][3] = 0;

    bb[3][0] = 0;
    bb[3][1] = 0;
    bb[3][2] = 0;
    bb[3][3] = 1;
   }

        Vec3Float64 v_min,v_max;
    v_min.x = 10000000;
    v_min.y = 10000000;
    v_min.z = 10000000;
    v_max.x = -10000000;
    v_max.y = -10000000;
    v_max.z = -10000000;

    //Shift + Translate + Rotate
    for(int i=0;i<geo.len;i++){
        
        geo.positions[i].x = geo.positions[i].x + pit[0][3];
        geo.positions[i].y = geo.positions[i].y + pit[1][3];
        geo.positions[i].z = geo.positions[i].z + pit[2][3];

        double t1 =
            geo.positions[i].x * b[0][0] +
            geo.positions[i].y * b[1][0] +
            geo.positions[i].z * b[2][0] +
            1 * b[3][0];
        double t2 =
            geo.positions[i].x * b[0][1] +
            geo.positions[i].y * b[1][1] +
            geo.positions[i].z * b[2][1] +
            1 * b[3][1];
        double t3 =
            geo.positions[i].x * b[0][2] +
            geo.positions[i].y * b[1][2] +
            geo.positions[i].z * b[2][2] +
            1 * b[3][2];

        if(trans_null == false){

            double tt1 =
                t1 * bb[0][0] +
                t2 * bb[1][0] +
                t3 * bb[2][0] +
                1 * bb[3][0];
            double tt2 =
                t1 * bb[0][1] +
                t2 * bb[1][1] +
                t3 * bb[2][1] +
                1 * bb[3][1];
            double tt3 =
                t1 * bb[0][2] +
                t2 * bb[1][2] +
                t3 * bb[2][2] +
                1 * bb[3][2];
            
            if(tt1 > v_max.x)
                v_max.x = tt1;
            if(tt1 < v_min.x)
                v_min.x = tt1;

            if(tt2 > v_max.y)
                v_max.y = tt2;
            if(tt2 < v_min.y)
                v_min.y = tt2;
            
            if(tt3 > v_max.z)
                v_max.z = tt3;
            if(tt3 < v_min.z)
                v_min.z = tt3;

            geo.positions[i].x = tt1;
            geo.positions[i].y = tt2;
            geo.positions[i].z = tt3;
        } else {
            if(t1 > v_max.x)
                v_max.x = t1;
            if(t1 < v_min.x)
                v_min.x = t1;

            if(t2 > v_max.y)
                v_max.y = t2;
            if(t2 < v_min.y)
                v_min.y = t2;
            
            if(t3 > v_max.z)
                v_max.z = t3;
            if(t3 < v_min.z)
                v_min.z = t3;

            geo.positions[i].x = t1;
            geo.positions[i].y = t2;
            geo.positions[i].z = t3;
        }
        //cout <<"#" << i << "  " << geo.positions[i].str() << endl;
    }

    //printf("%lf %lf %lf | %lf %lf %lf\n",v_min.x,v_min.y,v_min.z,v_max.x,v_max.y,v_max.z);
    
    double step = (v_max.z - v_min.z) / (1.0 + n_sections);
    Vec2F64 * cntr = (Vec2F64*)mt_global;
    
    int cntrs_len[12+1] = {0};
    int psum = 0;
    int goFurther = 1;

    for(int n=0;n<n_sections;n++){
        double z_sect = v_min.z + (n + 1.0) * step;
        int cntr_len = 0;
        get_contour_opt_cuda(&geo,z_sect,cntr + psum,cntr_len,(Vec2F64*)(mt_global + 1024 * sizeof(Vec2F64)),(short*)(mt_global+(1024+512) * sizeof(Vec2F64) ));

        if(cntr_len == 0){
            goFurther = 0;
            break;
        }
       // printf("%d\n",cntr_len);
        
        //Vec2F64 * cc = cntr + psum;

        
        /*
        printf("cntr #%d\n",cntr_len);
        for(int i=0;i<cntr_len;i++){
            printf("(%lf,%lf)  ",cc[i].x,cc[i].x);
        }
        printf("\n\n");
        */
        
        
        cntrs_len[n+1] = cntr_len;
        psum += cntr_len;
    }


    
    /*
    for(int i=0;i<psum;i++){
        printf("(%lf,%lf) ",cntr[i].x,cntr[i].y);
    }
    printf("\n");
    */
   out_hash_len_arr[idx]=-1;
    if(goFurther == 0)
    return;


    unsigned char out_hash[512];
    int out_hash_len = 0;
    //printf("%d\n",idx);

    //return;

    find_top_std_2_cuda(cntr,cntrs_len,10,n_sections,grid_size,v_min,v_max,out_hash,out_hash_len);

    out_hash_len_arr[idx] = out_hash_len;
    if(out_hash_len>0)
    memcpy(out_hash_arr + idx * 32,out_hash,32);


    //#define PRINT_DEBUG
     //printf("out hash len: %d\n",out_hash_len);
     #if defined(PRINT_DEBUG)
    if(idx <100){
        for(int i=0;i<out_hash_len;i++){
        printf("(%d)  hash %d | ",idx,i);
        for(int j=0;j<32;j++){
                unsigned char c = out_hash[i * 32 + j];
                char c1,c2;
                c1 = c/16;
                if(c1<10){
                    c1 = '0' + c1;
                } else{
                    c1 = 'a'+(c1-10);
                }
                c2 = c%16;
                if(c2<10){
                    c2 = '0' + c2;
                } else {
                    c2 = 'a' + (c2-10);
                }
                printf("%c%c",c1,c2);
            }
            printf("\n");
        }
    }
    #endif

}

__global__ void initCurand(curandState *state,unsigned long seed) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, idx, 0, &state[idx]);
}

#define MT_SIZE 128

__global__ void kernel(GpuContext context,Sphere * sphere_gpu,curandState * rand_states, unsigned char * buffer,unsigned char * mt_global,unsigned char * outhash,int * outlen,RockObjParams * output_rocks
,unsigned char * best_hash,unsigned char * pre_hash,unsigned char * diffBytes,unsigned char * cmpBytes,unsigned char * valid){
    const int THIS = (blockIdx.x * blockDim.x + threadIdx.x);
    int idx = THIS;
    Sphere * sphere = &context.sphere[THIS];
    sphere->len = sphere_gpu->len;
    sphere->len_indicies = sphere_gpu->len_indicies;
    /*if(idx == 0){
        printf("init sphere %d %d\n",sphere_gpu->len,sphere_gpu->len_indicies);
    }
    */
    for(int i=0;i<sphere->len;i++){
        sphere->vertices[i].x = sphere_gpu->vertices[i].x;
        sphere->vertices[i].y = sphere_gpu->vertices[i].y;
        sphere->vertices[i].z = sphere_gpu->vertices[i].z;
        sphere->normals[i].x = sphere_gpu->normals[i].x;
        sphere->normals[i].y = sphere_gpu->normals[i].y;
        sphere->normals[i].z = sphere_gpu->normals[i].z;
    }
    for(int i=0;i<sphere->len_indicies;i++){
        sphere->indices[i].x = sphere_gpu->indices[i].x;
        sphere->indices[i].y = sphere_gpu->indices[i].y;
        sphere->indices[i].z = sphere_gpu->indices[i].z;
    }
    //CreateSphere(sphere,1,25,25);
    RockCuda(pre_hash,sphere,&context.sphere_keep[THIS],rand_states,idx,buffer,mt_global + 2048 * sizeof(Vec2F64) * THIS,outhash,outlen,&output_rocks[THIS]);

    if(outlen[THIS] <= 0)
        return;

    uint2 keccak_gpu_state[25];

    unsigned char sealPre[96],seal[32];
    memcpy(sealPre,pre_hash,32);
    memcpy(sealPre+32,outhash+32*THIS,32);

    sha3_cuda(keccak_gpu_state,sealPre,64,seal);

    memcpy(sealPre,diffBytes,32);
    memcpy(sealPre+32,pre_hash,32);
    memcpy(sealPre+64,seal,32);

    sha3_cuda(keccak_gpu_state,sealPre,96,seal);

    int cmp=1;
    for (int ii = 0; ii < 32; ii++) {
        if (cmpBytes[ii] > seal[ii]) {
            cmp = 1;
            break;
        }
        else if (cmpBytes[ii] < seal[ii]) {
            cmp = -1;
            break;
        }
    }
    bool v = (cmp > 0);
    if(v){
        valid[THIS] = 1;
    }
    
}

__global__ void init(Sphere * sphere,curandState * rand_states, unsigned char * buffer,int stacks=12,int slices=19){
    const int THIS = (blockIdx.x * blockDim.x + threadIdx.x);
    int idx = THIS;
    //Sphere * sphere = &context.sphere[THIS];
    CreateSphere(sphere,1,stacks,slices);
    //printf("len %d\n",sphere->len);
    //printf("len ind %d\n",sphere->len_indicies);
    CellRet adjacentVertices;
    GetNeighbours(sphere->len,sphere->len_indicies,sphere->indices,&adjacentVertices,buffer + 0 * (602 * 65 * 4));
    
}

GpuContext gpu_contexts[16];
Sphere * sphere_gpu_list[16];
RockObjParams * obj_params_gpu[16];
curandState * devStateList[16];
unsigned char * bigDevBufferList[16];
unsigned char * mt_global_list[16];

void initGpuData(int x,int blocks,int threads,int sp_stacks,int sp_slices){
    cudaSetDevice(x);
    size_t lim;
    size_t stack_size;

    size_t stack_size_thread=0;
    cudaDeviceGetLimit(&lim, cudaLimitMallocHeapSize);
    cudaDeviceGetLimit(&stack_size, cudaLimitStackSize);
    cudaThreadGetLimit(&stack_size_thread,cudaLimitStackSize);

    //cout << lim << " " << stack_size << endl;
    

    cudaError_t err = cudaDeviceSetLimit(cudaLimitMallocHeapSize, 8388608 * 12);
    err = cudaDeviceSetLimit(cudaLimitStackSize, 1024 * 20);

    //Sphere * sphere_gpu;

    //curandState *devState;
    cudaMalloc((void**)&sphere_gpu_list[x],sizeof(Sphere));
    (cudaMalloc((void**)&devStateList[x], blocks * threads *sizeof(curandState)));
    std::chrono::high_resolution_clock m_clock;
    long long seed = std::chrono::duration_cast<std::chrono::milliseconds>
              (m_clock.now().time_since_epoch()).count() % 100000000000LL;
    initCurand<<<blocks,threads>>>(devStateList[x], seed);
    cudaDeviceSynchronize();
    CHECK_LAST_CUDA_ERROR();
    //cout << "Cuda rand ok\n";

    cudaMalloc((void**)&gpu_contexts[x].sphere,sizeof(Sphere) * blocks * threads);
    cudaMalloc((void**)&gpu_contexts[x].sphere_keep,sizeof(Sphere) * blocks * threads);


    cudaMalloc((void**)&bigDevBufferList[x],1024 * 1024 * 16 * 1);
    cudaMalloc((void**)&mt_global_list[x],sizeof(Vec2F64) * 2048 * blocks*threads);

    cudaMalloc((void**)&obj_params_gpu[x],sizeof(RockObjParams) * blocks * threads);

    //cout << "alloc " << sizeof(Sphere) * blocks * threads << endl;
    

    //cout << "Init on blocks:" << blocks << " threads:" << threads << endl;
    

    init << <1, 1 >> > (sphere_gpu_list[x],devStateList[x],bigDevBufferList[x],sp_stacks,sp_slices);
    CHECK_LAST_CUDA_ERROR();
    cudaDeviceSynchronize();
}

vector<GPUSolution> doGpuBatch(int x,int blocks,int threads,unsigned char * outhash,int * outlen,unsigned char * best_hash,unsigned char * pre_hash,unsigned char * diffBytes,unsigned char * cmpBytes){
    int bench = 0;
    if(x < 0){
        bench = 1;
        x=x+100;
    }
    cudaSetDevice(x);
    unsigned char * d_outhash;
    int * d_outlen;

    unsigned char * best_hash_gpu;
    unsigned char * pre_hash_gpu;
    unsigned char * diffBytes_gpu;
    unsigned char * cmpBytes_gpu;
    unsigned char * valid;


    cudaMalloc((void**)&best_hash_gpu,32);
    cudaMalloc((void**)&pre_hash_gpu,32);
    cudaMalloc((void**)&diffBytes_gpu,32);
    cudaMalloc((void**)&cmpBytes_gpu,32);
    cudaMalloc((void**)&valid,blocks*threads);

    cudaMemcpy(best_hash_gpu,best_hash,32,cudaMemcpyHostToDevice);
    cudaMemcpy(pre_hash_gpu,pre_hash,32,cudaMemcpyHostToDevice);
    cudaMemcpy(diffBytes_gpu,diffBytes,32,cudaMemcpyHostToDevice);
    cudaMemcpy(cmpBytes_gpu,cmpBytes,32,cudaMemcpyHostToDevice);
    cudaMemset(valid,0,blocks*threads);

    cudaMalloc((void**)&d_outhash,blocks * threads * 32);
    cudaMalloc((void**)&d_outlen,blocks*threads*4);
    kernel << <blocks, threads >> > (gpu_contexts[x],sphere_gpu_list[x],devStateList[x],bigDevBufferList[x],mt_global_list[x],d_outhash,d_outlen,obj_params_gpu[x],best_hash_gpu,pre_hash_gpu,diffBytes_gpu,cmpBytes_gpu,valid);
    CHECK_LAST_CUDA_ERROR();
    cudaDeviceSynchronize();

    if(bench){
    cudaMemcpy(outhash,d_outhash,32 * blocks*threads,cudaMemcpyDeviceToHost);
    cudaMemcpy(outlen,d_outlen,4 * blocks* threads,cudaMemcpyDeviceToHost);
    }

    unsigned char  * valid_host = new unsigned char[blocks*threads];
    cudaMemcpy(valid_host,valid,blocks*threads,cudaMemcpyDeviceToHost);

    vector<GPUSolution> sol;

    for(int i=0;i<blocks*threads;i++){
        if(valid_host[i]== 1){
            GPUSolution s;
            cudaMemcpy(&s.param,&obj_params_gpu[x][i],sizeof(RockObjParams),cudaMemcpyDeviceToHost);
            cudaMemcpy(s.obj_hash,d_outhash + 32 * i,32,cudaMemcpyDeviceToHost);
            //printf("%d  %lf %lf %lf\n",s.param.scrapeCount,s.param.scale[0],s.param.scale[1],s.param.scale[2]);
            //printf("solution at thread %d\n",i);
            sol.push_back(s);
        }
    }

    cudaFree(best_hash_gpu);
    cudaFree(pre_hash_gpu);
    cudaFree(diffBytes_gpu);
    cudaFree(cmpBytes_gpu);
    cudaFree(valid);

    cudaFree(d_outhash);
    cudaFree(d_outlen);

    delete[] valid_host;

    return sol;
}