#include "sphere.h"
#include "cuda_runtime.h"
#include <math.h>


void InitSphere(Sphere * sphere){
    /*
    sphere->vertices = (Vec3Float64*)malloc(1806 * sizeof(Vec3Float64));
    sphere->indices = (Vec3Uint*)malloc(3600 * sizeof(Vec3Uint));
    sphere->normals = (Vec3Float64*)malloc(1806 * sizeof(Vec3Float64));
    */
}

void InitSphereCuda(Sphere * sphere){
    sphere->run_mode = RUN_ON_GPU;
    /*
    cudaMalloc((void**)&sphere->vertices, 1806 * sizeof(Vec3Float64));
    cudaMalloc((void**)&sphere->indices, 3600 * sizeof(Vec3Uint));
    cudaMalloc((void**)&sphere->normals, 1806 * sizeof(Vec3Float64));
    */
}

__host__ __device__ double noise(){
    return (rand()% 1000000)/1000000.0 * 0.1;
}

__host__ __device__ void CreateSphere(Sphere * sphere,double radius, unsigned int stacks, unsigned int slices,int shape) {


    if(shape == SPHERE_NORMAL){
        int vertices_len = 0;
        int normals_len = 0;
        int indices_len=0;

        // keeps track of the index of the next vertex that we create.
        unsigned int index = 0;

        /*
        First of all, we create all the faces that are NOT adjacent to the
        bottom(0,-R,0) and top(0,+R,0) vertices of the sphere->

        (it's easier this way, because for the bottom and top vertices, we need to add triangle faces.
        But for the faces between, we need to add quad faces. )
        */

        // loop through the stacks.

        double base_radius = radius;

        for (int i=1;i<stacks;i++) {
            double u = (double)i / (double)stacks;
            double phi = u * M_PI;

            int stackBaseIndex = indices_len >> 1;


            // loop through the slices.
            for(int j=0;j<slices;j++) {
                double v = (double)j / (double)slices;
                double theta = v * (M_PI * 2.0);
                double sgn = 1.0;
                if((i*slices+j)%2==1)sgn = -1.0;
                double R = radius;
                // use spherical coordinates to calculate the positions.
                double x = cos(theta) * sin(phi);
                double y = cos(phi);//* ((double)j/(double)slices * (double)i/(double)stacks*0.05 * sgn + 0.95);
                double z = sin(theta) * sin(phi);
                sphere->vertices[vertices_len].x = R *x;
                sphere->vertices[vertices_len].y = R *y;
                sphere->vertices[vertices_len].z = R *z;
                sphere->normals[normals_len].x = x;
                sphere->normals[normals_len].y = y;
                sphere->normals[normals_len].z = z;
                vertices_len++;
                normals_len++;

                if(i + 1 != stacks) {
                    // for the last stack, we don't need to add faces.

                    unsigned int i1,i2,i3,i4;
                    if(j + 1 == slices) {
                        // for the last vertex in the slice, we need to wrap around to create the face.
                        i1 = index;
                        i2 = (unsigned int)stackBaseIndex;
                        i3 = index + slices;
                        i4 = (unsigned int)stackBaseIndex + slices;
                    } else {
                        // use the indices from the current slice, and indices from the next slice, to create the face.
                        i1 = index;
                        i2 = index + 1;
                        i3 = index + slices;
                        i4 = index + slices + 1;
                    }

                    // add quad face
                    sphere->indices[indices_len].x = i1;
                    sphere->indices[indices_len].y = i2;
                    sphere->indices[indices_len].z = i3;
                    indices_len++;
                    sphere->indices[indices_len].x = i4;
                    sphere->indices[indices_len].y = i3;
                    sphere->indices[indices_len].z = i2;
                    indices_len++;
                }

                index=index+1;
            }
        }

        /*
        Next, we finish the sphere by adding the faces that are adjacent to the top and bottom vertices.
        */

        unsigned int topIndex = index; index=index+1;
        sphere->vertices[vertices_len].x = 0;
        sphere->vertices[vertices_len].y = radius;
        sphere->vertices[vertices_len].z = 0; 
        vertices_len++;
        sphere->normals[normals_len].x = 0;
        sphere->normals[normals_len].y = 1;
        sphere->normals[normals_len].z = 0;
        normals_len++;

        unsigned int bottomIndex = index; //index=index+1;

        sphere->vertices[vertices_len].x = 0;
        sphere->vertices[vertices_len].y = -radius;
        sphere->vertices[vertices_len].z = 0;
        vertices_len++;
        sphere->normals[normals_len].x = 0;
        sphere->normals[normals_len].y = -1;
        sphere->normals[normals_len].z = 0;
        normals_len++;

        for(int i=0;i<slices;i++) {
            unsigned int i1 = topIndex;
            unsigned int i2 = i + 0;
            unsigned int i3 = (i + 1) % slices;

            sphere->indices[indices_len].x = i3;
            sphere->indices[indices_len].y = i2;
            sphere->indices[indices_len].z = i1;
            indices_len++;

            i1 = bottomIndex;
            i2 = bottomIndex - 1 - slices + (i + 0);
            i3 = bottomIndex - 1 - slices + ((i + 1) % slices);

            sphere->indices[indices_len].x = i1;
            sphere->indices[indices_len].y = i2;
            sphere->indices[indices_len].z = i3;
            indices_len++;
        }

        sphere->len = vertices_len;
        sphere->len_indicies = indices_len;
    } else if(shape == SPHERE_RANDOM){
        int vertices_len = 0;
        int normals_len = 0;
        int indices_len=0;
        for(int i=0;i<12;i++){
            double z = i * 0.2;
            double ang = 0;
            double rp= 8;
            double ang_step = 360/rp;
            double center_x = 0;
            double center_y = 0;
            //len_verts = verticie
            double x_min = 100000;
            double y_min = 100000;
            double x_max = -100000;
            double y_max = -100000;
            while (ang < 360){
                sphere->vertices[vertices_len].x = center_x + cos(M_PI * ang/180) * 1 + noise();
                sphere->vertices[vertices_len].y = center_y + sin(M_PI * ang/180) * 1 + noise();
                sphere->vertices[vertices_len].z = z;
                sphere->normals[normals_len].x = 0;
                sphere->normals[normals_len].y = 0;
                sphere->normals[normals_len].z = 0;
                vertices_len++;
                normals_len++;
                x_min = min(x_min,sphere->vertices[vertices_len-1].x);
                y_min = min(y_min,sphere->vertices[vertices_len-1].y);
                x_max = max(x_max,sphere->vertices[vertices_len-1].x);
                y_max = max(y_max,sphere->vertices[vertices_len-1].y);
                ang += ang_step;
                if(vertices_len >= 3){
                    int lv = vertices_len-1;

                    sphere->indices[indices_len].x = lv-2;
                    sphere->indices[indices_len].y = lv-1;
                    sphere->indices[indices_len].z = lv;
                    indices_len++;
                }
            }
            if(vertices_len >= 3){
                    int lv = vertices_len-1;

                    sphere->indices[indices_len].x = lv-(rp-1);
                    sphere->indices[indices_len].y = lv-(rp-2);
                    sphere->indices[indices_len].z = lv;
                    indices_len++;
                    sphere->indices[indices_len].x = lv-(rp-1);
                    sphere->indices[indices_len].y = lv-1;
                    sphere->indices[indices_len].z = lv;
                    indices_len++;
            }


            if(i>=2)
            continue;
            for(int stack=0;stack<stacks;stack++){
                for(int slice=0;slice<slices;slice++){
                    double px = ((rand()% 100000000)/100000000.0) * (x_max - x_min) + x_min;
                    double py = ((rand()% 100000000)/100000000.0) * (y_max - y_min) + y_min;
                    sphere->vertices[vertices_len].x = px;
                    sphere->vertices[vertices_len].y = py;
                    sphere->vertices[vertices_len].z = z;
                    //printf("%lf %lf %lf\n",px,py,z);
                    vertices_len++;
                    if(slice >= 2){
                        int lv = vertices_len - 1;
                        sphere->indices[indices_len].x = lv-1;
                        sphere->indices[indices_len].y = lv-2;
                        sphere->indices[indices_len].z = lv;
                        indices_len++;
                    }
                }
            }
            
        }
        sphere->len = vertices_len;
        sphere->len_indicies = indices_len;
    }
}
