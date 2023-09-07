#include <iostream>
#include "scrape.h"
#include <vector>
#include "math_utils.h"
#include "simple_deque.h"
#include "cuda_runtime.h"
#include "buffer_geometry.h"
#include <unistd.h>
#include <deque>

__host__ __device__  int wrap(int i) {
    if(i < 0) {
        return 3 + i;
    } else {
        return i % 3;
    }
}

__host__ __device__ int CellRet::pos(int pos1,int pos2){
    int p = pos1 * size2 + pos2;
    return p;
}

__host__ __device__ void GetNeighbours(unsigned int positions_len,unsigned int len_ind, Vec3Uint * cells,CellRet * adjacentVertices,unsigned char * buffer ) {
    /*
     adjacentVertices[i] contains a set containing all the indices of the neighbours of the vertex with
     index i.
     A set is used because it makes the algorithm more convenient.
     */
    //let adjacentVertices = [cell; positions.len()];
    
    if(buffer == NULL){
        adjacentVertices->vec = (unsigned int*) malloc(adjacentVertices->size * adjacentVertices->size2 * 4);
    } else {
        adjacentVertices->vec = (unsigned int*)buffer;
    }

    for(int i=0;i<positions_len;i++){
        adjacentVertices->vec[ adjacentVertices->pos(i,adjacentVertices->size2-1)  ] = 0;
    }


    //memset(adjacentVertices->vec,0,adjacentVertices->size * adjacentVertices->size2 * 4);
    // go through all faces.
    
    for(int iCell = 0;iCell <len_ind;iCell++){
        Vec3Uint cellPositions = cells[iCell];
        // go through all the points of the face.
        for(int iPosition=0; iPosition<3;iPosition++) {
            // the neighbours of this points are the previous and next points(in the array)
            int p1 = wrap(iPosition + 0);
            int p2 = wrap(iPosition - 1);
            int p3 = wrap(iPosition + 1);
            unsigned int cur,prev,next;        

            if(p1 == 0)
                cur = cellPositions.x;
            else if(p1 == 1)
                cur = cellPositions.y;
            else if(p1 == 2)
                cur = cellPositions.z;
            
            if(p2 == 0)
                prev = cellPositions.x;
            else if(p2 == 1)
                prev = cellPositions.y;
            else if(p2 == 2)
                prev = cellPositions.z;
            
            if(p3 == 0)
                next = cellPositions.x;
            else if(p3 == 1)
                next = cellPositions.y;
            else if(p3 == 2)
                next = cellPositions.z;

            // create set on the fly if necessary.
            // if (adjacentVertices[cur] == null) {
            //     adjacentVertices[cur] = cell::new();
            // }
            // add adjacent vertices.

            adjacentVertices->vec[adjacentVertices->pos(cur, adjacentVertices->vec[adjacentVertices->pos(cur,adjacentVertices->size2-1) ] )]= prev;
            adjacentVertices->vec[adjacentVertices->pos(cur,adjacentVertices->size2-1)] ++;
            adjacentVertices->vec[adjacentVertices->pos(cur, adjacentVertices->vec[adjacentVertices->pos(cur,adjacentVertices->size2-1) ] )]= next;
            adjacentVertices->vec[adjacentVertices->pos(cur,adjacentVertices->size2-1)]++;

           
        }
    }
}


/*
Projects the point `p` onto the plane defined by the normal `n` and the point `r0`
 */


__host__ __device__  Vec3Float64 Project( Vec3Float64 n, Vec3Float64 r0, Vec3Float64 p){
    // For an explanation of the math, see http://math.stackexchange.com/a/100766

    Vec3Float64 o;
    o.x = r0.x - p.x;
    o.y = r0.y - p.y;
    o.z = r0.z - p.z;
    double t =  VECf64_DOT(n,o) / VECf64_DOT(n,n);
    Vec3Float64 ret;
    ret.x = p.x + n.x * t;
    ret.y = p.y + n.y * t;
    ret.z = p.z + n.z * t;
     
    return ret;
}

// scrape at vertex with index `positionIndex`.
__host__ __device__ void scrapeMain(
    unsigned int positionIndex,
    Vec3Float64 * positions,
    Vec3Float64 * normals,
    CellRet * adjacentVertices,
    double strength,
    double radius,
    bool * traversed,
    Deque * stack
  //  stack:&mut VecDeque<usize>,
)
{

    Vec3Float64 centerPosition = positions[positionIndex];
    // to scrape, we simply project all vertices that are close to `centerPosition`
    // onto a plane. The equation of this plane is given by dot(n, r-r0) = 0,
    // where n is the plane normal, r0 is a point on the plane(in our case we set this to be the projected center),
    // and r is some arbitrary point on the plane.
    Vec3Float64 n = normals[positionIndex];
    Vec3Float64 r0;
    r0.x = centerPosition.x + n.x * -strength;
    r0.y = centerPosition.y + n.y * -strength;
    r0.z = centerPosition.z + n.z * -strength;
    stack->insertrear(positionIndex);
    /*
     We use a simple flood-fill algorithm to make sure that we scrape all vertices around the center.
     This will be fast, since all vertices have knowledge about their neighbours.
     */
    while (stack->isEmpty() == false) {
        unsigned int topIndex = stack->getFront();
        stack->deletefront();
        if(traversed[topIndex]){continue;} // already traversed; look at next element in stack.
        traversed[topIndex] = true;
        // project onto plane.
        Vec3Float64 p = positions[topIndex];
        Vec3Float64 projectedP;
        
        
        projectedP = Project(n, r0, p);


        double dist = (projectedP.x - r0.x) * (projectedP.x - r0.x)  +  (projectedP.y - r0.y) * (projectedP.y - r0.y)  +  (projectedP.z - r0.z) * (projectedP.z - r0.z);
        if(dist < radius) {
            positions[topIndex] = projectedP;
            normals[topIndex] = n;
        } else {
            continue;
        }
        unsigned int * v = adjacentVertices->vec + adjacentVertices->size2 * topIndex;
        for(int i=0; i < v[adjacentVertices->size2-1];i++) {
            stack->insertrear(v[i]);
        }
    }
}

void scrapeMainStd(
    unsigned int positionIndex,
    Vec3Float64 * positions,
    Vec3Float64 * normals,
    CellRet * adjacentVertices,
    double strength,
    double radius,
    bool * traversed,
    std::deque<int> &stack
  //  stack:&mut VecDeque<usize>,
)
{

    Vec3Float64 centerPosition = positions[positionIndex];
    // to scrape, we simply project all vertices that are close to `centerPosition`
    // onto a plane. The equation of this plane is given by dot(n, r-r0) = 0,
    // where n is the plane normal, r0 is a point on the plane(in our case we set this to be the projected center),
    // and r is some arbitrary point on the plane.
    Vec3Float64 n = normals[positionIndex];
    Vec3Float64 r0;
    r0.x = centerPosition.x + n.x * -strength;
    r0.y = centerPosition.y + n.y * -strength;
    r0.z = centerPosition.z + n.z * -strength;
    stack.push_back(positionIndex);
    /*
     We use a simple flood-fill algorithm to make sure that we scrape all vertices around the center.
     This will be fast, since all vertices have knowledge about their neighbours.
     */
    while (stack.empty() == false) {
        unsigned int topIndex = stack.front();
        stack.pop_front();
        if(traversed[topIndex]){continue;} // already traversed; look at next element in stack.
        traversed[topIndex] = true;
        // project onto plane.
        Vec3Float64 p = positions[topIndex];
        Vec3Float64 projectedP;
        
        
        projectedP = Project(n, r0, p);


        double dist = (projectedP.x - r0.x) * (projectedP.x - r0.x)  +  (projectedP.y - r0.y) * (projectedP.y - r0.y)  +  (projectedP.z - r0.z) * (projectedP.z - r0.z);
        if(dist < radius) {
            positions[topIndex] = projectedP;
            normals[topIndex] = n;
        } else {
            continue;
        }
        unsigned int * v = adjacentVertices->vec + adjacentVertices->size2 * topIndex;
        for(int i=0; i < v[adjacentVertices->size2-1];i++) {
            stack.push_back(v[i]);
        }
    }
}

__host__ __device__ void get_contour(BufferGeometry * geo,double z_sect,Vec2F64 * cntr,int &cntr_len){
    
    Vec2F64 sect[512];
    float mt[512][512];
    short ii[512];
    int sect_len = 0;
    for(int vertex_id = 0;vertex_id < geo->len;vertex_id++){
        Vec3Float64 p = geo->positions[vertex_id];
        if( abs(p.z - z_sect) < 0.15) {
           // printf("add to contour %d -> %lf %lf %lf\n",vertex_id,p.x,p.y,p.z );
            if(sect_len>=512){
                printf("OVERFLOW ERROR, buff size of 512 exceeded\nReport this !!!\n");
                return;
            }
            sect[sect_len].x = p.x;
            sect[sect_len].y = p.y;
            ii[sect_len] = sect_len;
            sect_len ++;
                // sect.push(Vec2{x: p.x,y:p.y});
        }
    }
    ii[sect_len] = sect_len;
    if(sect_len == 0){
        return;
    }
    // return ?
    for(int i=0;i<sect_len;i++){
        for(int j=0;j<sect_len;j++){
            mt[i][j] = (sect[i].x - sect[j].x) * (sect[i].x - sect[j].x) +
                        (sect[i].y - sect[j].y) * (sect[i].y - sect[j].y);
        }
    }
    for(int i=0;i<sect_len-1;i++){ // am bagat aici mare -1 ?????
        float * v = mt[ii[i]];
        //float mn = __FLT_MAX__;
        unsigned int mn = UINT32_MAX;
        int j = -1;
        for(int k=i+1;k<sect_len;k++){
            unsigned int elm = (unsigned int)(v[ii[k]] * 10000.0);
            if(elm < mn){
                //mn = v[ii[k]];
                mn = elm;
                j = k;
            } 
        }
        short tmp = ii[i+1];
        ii[i+1] = ii[j];
        ii[j] = tmp;
    }
    cntr_len = 0;
    for(int i=0;i<sect_len;i++){
        cntr[i].x = sect[ii[i]].x;
        cntr[i].y = sect[ii[i]].y;
        cntr_len++;
    }
    cntr[cntr_len].x = cntr[0].x;
    cntr[cntr_len].y = cntr[0].y;
    cntr_len++;

    Vec2F64 p0 = cntr[0];
    Vec2F64 p1 = cntr[1];
    Vec2F64 pn = cntr[cntr_len-1];
    double d = sqrt( sqrt( (p0.x-pn.x) * (p0.x - pn.x) + (p0.y - pn.y) * (p0.y - pn.y) ) );
    double d2 = sqrt( sqrt( (p0.x -p1.x) * (p0.x -p1.x) + (p0.y - p1.y) * (p0.y - p1.y) ));
    int nn = (int)round(d/d2);

    //printf("cntr last round = %d   %lf %lf\n",nn,d,d2);

    for(int n=0;n<nn;n++){
        double k = (pn.y - p0.y) / (pn.x - p0.x);
        cntr[cntr_len].x = p0.x + (double)n * d2;
        cntr[cntr_len].y = p0.y + (double)n * d2 * k;
        cntr_len++;
    }
}


__host__ __device__ void get_contour_opt(BufferGeometry * geo,double z_sect,Vec2F64 * cntr,int &cntr_len){
    
    Vec2F64 sect[512];
    //float mt[512][512];
    short ii[512];
    int sect_len = 0;
    for(int vertex_id = 0;vertex_id < geo->len;vertex_id++){
        Vec3Float64 p = geo->positions[vertex_id];
        if( abs(p.z - z_sect) < 0.15) {
           // printf("add to contour %d -> %lf %lf %lf\n",vertex_id,p.x,p.y,p.z );
            if(sect_len>=512){
                printf("OVERFLOW ERROR, buff size of 512 exceeded\nReport this !!!\n");
                return;
            }
            sect[sect_len].x = p.x;
            sect[sect_len].y = p.y;
            ii[sect_len] = sect_len;
            sect_len ++;
                // sect.push(Vec2{x: p.x,y:p.y});
        }
    }
    ii[sect_len] = sect_len;
    if(sect_len == 0){
        return;
    }
    // return ?
    /*
    for(int i=0;i<sect_len;i++){
        for(int j=0;j<sect_len;j++){
            mt[i][j] = (sect[i].x - sect[j].x) * (sect[i].x - sect[j].x) +
                        (sect[i].y - sect[j].y) * (sect[i].y - sect[j].y);
        }
    }
    */
    #define DIST(i,j) ((sect[i].x - sect[j].x) * (sect[i].x - sect[j].x) + (sect[i].y - sect[j].y) * (sect[i].y - sect[j].y))
    for(int i=0;i<sect_len-1;i++){ // am bagat aici mare -1 ?????
        //float * v = mt[ii[i]];
        //float mn = __FLT_MAX__;
        unsigned int mn = UINT32_MAX;
        int j = -1;
        for(int k=i+1;k<sect_len;k++){
            unsigned int elm = (unsigned int)(DIST(ii[i],ii[k]) * 10000.0);
            //unsigned int elm = (unsigned int)(v[ii[k]] * 10000.0);
            if(elm < mn){
                //mn = v[ii[k]];
                mn = elm;
                j = k;
            } 
        }
        short tmp = ii[i+1];
        ii[i+1] = ii[j];
        ii[j] = tmp;
    }

    #undef DIST
    cntr_len = 0;
    for(int i=0;i<sect_len;i++){
        cntr[i].x = sect[ii[i]].x;
        cntr[i].y = sect[ii[i]].y;
        cntr_len++;
    }
    cntr[cntr_len].x = cntr[0].x;
    cntr[cntr_len].y = cntr[0].y;
    cntr_len++;

    Vec2F64 p0 = cntr[0];
    Vec2F64 p1 = cntr[1];
    Vec2F64 pn = cntr[cntr_len-1];
    double d = sqrt( sqrt( (p0.x-pn.x) * (p0.x - pn.x) + (p0.y - pn.y) * (p0.y - pn.y) ) );
    double d2 = sqrt( sqrt( (p0.x -p1.x) * (p0.x -p1.x) + (p0.y - p1.y) * (p0.y - p1.y) ));
    int nn = (int)round(d/d2);

    //printf("cntr last round = %d   %lf %lf\n",nn,d,d2);

    for(int n=0;n<nn;n++){
        double k = (pn.y - p0.y) / (pn.x - p0.x);
        cntr[cntr_len].x = p0.x + (double)n * d2;
        cntr[cntr_len].y = p0.y + (double)n * d2 * k;
        cntr_len++;
    }
}

__host__ __device__ void get_contour_opt_cuda(BufferGeometry * geo,double z_sect,Vec2F64 * cntr,int &cntr_len,Vec2F64 * sect,short * ii){
    
    //float mt[512][512];
    int sect_len = 0;
    Vec3Float64 p;
    for(int vertex_id = 0;vertex_id < geo->len;vertex_id++){
        p = geo->positions[vertex_id];
        if( abs(p.z - z_sect) < 0.15) {
           // printf("add to contour %d -> %lf %lf %lf\n",vertex_id,p.x,p.y,p.z );
            if(sect_len>=256){
                printf("OVERFLOW ERROR, buff size of 512 exceeded\nReport this !!!\n");
                return;
            }
            sect[sect_len].x = p.x;
            sect[sect_len].y = p.y;
            ii[sect_len] = sect_len;
            sect_len ++;
                // sect.push(Vec2{x: p.x,y:p.y});
        }
    }
    ii[sect_len] = sect_len;
    if(sect_len == 0){
        return;
    }
    // return ?
    /*
    for(int i=0;i<sect_len;i++){
        for(int j=0;j<sect_len;j++){
            mt[i][j] = (sect[i].x - sect[j].x) * (sect[i].x - sect[j].x) +
                        (sect[i].y - sect[j].y) * (sect[i].y - sect[j].y);
        }
    }
    */
    #define DIST(i,j) ((sect[i].x - sect[j].x) * (sect[i].x - sect[j].x) + (sect[i].y - sect[j].y) * (sect[i].y - sect[j].y))
    for(int i=0;i<sect_len-1;i++){ // am bagat aici mare -1 ?????
        //float * v = mt[ii[i]];
        //float mn = __FLT_MAX__;
        unsigned int mn = UINT32_MAX;
        int j = -1;
        for(int k=i+1;k<sect_len;k++){
            unsigned int elm = (unsigned int)(DIST(ii[i],ii[k]) * 10000.0);
            //unsigned int elm = (unsigned int)(v[ii[k]] * 10000.0);
            if(elm < mn){
                //mn = v[ii[k]];
                mn = elm;
                j = k;
            } 
        }
        short tmp = ii[i+1];
        ii[i+1] = ii[j];
        ii[j] = tmp;
    }
    #undef DIST

    cntr_len = 0;
    for(int i=0;i<sect_len;i++){
        cntr[i].x = sect[ii[i]].x;
        cntr[i].y = sect[ii[i]].y;
        cntr_len++;
    }
    cntr[cntr_len].x = cntr[0].x;
    cntr[cntr_len].y = cntr[0].y;
    cntr_len++;
    

    Vec2F64 p0 = cntr[0];
    Vec2F64 p1 = cntr[1];
    Vec2F64 pn = cntr[cntr_len-1];
    double d = sqrt( sqrt( (p0.x-pn.x) * (p0.x - pn.x) + (p0.y - pn.y) * (p0.y - pn.y) ) );
    double d2 = sqrt( sqrt( (p0.x -p1.x) * (p0.x -p1.x) + (p0.y - p1.y) * (p0.y - p1.y) ));
    int nn = (int)round(d/d2);

    //printf("cntr last round = %d   %lf %lf\n",nn,d,d2);

    for(int n=0;n<nn;n++){
        double k = (pn.y - p0.y) / (pn.x - p0.x);
        cntr[cntr_len].x = p0.x + (double)n * d2;
        cntr[cntr_len].y = p0.y + (double)n * d2 * k;
        cntr_len++;
    }
}

__host__ __device__ void get_contour2(BufferGeometry * geo,double z_sect,Vec2F64 * cntr,int &cntr_len,float * mt_global){
    const int MT_SIZE=128;
    Vec2F64 sect[MT_SIZE];
    //float mt[128][128];
    short ii[MT_SIZE];
    int sect_len = 0;
    for(int vertex_id = 0;vertex_id < geo->len;vertex_id++){
        Vec3Float64 p = geo->positions[vertex_id];
        if( abs(p.z - z_sect) < 0.15) {
            if(sect_len>=MT_SIZE){
                printf("OVERFLOW ERROR, buff size of %d exceeded\nReport this !!!\n",MT_SIZE);
                return;
            }
            sect[sect_len].x = p.x;
            sect[sect_len].y = p.y;
            ii[sect_len] = sect_len;
            sect_len ++;
                // sect.push(Vec2{x: p.x,y:p.y});
        }
    }
    // return ?
    for(int i=0;i<sect_len;i++){
        for(int j=0;j<sect_len;j++){
            mt_global[i * MT_SIZE + j] = (sect[i].x - sect[j].x) * (sect[i].x - sect[j].x) +
                        (sect[i].y - sect[j].y) * (sect[i].y - sect[j].y);
        }
    }
    for(int i=0;i<sect_len;i++){
        float * v = mt_global + (ii[i] * MT_SIZE) ;// mt[ii[i]];
        float mn = __FLT_MAX__;
        int j = -1;
        for(int k=i+1;k<sect_len;k++){
            if(v[ii[k]] < mn){
                mn = v[ii[k]];
                j = k;
            } 
        }
        short tmp = ii[i+1];
        ii[i+1] = ii[j];
        ii[j] = tmp;
    }
    cntr_len = 0;
    for(int i=0;i<sect_len;i++){
        cntr[i].x = sect[ii[i]].x;
        cntr[i].y = sect[ii[i]].y;
        cntr_len++;
    }
    cntr[cntr_len].x = cntr[0].x;
    cntr[cntr_len].y = cntr[0].y;
    cntr_len++;

    Vec2F64 p0 = cntr[0];
    Vec2F64 p1 = cntr[1];
    Vec2F64 pn = cntr[cntr_len-1];
    double d = sqrt( sqrt( (p0.x-pn.x) * (p0.x - pn.x) + (p0.y - pn.y) * (p0.y - pn.y) ) );
    double d2 = sqrt( sqrt( (p0.x -p1.x) * (p0.x -p1.x) + (p0.y - p1.y) * (p0.y - p1.y) ));
    int nn = (int)round(d/d2);
    for(int n=0;n<nn;n++){
        double k = (pn.y - p0.y) / (pn.x - p0.x);
        cntr[cntr_len].x = p0.x + n * d2;
        cntr[cntr_len].y = p0.y + n * d2 * k;
    }
}