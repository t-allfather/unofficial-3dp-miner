#include "math_utils.h"
#include <string>
#include <float.h>

__host__ __device__ Vec3Float64 Vec3Float64::zero(){
    Vec3Float64 tmp;
    tmp.x =0;
    tmp.y = 0;
    tmp.z = 0;
    return tmp;
}

__host__ __device__ Vec3Float64 Vec3Float64::create(double x,double y,double z){
    Vec3Float64 tmp;
    tmp.x = x;
    tmp.y = y;
    tmp.z = z;
    return tmp;
}

__host__ __device__ Vec3Uint Vec3Uint::create(unsigned int x,unsigned int y,unsigned int z){
    Vec3Uint tmp;
    tmp.x = x;
    tmp.y = y;
    tmp.z = z;
    return tmp;
}

__host__ __device__ Vec3Float64 Vec3Float64::sub(Vec3Float64 a,Vec3Float64 b){
    Vec3Float64 tmp;
    tmp.x = a.x - b.x;
    tmp.y = a.y-b.y;
    tmp.z = a.z - b.z;
    return tmp;
}

__host__ __device__ Vec3Float64 Vec3Float64::add(Vec3Float64 a,Vec3Float64 b){
    Vec3Float64 tmp;
    tmp.x = a.x + b.x;
    tmp.y = a.y + b.y;
    tmp.z = a.z + b.z;
    return tmp;
}

__host__ __device__ Vec3Float64 Vec3Float64::addref(Vec3Float64 * a,Vec3Float64 b){
    a->x = a->x + b.x;
    a->y = a->y + b.y;
    a->z = a->z + b.z;
}

__host__ __device__ Vec3Float64 Vec3Float64::cross(Vec3Float64 a,Vec3Float64 b){
    Vec3Float64 tmp;
    tmp.x = a.y*b.z - a.z*b.y;
    tmp.y = a.z*b.x - a.x*b.z;
    tmp.z = a.x*b.y - a.y*b.x;
    return tmp;
}

__host__ __device__ double Vec3Float64::dot(Vec3Float64 *a,Vec3Float64 *b){
    return a->x * b->x + a->y * b->y + a->z* b->z;
}

__host__ std::string Vec3Float64::str(){
    return std::to_string(x)+" "+std::to_string(y) +" " + std::to_string(z);
}

__host__ __device__ Triangle::Triangle(Vec3Float64 _p1,Vec3Float64 _p2,Vec3Float64 _p3){
    set(_p1,_p2,_p3);
}

__host__ __device__ void Triangle::set(Vec3Float64 _p1,Vec3Float64 _p2,Vec3Float64 _p3){
    p1.x = _p1.x;
    p1.y = _p1.y;
    p1.z = _p1.z;

    p2.x = _p2.x;
    p2.y = _p2.y;
    p2.z = _p2.z;

    p3.x = _p3.x;
    p3.y = _p3.y;
    p3.z = _p3.z;
}
__host__ __device__ Triangle::Triangle(){
    p1.x = 0;
    p1.y = 0;
    p1.z = 0;
    p2.x = 0;
    p2.y = 0;
    p2.z = 0;
    p3.x = 0;
    p3.y = 0;
    p3.z = 0;
}

template <size_t rows, size_t cols>
__host__ __device__ void eigsrt(double * d, double (&v)[rows][cols]) {
    unsigned int k;
    unsigned int n = 3;
    for(int i=0;i<n - 1;i++) {
        k = i;
        double p = d[k];
        for(int j=i;j<n;j++) {
            if(d[j] >= p){
                k = j;
                p = d[k];
            }
        }
        if(k != i) {
            d[k] = d[i];
            d[i] = p;
            for(int j=0;j<n;j++) {
                p = v[j][i];
                v[j][i] = v[j][k];
                v[j][k] = p;
            }
        }
    }
}

__host__ __device__ void Eigen::Create(double mat[3][3]){
    n =3;
    d[0] = mat[0][0];
    d[1] = mat[1][1];
    d[2] = mat[2][2];
    for(int i=0;i<3;i++)
        for(int j=0;j<3;j++)
            a[i][j] = mat[i][j];
}

template <size_t rows, size_t cols>
__host__ __device__ void rot(double (&a)[rows][cols], double s, double tau, unsigned int i,unsigned int j,unsigned int k, unsigned int l) {
    double g = a[i][j];
    double h = a[k][l];
    a[i][j] = g - s * (h + g * tau);
    a[k][l] = h + s * (g - h * tau);
}

__host__ __device__ void Eigen::Solve(){

    double h;
    double z[3]={0};
    double b[3];
    b[0] = a[0][0];
    b[1] = a[1][1];
    b[2] = a[2][2];
    unsigned int _n_rot = n_rot;

    for(int i = 1;i<51;i++){
        double sm = 0;
        for(int ip = 0;ip < n-1;ip++){
            for(int iq = ip+1;iq < n;iq++){
                sm += abs(a[ip][iq]);
            }
        }
        if(sm == 0){
            eigsrt(d,v);
            return;
        }
        double tresh = 0;
        if(i<4){
            tresh = 0.2 * sm/ ((double)(n * n));
        } else {
            tresh = 0;
        }
        for(int ip=0;ip<n-1;ip++){
            for(int iq=ip+1;iq<n;iq++){
                double g = 100.0 * abs(a[ip][iq]);
                if(i > 4 && g <= DBL_EPSILON * abs(d[ip]) && g <= DBL_EPSILON * abs(d[iq]) ) {
                    a[ip][iq] = 0;
                } else if(abs(a[ip][iq]) > tresh) {
                    h = d[iq] - d[ip];
                    double t = 0;
                    if(g <= DBL_EPSILON * abs(h)){
                        t = a[ip][iq] / h;
                    } else {
                        double theta = 0.5 * h / a[ip][iq];
                        double temp = 1.0 / (abs(theta) + sqrt(1.0 + theta * theta));
                        if(theta < 0){
                            temp = -temp;
                        }
                        t= temp;
                    }
                    double c = 1.0 / sqrt(1.0 + t * t);
                    double s = t * c;
                    double tau = s / (1.0 + c);
                    h = t * a[ip][iq];
                    z[ip] -= h;
                    z[iq] += h;
                    d[ip] -= h;
                    d[iq] += h;
                    a[ip][iq] = 0;
                        for(int j=0;j<ip;j++) {
                            rot(a, s, tau, j, ip, j, iq);
                        }
                        for(int j=ip+1;j<iq;j++) {
                            rot(a, s, tau, ip, j, j, iq);
                        }
                        for(int j=iq + 1;j<n;j++) {
                            rot(a, s, tau, ip, j, iq, j);
                        }
                        for(int j=0;j<n;j++) {
                            rot(v, s, tau, j, ip, j, iq);
                        }
                        _n_rot += 1;
                }
            }
        }
        for(int ip=0;ip<n;ip++){
            b[ip] += z[ip];
            d[ip] = b[ip];
            z[ip] = 0;
        }
    }

}

    __host__ __device__ void PolyLine::push(int x,int y,bool allowed_mat){
        if(len == 40){
            printf("Overflow at polyline\n");
            return;
        }
        nodes[len].x = x;
        nodes[len].y = y;
        if(allowed_mat){
            const char di[9] = {0,0,1,1,1,-1,-1,-1,0};
            const char dj[9] = {-1,1,-1,1,0,-1,0,1,0};
            for(int i=0;i<9;i++){
                int xx,yy;
                xx = x + di[i];
                yy = y + dj[i];
                if(xx>=0 && xx<8 && yy>=0 && yy<8 && allowed[xx][yy] == 0){
                    allowed[xx][yy]=len+1;
                }
            }
        }
        /*
        for(int i=0;i<8;i++){
            for(int j=0;j<8;j++){
                printf("%d ",allowed[i][j]);
            }
            printf("\n");
        }
        printf("\n");
        */
        len++;
    }
    
    __host__ __device__ void PolyLine::push(Vec2Int a){ 
        if(len == 40){
            printf("Overflow at polyline\n");
            return;
        }
        nodes[len].x = a.x;
        nodes[len].y = a.y;
        //if(allowed[a.x][a.y] != 0){
        //    printf("INVALID POS\n");
        //}

        
        const char di[9] = {  0,0, 1,1,1,-1,-1,-1,0};
        const char dj[9] = { -1,1,-1,1,0,-1, 0, 1,0};

        int len2 = len+1;
        for(char i=0;i<9;i++){
            char xx,yy;
            xx = a.x + di[i];
            yy = a.y + dj[i];
            if(xx>=0 && xx<8 && yy>=0 && yy<8 && (allowed[xx][yy] == 0)){// || (allowed[xx][yy] >= len2))){
                allowed[xx][yy]=len2;
            }
        }
        

        /*
        for(int i=0;i<8;i++){
            for(int j=0;j<8;j++){
                printf("%d ",allowed[i][j]);
            }
            printf("\n");
        }
        printf("\n");
        */
        len++;
    }

    __host__ __device__ void PolyLine::pop(){
        if(len>0){
            
            const char di[9] = {0,0,1,1,1,-1,-1,-1,0};
            const char dj[9] = {-1,1,-1,1,0,-1,0,1,0};
            char x = nodes[len-1].x;
            char y = nodes[len-1].y;

            for(char i=0;i<9;i++){
               char xx,yy;
               xx = x + di[i];
               yy = y + dj[i];
               if(xx>=0 && xx<8 && yy>=0 && yy<8 && allowed[xx][yy] == len){
                   allowed[xx][yy]=0;
                }
            }
            
            /*
            for(int i=0;i<8;i++){
                for(int j=0;j<8;j++){
                   printf("%d ",allowed[i][j]);
                }
                printf("\n");
            }
            printf("\n");
            */
            len--;
        }
        else{
            printf("Polyline underflow\n");
        }
    }

__host__ __device__ int int_to_be(int val){
    return (val % 256) * 256 * 256 * 256 +
            ((val/256) % 256) * 256 * 256+
            ((val/(256*256)) % 256) * 256;
            ((val/(256*256*256))%256);
}

__host__ __device__ Vec2Int PolyLineCompress::decomp(unsigned char elem){
   Vec2Int r;
   int y = elem % 8;
   int x = (elem - y) / 8;
   r.x = x;
   r.y = y;
   return r;
}
__host__ __device__ unsigned char PolyLineCompress::comp(Vec2Int elem){
  return (unsigned char)(elem.x * 8 + elem.y);
}