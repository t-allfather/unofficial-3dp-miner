#include "perlin.h"
#include <iostream>



__host__ __device__  void Perlin::Init(int seed){
    _seedValue = XorShift(seed) * 1.0;
}

__host__ __device__  double Perlin::Noise(double a,double b,double c){
    double x,y,z;
    x = a + _seedValue;
    y = b + _seedValue;
    z = c + _seedValue;

    unsigned int X,Y,Z;
    X = (((int)floor(x)) & 255);
    Y = (((int)floor(y)) & 255);
    Z = (((int)floor(z)) & 255);

    x -= floor(x);
    y -= floor(y);
    z -= floor(z);

    double u,v,w;

    u = Fade(x);
    v = Fade(y);
    w = Fade(z);


    unsigned int A,AA,AB,B,BA,BB;

    A = (unsigned int)P[X] + Y;
    AA = (unsigned int)P[A] + Z;
    AB = (unsigned int)P[A+1] + Z;
    B = (unsigned int)P[X+1] + Y;
    BA = (unsigned int)P [B] + Z;
    BB = (unsigned int)P[B+1] + Z;

    return Lerp(
            w,
            Lerp(v, Lerp(u, Grad(P[AA    ], x, y, z         ), Grad(P[BA    ], x - 1.0, y, z        )), Lerp(u, Grad(P[AB    ], x, y - 1.0, z        ), Grad(P[BB    ], x - 1.0, y - 1.0, z        ))),
            Lerp(v, Lerp(u, Grad(P[AA + 1], x, y, z - 1.0), Grad(P[BA + 1], x - 1.0, y, z - 1.0)), Lerp(u, Grad(P[AB + 1], x, y - 1.0, z - 1.0), Grad(P[BB + 1], x - 1.0, y - 1.0, z - 1.0)))
        );
}

__host__ __device__ int Perlin::XorShift(int value) {
        int x = value ^ (value >> 12);
        x = x ^ (x << 25);
        x = x ^ (x >> 27);
        return x * 2;
}
    
 __host__ __device__    double Perlin::Lerp( double t,  double a,  double b)
    {
        return a + t * (b - a);
    }
    
  __host__ __device__   double Perlin::Fade(double t)
    {
        return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
    }
    
  __host__ __device__   double Perlin::Grad( unsigned char hash,  double x,double y,double z)
    {
        unsigned char h = hash & 15;
        double  u,v;
        if(h < 8)
        {
            u = x;
        } else {
            u = y;
        }
        if(h < 4){
            v = y;
        } else {
            if(h == 12 || h== 14){
                v = x;
            } else {
                v = z;
            }
        }

        double r1 = 0;
        double r2 = 0;

        if((h & 1) == 0 ){
            r1 = u;
        } else {
            r1 = -u;
        }

        if((h&2) == 0){
            r2 = v;
        } else {
            r2 = -v;
        }
        return r1 + r2;
    }