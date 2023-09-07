#ifndef SIMPLE_DEQUE

#define SIMPLE_DEQUE
#include <iostream>
#include "cuda_runtime.h"
#include "math_utils.h"
using namespace std;
 
#define MAX 400

struct Deque {
    int arr[MAX];
    int front;
    int rear;
    int size;

    __host__ __device__ Deque(int size)
    {
        front = -1;
        rear = 0;
        this->size = size;
    }
    __host__ __device__  void insertfront(int key);
    __host__ __device__  void insertrear(int key);
    __host__ __device__  void deletefront();
    __host__ __device__  void deleterear();
    __host__ __device__  bool isFull();
    __host__ __device__  bool isEmpty();
    __host__ __device__  int getFront();
    __host__ __device__  int getRear();
};

struct Deque_F64_PolyLine {
    T64_PolyLine arr[1];
    int front;
    int rear;
    int size;
    int len=0;

    int pushes=0;
    __host__ __device__ Deque_F64_PolyLine(int ssize)
    {
        front = -1;
        rear = 0;
        len=0;
        this->size = ssize;
    }
    __host__ __device__  void insertfront(double f,PolyLine * line);
    __host__ __device__  void insertrear(double f,PolyLine * line);
    __host__ __device__  void deletefront();
    __host__ __device__  void deleterear();
    __host__ __device__  bool isFull();
    __host__ __device__  bool isEmpty();
    __host__ __device__  int getFront();
    __host__ __device__  int getRear();
    __host__ __device__ T64_PolyLine * get(int index);
    __host__ __device__ void sort_by_val();
};


struct Deque_F64_Hash384 {
    F64_Hash384 arr[20];
    int front;
    int rear;
    int size;
    int len=0;

    __host__ __device__ Deque_F64_Hash384(int size)
    {
        front = -1;
        rear = 0;
        this->size = size;
    }
    __host__ __device__  void insertfront(F64_Hash384 * obj);
    __host__ __device__  void insertrear(F64_Hash384 * obj);
    __host__ __device__  void deletefront();
    __host__ __device__  void deleterear();
    __host__ __device__  bool isFull();
    __host__ __device__  bool isEmpty();
    __host__ __device__  int getFront();
    __host__ __device__  int getRear();
    __host__ __device__ F64_Hash384 * get(int index);
    __host__ __device__ void sort_by_val();
};


template<typename T,size_t elems>
struct DequeT {
    T arr[elems];
    int front;
    int rear;
    int size;
    int len=0;

    __host__ __device__ DequeT(int size)
    {
        front = -1;
        rear = 0;
        this->size = size;
    }
    __host__ __device__  void insertfront(T * obj);
    __host__ __device__  void insertrear(T * obj);
    __host__ __device__  void deletefront();
    __host__ __device__  void deleterear();
    __host__ __device__  bool isFull();
    __host__ __device__  bool isEmpty();
    __host__ __device__  int getFront();
    __host__ __device__  int getRear();
    __host__ __device__ T * get(int index);
     __host__ __device__ void sort_by_val();
    //__host__ __device__ void sort_by_val();
};

#endif