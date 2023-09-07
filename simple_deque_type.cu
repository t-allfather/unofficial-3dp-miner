#include "math_utils.h"
#include "simple_deque.h"


template<typename T,size_t elems>
 __host__ __device__ bool DequeT<T,elems>::isFull()
{
    return ((front == 0 && rear == size - 1)
            || front == rear + 1);
}
 
// Checks whether Deque is empty or not.
template<typename T,size_t elems>
__host__ __device__ bool DequeT<T,elems>::isEmpty() { return (front == -1); }
 
template<typename T,size_t elems>
__host__ __device__ T * DequeT<T,elems>::get(int index){
    return &arr[(front+index)%(size-1)];
    /*
    //printf("front=%d rear=%d\n",front,rear);
    int x = front;
    int s=0;
    for(int i=0;i<index;i++){
        if(x==size-1){
            x=0;
        } else 
            x++;
    }
    //printf("final=%d\n",x);
    return &arr[x];
    */
}

//__host__ __device__ void DequeT::sort_by_val(){
    
//}

// Inserts an element at front
template<typename T,size_t elems>
__host__ __device__ void DequeT<T,elems>::insertfront(T* obj)
{
    // check whether Deque if  full or not
    if (isFull()) {
        printf("Overflow\n");
        return;
    }
 
    // If queue is initially empty
    if (front == -1) {
        front = 0;
        rear = 0;
    }
 
    // front is at first position of queue
    else if (front == 0)
        front = size - 1;
 
    else // decrement front end by '1'
        front = front - 1;
 
    // insert current element into Deque
    //arr[front] = f;
    memcpy(&arr[front],obj,sizeof(T));
    len++;
}
 
// function to inset element at rear end
// of Deque.
template<typename T,size_t elems>
__host__ __device__ void DequeT<T,elems>::insertrear(T * obj)
{
    if (isFull()) {
        printf(" Overflow\n ");
        return;
    }
 
    // If queue is initially empty
    if (front == -1) {
        front = 0;
        rear = 0;
    }
 
    // rear is at last position of queue
    else if (rear == size - 1)
        rear = 0;
 
    // increment rear end by '1'
    else
        rear = rear + 1;
 
    // insert current element into Deque
    //arr[rear].val = f;
    memcpy(&arr[rear],obj,sizeof(T));
    len++;
}
 
// Deletes element at front end of Deque
template<typename T,size_t elems>
__host__ __device__ void DequeT<T,elems>::deletefront()
{
    // check whether Deque is empty or not
    if (isEmpty()) {
        printf("Queue Underflow\n");
        return;
    }
 
    // Deque has only one element
    if (front == rear) {
        front = -1;
        rear = -1;
    }
    else
        // back to initial position
        if (front == size - 1)
        front = 0;
 
    else // increment front by '1' to remove current
        // front value from Deque
        front = front + 1;
    len --;
}
 
// Delete element at rear end of Deque
template<typename T,size_t elems>
__host__ __device__ void DequeT<T,elems>::deleterear()
{
    if (isEmpty()) {
        printf(" Underflow\n");
        return;
    }
 
    // Deque has only one element
    if (front == rear) {
        front = -1;
        rear = -1;
    }
    else if (rear == 0)
        rear = size - 1;
    else
        rear = rear - 1;
    len--;
}
 
// Returns front element of Deque
template<typename T,size_t elems>
__host__ __device__ int DequeT<T,elems>::getFront()
{
    // check whether Deque is empty or not
    if (isEmpty()) {
        printf(" Underflow\n");
        return -1;
    }
    return 0;//arr[front];
}
 
// function return rear element of Deque
template<typename T,size_t elems>
__host__ __device__ int DequeT<T,elems>::getRear()
{
    // check whether Deque is empty or not
    if (isEmpty() || rear < 0) {
        printf(" Underflow\n");
        return -1;
    }
    return 0;//arr[rear];
}
 
template<typename T,size_t elems>
 __host__ __device__ void DequeT<T,elems>::sort_by_val(){
    int x = front;
    int s=0;
    for(int i=0;i<len;i++){
        int y = x;
        for(int j=i+1;j<len;j++){
            if(y == size-1)
                y = 0;
            else 
                y++;
            if(arr[x].val < arr[y].val){
                T temp;
                memcpy(&temp,&arr[x],sizeof(T));
                arr[x] = arr[y];
                arr[y] = temp;
            }
        }
        if(x==size-1){
            x=0;
        } else 
            x++;
    }
}

