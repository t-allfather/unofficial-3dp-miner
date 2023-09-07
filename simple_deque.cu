#include "simple_deque.h"
#include "math_utils.h"

// Checks whether Deque is full or not.
__host__ __device__ bool Deque::isFull()
{
    return ((front == 0 && rear == size - 1)
            || front == rear + 1);
}
 
// Checks whether Deque is empty or not.
__host__ __device__ bool Deque::isEmpty() { return (front == -1); }
 
// Inserts an element at front
__host__ __device__ void Deque::insertfront(int key)
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
    arr[front] = key;
}
 
// function to inset element at rear end
// of Deque.
__host__ __device__ void Deque ::insertrear(int key)
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
    arr[rear] = key;
}
 
// Deletes element at front end of Deque
__host__ __device__ void Deque ::deletefront()
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
}
 
// Delete element at rear end of Deque
__host__ __device__ void Deque::deleterear()
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
}
 
// Returns front element of Deque
__host__ __device__ int Deque::getFront()
{
    // check whether Deque is empty or not
    if (isEmpty()) {
        printf(" Underflow\n");
        return -1;
    }
    return arr[front];
}
 
// function return rear element of Deque
__host__ __device__ int Deque::getRear()
{
    // check whether Deque is empty or not
    if (isEmpty() || rear < 0) {
        printf(" Underflow\n");
        return -1;
    }
    return arr[rear];
}
 



 // Checks whether Deque is full or not.
__host__ __device__ bool Deque_F64_PolyLine::isFull()
{
    return ((front == 0 && rear == size - 1)
            || front == rear + 1);
}
 
// Checks whether Deque is empty or not.
__host__ __device__ bool Deque_F64_PolyLine::isEmpty() { return (front == -1); }
 
__host__ __device__ T64_PolyLine * Deque_F64_PolyLine::get(int index){
    //printf("front=%d rear=%d index=%d target=%d\n",front,rear,index,(front+index)%(size-1));
    return &arr[(front+index)%(size-1)];
}

__host__ __device__ void Deque_F64_PolyLine::sort_by_val(){
    int x = front;
    int s=0;
    PolyLine temp;
    for(int i=0;i<len;i++){
        int y = x;
        for(int j=i+1;j<len;j++){
            if(y == size-1)
                y = 0;
            else 
                y++;
            if(arr[x].val < arr[y].val){ 
                memcpy(&temp,&arr[x].line,sizeof(PolyLine));
                double temp_val = arr[x].val;
                arr[x].val = arr[y].val;
                memcpy(&arr[x].line,&arr[y].line,sizeof(PolyLine));
                arr[y].val = temp_val;
                memcpy(&arr[y].line,&temp,sizeof(PolyLine));
            }
        }
        if(x==size-1){
            x=0;
        } else 
            x++;
    }
}

// Inserts an element at front
__host__ __device__ void Deque_F64_PolyLine::insertfront(double f,PolyLine * line)
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
    arr[front].val = f;
    len++;
    memcpy(&arr[front].line,line,sizeof(PolyLine));
}
 
// function to inset element at rear end
// of Deque.
__host__ __device__ void Deque_F64_PolyLine ::insertrear(double f,PolyLine * line)
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
    arr[rear].val = f;
    
    arr[rear].line.len = line->len;
    for(int i=0;i<line->len;i++){
        arr[rear].line.nodes[i].x = line->nodes[i].x;
        arr[rear].line.nodes[i].y = line->nodes[i].y;
    }
    for(int i=0;i<8;i++){
        for(int j=0;j<8;j++){
            arr[rear].line.allowed[i][j] = line->allowed[i][j];
        }
    }
    len++;
    /*
    pushes++;
    if(pushes>150){
        printf("%d\n",pushes);
    }*/
    //memcpy(&arr[rear].line,line,sizeof(PolyLine));
}
 
// Deletes element at front end of Deque
__host__ __device__ void Deque_F64_PolyLine ::deletefront()
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
__host__ __device__ void Deque_F64_PolyLine::deleterear()
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
__host__ __device__ int Deque_F64_PolyLine::getFront()
{
    // check whether Deque is empty or not
    if (isEmpty()) {
        printf(" Underflow\n");
        return -1;
    }
    return 0;//arr[front];
}
 
// function return rear element of Deque
__host__ __device__ int Deque_F64_PolyLine::getRear()
{
    // check whether Deque is empty or not
    if (isEmpty() || rear < 0) {
        printf(" Underflow\n");
        return -1;
    }
    return 0;//arr[rear];
}
 























 // Checks whether Deque is full or not.
__host__ __device__ bool Deque_F64_Hash384::isFull()
{
    return ((front == 0 && rear == size - 1)
            || front == rear + 1);
}
 
// Checks whether Deque is empty or not.
__host__ __device__ bool Deque_F64_Hash384::isEmpty() { return (front == -1); }
 
__host__ __device__ F64_Hash384 * Deque_F64_Hash384::get(int index){
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

__host__ __device__ void Deque_F64_Hash384::sort_by_val(){
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
                F64_Hash384 temp;
                memcpy(&temp,&arr[x],sizeof(F64_Hash384));
                arr[x] = arr[y];
                memcpy(&arr[y],&temp,sizeof(F64_Hash384));
            }
        }
        if(x==size-1){
            x=0;
        } else 
            x++;
    }
}

// Inserts an element at front
__host__ __device__ void Deque_F64_Hash384::insertfront(F64_Hash384 * obj)
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
    len++;
    memcpy(&arr[front],obj,sizeof(F64_Hash384));
}
 
// function to inset element at rear end
// of Deque.
__host__ __device__ void Deque_F64_Hash384::insertrear(F64_Hash384 * obj)
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
    len++;
    memcpy(&arr[rear],obj,sizeof(F64_Hash384));
}
 
// Deletes element at front end of Deque
__host__ __device__ void Deque_F64_Hash384 ::deletefront()
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
__host__ __device__ void Deque_F64_Hash384::deleterear()
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
__host__ __device__ int Deque_F64_Hash384::getFront()
{
    // check whether Deque is empty or not
    if (isEmpty()) {
        printf(" Underflow\n");
        return -1;
    }
    return 0;//arr[front];
}
 
// function return rear element of Deque
__host__ __device__ int Deque_F64_Hash384::getRear()
{
    // check whether Deque is empty or not
    if (isEmpty() || rear < 0) {
        printf(" Underflow\n");
        return -1;
    }
    return 0;//arr[rear];
}
 
