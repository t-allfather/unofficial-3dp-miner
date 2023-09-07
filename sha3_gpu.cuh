#ifndef SHA3_GPU
#define SHA3_GPU
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_functions.h"
#include "device_launch_parameters.h"

typedef unsigned char BYTE;             // 8-bit byte
typedef unsigned int  WORD;             // 32-bit word, change to "long" for 16-bit machines
typedef unsigned long long LONG;

#define checkCudaErrors(x) \
{ \
    cudaGetLastError(); \
    x; \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) {\
        printf("CUDA Error Occurred %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(err)); \
    } \
}
#define KECCAK_ROUND 24
#define KECCAK_STATE_SIZE 25
#define KECCAK_Q_SIZE 192



typedef struct {

    BYTE sha3_flag=1;
    WORD digestbitlen=256;
    LONG rate_bits=1088;
    LONG rate_BYTEs=136;
    LONG absorb_round=17;

    int64_t state[KECCAK_STATE_SIZE]={0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    BYTE q[KECCAK_Q_SIZE]={
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
    };

    LONG bits_in_queue=0;

} cuda_keccak_ctx_t;
typedef cuda_keccak_ctx_t CUDA_KECCAK_CTX;

__device__ void cuda_keccak_init(cuda_keccak_ctx_t *ctx, WORD digestbitlen);
__device__ void cuda_keccak_update(cuda_keccak_ctx_t *ctx, BYTE *in, LONG inlen);
__device__ void cuda_keccak_final(cuda_keccak_ctx_t *ctx, BYTE *out);

__device__ void keccak256(unsigned char* data, unsigned long size, unsigned char* digest);

__device__ void sha3_cuda(uint2 * keccak_gpu_state,unsigned char * data,int data_len,unsigned char * output);


#endif