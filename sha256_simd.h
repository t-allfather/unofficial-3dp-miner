#ifndef SHA256_SIMD
#define SHA256_SIMD
#include <stdint.h>

#define SHA256_MODE_NORMAL 0
#define SHA256_MODE_SHA 1

extern int sha256Mode;
void sha256_sha_full(uint8_t* data, int len, uint8_t* output);
#endif