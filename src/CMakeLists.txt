cmake_minimum_required(VERSION 3.5)

project("3dp-miner" LANGUAGES C CXX CUDA)
enable_language(CUDA)
enable_language(CXX)
enable_language(C)

set(USE_CLANG FALSE)
set(USE_CUDA TRUE)
set(USE_STATIC TRUE)

set(CLANG_FLAGS "")

set(CUDA_FILES "")
set(STATIC_FLAGS "")

if(USE_STATIC)
    set(STATIC_FLAGS "-static-libgcc -static-libstdc++")
endif()

if(USE_CLANG)
    set(CMAKE_C_COMPILER /usr/bin/clang)
    set(CMAKE_CXX_COMPILER /usr/bin/clang++)
    set(CLANG_FLAGS "-Ofast")
else()
    set(CLANG_FLAGS "-O3")
    set(CMAKE_C_COMPILER /usr/bin/gcc)
    set(CMAKE_CXX_COMPILER /usr/bin/g++)
endif()

find_package(CUDA REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")

set(CMAKE_CXX_FLAGS "${CLANG_FLAGS} -m64 -s -std=c++11 -fPIC -funroll-all-loops")
set(CMAKE_C_FLAGS "${CLANG_FLAGS} -m64 -s -funroll-all-loops -fPIC")

set(CUDA_NVCC_FLAGS "${CLANG_FLAGS} -std=c++11 --use_fast_math --relocatable-device-code=true -v -Xptxas -O3")

string(APPEND CMAKE_CUDA_FLAGS " ${STATIC_FLAGS} -gencode arch=compute_86,code=sm_86")

if(USE_CUDA)
    list(APPEND CUDA_FILES kernel.cu)
    list(APPEND CUDA_FILES grid_cuda.cu)
    list(APPEND CUDA_FILES sha3_gpu.cu)
endif()

set_source_files_properties(sha256_simd.cpp PROPERTIES COMPILE_FLAGS "-msse4 -msha -s -O3")

add_executable(3dp-miner ${CUDA_FILES} sha256_simd.cpp main.cpp sphere.cu math_utils.cu buffer_geometry.cu scrape.cu simple_deque.cu grid.cpp sha256.cu perlin.cu Http.cpp json.cpp utils.cpp sha3.cpp int128.cpp int128.h uint128_t.cpp uint128_t.h uint256_t.cpp uint256_t.h uint256_t_include.h uint256_t_build.h uint128_t_include.h uint256_t_build.h uint256_t_config_include.h uint128_t_config_include.h)

target_compile_options(3dp-miner PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    -O3
    --use_fast_math
    --relocatable-device-code=true
    "-Xptxas=-O3 -v"
>)

set_target_properties("3dp-miner"
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Set the linker flags to use static GLIBC
if(USE_STATIC)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${STATIC_FLAGS}")
endif()
