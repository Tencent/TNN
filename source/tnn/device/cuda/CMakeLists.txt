cmake_minimum_required(VERSION 3.8)

enable_language(CUDA)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
include_directories(thirdparty/cub-1.7.4)

if(NOT DEFINED ENV{CUDNN_ROOT_DIR})
    message(FATAL_ERROR "not defined environment variable:CUDNN_ROOT_DIR")
endif()
include_directories($ENV{CUDNN_ROOT_DIR}/include)

set(TARGET_ARCH "-gencode arch=compute_75,code=sm_75 \
                 -gencode arch=compute_70,code=sm_70 \
                 -gencode arch=compute_61,code=sm_61 \
                 -gencode arch=compute_60,code=sm_60")

message(${TARGET_ARCH})

set(CMAKE_CUDA_FLAGS "${CUDA_NVCC_FLAGS} ${CUDA_OPT_FLAG} -Xcompiler -fPIC --compiler-options -fno-strict-aliasing \
    -lineinfo -Xptxas -dlcm=cg -use_fast_math ${TARGET_ARCH}")

file(GLOB CUDA_SRCS *.cc *.h *.cu acc/*.cc acc/*.h acc/*.cu)
message(${CUDA_SRCS})

add_library(TNNCuda OBJECT ${CUDA_SRCS})

set_property(TARGET TNNCuda PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET TNNCuda PROPERTY CUDA_STANDARD 11)

