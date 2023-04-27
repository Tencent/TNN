cmake_minimum_required(VERSION 3.8)

enable_language(CUDA)

add_definitions(-DTHRUST_IGNORE_CUB_VERSION_CHECK)
add_definitions(-DTHRUST_IGNORE_DEPRECATED_CPP_DIALECT)
add_definitions(-DCUB_IGNORE_DEPRECATED_CPP11)

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
include_directories(thirdparty/cub-1.9.10-1)

if(NOT DEFINED ENV{CUDNN_ROOT_DIR})
    message(FATAL_ERROR "not defined environment variable:CUDNN_ROOT_DIR")
endif()
include_directories($ENV{CUDNN_ROOT_DIR}/include)

set(TARGET_ARCH "-gencode arch=compute_75,code=sm_75 \
                 -gencode arch=compute_70,code=sm_70 \
                 -gencode arch=compute_61,code=sm_61 \
                 -gencode arch=compute_60,code=sm_60 \
                 -gencode arch=compute_53,code=sm_53")

message(${TARGET_ARCH})

if(TNN_GLIBCXX_USE_CXX11_ABI_ENABLE)
    set(CMAKE_CUDA_FLAGS "${CUDA_NVCC_FLAGS} ${CUDA_OPT_FLAG} -Xcompiler -fPIC --compiler-options -fno-strict-aliasing \
        -lineinfo -Xptxas -dlcm=cg -use_fast_math -D_GLIBCXX_USE_CXX11_ABI=1 ${TARGET_ARCH}")
else()
    set(CMAKE_CUDA_FLAGS "${CUDA_NVCC_FLAGS} ${CUDA_OPT_FLAG} -Xcompiler -fPIC --compiler-options -fno-strict-aliasing \
        -lineinfo -Xptxas -dlcm=cg -use_fast_math -D_GLIBCXX_USE_CXX11_ABI=0 ${TARGET_ARCH}")
endif()

file(GLOB CUDA_SRCS *.cc *.h *.cu acc/*.cc acc/*.h acc/*.cu)
message(${CUDA_SRCS})

add_library(TNNCuda OBJECT ${CUDA_SRCS})

set_property(TARGET TNNCuda PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET TNNCuda PROPERTY CUDA_STANDARD 11)

