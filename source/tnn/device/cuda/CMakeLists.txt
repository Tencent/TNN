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

message(${CMAKE_CUDA_COMPILER_VERSION})

# sm_70: V100
# sm_75: T4, RTX 2080
set(TARGET_ARCH "-gencode arch=compute_75,code=sm_75 \
                 -gencode arch=compute_70,code=sm_70")

if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 11.0)
    # sm_60: P100
    # sm_61: P4, P40, GTX 1080
    string(APPEND TARGET_ARCH " -gencode=arch=compute_60,code=sm_60")
    string(APPEND TARGET_ARCH " -gencode=arch=compute_61,code=sm_61")
endif()

if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.0)
    # sm_80: A100, A800
    string(APPEND TARGET_ARCH " -gencode=arch=compute_80,code=sm_80")
endif()

if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.1)
    # sm_86: A20, A40, RTX 3080
    string(APPEND TARGET_ARCH " -gencode=arch=compute_86,code=sm_86")
endif()

if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.4 AND TNN_CUDA_JETSON_ENABLE)
    # sm_87: Jetson Orin
    string(APPEND TARGET_ARCH " -gencode=arch=compute_87,code=sm_87")
endif()

if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.8)
    # sm_89: L4, L20, L40, RTX 4090
    string(APPEND TARGET_ARCH " -gencode=arch=compute_89,code=sm_89")
endif()

if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.0)
    # sm_90: H20, H100, H200
    string(APPEND TARGET_ARCH " -gencode=arch=compute_90,code=sm_90")
endif()

if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.6)
    # sm_100: B40, B100, B200, RTX 5080
    string(APPEND TARGET_ARCH " -gencode=arch=compute_100,code=sm_100")
endif()

# ptx forward-compatible for cuda12 or later
if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.0)
    string(APPEND TARGET_ARCH " -gencode=arch=compute_80,code=compute_80")
endif()

message(${TARGET_ARCH})

if(TNN_GLIBCXX_USE_CXX11_ABI_ENABLE)
    set(CMAKE_CUDA_FLAGS "${CUDA_NVCC_FLAGS} ${CUDA_OPT_FLAG} -Xcompiler -fPIC --compiler-options -fno-strict-aliasing \
        -lineinfo -Xptxas -dlcm=cg -use_fast_math -D_GLIBCXX_USE_CXX11_ABI=1 ${TARGET_ARCH}")
else()
    set(CMAKE_CUDA_FLAGS "${CUDA_NVCC_FLAGS} ${CUDA_OPT_FLAG} -Xcompiler -fPIC --compiler-options -fno-strict-aliasing \
        -lineinfo -Xptxas -dlcm=cg -use_fast_math -D_GLIBCXX_USE_CXX11_ABI=0 ${TARGET_ARCH}")
endif()

file(GLOB CUDA_SRCS *.cc *.h *.cu acc/*.cc acc/*.h acc/*.cu acc/compute/*.h acc/compute/*.cuh acc/compute/*.cu)
message(${CUDA_SRCS})

#file(GLOB FUSED_ATTENTION_SRCS acc/compute/trt_fused_multihead_attention/*.cpp acc/compute/trt_fused_multihead_attention/*.cu acc/compute/trt_unfused_multihead_attention/*.cu)
#file(GLOB ATTENTION_FLASH_SRCS acc/compute/trt_multihead_flash_attention/*.cpp acc/compute/trt_multihead_flash_attention/*.h acc/compute/trt_multihead_flash_attention/fmha_flash_attention/include/*.h acc/compute/trt_multihead_flash_attention/fmha_flash_attention/src/*.cpp)
#file(GLOB ATTENTION_CROSS_SRCS acc/compute/trt_multihead_cross_attention/*.cpp acc/compute/trt_multihead_cross_attention/*.h acc/compute/trt_multihead_cross_attention/fmha_cross_attention/include/*.h acc/compute/trt_multihead_cross_attention/fmha_cross_attention/src/*.cpp)
#message(${ATTENTION_FLASH_SRCS})
#file(GLOB ATTENTION_CROSS_SRCS acc/compute/trt_multihead_cross_attention/*.cpp acc/compute/trt_multihead_cross_attention/*.cu)
# message(${FUSED_ATTENTION_SRCS})
#set(CUDA_SRCS ${CUDA_SRCS} ${FUSED_ATTENTION_SRCS} ${ATTENTION_FLASH_SRCS} ${ATTENTION_CROSS_SRCS})
set(CUDA_SRCS ${CUDA_SRCS})

add_library(TNNCuda OBJECT ${CUDA_SRCS})

set_property(TARGET TNNCuda PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET TNNCuda PROPERTY CUDA_STANDARD 11)

