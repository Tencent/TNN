cmake_minimum_required(VERSION 3.8)

if(NOT DEFINED ENV{TENSORRT_ROOT_DIR})
    message(FATAL_ERROR "not defined environment variable:TENSORRT_ROOT_DIR")
endif()

enable_language(CUDA)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

file(GLOB_RECURSE TENSORRT_SRC *.h *.cc)
add_library(TNNTensorRT OBJECT ${TENSORRT_SRC})

include_directories($ENV{TENSORRT_ROOT_DIR}/include)
include_directories($ENV{CUDNN_ROOT_DIR}/include)
