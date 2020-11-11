cmake_minimum_required(VERSION 3.8)

enable_language(CUDA)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

file(GLOB_RECURSE TENSORRT_SRC *.h *.cc)
add_library(TNNTensorRT OBJECT ${TENSORRT_SRC})

include_directories(thirdparty/TensorRT7.1/include)
include_directories(thirdparty/md5)