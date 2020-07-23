cmake_minimum_required(VERSION 3.8)

enable_language(CUDA)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

set(TARGET_ARCH "-gencode arch=compute_75,code=sm_75
                 -gencode arch=compute_70,code=sm_70
                 -gencode arch=compute_61,code=sm_61")

message(${TARGET_ARCH})
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" "${CUDA_OPT_FLAG}" "-Xcompiler -fPIC --compiler-options -fno-strict-aliasing -lineinfo -Xptxas -dlcm=cg -use_fast_math" "${TARGET_ARCH}")

file(GLOB_RECURSE CUDA_SRCS *.cc *.hpp *.h *.cpp *.cu acc/*.cc acc/*.h acc/*.cu acc/*.cpp)
message(${CUDA_SRCS})

add_library(TNNCuda OBJECT ${CUDA_SRCS})

set_property(TARGET TNNCuda PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET TNNCuda PROPERTY CUDA_STANDARD 11)
