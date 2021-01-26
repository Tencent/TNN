cmake_minimum_required(VERSION 3.1)
project(TNN-demo)
message(${CMAKE_SOURCE_DIR})
message(${TNN_LIB_PATH})

if(NOT DEFINED ENV{TENSORRT_ROOT_DIR})
    message(FATAL_ERROR "not defined environment variable:TENSORRT_ROOT_DIR")
endif()

if(NOT DEFINED ENV{CUDNN_ROOT_DIR})
    message(FATAL_ERROR "not defined environment variable:CUDNN_ROOT_DIR")
endif()

find_package(CUDA REQUIRED)

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_CXX_STANDARD 11)
set(TNN_CUDA_ENABLE ON)
set(TNN_TENSORRT_ENABLE ON)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -std=c++11 -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++11 -pthread -fPIC")

set(TNNRoot ${CMAKE_SOURCE_DIR}/../../)
set(TNNInclude ${TNNRoot}/include)

include_directories(${TNNInclude})
include_directories(${CMAKE_SOURCE_DIR}/../base)
include_directories(${CMAKE_SOURCE_DIR}/../utils)
include_directories(${CUDA_INCLUDE_DIRS})

link_directories(${TNN_LIB_PATH})
link_directories($ENV{TENSORRT_ROOT_DIR}/lib)
link_directories($ENV{CUDNN_ROOT_DIR}/lib)
set(CUDA_LINK_LIBS ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})
link_libraries(TNN)
link_libraries(${CUDA_LINK_LIBS})

file(GLOB_RECURSE BASE_SRC
    "${CMAKE_SOURCE_DIR}/../base/*.cc"
    "${CMAKE_SOURCE_DIR}/../utils/*.cc")

add_executable(demo_cuda_imageclassify src/TNNImageClassify/TNNImageClassify.cc ${BASE_SRC})
add_executable(demo_cuda_facedetector src/TNNFaceDetector/TNNFaceDetector.cc ${BASE_SRC})

