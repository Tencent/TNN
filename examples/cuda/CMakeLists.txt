cmake_minimum_required(VERSION 3.1)
project(TNN-demo)
message(${CMAKE_SOURCE_DIR})
message(${TNN_LIB_PATH})
message(${TENSORRT_ROOT})
find_package(CUDA REQUIRED)

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_CXX_STANDARD 11)
set(TNN_CUDA_ENABLE ON)
set(TNN_TENSORRT_ENABLE ON)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -std=c++11 -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++11 -pthread -fPIC")

set(TNNRoot ${CMAKE_SOURCE_DIR}/../../)
set(TNNInclude ${TNNRoot}/include)
set(LINK_LIBS ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} nvinfer)

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${TNNInclude})
link_directories(${TNN_LIB_PATH})
link_directories(${TENSORRT_ROOT}/lib)

file(GLOB_RECURSE SRC "./*.cc")

add_executable(demo_cuda_linux ${SRC})
target_link_libraries(demo_cuda_linux TNN ${LINK_LIBS})
