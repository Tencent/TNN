
cmake_minimum_required(VERSION 3.1)
project(TNN-demo)
message(${CMAKE_SOURCE_DIR})
message(${TNN_LIB_PATH})
set(CMAKE_SYSTEM_NAME Linux)
set(TNN_OPENMP_ENABLE ON)
set(CMAKE_CXX_STANDARD 11)
set(TNN_ARM_ENABLE ON)
set(CMAKE_SYSTEM_PROCESSOR aarch64)


set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -std=c++11 -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++11 -pthread -fPIC")

set(TNNRoot ${CMAKE_SOURCE_DIR}/../../)
set(TNNInclude ${TNNRoot}/include)

include_directories(${TNNInclude})
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/../base)
link_directories(${TNN_LIB_PATH})

link_libraries(-Wl,--whole-archive TNN -Wl,--no-whole-archive)
file(GLOB_RECURSE SRC "${CMAKE_SOURCE_DIR}/../samples/*.cc")

add_executable(demo_arm_linux_imageclassify src/TNNImageClassify/TNNImageClassify.cc ${SRC})
add_executable(demo_arm_linux_facedetector src/TNNFaceDetector/TNNFaceDetector.cc ${SRC})
