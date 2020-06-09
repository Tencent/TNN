
cmake_minimum_required(VERSION 3.1)
project(TNN-demo)
message(${CMAKE_SOURCE_DIR})

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

link_directories(${TNN_LIB_PATH})


file(GLOB_RECURSE SRC "src/*.cpp")

add_executable(demo_arm_linux ${SRC})
target_link_libraries(demo_arm_linux TNN)
