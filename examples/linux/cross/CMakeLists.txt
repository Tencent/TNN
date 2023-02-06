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

set(TNNRoot ${CMAKE_CURRENT_SOURCE_DIR}/../../../)
set(TNNInclude ${TNNRoot}/include)

include_directories(${TNNInclude})
include_directories(${TNNRoot}/third_party/stb)
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/../../base)
include_directories(${CMAKE_SOURCE_DIR}/../../utils)
link_directories(${TNN_LIB_PATH})

add_subdirectory(${TNNRoot}/third_party/gflags ${TNNRoot}/third_party/gflags)
get_target_property(GFLAGS_INCLUDE_DIRS gflags INTERFACE_INCLUDE_DIRECTORIES)
include_directories(BEFORE "${GFLAGS_INCLUDE_DIRS}")
link_libraries(gflags)

link_libraries(-Wl,--whole-archive TNN rt -Wl,--no-whole-archive)

if(TNN_OPENMP_ENABLE)
    FIND_PACKAGE(OpenMP REQUIRED)
    if(OPENMP_FOUND)
        link_libraries(${OpenMP_C_LIBRARIES} ${OpenMP_CXX_LIBRARIES})
    endif()
endif()

file(GLOB FLAG_SRC "${CMAKE_SOURCE_DIR}/../src/*.cc")

file(GLOB_RECURSE BASE_SRC "${CMAKE_SOURCE_DIR}/../../base/*.cc")
file(GLOB_RECURSE UTIL_SRC "${CMAKE_SOURCE_DIR}/../../utils/*.cc")

add_definitions(-D_ARM_)
add_executable(demo_arm_imageclassify ../src/TNNImageClassify/TNNImageClassify.cc ${BASE_SRC} ${UTIL_SRC} ${FLAG_SRC})
add_executable(demo_arm_facedetector ../src/TNNFaceDetector/TNNFaceDetector.cc ${BASE_SRC} ${UTIL_SRC} ${FLAG_SRC})
add_executable(demo_arm_objectdetector ../src/TNNObjectDetector/TNNObjectDetector.cc ${BASE_SRC} ${UTIL_SRC} ${FLAG_SRC})
