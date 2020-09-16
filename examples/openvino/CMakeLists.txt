
cmake_minimum_required(VERSION 3.1)
project(TNN-demo)
message(${CMAKE_SOURCE_DIR})
message(${TNN_LIB_PATH})
option(TNN_OPENVINO_LIB_PATH "openvino lib path" "")
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_CXX_STANDARD 11)
set(TNN_CPU_ENABLE ON)
set(TNN_OPENVINO_ENABLE ON)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -std=c++11 -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++11 -pthread -fPIC")

set(TNNRoot ${CMAKE_SOURCE_DIR}/../../)
set(TNNInclude ${TNNRoot}/include)

include_directories(${TNNInclude})
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/../base)
link_directories(${TNN_LIB_PATH})
if (${TNN_OPENVINO_ENABLE})
    message(${TNN_OPENVINO_LIB_PATH})
    link_directories(${TNN_OPENVINO_LIB_PATH})
endif()


file(GLOB_RECURSE SRC "src/*.cc" "${CMAKE_SOURCE_DIR}/../base/*.cc")

add_executable(demo_x86_linux ${SRC})
target_link_libraries(demo_x86_linux TNN MKLDNNPlugin)
