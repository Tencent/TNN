cmake_minimum_required(VERSION 3.0)
project(onnx2tnn)

include(script/protobuf.cmake)

add_subdirectory(pybind11)

if(APPLE)
    find_library(MAC_ACCE Accelerate)
    if (NOT MAC_ACCE)
        message(FATAL_ERROR "Accelerate not found")
    endif()
endif(APPLE)

include_directories(./../src/core)
include_directories(./../src/core/objseri)
include_directories(./../src/core/half)
include_directories(./../src/core/layer)
include_directories(./../src/core/onnx_fuse)
include_directories(./../src/core/onnx_remove)
include_directories(./../src/core/onnx_process_tf)

# if convert onnx model from tf , you should uncomment below code .
# add_definitions(-DPROCESS_TF)

add_definitions(-DDEBUG)

file(GLOB_RECURSE ONNX2TNN_SRC ./../src/core/*.h ./../src/core/*.cc)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-deprecated-declarations -Wno-ignored-attributes")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations -Wno-ignored-attributes")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++11")
SET(CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS "${CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS} -rdynamic")

add_library(onnx2tnn MODULE
                             onnx2tnn_convert.cc
                             ${ONNX2TNN_SRC}
                             ${ONNX_PROTO_SRC}
                             ${ONNX_PROTO_HEAD})
set_target_properties(onnx2tnn PROPERTIES CXX_STANDARD 11)
if (MAC_ACCE)
    target_link_libraries(onnx2tnn PRIVATE ${MAC_ACCE})
endif()
target_link_libraries(onnx2tnn PRIVATE pybind11::module)
target_link_libraries(onnx2tnn PRIVATE ${PROTOBUF_LIBRARIES})
set_target_properties(onnx2tnn PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
                                         SUFFIX "${PYTHON_MODULE_EXTENSION}")
