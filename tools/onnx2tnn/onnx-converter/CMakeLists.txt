get_filename_component(TNN_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..  ABSOLUTE)

# include(script/protobuf.cmake)
find_package(Protobuf REQUIRED)
if (PROTOBUF_FOUND)
    include_directories(${Protobuf_INCLUDE_DIRS})
    include_directories(${CMAKE_CURRENT_BINARY_DIR})
    protobuf_generate_cpp(ONNX_PROTO_SRC ONNX_PROTO_HEAD ${TNN_ROOT}/tools/converter/source/onnx//onnx-proto/onnx.proto)
else ()
    message(FATAL_ERROR "Protobuf not found, must install first")
endif ()

add_subdirectory(pybind11)

if(APPLE)
    find_library(MAC_ACCE Accelerate)
    if (NOT MAC_ACCE)
        message(FATAL_ERROR "Accelerate not found")
    endif()
endif(APPLE)

file(GLOB_RECURSE ONNX2TNN_SRC ${TNN_ROOT}/tools/onnx2tnn/src/core/*.h ${TNN_ROOT}/tools/onnx2tnn/src/core/*.cc)

include_directories(${TNN_ROOT}/tools/onnx2tnn/src/core)
include_directories(${TNN_ROOT}/tools/onnx2tnn/src/core/objseri)
include_directories(${TNN_ROOT}/tools/onnx2tnn/src/core/half)
include_directories(${TNN_ROOT}/tools/onnx2tnn/src/core/layer)
include_directories(${TNN_ROOT}/tools/onnx2tnn/src/core/onnx_fuse)
include_directories(${TNN_ROOT}/tools/onnx2tnn/src/onnx_remove)
include_directories(${TNN_ROOT}/tools/onnx2tnn/src/onnx_process_tf)


include_directories(${TNN_ROOT}/include)
include_directories(${TNN_ROOT}/source)

# if convert onnx model from tf , you should uncomment below code .
# add_definitions(-DPROCESS_TF)

add_definitions(-DDEBUG)

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
target_link_libraries(onnx2tnn PRIVATE ${Protobuf_LIBRARIES})
set_target_properties(onnx2tnn PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
                                         SUFFIX "${PYTHON_MODULE_EXTENSION}")

if(TNN_BUILD_SHARED)
    target_link_libraries(onnx2tnn PRIVATE TNN)
elseif(SYSTEM.iOS OR SYSTEM.Darwin)
    #use -w to fix ld: warning: direct access to global weak symbol
    target_link_libraries(onnx2tnn PRIVATE -w -Wl,-force_load TNN)
else()
    message("target link libraries whole-archive")
    target_link_libraries(onnx2tnn PRIVATE -Wl,--whole-archive TNN -Wl,--no-whole-archive)
endif()
