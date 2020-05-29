
set(Protobuf_ROOT ${CMAKE_SOURCE_DIR}/3rdparty/protobuf)

if(EXISTS ${Protobuf_ROOT}/lib/cmake/protobuf)
    set(Protobuf_DIR ${Protobuf_ROOT}/lib/cmake/protobuf)
else()
    set(Protobuf_DIR ${Protobuf_ROOT}/lib64/cmake/protobuf)
endif()

include(${Protobuf_DIR}/protobuf-options.cmake)
include(${Protobuf_DIR}/protobuf-targets.cmake)
include(${Protobuf_DIR}/protobuf-config.cmake)
include(${Protobuf_DIR}/protobuf-module.cmake)

# find_package(Protobuf REQUIRED HINTS ${Protobuf_DIR})

if(PROTOBUF_FOUND)
    include_directories(${PROTOBUF_INCLUDE_DIR})
    include_directories(${PROTOBUF_INCLUDE_DIRS})
    include_directories(${CMAKE_CURRENT_BINARY_DIR})
    message(STATUS "Protobuf ${PROTOBUF_LIBRARIES}")
    # protobuf_generate_cpp(ONNX_PROTO_SRC ONNX_PROTO_HEAD onnx-proto/onnx.proto)
    set(ONNX_PROTO ${CMAKE_SOURCE_DIR}/../src/onnx-proto/onnx.proto)
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/proto)
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/proto/onnx.pb.h ${CMAKE_CURRENT_BINARY_DIR}/proto/onnx.pb.cc
        COMMAND protobuf::protoc --cpp_out=${CMAKE_CURRENT_BINARY_DIR}/proto ${ONNX_PROTO} -I ${CMAKE_SOURCE_DIR}/../src/onnx-proto)
    set(ONNX_PROTO_HEAD ${CMAKE_BINARY_DIR}/proto/onnx.pb.h)
    set(ONNX_PROTO_SRC  ${CMAKE_BINARY_DIR}/proto/onnx.pb.cc)
    include_directories(${CMAKE_BINARY_DIR}/proto)
else()
    message(FATAL_ERROR "Protobuf not found, must install first")
endif()

