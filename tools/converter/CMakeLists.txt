find_package(Protobuf REQUIRED)
if (PROTOBUF_FOUND)
    include_directories(${Protobuf_INCLUDE_DIRS})
    include_directories(${CMAKE_CURRENT_BINARY_DIR})
    protobuf_generate_cpp(ONNX_PROTO_SRC ONNX_PROTO_HEAD source/onnx/onnx-proto/onnx.proto)

else ()
    message(FATAL_ERROR "Protobuf not found, must install first")
endif ()


file(GLOB_RECURSE TNN_CONVERTER_SRC source/*.cc)
include_directories(source/utils)
include_directories(source/caffe)
include_directories(source/onnx)
include_directories(source/tflite)
include_directories(source/tensorflow)

add_executable(TnnConverter ${TNN_CONVERTER_SRC} ${ONNX_PROTO_SRC} ${ONNX_PROTO_HEAD})
target_link_libraries(TnnConverter TNN gflags ${Protobuf_LIBRARIES})
