
get_filename_component(TNN_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..  ABSOLUTE)
message("TNN ROOT : ${TNN_ROOT}")
include_directories(${TNN_ROOT})

find_package(Protobuf REQUIRED)
if (PROTOBUF_FOUND)
    include_directories(${Protobuf_INCLUDE_DIRS})
    include_directories(${CMAKE_CURRENT_BINARY_DIR})
    protobuf_generate_cpp(ONNX_PROTO_SRC ONNX_PROTO_HEAD source/onnx/onnx-proto/onnx.proto)

else ()
    message(FATAL_ERROR "Protobuf not found, must install first")
endif ()

message("CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR}")
file(GLOB_RECURSE TNN_CONVERTER_SRC source/*.cc source/*.h)
include_directories(source)
include_directories(${CMAKE_SOURCE_DIR}/source/tnn)

#add_subdirectory(${CMAKE_SOURCE_DIR}/third_party/gflags  gflags.output EXCLUDE_FROM_ALL)

add_executable(TnnConverter ${TNN_CONVERTER_SRC} ${ONNX_PROTO_SRC} ${ONNX_PROTO_HEAD})
if (TNN_BUILD_SHARED)
    message("build shard library")
endif()
target_link_libraries(TnnConverter
        TNN
        gflags
        ${Protobuf_LIBRARIES})


#if(TNN_BUILD_SHARED)
#    target_link_libraries(TnnConverter
#            TNN
#            gflags
#            ${Protobuf_LIBRARIES}
#            )
#elseif(SYSTEM.iOS OR SYSTEM.Darwin)
#    target_link_libraries(TnnConverter
#            -Wl,-force_load TNN
#            gflags
#            ${Protobuf_LIBRARIES}
#            )
#else()
#    message("target link libraries whole-archive")
#    target_link_libraries(TnnConverter
#            -Wl,--whole-archive TNN -Wl,--no-whole-archive
#            gflags
#            ${Protobuf_LIBRARIES}
#            )
#endif()

