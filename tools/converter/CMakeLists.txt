find_package(Protobuf REQUIRED)
if (PROTOBUF_FOUND)
    include_directories(${Protobuf_INCLUDE_DIRS})
    include_directories(${CMAKE_CURRENT_BINARY_DIR})
    protobuf_generate_cpp(ONNX_PROTO_SRC ONNX_PROTO_HEAD source/onnx/onnx-proto/onnx.proto)
else ()
    message(FATAL_ERROR "Protobuf not found, must install first")
endif ()

file(GLOB_RECURSE TNN_CONVERTER_SRC source/*.cc source/*.h)
file(GLOB_RECURSE REMOVE_SOURCES source/torch/*.cc source/torch/*.h)
list(REMOVE_ITEM TNN_CONVERTER_SRC ${REMOVE_SOURCES})

get_filename_component(TNN_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..  ABSOLUTE)
include_directories(${TNN_ROOT})
include_directories(source)

if (TNN_CONVERTER_TORCH_ENABLE)
    execute_process(COMMAND python3 -c "import torch;print(torch.utils.cmake_prefix_path)"
            OUTPUT_VARIABLE LIB_TORCH_PATH
            RESULT_VARIABLE EXIT_CODE
            OUTPUT_STRIP_TRAILING_WHITESPACE)
    if (NOT ${EXIT_CODE} EQUAL 0)
        message(FATAL_ERROR
                "The \"torch\" Python3 package is not installed. Please install it using the following command: \"pip3 install torch\".")
    endif()
    message(STATUS "LibTorch Path is : " ${LIB_TORCH_PATH})
    find_package(Torch REQUIRED PATHS ${LIB_TORCH_PATH} NO_DEFAULT_PATH)
    include_directories(${TORCH_INCLUDE_DIRS})
    set(CMAKE_CXX_STANDARD 14)
    file(GLOB_RECURSE TNN_CONVERTER_SRC
            ${TNN_CONVERTER_SRC}
            source/torch/*.cc
            source/torch/*.h
            )
    add_definitions(-DTNN_CONVERTER_TORCH)
endif()

add_executable(TnnConverter ${TNN_CONVERTER_SRC} ${ONNX_PROTO_SRC} ${ONNX_PROTO_HEAD})

if(TNN_CONVERTER_TORCH_ENABLE)
    target_link_libraries(TnnConverter ${TORCH_LIBRARIES})
endif()

if(TNN_BUILD_SHARED)
    target_link_libraries(TnnConverter
            TNN
            gflags
            flatbuffers
            ${Protobuf_LIBRARIES}
            )
elseif(SYSTEM.iOS OR SYSTEM.Darwin)
    target_link_libraries(TnnConverter
            -Wl,-force_load TNN
            gflags
            flatbuffers
            ${Protobuf_LIBRARIES}
            )
else()
    message("target link libraries whole-archive")
    target_link_libraries(TnnConverter
            -Wl,--whole-archive TNN -Wl,--no-whole-archive
            gflags
            flatbuffers
            ${Protobuf_LIBRARIES}
            )
endif()
