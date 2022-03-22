file(GLOB DYNAMIC_RANGE_QUANTIZATION_SRCS *.h *.cc)

message(${DYNAMIC_RANGE_QUANTIZATION_SRCS})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../dynamic_range_quantization)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../source/tnn/interpreter/tnn)

if (SYSTEM.Windows)
    file(GLOB INTERPRETER_SRCS
            ${CMAKE_CURRENT_SOURCE_DIR}/../../source/tnn/core/layer_type*
            ${CMAKE_CURRENT_SOURCE_DIR}/../../source/tnn/interpreter/*interpreter*
            ${CMAKE_CURRENT_SOURCE_DIR}/../../source/tnn/interpreter/layer_param.h
            ${CMAKE_CURRENT_SOURCE_DIR}/../../source/tnn/interpreter/layer_resource.h
            ${CMAKE_CURRENT_SOURCE_DIR}/../../source/tnn/interpreter/net_resource*
            ${CMAKE_CURRENT_SOURCE_DIR}/../../source/tnn/interpreter/net_structure*
            ${CMAKE_CURRENT_SOURCE_DIR}/../../source/tnn/interpreter/*packer*
            ${CMAKE_CURRENT_SOURCE_DIR}/../../source/tnn/interpreter/raw_buffer*
            ${CMAKE_CURRENT_SOURCE_DIR}/../../source/tnn/interpreter/tnn/*.h
            ${CMAKE_CURRENT_SOURCE_DIR}/../../source/tnn/interpreter/tnn/*.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/../../source/tnn/utils/split_utils*
            ${CMAKE_CURRENT_SOURCE_DIR}/../../source/tnn/utils/md5*)
    add_executable(dynamic_range_quantization ${DYNAMIC_RANGE_QUANTIZATION_SRCS} ${INTERPRETER_SRCS})
else ()
    add_executable(dynamic_range_quantization ${DYNAMIC_RANGE_QUANTIZATION_SRCS})
endif ()

if(TNN_BUILD_SHARED)
    target_link_libraries(dynamic_range_quantization TNN gflags)
elseif(SYSTEM.Darwin OR SYSTEM.iOS)
    message("target link libraries -force_load")
    target_link_libraries(dynamic_range_quantization -Wl,-force_load TNN gflags)
elseif(SYSTEM.Windows)
    target_link_libraries(dynamic_range_quantization -Wl,--whole-archive TNN -Wl,--no-whole-archive gflags)
else()
    message("target link libraries whole-archive")
    target_link_libraries(dynamic_range_quantization -Wl,--whole-archive TNN -Wl,--no-whole-archive gflags)
endif()

set_target_properties(dynamic_range_quantization PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
