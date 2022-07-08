file(GLOB DYNAMIC_RANGE_QUANTIZATION_SRCS *.h *.cc)

message(${DYNAMIC_RANGE_QUANTIZATION_SRCS})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../dynamic_range_quantization)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../source/tnn/interpreter/tnn)

add_executable(dynamic_range_quantization ${DYNAMIC_RANGE_QUANTIZATION_SRCS})

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
