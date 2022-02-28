file(GLOB FAKE_QUANTIZATION_SRCS *.h *.cc)

message(${FAKE_QUANTIZATION_SRCS})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../fake_quantization)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../source/tnn/interpreter/tnn)

add_executable(fake_quantization ${FAKE_QUANTIZATION_SRCS})

if(TNN_BUILD_SHARED)
    target_link_libraries(fake_quantization TNN gflags)
elseif(SYSTEM.Darwin OR SYSTEM.iOS)
    message("target link libraries -force_load")
    target_link_libraries(fake_quantization -Wl,-force_load TNN gflags)
elseif(SYSTEM.Windows)
    target_link_libraries(fake_quantization -Wl,--whole-archive TNN -Wl,--no-whole-archive gflags)
else()
    message("target link libraries whole-archive")
    target_link_libraries(fake_quantization -Wl,--whole-archive TNN -Wl,--no-whole-archive gflags)
endif()

set_target_properties(fake_quantization PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
