file(GLOB COMMON_SRCS ${CMAKE_SOURCE_DIR}/tools/common/*.cc)
file(GLOB ALIGN_TOOL_SRCS *.h *.cc)

message(${ALIGN_TOOL_SRCS})

include_directories(${CMAKE_SOURCE_DIR}/source/include/tnn/core)
include_directories(${CMAKE_SOURCE_DIR}/source/include/tnn/utils)

get_filename_component(TNN_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..  ABSOLUTE)

include_directories(${TNN_ROOT})

include_directories(${CMAKE_SOURCE_DIR}/tools/common)
include_directories(${CMAKE_SOURCE_DIR}/tools/convert2tnn/align_tool/run_tnn_model)
include_directories(${CMAKE_SOURCE_DIR}/source/tnn/interpreter/tnn)
include_directories(${CMAKE_SOURCE_DIR}/third_party/stb)

add_executable(align_tool ${ALIGN_TOOL_SRCS} ${COMMON_SRCS})

if(TNN_BUILD_SHARED)
    target_link_libraries(align_tool TNN)
elseif(SYSTEM.Darwin OR SYSTEM.iOS)
    target_link_libraries(align_tool -Wl,-force_load TNN)
else()
    target_link_libraries(align_tool -Wl,--whole-archive TNN -Wl,--no-whole-archive)
endif()

set_target_properties(align_tool PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
