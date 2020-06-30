file(GLOB COMMON_SRCS ${CMAKE_SOURCE_DIR}/tools/common/*.cc)
file(GLOB MODEL_CHECK_SRCS *.cc)

message(${MODEL_CHECK_SRCS})

include_directories(${CMAKE_SOURCE_DIR}/tools/common)
include_directories(${CMAKE_SOURCE_DIR}/tools/model_check)
include_directories(${CMAKE_SOURCE_DIR}/source/tnn/interpreter/tnn)
include_directories(${CMAKE_SOURCE_DIR}/third_party/stb)

add_executable(model_check ${MODEL_CHECK_SRCS} ${COMMON_SRCS})

if(TNN_BUILD_SHARED)
    target_link_libraries(model_check TNN)
elseif(SYSTEM.Darwin OR SYSTEM.iOS)
    target_link_libraries(model_check -Wl,-force_load TNN)
else()
    target_link_libraries(model_check -Wl,--whole-archive TNN -Wl,--no-whole-archive)
endif()

set_target_properties(model_check PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
