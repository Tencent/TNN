file(GLOB COMMON_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/../common/*.cc)
file(GLOB QUANTIZATION_SRCS *.cc)

message(${QUANTIZATION_SRCS})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../common)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../quantization)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../source/tnn/interpreter/tnn)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/stb)

add_executable(quantization_cmd ${QUANTIZATION_SRCS} ${COMMON_SRCS})
target_link_libraries(quantization_cmd TNN)
set_target_properties(quantization_cmd PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
