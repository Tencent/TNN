file(GLOB COMMON_SRCS ${CMAKE_SOURCE_DIR}/tools/common/*.cc)
file(GLOB evaluation_SRCS *.cc)
file(GLOB_RECURSE BASE_SRC ${CMAKE_SOURCE_DIR}/examples/base/*.cc)
file(GLOB_RECURSE UTIL_SRC ${CMAKE_SOURCE_DIR}/examples/utils/*.cc)

message(${evaluation_SRCS})

include_directories(${CMAKE_SOURCE_DIR}/tools/common)
include_directories(${CMAKE_SOURCE_DIR}/tools/evaluation)
include_directories(${CMAKE_SOURCE_DIR}/source/tnn/interpreter/tnn)
include_directories(${CMAKE_SOURCE_DIR}/examples/base)
include_directories(${CMAKE_SOURCE_DIR}/examples/base/utils)
include_directories(${CMAKE_SOURCE_DIR}/examples/utils)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/stb)

add_executable(eval_cmd ${evaluation_SRCS} ${COMMON_SRCS} ${BASE_SRC} ${UTIL_SRC})

if(TNN_TENSORRT_ENABLE)
    find_package(CUDA REQUIRED)
    set(CUDA_LINK_LIBS ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})
    target_link_libraries(eval_cmd TNN ${CUDA_LINK_LIBS})
    add_definitions(-D_CUDA_)
else()
    target_link_libraries(eval_cmd TNN)
endif()

set_target_properties(eval_cmd PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})