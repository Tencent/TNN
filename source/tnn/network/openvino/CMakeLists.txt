file(GLOB CPU_BASE_SRC *.h *.cc)
file(GLOB_RECURSE X86_SRC *.h *.cc)

add_library(TNNOpenVINO OBJECT ${CPU_BASE_SRC} ${X86_SRC})
set(CMAKE_CXX_FLAGS, "${CMAKE_CXX_FLAGS} -mavx2 -mavx -mfma")
add_definitions(-mavx2 -mavx -mfma)

include_directories(thirdparty/openvino/include)
include_directories(thirdparty/ngraph/include)
