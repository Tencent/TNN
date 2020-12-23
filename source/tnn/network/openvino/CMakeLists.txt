file(GLOB CPU_BASE_SRC *.h *.cc)
file(GLOB_RECURSE X86_SRC *.h *.cc)

add_library(TNNOpenVINO OBJECT ${CPU_BASE_SRC} ${X86_SRC})
if (CMAKE_SYSTEM_NAME MATCHES "Windows")
    set(CMAKE_CXX_FLAGS, "${CMAKE_CXX_FLAGS} /w -mavx2 -mfma")
else ()
    set(CMAKE_CXX_FLAGS, "${CMAKE_CXX_FLAGS} -mavx2 -mavx -mfma")
endif()
add_definitions(-mavx2 -mavx -mfma)

if(NOT DEFINED ENV{OPENVINO_ROOT_DIR})
    message(FATAL_ERROR "not defined environment variable:OPENVINO_ROOT_DIR")
endif()

include_directories($ENV{OPENVINO_ROOT_DIR}/deployment_tools/inference_engine/include)
include_directories($ENV{OPENVINO_ROOT_DIR}/deployment_tools/ngraph/include)
