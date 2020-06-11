file(GLOB_RECURSE NPU_SRC *.h *.cc)

add_library(TNNNPU OBJECT ${NPU_SRC})

include_directories(thirdparty/hiai_ddk_200/include)
