file(GLOB NPU_SRC
        *.h
        *.cc
        convert/*.h
        convert/*.cc
        convert/math/*.h
        convert/math/*.cc)
add_library(TNNNPU OBJECT ${NPU_SRC})
#include_directories(../../../../third_party/huawei_npu/hiai_ddk_latest/include) #for old DDK 100.320.XXX.XXX
include_directories(../../../../third_party/huawei_npu/hiai_ddk_latest/ddk/ai_ddk_lib/include)
