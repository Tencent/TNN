file(GLOB NPU_SRC
        *.h
        *.cc
        convert/*.h
        convert/*.cc
        convert/math/*.h
        convert/math/*.cc)
add_library(TNNNPU OBJECT ${NPU_SRC})
if(TNN_CPU_ENABLE)
else()
    add_library(TNNBLOB OBJECT ../cpu/cpu_blob_converter.cc
            ../cpu/cpu_blob_converter.h
            ../cpu/acc/compute/compute_int8.h
            ../cpu/acc/compute/compute_int8.cc)
endif()
include_directories(../../../../third_party/huawei_npu/hiai_ddk_latest/include)
