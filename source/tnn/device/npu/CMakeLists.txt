option(NPU_V200_ENABLE "Enable NPU 200" OFF)
option(NPU_V300_ENABLE "Enable NPU 300" OFF)
option(NPU_V320_ENABLE "Enable NPU 320" OFF)
option(NPU_IR_ENABLE "Enable NPU IR" ON)


if(NPU_V200_ENABLE)
  file(GLOB_RECURSE NPU_SRC
	${CMAKE_CURRENT_SOURCE_DIR}/src_v200/*.h
	${CMAKE_CURRENT_SOURCE_DIR}/src_v200/*.cc)
  add_library(TNNNPU OBJECT ${NPU_SRC})
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/hiai_ddk_200/include)
endif()

if(NPU_V300_ENABLE)
  file(GLOB_RECURSE NPU_SRC
          ${CMAKE_CURRENT_SOURCE_DIR}/src_v300/*.h
          ${CMAKE_CURRENT_SOURCE_DIR}/src_v300/*.cc)
  add_library(TNNNPU OBJECT ${NPU_SRC})
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/hiai_ddk_300/include)
endif()

if(NPU_V320_ENABLE)
  file(GLOB_RECURSE NPU_SRC
          ${CMAKE_CURRENT_SOURCE_DIR}/src_v300/*.h
          ${CMAKE_CURRENT_SOURCE_DIR}/src_v300/*.cc)
  add_library(TNNNPU OBJECT ${NPU_SRC})
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/hiai_ddk_320/include)
endif()

if(NPU_IR_ENABLE)
  file(GLOB NPU_SRC
          *.h
          *.cc
          convert/*.h
          convert/*.cc
          convert/math/*.h
          convert/math/*.cc)
  add_library(TNNNPU OBJECT ${NPU_SRC})
  add_library(TNNBLOB OBJECT ../cpu/cpu_blob_converter.cc
          ../cpu/cpu_blob_converter.h
          ../cpu/acc/compute/compute_int8.h
          ../cpu/acc/compute/compute_int8.cc)
  include_directories(../../../../third_party/npu/hiai_ddk_320/include)
endif()
