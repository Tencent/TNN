cmake_minimum_required(VERSION 3.1)

project(TNN)

ENABLE_LANGUAGE(ASM)

set(TNN_MAJOR_VERSION 0)
set(TNN_MINOR_VERSION 1)
set(TNN_PATCH_VERSION 0)
set(TNN_BUILD_VERSION 0)
set(TNN_VERSION "${TNN_MAJOR_VERSION}.${TNN_MINOR_VERSION}.${TNN_PATCH_VERSION}.${TNN_BUILD_VERSION}")

option(TNN_CPU_ENABLE "Enable Cpu" OFF)
option(TNN_X86_ENABLE  "Enable X86" OFF)
option(TNN_ARM_ENABLE "Enable Arm" OFF)
option(TNN_METAL_ENABLE "Enable Metal" OFF)
option(TNN_OPENCL_ENABLE "Enable OpenCL" OFF)
option(TNN_CUDA_ENABLE "Enable CUDA" OFF)
option(TNN_HUAWEI_NPU_ENABLE "Enable NPU" OFF)
option(TNN_RK_NPU_ENABLE "Enable RKNPU" OFF)

option(TNN_SYMBOL_HIDE "Enable Hide Symbol Visibility" ON)

option(TNN_OPENMP_ENABLE "Enable OpenMP" OFF)
option(TNN_BUILD_SHARED "Build Shared Library" ON)
option(TNN_TEST_ENABLE "Enable Test" OFF)
option(TNN_UNIT_TEST_ENABLE "Enable Test" OFF)
option(TNN_PROFILER_ENABLE "Enable Test" OFF)
option(TNN_QUANTIZATION_ENABLE "Enable Test" OFF)
option(TNN_MODEL_CHECK_ENABLE "Enable Test" OFF)
option(TNN_BENCHMARK_MODE "Enable Benchmark" OFF)
option(TNN_UNIT_TEST_BENCHMARK "Enable Benchmark Layer" OFF)
option(TNN_CONVERTER_ENABLE "Enable Model Converter" OFF)
option(TNN_TNN2MEM_ENABLE "Enable tnn2mem" OFF)

message(${CMAKE_SOURCE_DIR})
message(${CMAKE_CURRENT_SOURCE_DIR})

include(cmake/macros.cmake)

if (SYSTEM.Windows)
    add_definitions(-DBUILDING_DLL)
endif()

if(TNN_PROFILER_ENABLE)
    add_definitions(-DTNN_PROFILE)
    set(TNN_SYMBOL_HIDE OFF)
endif()

if(TNN_BENCHMARK_MODE)
    add_definitions(-DBENCHMARK)
endif()

# ignore loop-vectorize warning
if(SYSTEM.Windows)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
else()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-pass-failed")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-pass-failed")
endif()

# ignore deprecated warning
if(SYSTEM.Windows)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
else()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-deprecated-declarations -Wno-ignored-attributes")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations -Wno-ignored-attributes")
endif()

if(DEBUG)
    set(TNN_SYMBOL_HIDE OFF)
    add_definitions(-DDEBUG)
    if (NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
        set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "set build type to debug" FORCE)
    endif()
else()
    if (NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
        set(CMAKE_BUILD_TYPE "Release" CACHE STRING "set build type to release" FORCE)
    endif()
    if(BUILD_FOR_ANDROID_COMMAND)
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -s -Wl,--gc-sections")
    endif()
endif()

if(TNN_TEST_ENABLE)
    option(TNN_METAL_FLOAT32 "Enable Metal Float32" ON)
else()
    set(TNN_UNIT_TEST_ENABLE OFF)
endif()

if(TNN_UNIT_TEST_ENABLE)
    enable_testing()
    if (TNN_UNIT_TEST_BENCHMARK MATCHES OFF)
        add_definitions(-DOPENCL_FORCE_FP32)
    endif()
    set(TNN_CPU_ENABLE ON)
    set(TNN_SYMBOL_HIDE OFF)
endif()

if(TNN_CONVERTER_ENABLE)
    set(TNN_SYMBOL_HIDE OFF)
endif()

if(TNN_QUANTIZATION_ENABLE OR TNN_MODEL_CHECK_ENABLE)
    set(TNN_SYMBOL_HIDE OFF)
    add_definitions(-DFORWARD_CALLBACK_ENABLE)
endif()

if(TNN_MODEL_CHECK_ENABLE)
    add_definitions(-DOPENCL_FORCE_FP32)
    option(TNN_METAL_FLOAT32 "Enable Metal Float32" ON)
endif()

if(TNN_METAL_FLOAT32)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DTNN_METAL_FULL_PRECISION=1")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTNN_METAL_FULL_PRECISION=1")

    if(TNN_PROFILER_ENABLE OR TNN_MODEL_CHECK_ENABLE)
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DTNN_METAL_BENCHMARK=1 -DTNN_METAL_DEBUG=1")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTNN_METAL_BENCHMARK=1 -DTNN_METAL_DEBUG=1")
    endif()
endif()

if(TNN_OPENMP_ENABLE)
    FIND_PACKAGE(OpenMP REQUIRED)
    if(OPENMP_FOUND)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    else()
        error("OpenMP Not Found.")
    endif()
endif()


if(UNIX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
endif()

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(TNN_METAL_ENABLE)
    add_compile_options(-x objective-c++)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fobjc-arc")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fobjc-arc")
endif()

if(TNN_TNN2MEM_ENABLE)
    add_subdirectory(tools/tnn2mem)
endif()

message(STATUS ">>>>>>>>>>>>>")
message(STATUS "TNN BUILD INFO:")
message(STATUS "\tSystem: ${CMAKE_SYSTEM_NAME}")
message(STATUS "\tProcessor: ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "\tCpu:\t${TNN_CPU_ENABLE}")
message(STATUS "\tX86:\t${TNN_X86_ENABLE}")
message(STATUS "\tArm:\t${TNN_ARM_ENABLE}")
message(STATUS "\tMetal:\t${TNN_METAL_ENABLE}")
message(STATUS "\tOpenCL:\t${TNN_OPENCL_ENABLE}")
message(STATUS "\tCUDA:\t${TNN_CUDA_ENABLE}")
message(STATUS "\tHuaweiNPU:\t${TNN_HUAWEI_NPU_ENABLE}")
message(STATUS "\tRKNPU:\t${TNN_RK_NPU_ENABLE}")
message(STATUS "\tOpenMP:\t${TNN_OPENMP_ENABLE}")
message(STATUS "\tTEST:\t${TNN_TEST_ENABLE}")
message(STATUS "\t--Unit Test:\t${TNN_UNIT_TEST_ENABLE}")
message(STATUS "\tQantization:\t${TNN_QUANTIZATION_ENABLE}")
message(STATUS "\tModelCheck:\t${TNN_MODEL_CHECK_ENABLE}")
message(STATUS "\tDEBUG:\t${DEBUG}")
message(STATUS "\tPROFILE:\t${TNN_PROFILER_ENABLE}")
message(STATUS "\tBENCHMARK:\t${TNN_BENCHMARK_MODE}")
message(STATUS "\tBENCHMARK Layer:\t${TNN_UNIT_TEST_BENCHMARK}")
message(STATUS "\tModel Converter:\t${TNN_CONVERTER_ENABLE}")
message(STATUS "\tTNN2MEM:\t${TNN_TNN2MEM_ENABLE}")

include_directories(include)
include_directories(source)

file(GLOB_RECURSE SRC "source/tnn/core/*.h"
                      "source/tnn/core/*.cc"
                      "source/tnn/layer/*.h"
                      "source/tnn/layer/*.cc"
                      "source/tnn/utils/*.h"
                      "source/tnn/utils/*.cc"
                      "source/tnn/interpreter/*.h"
                      "source/tnn/interpreter/*.cc"
                      "source/tnn/optimizer/*.h"
                      "source/tnn/optimizer/*.cc"
                      "source/tnn/memory_manager/*.h"
                      "source/tnn/memory_manager/*.cc")

if(TNN_SYMBOL_HIDE AND UNIX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden -fvisibility-inlines-hidden")
endif()

if(TNN_CPU_ENABLE)
    add_subdirectory(source/tnn/device/cpu)
    set(TARGET_OBJECTS ${TARGET_OBJECTS} "$<TARGET_OBJECTS:TNNCpu>")
endif()

if(TNN_ARM_ENABLE)
    add_subdirectory(source/tnn/device/arm)
    set(TARGET_OBJECTS ${TARGET_OBJECTS} "$<TARGET_OBJECTS:TNNArm>")
endif()

if(TNN_X86_ENABLE)
    add_subdirectory(source/tnn/device/x86)
    set(TARGET_OBJECTS ${TARGET_OBJECTS} "$<TARGET_OBJECTS:TNNX86>")
endif()

if(TNN_OPENCL_ENABLE)
    if(SHARING_MEM_WITH_OPENGL)
        add_definitions(-DSHARING_MEM_WITH_OPENGL)
        add_definitions(-DCL_HPP_TARGET_OPENCL_VERSION=120)
    endif()
    add_subdirectory(source/tnn/device/opencl)
    set(TARGET_OBJECTS ${TARGET_OBJECTS} "$<TARGET_OBJECTS:TNNOpenCL>")
endif()

if(TNN_METAL_ENABLE)
    add_subdirectory(source/tnn/device/metal)
    set(TARGET_OBJECTS ${TARGET_OBJECTS} "$<TARGET_OBJECTS:TNNMetal>")
endif()

if(TNN_CUDA_ENABLE)
    add_subdirectory(source/tnn/device/cuda)
    set(TARGET_OBJECTS ${TARGET_OBJECTS} "$<TARGET_OBJECTS:TNNCuda>")
endif()

if(TNN_HUAWEI_NPU_ENABLE)
    if(ANDROID_ABI STREQUAL "armeabi-v7a")
        link_directories(
                third_party/huawei_npu/hiai_ddk_latest/armeabi-v7a/
        )
    else()
        link_directories(
                third_party/huawei_npu/hiai_ddk_latest/arm64-v8a/
        )
    endif()
    add_subdirectory(source/tnn/device/huawei_npu)
    set(TARGET_OBJECTS ${TARGET_OBJECTS} "$<TARGET_OBJECTS:TNNNPU>")
endif()

if(TNN_RK_NPU_ENABLE)
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        link_directories(
                ./third_party/rknpu/rknpu_ddk/lib64/
        )
    else()
        link_directories(
                ./third_party/rknpu/rknpu_ddk/lib/
        )
    endif()
    add_subdirectory(source/tnn/device/rknpu)
    set(TARGET_OBJECTS ${TARGET_OBJECTS} "$<TARGET_OBJECTS:TNNRKNPU>")
endif()

if(TNN_BUILD_SHARED)
    add_library(TNN SHARED ${SRC} ${TARGET_OBJECTS})
    set_target_properties(TNN PROPERTIES VERSION ${TNN_VERSION} SOVERSION ${TNN_MAJOR_VERSION})
    if(SHARING_MEM_WITH_OPENGL)
        target_link_libraries(TNN -lEGL -lGLESv2)
    endif()
else()
    add_library(TNN STATIC ${SRC} ${TARGET_OBJECTS})
    set_target_properties(TNN PROPERTIES VERSION ${TNN_VERSION})
    if(SHARING_MEM_WITH_OPENGL)
        target_link_libraries(TNN -lEGL -lGLESv2)
    endif()
endif()

if(TNN_QUANTIZATION_ENABLE)
    add_subdirectory(tools/quantization)
endif()

if(SYSTEM.Linux)
    include(platforms/linux/CMakeLists.txt)
elseif(SYSTEM.Android)
    include(platforms/android/CMakeLists.txt)
elseif(SYSTEM.iOS)
    include(platforms/ios/CMakeLists.txt)
elseif(SYSTEM.Darwin)
    include(platforms/mac/CMakeLists.txt)
endif()

if(TNN_MODEL_CHECK_ENABLE)
    add_subdirectory(tools/model_check)
endif()


if (TNN_TEST_ENABLE OR TNN_CONVERTER_ENABLE)
    add_subdirectory(third_party/gflags)
    get_target_property(GFLAGS_INCLUDE_DIRS gflags INTERFACE_INCLUDE_DIRECTORIES)
    include_directories(BEFORE "${GFLAGS_INCLUDE_DIRS}")
endif ()


if(TNN_TEST_ENABLE)
    add_subdirectory(test)
endif()

if(TNN_CONVERTER_ENABLE)
    add_subdirectory(third_party/flatbuffers)
    add_subdirectory(tools/converter)
endif()

if(TNN_COVERAGE)
    if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-instr-generate -fcoverage-mapping")
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -coverage -fprofile-arcs -ftest-coverage")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -coverage -lgcov")
    endif()
endif()
