cmake_minimum_required(VERSION 3.1)

project(TNN)

ENABLE_LANGUAGE(ASM)

set(TNN_MAJOR_VERSION 0)
set(TNN_MINOR_VERSION 1)
set(TNN_PATCH_VERSION 0)
set(TNN_BUILD_VERSION 0)
set(TNN_VERSION "${TNN_MAJOR_VERSION}.${TNN_MINOR_VERSION}.${TNN_PATCH_VERSION}.${TNN_BUILD_VERSION}")

option(TNN_CPU_ENABLE "Enable Cpu" ON)
option(TNN_X86_ENABLE  "Enable X86" OFF)
option(TNN_ARM_ENABLE "Enable Arm" OFF)
option(TNN_ARM82_ENABLE "Enable Arm82" OFF)
option(TNN_METAL_ENABLE "Enable Metal" OFF)
option(TNN_OPENCL_ENABLE "Enable OpenCL" OFF)
option(TNN_CUDA_ENABLE "Enable CUDA" OFF)
option(TNN_DSP_ENABLE "Enable DSP" OFF)
option(TNN_ATLAS_ENABLE "Enable Atlas" OFF)
option(TNN_TENSORRT_ENABLE "Enable TensorRT" OFF)
option(TNN_OPENVINO_ENABLE  "Enable OPENVINO" OFF)
option(TNN_NPU_ENABLE "Enable NPU" OFF)
option(TNN_HUAWEI_NPU_ENABLE "Enable NPU" OFF)
option(TNN_RK_NPU_ENABLE "Enable RKNPU" OFF)
option(TNN_SYMBOL_HIDE "Enable Hide Symbol Visibility" ON)
option(TNN_OPENMP_ENABLE "Enable OpenMP" OFF)
option(TNN_BUILD_SHARED "Build Shared Library" ON)
option(TNN_OPENVINO_BUILD_SHARED "Build Shared Openvino Library" OFF)
option(TNN_TEST_ENABLE "Enable Test" OFF)
option(TNN_UNIT_TEST_ENABLE "Enable Test" OFF)
option(TNN_PROFILER_ENABLE "Enable Test" OFF)
option(TNN_QUANTIZATION_ENABLE "Enable Test" OFF)
option(TNN_MODEL_CHECK_ENABLE "Enable Test" OFF)
option(TNN_BENCHMARK_MODE "Enable Benchmark" OFF)
option(TNN_UNIT_TEST_BENCHMARK "Enable Benchmark Layer" OFF)
option(TNN_CONVERTER_ENABLE "Enable Model Converter" OFF)
option(TNN_ONNX2TNN_ENABLE "Enable ONNX2TNN Converter" OFF)
option(TNN_TNN2MEM_ENABLE "Enable tnn2mem" OFF)
option(TNN_BUILD_BENCHMARK_TEST_LIB_ENABLE "Enable Build Benchmark Test Lib" OFF)
option(TNN_GLIBCXX_USE_CXX11_ABI_ENABLE "Enable Use CXX11 ABI" ON)

set(TNN_USE_GFLAGS OFF)

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
    add_definitions(-DGENERATE_RESOURCE)
endif()

if(MSVC)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /wd4003 /wd4819 /wd4244 /wd4018 /utf-8")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4003 /wd4819 /wd4244 /wd4018 /utf-8")
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
    set(TNN_CPU_ENABLE ON)
    set(TNN_SYMBOL_HIDE OFF)
    add_definitions(-DGENERATE_RESOURCE)
endif()

if(TNN_CONVERTER_ENABLE OR TNN_ONNX2TNN_ENABLE)
    set(TNN_SYMBOL_HIDE OFF)
    add_definitions(-DTNN_CONVERTER_RUNTIME)
endif()

if(TNN_QUANTIZATION_ENABLE OR TNN_MODEL_CHECK_ENABLE)
    set(TNN_SYMBOL_HIDE OFF)
    add_definitions(-DFORWARD_CALLBACK_ENABLE)
endif()

if(TNN_QUANTIZATION_ENABLE OR TNN_UNIT_TEST_ENABLE)
    add_definitions(-DGET_INTERP_ENABLE)
endif()

if(TNN_MODEL_CHECK_ENABLE)
    option(TNN_METAL_FLOAT32 "Enable Metal Float32" ON)
endif()

if(TNN_ARM82_ENABLE)
    add_definitions(-DTNN_ARM82=1)
endif()

# only used to simulate arm82 computation in the unit test
option(TNN_ARM82_SIMU "Enable arm82 simulation" OFF)
if(TNN_ARM82_SIMU)
    add_definitions(-DTNN_ARM82_SIMU)
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
        if(MSVC)
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /openmp")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /openmp")
        else()
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
            include_directories(${OpenMP_C_INCLUDE_DIRS} ${OpenMP_CXX_INCLUDE_DIRS})
            if(${ANDROID_NDK_MAJOR})
                if(${ANDROID_NDK_MAJOR} GREATER 20)

                else()
                    link_libraries(${OpenMP_C_LIBRARIES} ${OpenMP_CXX_LIBRARIES})
                endif()
            else()
                link_libraries(${OpenMP_C_LIBRARIES} ${OpenMP_CXX_LIBRARIES})
            endif()
        endif()
    else()
        error("OpenMP Not Found.")
    endif()
endif()


if(UNIX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
    if(TNN_GLIBCXX_USE_CXX11_ABI_ENABLE)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=1")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
    endif()
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
message(STATUS "\tArm82:\t${TNN_ARM82_ENABLE}")
message(STATUS "\tMetal:\t${TNN_METAL_ENABLE}")
message(STATUS "\tOpenCL:\t${TNN_OPENCL_ENABLE}")
message(STATUS "\tCUDA:\t${TNN_CUDA_ENABLE}")
message(STATUS "\tDSP:\t${TNN_DSP_ENABLE}")
message(STATUS "\tAtlas:\t${TNN_ATLAS_ENABLE}")
message(STATUS "\tTensorRT:\t${TNN_TENSORRT_ENABLE}")
message(STATUS "\tHuaweiNPU:\t${TNN_HUAWEI_NPU_ENABLE}")
message(STATUS "\tRKNPU:\t${TNN_RK_NPU_ENABLE}")
message(STATUS "\tOpenVINO:\t${TNN_OPENVINO_ENABLE}")
message(STATUS "\tOpenMP:\t${TNN_OPENMP_ENABLE}")
message(STATUS "\tTEST:\t${TNN_TEST_ENABLE}")
message(STATUS "\t--Unit Test:\t${TNN_UNIT_TEST_ENABLE}")
message(STATUS "\tQuantization:\t${TNN_QUANTIZATION_ENABLE}")
message(STATUS "\tModelCheck:\t${TNN_MODEL_CHECK_ENABLE}")
message(STATUS "\tDEBUG:\t${DEBUG}")
message(STATUS "\tPROFILE:\t${TNN_PROFILER_ENABLE}")
message(STATUS "\tBENCHMARK:\t${TNN_BENCHMARK_MODE}")
message(STATUS "\tBENCHMARK Layer:\t${TNN_UNIT_TEST_BENCHMARK}")
message(STATUS "\tModel Converter:\t${TNN_CONVERTER_ENABLE}")
message(STATUS "\tONNX2TNN Converter:\t${TNN_ONNX2TNN_ENABLE}")
message(STATUS "\tTNN2MEM:\t${TNN_TNN2MEM_ENABLE}")
message(STATUS "\tBENCHMARK Test Lib:\t${TNN_BUILD_BENCHMARK_TEST_LIB_ENABLE}")

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
                      "source/tnn/extern_wrapper/*.h"
                      "source/tnn/extern_wrapper/*.cc"
                      "source/tnn/memory_manager/*.h"
                      "source/tnn/memory_manager/*.cc")

if(TNN_SYMBOL_HIDE AND UNIX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden -fvisibility-inlines-hidden")
endif()

if(TNN_X86_ENABLE)
    add_subdirectory(source/tnn/device/x86)
    set(TARGET_OBJECTS ${TARGET_OBJECTS} "$<TARGET_OBJECTS:TNNX86>")
endif()

if(TNN_CPU_ENABLE)
    add_subdirectory(source/tnn/device/cpu)
    set(TARGET_OBJECTS ${TARGET_OBJECTS} "$<TARGET_OBJECTS:TNNCpu>")
endif()

if(TNN_ARM_ENABLE)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")

    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm")
        add_definitions( -mfpu=neon )
    endif()
    add_subdirectory(source/tnn/device/arm)
    set(TARGET_OBJECTS ${TARGET_OBJECTS} "$<TARGET_OBJECTS:TNNArm>")
    if(TNN_ARM82_ENABLE)
        set(TARGET_OBJECTS ${TARGET_OBJECTS} "$<TARGET_OBJECTS:TNNArm82>")
    endif()
endif()

if(TNN_OPENVINO_ENABLE)
    add_subdirectory(source/tnn/network/openvino)
    set(TARGET_OBJECTS ${TARGET_OBJECTS} "$<TARGET_OBJECTS:TNNOpenVINO>")
endif()

if(TNN_OPENCL_ENABLE)
    include(FindPythonInterp REQUIRED)
    if (NOT PYTHON_EXECUTABLE)
        message (FATAL_ERROR "No Python installation found! It is required by OpenCL codegen.")
    endif ()

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
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
    add_subdirectory(source/tnn/device/cuda)
    if(TNN_TENSORRT_ENABLE)
        add_subdirectory(source/tnn/network/tensorrt)
        set(TARGET_OBJECTS ${TARGET_OBJECTS} "$<TARGET_OBJECTS:TNNTensorRT>")
    endif()
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
elseif(SYSTEM.Windows)
    include(platforms/windows/CMakeLists.txt)
endif()

if (TNN_TEST_ENABLE OR TNN_CONVERTER_ENABLE OR TNN_MODEL_CHECK_ENABLE)
    set(TNN_USE_GFLAGS ON)
endif ()

if (TNN_USE_GFLAGS)
    add_subdirectory(third_party/gflags)
    get_target_property(GFLAGS_INCLUDE_DIRS gflags INTERFACE_INCLUDE_DIRECTORIES)
    include_directories(BEFORE "${GFLAGS_INCLUDE_DIRS}")
endif ()

if(TNN_MODEL_CHECK_ENABLE)
    add_subdirectory(tools/model_check)
endif()

if(TNN_TEST_ENABLE)
    add_subdirectory(test)
endif()

if(TNN_CONVERTER_ENABLE)
    add_subdirectory(third_party/flatbuffers)
    add_subdirectory(tools/converter)
endif()

if(TNN_ONNX2TNN_ENABLE)
    add_subdirectory(tools/onnx2tnn/onnx-converter)
endif()

if(TNN_COVERAGE)
    if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-instr-generate -fcoverage-mapping")
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -coverage -fprofile-arcs -ftest-coverage")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -coverage -lgcov")
    endif()
endif()
