# Sets the minimum version of CMake required to build the native
# library. You should either keep the default value or only pass a
# value of 3.4.0 or lower.

cmake_minimum_required(VERSION 3.4.1)

# Creates and names a library, sets it as either STATIC
# or SHARED, and provides the relative paths to its source code.
# You can define multiple libraries, and CMake builds it for you.
# Gradle automatically packages shared libraries with your APK.

get_filename_component(TNN_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../../..  ABSOLUTE)
include_directories(${TNN_ROOT})
include_directories(${TNN_ROOT}/include)
include_directories(${TNN_ROOT}/test)
include_directories(${TNN_ROOT}/third_party/opencl/include)
include_directories(../benchmark/src/main/jni/cc/)

set(CMAKE_BUILD_TYPE Release)

file(GLOB_RECURSE WRAPPER_SRCS ../benchmark/src/main/jni/cc/*.cc)
file(GLOB_RECURSE THIRDPARTY_SRCS ../benchmark/src/main/jni/thirdparty/kannarotate-android-lib/src/*.c)
add_library(tnn_wrapper SHARED ${WRAPPER_SRCS} ${THIRDPARTY_SRCS})

find_library( # Sets the name of the path variable.
              log-lib

              # Specifies the name of the NDK library that
              # you want CMake to locate.
              log)

# Specifies libraries CMake should link to your target library. You
# can link multiple libraries, such as libraries you define in the
# build script, prebuilt third-party libraries, or system libraries.

target_link_libraries( # Specifies the target library.
                        tnn_wrapper
                        -ljnigraphics
                        -Wl,--whole-archive
                        ${CMAKE_CURRENT_SOURCE_DIR}/../../build/libTNN.a
                        ${CMAKE_CURRENT_SOURCE_DIR}/../../build/test/libTNNBenchmarkTest.a
                        -Wl,--no-whole-archive
                        ${CMAKE_CURRENT_SOURCE_DIR}/../../build/third_party/gflags/libgflags_nothreads.a
                       # Links the target library to the log library
                       # included in the NDK.
                       ${log-lib})

if(TNN_HUAWEI_NPU_ENABLE)
    target_link_libraries( # Specifies the target library.
            tnn_wrapper hiai hiai_ir hiai_ir_build)
endif()
