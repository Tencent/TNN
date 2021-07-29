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

if(TNN_HUAWEI_NPU_ENABLE)
    add_library(hiai
            SHARED
            IMPORTED)
    set_target_properties(hiai
            PROPERTIES
            IMPORTED_LOCATION
            ${TNN_ROOT}/thirdparty/hiai_ddk/${ANDROID_ABI}/libhiai.so)

    add_library(hiai_ir
            SHARED
            IMPORTED)
    set_target_properties(hiai_ir
            PROPERTIES
            IMPORTED_LOCATION
            ${TNN_ROOT}/thirdparty/hiai_ddk/${ANDROID_ABI}/libhiai_ir.so)

    add_library(hiai_ir_build
            SHARED
            IMPORTED)
    set_target_properties(hiai_ir_build
            PROPERTIES
            IMPORTED_LOCATION
            ${TNN_ROOT}/thirdparty/hiai_ddk/${ANDROID_ABI}/libhiai_ir_build.so)

endif()

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
                        ${CMAKE_CURRENT_SOURCE_DIR}/../../build/libTNN.so
                        ${CMAKE_CURRENT_SOURCE_DIR}/../../build/test/libTNNBenchmarkTest.so
                        ${CMAKE_CURRENT_SOURCE_DIR}/../../build/third_party/gflags/libgflags_nothreads.a
                       # Links the target library to the log library
                       # included in the NDK.
                       ${log-lib})

if(TNN_HUAWEI_NPU_ENABLE)
    target_link_libraries( # Specifies the target library.
            tnn_wrapper hiai hiai_ir hiai_ir_build)
endif()
