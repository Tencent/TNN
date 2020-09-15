# Sets the minimum version of CMake required to build the native
# library. You should either keep the default value or only pass a
# value of 3.4.0 or lower.

cmake_minimum_required(VERSION 3.4.1)

# Creates and names a library, sets it as either STATIC
# or SHARED, and provides the relative paths to its source code.
# You can define multiple libraries, and CMake builds it for you.
# Gradle automatically packages shared libraries with your APK.

set(TNN_ROOT ../../..)
set(BASE_SRC ../../base)
include_directories(${TNN_ROOT}/include)
include_directories(${TNN_ROOT}/third_party/opencl/include)
include_directories(../demo/src/main/jni/thirdparty)
include_directories(../demo/src/main/jni/thirdparty/kannarotate-android-lib/include)
include_directories(../demo/src/main/jni/thirdparty/kannarotate-android-lib/src)
include_directories(${BASE_SRC}/)
include_directories(../demo/src/main/jni/cc/)

set(TNN_OPENCL_ENABLE ON CACHE BOOL "" FORCE)
set(TNN_ARM_ENABLE ON CACHE BOOL "" FORCE)
set(TNN_BUILD_SHARED OFF CACHE BOOL "" FORCE)

add_subdirectory(${TNN_ROOT}/ ../build)

file(GLOB_RECURSE WRAPPER_SRCS ../demo/src/main/jni/cc/*.cc)
file(GLOB_RECURSE BASE_SRC ${BASE_SRC}/*.cc)
file(GLOB_RECURSE THIRDPARTY_SRCS ../demo/src/main/jni/thirdparty/kannarotate-android-lib/src/*.c)
add_library(tnn_wrapper SHARED ${WRAPPER_SRCS} ${BASE_SRC} ${THIRDPARTY_SRCS})

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
                        -Wl,--whole-archive TNN -Wl,--no-whole-archive
                       # Links the target library to the log library
                       # included in the NDK.
                       ${log-lib})




