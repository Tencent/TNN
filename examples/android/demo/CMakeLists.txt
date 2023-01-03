# Sets the minimum version of CMake required to build the native
# library. You should either keep the default value or only pass a
# value of 3.4.0 or lower.

cmake_minimum_required(VERSION 3.4.1)

# Creates and names a library, sets it as either STATIC
# or SHARED, and provides the relative paths to its source code.
# You can define multiple libraries, and CMake builds it for you.
# Gradle automatically packages shared libraries with your APK.

set(TNN_ROOT ../../..)
set(BASE_SRC ${TNN_ROOT}/examples/base)
set(UTILS_SRC ${TNN_ROOT}/examples/utils)
set(OPENCV_ANDROID_SDK_PATH ${TNN_ROOT}/third_party/opencv/Android/OpenCV-android-sdk)
include_directories(${TNN_ROOT}/include)
include_directories(${TNN_ROOT}/third_party/opencl/include)
include_directories(${OPENCV_ANDROID_SDK_PATH}/sdk/native/jni/include)
include_directories(${CMAKE_SOURCE_DIR}/src/main/jni/thirdparty)
include_directories(${CMAKE_SOURCE_DIR}/src/main/jni/thirdparty/kannarotate-android-lib/include)
include_directories(${CMAKE_SOURCE_DIR}/src/main/jni/thirdparty/kannarotate-android-lib/src)
include_directories(${BASE_SRC}/)
include_directories(${UTILS_SRC}/)
include_directories(${CMAKE_SOURCE_DIR}/src/main/jni/cc/)
include_directories(${CMAKE_SOURCE_DIR}/src/main/jni/thirdparty/hiai_ddk/include)

set(CMAKE_BUILD_TYPE Release)

set(TNN_OPENCL_ENABLE ON CACHE BOOL "" FORCE)
set(TNN_ARM_ENABLE ON CACHE BOOL "" FORCE)
if(ANDROID_ABI STREQUAL "arm64-v8a")
    set(TNN_ARM82_ENABLE ON CACHE BOOL "" FORCE)
endif()
set(TNN_BUILD_SHARED OFF CACHE BOOL "" FORCE)
set(TNN_HUAWEI_NPU_ENABLE OFF CACHE BOOL "" FORCE)
set(TNN_CPU_ENABLE ON CACHE BOOL "" FORCE)
set(TNN_OPENCV_ENABLE OFF CACHE BOOL "" FORCE)
set(SHARING_MEM_WITH_OPENGL OFF CACHE BOOL "" FORCE)

add_subdirectory(${TNN_ROOT}/ ../build)

if(TNN_OPENCV_ENABLE)
    add_definitions(-DHAS_OPENCV)
endif()

file(GLOB_RECURSE WRAPPER_SRCS ${CMAKE_SOURCE_DIR}/src/main/jni/cc/*.cc)
file(GLOB_RECURSE BASE_SRC ${BASE_SRC}/*.cc)
file(GLOB_RECURSE UTILS_SRC ${UTILS_SRC}/*.cc)
file(GLOB_RECURSE THIRDPARTY_SRCS ${CMAKE_SOURCE_DIR}/src/main/jni/thirdparty/kannarotate-android-lib/src/*.c)
add_library(tnn_wrapper SHARED ${WRAPPER_SRCS} ${BASE_SRC} ${THIRDPARTY_SRCS} ${UTILS_SRC})

if(TNN_OPENCV_ENABLE)
    add_library(opencv
            SHARED
            IMPORTED)
    set_target_properties(opencv
            PROPERTIES
            IMPORTED_LOCATION
            ${OPENCV_ANDROID_SDK_PATH}/sdk/native/libs/${ANDROID_ABI}/libopencv_java3.so)
endif()

if(TNN_HUAWEI_NPU_ENABLE)
    add_library(hiai
            SHARED
            IMPORTED)
    set_target_properties(hiai
            PROPERTIES
            IMPORTED_LOCATION
            ${CMAKE_SOURCE_DIR}/src/main/jni/thirdparty/hiai_ddk/${ANDROID_ABI}/libhiai.so)

    add_library(hiai_ir
            SHARED
            IMPORTED)
    set_target_properties(hiai_ir
            PROPERTIES
            IMPORTED_LOCATION
            ${CMAKE_SOURCE_DIR}/src/main/jni/thirdparty/hiai_ddk/${ANDROID_ABI}/libhiai_ir.so)

    add_library(hiai_ir_build
            SHARED
            IMPORTED)
    set_target_properties(hiai_ir_build
            PROPERTIES
            IMPORTED_LOCATION
            ${CMAKE_SOURCE_DIR}/src/main/jni/thirdparty/hiai_ddk/${ANDROID_ABI}/libhiai_ir_build.so)

endif()


find_library( # Sets the name of the path variable.
              log-lib

              # Specifies the name of the NDK library that
              # you want CMake to locate.
              log)

# Specifies libraries CMake should link to your target library. You
# can link multiple libraries, such as libraries you define in the
# build script, prebuilt third-party libraries, or system libraries.

if(SHARING_MEM_WITH_OPENGL)
    add_definitions(-DSHARING_MEM_WITH_OPENGL)
    add_definitions(-DEGL_EGLEXT_PROTOTYPES)
    add_definitions(-DGL_GLEXT_PROTOTYPES)
    target_link_libraries( # Specifies the target library.
                            tnn_wrapper
                            -ljnigraphics
                            -Wl,--whole-archive TNN -Wl,--no-whole-archive
                            -lEGL
                            -lGLESv2
                            -lGLESv3
                            # Links the target library to the log library
                            # included in the NDK.
                            ${log-lib})
else()
    target_link_libraries( # Specifies the target library.
                            tnn_wrapper
                            -ljnigraphics
                            -Wl,--whole-archive TNN -Wl,--no-whole-archive
                            # Links the target library to the log library
                            # included in the NDK.
                            ${log-lib})
endif()

if(TNN_OPENCV_ENABLE)
    target_link_libraries(tnn_wrapper opencv)
endif()

if(TNN_HUAWEI_NPU_ENABLE)
    target_link_libraries( # Specifies the target library.
            tnn_wrapper hiai hiai_ir hiai_ir_build)
endif()
