if(PROCESSOR.arm)
    add_definitions(-mfloat-abi=softfp -mfpu=neon)
endif()
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pie -fPIE -s")

if(ANDROID_STL STREQUAL "gnustl_static")
    message(STATUS "\tHALF_ENABLE_CPP11_CMATH:\t0")
    add_definitions(-DHALF_ENABLE_CPP11_CMATH=0)
endif()

set(COMMON_SRC ${CMAKE_CURRENT_SOURCE_DIR}/platforms/android/src/test_common.cc)

if(${ANDROID_NDK_MAJOR})
    if(${ANDROID_NDK_MAJOR} GREATER 20)
        target_link_libraries(TNN log -fopenmp -static-openmp)
    else()
        target_link_libraries(TNN log -fopenmp)
    endif()
else()
    target_link_libraries(TNN log -fopenmp)
endif()

if(TNN_OPENCL_ENABLE)
    message(STATUS "Build TNN OpenCL")

    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/opencl/include)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/source/device/opencl)

elseif(TNN_DSP_ENABLE)
    message(STATUS "Build TNN DSP")

    target_link_libraries(TNN SNPE EGL GLESv2 symphony-cpu c++_shared)

    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/source/device/dsp)

    if(ANDROID_TEST_ENABLE)
        add_executable(AndroidTest ${COMMON_SRC} ${CMAKE_CURRENT_SOURCE_DIR}/platforms/android/src/test_dsp_android.cc)
        target_link_libraries(AndroidTest TNN)
    endif()

elseif(TNN_NPU_ENABLE)
    message(STATUS "Build TNN NPU")

    target_link_libraries(TNN hiai)

    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/source/device/npu)

    if(ANDROID_TEST_ENABLE)
        add_executable(AndroidTest ${COMMON_SRC} ${CMAKE_CURRENT_SOURCE_DIR}/platforms/android/src/test_npu_android.cc)
        target_link_libraries(AndroidTest TNN)
    endif()

endif()
