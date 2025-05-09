if(TNN_OPENCL_ENABLE)
    find_package(OpenCL)
    target_link_libraries(TNN ${OpenCL_LIBRARY} ${CMAKE_THREAD_LIBS_INIT})
endif()

# Darwin third party
if(TNN_METAL_ENABLE OR TNN_APPLE_NPU_ENABLE)
    target_link_libraries(TNN "-framework Foundation")
    target_link_libraries(TNN "-framework Metal")
    target_link_libraries(TNN "-framework CoreML")
endif()


include_directories(${CMAKE_CURRENT_SOURCE_DIR}/source/tnn/device/opencl)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/opencl/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/stb)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/platforms/android/src)

set(COMMON_SRC ${CMAKE_CURRENT_SOURCE_DIR}/platforms/android/src/test_common.cc)


if(TNN_OPENVINO_ENABLE)

    if(NOT DEFINED ENV{OPENVINO_ROOT_DIR})
        message(FATAL_ERROR "not defined environment variable:OPENVINO_ROOT_DIR")
    endif()

    if (TNN_OPENVINO_BUILD_SHARED)
        set(LINK_TYPE "SHARED")
        set(LIB_EXT ".dylib")
    else()
        set(LINK_TYPE "STATIC")
        set(LIB_EXT ".a")
    endif()

    add_library(openvino ${LINK_TYPE} IMPORTED)
    #add_library(tbb ${LINK_TYPE} IMPORTED)

    set_target_properties(openvino PROPERTIES IMPORTED_LOCATION $ENV{OPENVINO_ROOT_DIR}/runtime/lib/intel64/Release/${LIB_PFX}openvino${LIB_EXT})
    #set_target_properties(tbb PROPERTIES IMPORTED_LOCATION $ENV{OPENVINO_ROOT_DIR}/runtime/3rdparty/tbb/lib/libtbb.so)

    target_link_libraries(TNN openvino)
endif()
