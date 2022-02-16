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

    add_library(inference_engine ${LINK_TYPE} IMPORTED)
    add_library(inference_engine_legacy ${LINK_TYPE} IMPORTED)
    add_library(inference_engine_transformations ${LINK_TYPE} IMPORTED)
    add_library(inference_engine_lp_transformations ${LINK_TYPE} IMPORTED)
    # add_library(MKLDNNPlugin SHARED IMPORTED)
    add_library(ngraph ${LINK_TYPE} IMPORTED)
    # add_library(pugixml STATIC IMPORTED)
    add_library(tbb ${LINK_TYPE} IMPORTED)

    set_target_properties(inference_engine PROPERTIES IMPORTED_LOCATION $ENV{OPENVINO_ROOT_DIR}/deployment_tools/inference_engine/lib/intel64/${LIB_PFX}inference_engine${LIB_EXT})
    set_target_properties(inference_engine_legacy PROPERTIES IMPORTED_LOCATION $ENV{OPENVINO_ROOT_DIR}/deployment_tools/inference_engine/lib/intel64/${LIB_PFX}inference_engine_legacy${LIB_EXT})
    set_target_properties(inference_engine_transformations PROPERTIES IMPORTED_LOCATION $ENV{OPENVINO_ROOT_DIR}/deployment_tools/inference_engine/lib/intel64/${LIB_PFX}inference_engine_transformations${LIB_EXT})
    set_target_properties(inference_engine_lp_transformations PROPERTIES IMPORTED_LOCATION $ENV{OPENVINO_ROOT_DIR}/deployment_tools/inference_engine/lib/intel64/${LIB_PFX}inference_engine_lp_transformations${LIB_EXT})
    # set_target_properties(MKLDNNPlugin PROPERTIES IMPORTED_LOCATION $ENV{OPENVINO_ROOT_DIR}/deployment_tools/inference_engine/lib/intel64/${LIB_PFX}MKLDNNPlugin.dylib)
    set_target_properties(ngraph PROPERTIES IMPORTED_LOCATION $ENV{OPENVINO_ROOT_DIR}/deployment_tools/ngraph/lib/${LIB_PFX}ngraph${LIB_EXT})
    # set_target_properties(pugixml PROPERTIES IMPORTED_LOCATION $ENV{OPENVINO_ROOT_DIR}/lib/${LIB_PFX}pugixml.a)
    set_target_properties(tbb PROPERTIES IMPORTED_LOCATION $ENV{OPENVINO_ROOT_DIR}/deployment_tools/inference_engine/external/tbb/lib/libtbb.so.2)

    target_link_libraries(TNN inference_engine inference_engine_legacy inference_engine_transformations inference_engine_lp_transformations ngraph)
endif()
