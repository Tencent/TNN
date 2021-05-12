include_directories(${CMAKE_CURRENT_SOURCE_DIR}/source/tnn/device/opencl)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/opencl/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/stb)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/platforms/android/src)

set(COMMON_SRC ${CMAKE_CURRENT_SOURCE_DIR}/platforms/android/src/test_common.cc)

if (TNN_OPENVINO_ENABLE)

    if(NOT DEFINED ENV{OPENVINO_ROOT_DIR})
        message(FATAL_ERROR "not defined environment variable:OPENVINO_ROOT_DIR")
    endif()

    set(LIB_EXT ".lib")

    add_library(inference_engine STATIC IMPORTED)
    add_library(inference_engine_legacy STATIC IMPORTED)
    add_library(inference_engine_transformations STATIC IMPORTED)
    add_library(inference_engine_lp_transformations STATIC IMPORTED)
    add_library(MKLDNNPlugin STATIC IMPORTED)
    add_library(ngraph STATIC IMPORTED)
    add_library(pugixml STATIC IMPORTED)

    set_target_properties(inference_engine PROPERTIES IMPORTED_LOCATION $ENV{OPENVINO_ROOT_DIR}/deployment_tools/inference_engine/lib/intel64/Release/${LIB_PFX}inference_engine${LIB_EXT})
    set_target_properties(inference_engine_legacy PROPERTIES IMPORTED_LOCATION $ENV{OPENVINO_ROOT_DIR}/deployment_tools/inference_engine/lib/intel64/Release/${LIB_PFX}inference_engine_legacy${LIB_EXT})
    set_target_properties(inference_engine_transformations PROPERTIES IMPORTED_LOCATION $ENV{OPENVINO_ROOT_DIR}/deployment_tools/inference_engine/lib/intel64/Release/${LIB_PFX}inference_engine_transformations${LIB_EXT})
    set_target_properties(inference_engine_lp_transformations PROPERTIES IMPORTED_LOCATION $ENV{OPENVINO_ROOT_DIR}/deployment_tools/inference_engine/lib/intel64/Release/${LIB_PFX}inference_engine_lp_transformations${LIB_EXT})
    set_target_properties(MKLDNNPlugin PROPERTIES IMPORTED_LOCATION $ENV{OPENVINO_ROOT_DIR}/deployment_tools/inference_engine/lib/intel64/Release/${LIB_PFX}MKLDNNPlugin${LIB_EXT})
    set_target_properties(ngraph PROPERTIES IMPORTED_LOCATION $ENV{OPENVINO_ROOT_DIR}/deployment_tools/ngraph/lib/${LIB_PFX}ngraph${LIB_EXT})
    set_target_properties(pugixml PROPERTIES IMPORTED_LOCATION $ENV{OPENVINO_ROOT_DIR}/lib/${LIB_PFX}pugixml.lib)

    target_link_libraries(TNN inference_engine inference_engine_legacy inference_engine_transformations inference_engine_lp_transformations MKLDNNPlugin ngraph pugixml)

endif()


if(TNN_CUDA_ENABLE)
    set(LINK_TYPE "STATIC")
    set(LIB_EXT ".lib")
    enable_language(CUDA)
    include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    add_library(nvinfer ${LINK_TYPE} IMPORTED)
    add_library(nvinfer_plugin ${LINK_TYPE} IMPORTED)
    add_library(cudnn ${LINK_TYPE} IMPORTED)
    add_library(cublas ${LINK_TYPE} IMPORTED)
    set_target_properties(nvinfer PROPERTIES IMPORTED_LOCATION $ENV{TENSORRT_ROOT_DIR}\\lib\\nvinfer${LIB_EXT})
    set_target_properties(nvinfer_plugin PROPERTIES IMPORTED_LOCATION $ENV{TENSORRT_ROOT_DIR}\\lib\\nvinfer_plugin${LIB_EXT})
    set_target_properties(cudnn PROPERTIES IMPORTED_LOCATION $ENV{CUDNN_ROOT_DIR}\\lib\\x64\\cudnn${LIB_EXT})
    set_target_properties(cublas PROPERTIES IMPORTED_LOCATION $ENV{CUDA_PATH}\\lib\\x64\\cublas${LIB_EXT})
    message($ENV{TENSORRT_ROOT_DIR}\\lib\\nvinfer${LIB_EXT})
    message($ENV{TENSORRT_ROOT_DIR}\\lib\\nvinfer_plugin${LIB_EXT})
    message($ENV{CUDNN_ROOT_DIR}\\lib\\x64\\cudnn${LIB_EXT})
    target_link_libraries(TNN nvinfer nvinfer_plugin cudnn cublas)
endif()
