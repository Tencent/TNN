include_directories(${CMAKE_CURRENT_SOURCE_DIR}/source/tnn/device/opencl)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/opencl/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/stb)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/platforms/android/src)

set(COMMON_SRC ${CMAKE_CURRENT_SOURCE_DIR}/platforms/android/src/test_common.cc)

if(TNN_OPENCL_ENABLE)
    find_package(OpenCL)
    target_link_libraries(TNN ${OpenCL_LIBRARY} ${CMAKE_THREAD_LIBS_INIT})
endif()

if(TNN_X86_ENABLE)
    add_library(inference_engine SHARED IMPORTED)
    set_target_properties(inference_engine PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/source/tnn/device/x86/thirdparty/openvino/lib/${LIB_PFX}inference_engine${LIB_EXT})
    target_link_libraries(TNN inference_engine)
endif()

if(TNN_CUDA_ENABLE)
    set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-10.0")
    find_package(CUDA REQUIRED)
    include(FindCUDA)
    include_directories("${CUDA_INCLUDE_DIRS}")
    set(CUDNN_DIR "${CMAKE_SOURCE_DIR}/../yt-rapidnet-gpu-deps/cloud/cuda-10.0/cudnn-7.3/")
    set(CUDNN_LIBS "${CUDNN_DIR}/lib64/libcudnn.so")
    target_link_libraries(TNN ${CUDNN_LIBS} ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})
endif()

if(TNN_ATLAS_ENABLE)
    target_link_libraries(TNN ${DDK_HOST_LIBRARIES})

    if(LINUX_TEST_ENABLE)
        add_executable(AtlasTest ${COMMON_SRC} ${CMAKE_SOURCE_DIR}/platforms/linux/src/test_atlas_x86.cc)
        target_link_libraries(AtlasTest TNN)

        add_executable(AtlasDvppTest ${COMMON_SRC} ${CMAKE_SOURCE_DIR}/platforms/linux/src/test_atlas_dvpp_x86.cc)
        target_link_libraries(AtlasDvppTest TNN)
    endif()
endif()
