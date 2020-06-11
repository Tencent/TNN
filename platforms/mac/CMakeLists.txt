 set(CMAKE_MACOSX_RPATH ON)

if(TNN_OPENCL_ENABLE)
    find_package(OpenCL)
    target_link_libraries(TNN ${OpenCL_LIBRARY} ${CMAKE_THREAD_LIBS_INIT})
endif()

# Darwin third party
if(TNN_METAL_ENABLE)
    target_link_libraries(TNN "-framework Foundation")
    target_link_libraries(TNN "-framework Metal")
    target_link_libraries(TNN "-framework CoreML")
endif()

if(TNN_X86_ENABLE)
    set(OPENVINO_ROOT ${CMAKE_SOURCE_DIR}/source/tnn/device/x86/thirdparty/openvino)
    set(CMAKE_SHARED_LIBRARY_RUNTIME_C_FLAG "-Wl,-rpath .")
    add_library(inference_engine SHARED IMPORTED)
    set_target_properties(inference_engine PROPERTIES IMPORTED_LOCATION ${OPENVINO_ROOT}/lib/${LIB_PFX}inference_engine${LIB_EXT})
    target_link_libraries(TNN inference_engine) 
endif()

