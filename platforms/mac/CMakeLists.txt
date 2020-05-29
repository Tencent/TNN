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
