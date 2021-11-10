file(GLOB_RECURSE METAL_SRCS *.h *.cc *.m *.mm)
file(GLOB_RECURSE METAL_SHADER_SRCS *.metal)

add_library(TNNMetal OBJECT ${METAL_SRCS})

if(TNN_METAL_ENABLE AND SYSTEM.Darwin)
    if(TNN_METAL_FLOAT32)
        add_custom_command(OUTPUT TNN_METAL_LIBRARY
                           COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/build_tnn_metallib.sh ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_BINARY_DIR}/tnn.metallib ${CMAKE_CURRENT_SOURCE_DIR}/../../.. 1
                           COMMENT "Compile metal shaderes: fp32")
    else()
        add_custom_command(OUTPUT TNN_METAL_LIBRARY
                           COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/build_tnn_metallib.sh ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_BINARY_DIR}/tnn.metallib ${CMAKE_CURRENT_SOURCE_DIR}/../../.. 0
                           COMMENT "Compile metal shaderes: fp16")
    endif()

    add_custom_target(tnn.metallib ALL DEPENDS TNN_METAL_LIBRARY)
endif()
