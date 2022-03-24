

if (MSVC)
    add_definitions(-DUNICODE -D_UNICODE)
endif()

if (MSVC)
    add_custom_command(
        OUTPUT TNN_DIRECTX_KERNELS ${CMAKE_CURRENT_SOURCE_DIR}/directx_kernels.cc
        COMMAND cmd \/C ${CMAKE_CURRENT_SOURCE_DIR}/build_hlsl.bat
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/build_hlsl.bat
        COMMENT "--------------- Compiling DirectX compute shaderes ------------------"
    )

endif()

file(GLOB_RECURSE DIRECTX_SRC *.h *.cc)
list(APPEND DIRECTX_SRC directx_kernels.cc)

add_library(TNNDIRECTX OBJECT ${DIRECTX_SRC})