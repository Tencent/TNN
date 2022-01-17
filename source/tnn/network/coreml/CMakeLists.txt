file(GLOB_RECURSE APPLE_NPU_SRCS *.h *.cc *.c *.m *.mm)

add_library(TNNAppleNPU OBJECT ${APPLE_NPU_SRCS})

target_include_directories(TNNAppleNPU PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/mlmodel/include
)
