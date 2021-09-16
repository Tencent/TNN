execute_process(COMMAND ${PYTHON_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/cl/opencl_codegen.py" "${CMAKE_CURRENT_SOURCE_DIR}/cl")

file(GLOB_RECURSE OPENCL_SRCS *.h *.cc)

add_library(TNNOpenCL OBJECT ${OPENCL_SRCS})

target_include_directories(TNNOpenCL PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../third_party/opencl/include
)


add_definitions(-DTNN_USE_OPENCL_WRAPPER)
