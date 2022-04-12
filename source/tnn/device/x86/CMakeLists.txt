file(GLOB_RECURSE X86_SRC *.h *.cc)
file(GLOB X86_ACC_SRC acc/* acc/compute/* acc/convolution/* acc/deconvolution/*)
file(GLOB X86_ACC_EXCLUDE acc/x86_unary2_layer_register.cc)
list(REMOVE_ITEM X86_ACC_SRC ${X86_ACC_EXCLUDE})
list(REMOVE_ITEM X86_SRC ${X86_ACC_SRC})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../../../third_party/xbyak)

add_library(TNNX86 OBJECT ${X86_SRC})
add_library(TNNX86ACC OBJECT ${X86_ACC_SRC})

if (MSVC)
    add_compile_options(/arch:AVX2)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D__SSE4_2__ -D__AVX__ -D__AVX2__ -D__FMA__")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__SSE4_2__ -D__AVX__ -D__AVX2__ -D__FMA__")
else()
    target_compile_options(TNNX86ACC PRIVATE -mavx -ffast-math)
endif()
