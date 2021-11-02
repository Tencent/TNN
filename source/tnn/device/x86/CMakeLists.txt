file(GLOB_RECURSE X86_SRC *.h *.cc)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../../../third_party/xbyak)

add_library(TNNX86 OBJECT ${X86_SRC})

if (MSVC)
    add_compile_options(/arch:AVX2)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D__SSE4_2__ -D__AVX__ -D__AVX2__ -D__FMA__")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__SSE4_2__ -D__AVX__ -D__AVX2__ -D__FMA__")
else()
    add_definitions(-mavx2 -mavx -mfma -ffast-math)
endif()
