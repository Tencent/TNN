file(GLOB_RECURSE X86_SRC *.h *.cc)

include_directories(${CMAKE_SOURCE_DIR}/third_party/xbyak)

add_library(TNNX86 OBJECT ${X86_SRC})

if (MSVC)
    add_compile_options(/arch:AVX2)
else()
    add_definitions(-mavx2 -mavx -mfma)
endif()
