file(GLOB_RECURSE X86_SRC *.h *.cc)

add_library(TNNX86 OBJECT ${X86_SRC})

if (MSVC)
    add_compile_options(/arch:AVX2)
else()
    add_compile_options(-mavx2 -mavx -mfma)
endif()
