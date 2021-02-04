file(GLOB_RECURSE CPU_PACK_SRC *.h *.cc)

if(TNN_X86_ENABLE)
    if (MSVC)
        add_compile_options(/arch:AVX2)
    else()
        add_compile_options(-mavx2 -mavx -mfma)
    endif()
    add_library(TNNX86 OBJECT ${X86_SRC})
endif()

if(TNN_ARM_ENABLE)
    if(TNN_ARM82_ENABLE)

    endif()
endif()
