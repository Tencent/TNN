file(GLOB ARM82_SRC *.h *.cc)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
    file(GLOB_RECURSE ARM82_SRC_ASM asm_64/*.S)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm")
    file(GLOB_RECURSE ARM82_SRC_ASM asm_32/*.S)
endif()

add_library(
    TNNArm82
    OBJECT
    ${ARM82_SRC}
    ${ARM82_SRC_ASM}
    )

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
    target_compile_options(TNNArm82 PRIVATE -march=armv8.2-a+fp16)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm")
    if(CMAKE_SYSTEM_NAME MATCHES "Android")    
        target_compile_options(TNNArm82 PRIVATE -march=armv8.2-a+fp16 -mfpu=neon-fp-armv8 -mfloat-abi=softfp)
    else()
        message(STATUS "error")
    endif()
endif()

# target_compile_options(TNNArm82 PRIVATE -march=armv7-a+fp16 -mfpu=neon-fp-armv8 -mfloat-abi=softfp)

