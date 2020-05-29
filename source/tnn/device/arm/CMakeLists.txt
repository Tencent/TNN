file(GLOB_RECURSE ARM_SRC *.h *.cc)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    file(GLOB_RECURSE ARM_SRC_ASM acc/compute/arm64/*.S)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm")
    message("--      enable armv7 neon")
    add_definitions( -mfpu=neon )
    file(GLOB_RECURSE ARM_SRC_ASM acc/compute/arm32/*.S)
endif()

add_library(TNNArm OBJECT ${ARM_SRC} ${ARM_SRC_ASM})

#if(SYSTEM.Android)
#    target_link_libraries(TNNArm log)
#endif()
