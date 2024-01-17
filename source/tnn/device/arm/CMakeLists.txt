file(GLOB_RECURSE ARM_SRC *.h *.cc)
file(GLOB_RECURSE REMOVE_SOURCES "*/compute_arm82/*")
list(REMOVE_ITEM ARM_SRC ${REMOVE_SOURCES})
if(TNN_ARM82_ENABLE)
    add_subdirectory(acc/compute_arm82)
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
    file(GLOB_RECURSE ARM_SRC_ASM acc/compute/arm64/*.S)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm")
    file(GLOB_RECURSE ARM_SRC_ASM acc/compute/arm32/*.S)
endif()

add_library(TNNArm OBJECT ${ARM_SRC} ${ARM_SRC_ASM})

#if(SYSTEM.Android)
#    target_link_libraries(TNNArm log)
#endif()
