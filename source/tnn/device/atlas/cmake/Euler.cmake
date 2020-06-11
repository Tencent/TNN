if(NOT DEFINED ENV{DDK_HOME})
    message(FATAL_ERROR "please define environment variable:DDK_HOME")
endif()

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(tools $ENV{DDK_HOME}/toolchains/Euler_compile_env_cross/arm/cross_compile/install)

set(CMAKE_SYSROOT ${tools}/sysroot)
set(CMAKE_C_COMPILER ${tools}/bin/aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER ${tools}/bin/aarch64-linux-gnu-g++)
set(CMAKE_AR ${tools}/bin/aarch64-linux-gnu-ar)
set(CMAKE_RANLIB ${tools}/bin/aarch64-linux-gnu-ranlib)

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
