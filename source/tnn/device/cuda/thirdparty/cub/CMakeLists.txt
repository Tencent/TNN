# 3.15 is the minimum for including the project with add_subdirectory.
# 3.21 is the minimum for the developer build.
# 3.27.5 is the minimum for MSVC build with RDC=true.
cmake_minimum_required(VERSION 3.15)

# This must be done before any languages are enabled:
if (CCCL_ENABLE_CUB)
  cmake_minimum_required(VERSION 3.21)
endif()

project(CUB LANGUAGES NONE)

# This must appear before the installation rules, as it is required by the
# GNUInstallDirs CMake module.
enable_language(CXX)

# Support adding CUB to a parent project via add_subdirectory.
# See examples/cmake/add_subdir/CMakeLists.txt for details.
if (NOT CCCL_ENABLE_CUB)
  include(cmake/CubAddSubdir.cmake)
  return()
endif()

option(CUB_ENABLE_HEADER_TESTING "Test that all public headers compile." ON)
option(CUB_ENABLE_TESTING "Build CUB testing suite." ON)
option(CUB_ENABLE_BENCHMARKS "Build CUB benchmarking suite." "${CCCL_ENABLE_BENCHMARKS}")
option(CUB_ENABLE_TUNING "Build CUB tuning suite." OFF)
option(CUB_ENABLE_EXAMPLES "Build CUB examples." ON)

# This is needed for NVCXX QA, which requires a static set of executable names.
# Only a single dialect may be enabled when this is off.
option(CUB_ENABLE_CPP_DIALECT_IN_NAMES
  "Include C++ dialect information in target/object/etc names."
  ON
)
mark_as_advanced(CUB_ENABLE_CPP_DIALECT_IN_NAMES)

# This option is only used when CUB is built stand-alone; otherwise the Thrust
# option has the same effect.
option(CUB_IGNORE_DEPRECATED_API
  "Suppress warnings about deprecated Thrust/CUB API."
  OFF
)

# Check if we're actually building anything before continuing. If not, no need
# to search for deps, etc. This is a common approach for packagers that just
# need the install rules. See GH issue NVIDIA/thrust#1211.
if (NOT (CUB_ENABLE_HEADER_TESTING OR
         CUB_ENABLE_TESTING OR
         CUB_ENABLE_EXAMPLES))
  return()
endif()

include(cmake/CubBuildCompilerTargets.cmake)
include(cmake/CubBuildTargetList.cmake)
include(cmake/CubCudaConfig.cmake)
include(cmake/CubUtilities.cmake)

if ("NVHPC" STREQUAL "${CMAKE_CXX_COMPILER_ID}")
  set(CUB_ENABLE_BENCHMARKS OFF)
  set(CUB_ENABLE_TUNING OFF)
endif()

if ("" STREQUAL "${CMAKE_BUILD_TYPE}")
  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Choose the type of build." FORCE)

  set_property(
    CACHE CMAKE_BUILD_TYPE
    PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel
  )
endif ()

set(CMAKE_CXX_EXTENSIONS OFF)

cub_build_compiler_targets()
cub_build_target_list()

if (CUB_ENABLE_HEADER_TESTING)
  include(cmake/CubHeaderTesting.cmake)
endif()

# Both testing and examples use ctest
if (CUB_ENABLE_TESTING OR CUB_ENABLE_EXAMPLES)
  include(CTest)
  enable_testing()
endif()

if (CUB_ENABLE_TESTING)
  add_subdirectory(test)
endif()

if (CUB_ENABLE_EXAMPLES)
  add_subdirectory(examples)
endif()

if (CUB_ENABLE_BENCHMARKS OR CUB_ENABLE_TUNING)
  add_subdirectory(benchmarks)
endif()
