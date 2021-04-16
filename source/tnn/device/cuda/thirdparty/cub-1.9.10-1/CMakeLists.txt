cmake_minimum_required(VERSION 3.8)

project(CUB CUDA CXX)

set(CUB_SOURCE ${CMAKE_SOURCE_DIR})
# include(cmake/common_variables.cmake)

if ("" STREQUAL "${CMAKE_BUILD_TYPE}")
  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Choose the type of build." FORCE)

  set_property(
    CACHE CMAKE_BUILD_TYPE
    PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel
  )
endif ()

if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.12)
  set(CMAKE_CONFIGURE_DEPENDS CONFIGURE_DEPENDS)
endif ()

list(INSERT CMAKE_MODULE_PATH 0 ${PROJECT_SOURCE_DIR}/cmake)
include(AppendOptionIfAvailable)

# Please note this also sets the default for the CUDA C++ version; see the comment below.
set(CMAKE_CXX_STANDARD 11 CACHE STRING "The C++ version to be used.")
set(CMAKE_CXX_EXTENSIONS OFF)

message("-- C++ Standard version: ${CMAKE_CXX_STANDARD}")

if (NOT "${CMAKE_CUDA_HOST_COMPILER}" STREQUAL "")
  unset(CMAKE_CUDA_HOST_COMPILER CACHE)
  message(FATAL_ERROR "CUB tests and examples require the C++ compiler"
      " and the CUDA host compiler to be the same; to set this compiler, please"
      " use the CMAKE_CXX_COMPILER variable, not the CMAKE_CUDA_HOST_COMPILER"
      " variable.")
endif ()
set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})

enable_language(CUDA)

# Force CUDA C++ standard to be the same as the C++ standard used.
#
# Now, CMake is unaligned with reality on standard versions: https://gitlab.kitware.com/cmake/cmake/issues/18597
# which means that using standard CMake methods, it's impossible to actually sync the CXX and CUDA versions for pre-11
# versions of C++; CUDA accepts 98 but translates that to 03, while CXX doesn't accept 03 (and doesn't translate that to 03).
# In case this gives You, dear user, any trouble, please escalate the above CMake bug, so we can support reality properly.
if (DEFINED CMAKE_CUDA_STANDARD)
    message(WARNING "You've set CMAKE_CUDA_STANDARD; please note that this variable is ignored, and CMAKE_CXX_STANDARD"
        " is used as the C++ standard version for both C++ and CUDA.")
endif()
unset(CMAKE_CUDA_STANDARD CACHE)
set(CMAKE_CUDA_STANDARD ${CMAKE_CXX_STANDARD})

set(CUB_HIGHEST_COMPUTE_ARCH 75)
set(CUB_KNOWN_COMPUTE_ARCHS 30 32 35 50 52 53 60 61 62 70 72 75)

option(CUB_DISABLE_ARCH_BY_DEFAULT "If ON, then all CUDA architectures are disabled on the initial CMake run." OFF)
set(OPTION_INIT ON)
if (CUB_DISABLE_ARCH_BY_DEFAULT)
  set(OPTION_INIT OFF)
endif ()

if (NOT ${CUB_HIGHEST_COMPUTE_ARCH} IN_LIST CUB_KNOWN_COMPUTE_ARCHS)
  message(FATAL_ERROR "When changing the highest compute version, don't forget to add it to the list!")
endif ()

foreach (COMPUTE_ARCH IN LISTS CUB_KNOWN_COMPUTE_ARCHS)
  option(CUB_ENABLE_COMPUTE_${COMPUTE_ARCH} "Enable code generation for tests for sm_${COMPUTE_ARCH}" ${OPTION_INIT})
  if (CUB_ENABLE_COMPUTE_${COMPUTE_ARCH})
    list(APPEND CUB_ENABLED_ARCH ${COMPUTE_ARCH})
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_${COMPUTE_ARCH},code=sm_${COMPUTE_ARCH}")
    set(COMPUTE_MESSAGE "${COMPUTE_MESSAGE} sm_${COMPUTE_ARCH}")
  endif ()
endforeach ()

option(CUB_ENABLE_COMPUTE_FUTURE "Enable code generation for tests for compute_${CUB_HIGHEST_COMPUTE_ARCH}" ${OPTION_INIT})
if (CUB_ENABLE_COMPUTE_FUTURE)
  list(APPEND CUB_ENABLED_ARCH ${CUB_HIGHEST_COMPUTE_ARCH})
  set(CMAKE_CUDA_FLAGS
    "${CMAKE_CUDA_FLAGS} -gencode arch=compute_${CUB_HIGHEST_COMPUTE_ARCH},code=compute_${CUB_HIGHEST_COMPUTE_ARCH}")
  set(COMPUTE_MESSAGE "${COMPUTE_MESSAGE} compute_${CUB_HIGHEST_COMPUTE_ARCH}")
endif ()

message("-- Enabled CUDA architectures:${COMPUTE_MESSAGE}")

# Create a variable containing the minimal target arch for tests
list(REMOVE_DUPLICATES CUB_ENABLED_ARCH)
list(SORT CUB_ENABLED_ARCH)
list(GET CUB_ENABLED_ARCH 0 CUB_MINIMAL_ENABLED_ARCH)


if ("MSVC" STREQUAL "${CMAKE_CXX_COMPILER_ID}")
  if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 19.00)
    message(FATAL_ERROR "This version of MSVC no longer supported.")
  endif ()
endif ()

if ("GNU" STREQUAL "${CMAKE_CXX_COMPILER_ID}")
  if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.4)
    message(FATAL_ERROR "This version of GCC no longer supported.")
  endif ()
endif ()

if ("MSVC" STREQUAL "${CMAKE_CXX_COMPILER_ID}")
  # TODO Enable /Wall
  append_option_if_available("/WX" CUB_CXX_WARNINGS)

  # Disabled loss-of-data conversion warnings.
  # TODO Re-enable.
  append_option_if_available("/wd4244" CUB_CXX_WARNINGS)
  append_option_if_available("/wd4267" CUB_CXX_WARNINGS)

  # Suppress numeric conversion-to-bool warnings.
  # TODO Re-enable.
  append_option_if_available("/wd4800" CUB_CXX_WARNINGS)

  # Disable warning about applying unary operator- to unsigned type.
  append_option_if_available("/wd4146" CUB_CXX_WARNINGS)

  set(CUB_TREAT_FILE_AS_CXX "/TP")
else ()
  append_option_if_available("-Werror" CUB_CXX_WARNINGS)
  append_option_if_available("-Wall" CUB_CXX_WARNINGS)
  append_option_if_available("-Wextra" CUB_CXX_WARNINGS)
  append_option_if_available("-Winit-self" CUB_CXX_WARNINGS)
  append_option_if_available("-Woverloaded-virtual" CUB_CXX_WARNINGS)
  append_option_if_available("-Wcast-qual" CUB_CXX_WARNINGS)
  append_option_if_available("-Wno-cast-align" CUB_CXX_WARNINGS)
  append_option_if_available("-Wno-long-long" CUB_CXX_WARNINGS)
  append_option_if_available("-Wno-variadic-macros" CUB_CXX_WARNINGS)
  append_option_if_available("-Wno-unused-function" CUB_CXX_WARNINGS)
  append_option_if_available("-Wno-unused-variable" CUB_CXX_WARNINGS)

  set(CUB_TREAT_FILE_AS_CXX "-x c++")
endif ()

if ("GNU" STREQUAL "${CMAKE_CXX_COMPILER_ID}")
  if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.5)
    # In GCC 4.4, the CUDA backend's kernel launch templates cause
    # impossible-to-decipher "'<anonymous>' is used uninitialized in this
    # function" warnings, so we disable uninitialized variable warnings.
    append_option_if_available("-Wno-uninitialized" CUB_CXX_WARNINGS)
  endif ()

  if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 4.5)
    # This isn't available until GCC 4.3, and misfires on TMP code until
    # GCC 4.5.
    append_option_if_available("-Wlogical-op" CUB_CXX_WARNINGS)
  endif ()

  if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 7.3)
    # GCC 7.3 complains about name mangling changes due to `noexcept`
    # becoming part of the type system; we don't care.
    append_option_if_available("-Wno-noexcept-type" CUB_CXX_WARNINGS)
  endif ()

  if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 8.1 AND CMAKE_CXX_STANDARD EQUAL 98)
    # thrust::complex can't really be made trivially copyable in pre-11.
    # Disable a warning about a non-trivially-copyable type being memmoved that was added to GCC 8.
    append_option_if_available("-Wno-class-memaccess" CUB_CXX_WARNINGS)
  endif ()
endif ()

if (("Clang" STREQUAL "${CMAKE_CXX_COMPILER_ID}") OR
    ("XL" STREQUAL "${CMAKE_CXX_COMPILER_ID}"))
  # xlC and Clang warn about unused parameters in uninstantiated templates.
  # This causes xlC to choke on the OMP backend, which is mostly #ifdef'd out
  # (and thus has unused parameters) when you aren't using it.
  append_option_if_available("-Wno-unused-parameters" CUB_CXX_WARNINGS)
endif ()

if ("Clang" STREQUAL "${CMAKE_CXX_COMPILER_ID}")
  # -Wunneeded-internal-declaration misfires in the unit test framework
  # on older versions of Clang.
  append_option_if_available("-Wno-unneeded-internal-declaration" CUB_CXX_WARNINGS)
endif ()

foreach (CXX_OPTION IN LISTS CUB_CXX_WARNINGS)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CXX_OPTION}")
endforeach ()

foreach (CXX_OPTION IN LISTS CUB_CXX_WARNINGS)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=${CXX_OPTION}")
endforeach ()

# For every public header, build a translation unit containing `#include <header>`
# to let the compiler try to figure out warnings in that header if it is not otherwise
# included in tests, and also to verify if the headers are modular enough.
# .inl files are not globbed for, because they are not supposed to be used as public
# entrypoints.
list(APPEND CUB_HEADER_GLOBS cub/*.cuh)

# Get all .cuh files...
file(
  GLOB_RECURSE CUB_HEADERS
  RELATIVE ${PROJECT_SOURCE_DIR}/cub
  ${CMAKE_CONFIGURE_DEPENDS}
  ${CUB_HEADER_GLOBS}
)

foreach (CUB_HEADER IN LISTS CUB_HEADERS)

  set(CUB_HEADER_TEST_EXT .cu)

  set(SOURCE_NAME headers/${CUB_HEADER}${CUB_HEADER_TEST_EXT})
  configure_file(cmake/header_test.in ${SOURCE_NAME})

  list(APPEND CUB_HEADER_TEST_SOURCES ${SOURCE_NAME})
endforeach ()

add_library(header-test OBJECT ${CUB_HEADER_TEST_SOURCES})
target_include_directories(
  header-test
  PUBLIC ${PROJECT_SOURCE_DIR}
)

#Create Header-Only Library/Target

add_library(CUB INTERFACE)
target_include_directories(CUB INTERFACE ${CMAKE_SOURCE_DIR})



include(CTest)
enable_testing()

# Handle tests.

math(EXPR CUB_TEST_ARCH ${CUB_MINIMAL_ENABLED_ARCH}*10)
message("-- CUB Test architecture (TEST_ARCH): ${CUB_TEST_ARCH}")

set(CUB_TEST_RUN_ARGUMENTS
  -DCUB_SOURCE=${CMAKE_SOURCE_DIR}
  -P "${CMAKE_SOURCE_DIR}/cmake/run_test.cmake")

list(APPEND CUB_TEST_GLOBS test/test_*.cu)

file(
  GLOB CUB_TESTS
  RELATIVE ${PROJECT_SOURCE_DIR}/test
  ${CMAKE_CONFIGURE_DEPENDS}
  ${CUB_TEST_GLOBS}
)

foreach (CUB_TEST_SOURCE IN LISTS CUB_TESTS)
  # TODO: Per-test flags.

  set(CUB_TEST_ADD_TO_CTEST ON)

  get_filename_component(CUB_TEST_CATEGORY ${CUB_TEST_SOURCE} DIRECTORY)
  if (NOT ("" STREQUAL "${CUB_TEST_CATEGORY}"))
    set(CUB_TEST_CATEGORY "${CUB_TEST_CATEGORY}.")
  endif ()

  get_filename_component(CUB_TEST_NAME ${CUB_TEST_SOURCE} NAME_WE)

  set(CUB_TEST "cub.test.${CUB_TEST_CATEGORY}${CUB_TEST_NAME}")

  add_executable(
    ${CUB_TEST}
    ${PROJECT_SOURCE_DIR}/test/${CUB_TEST_SOURCE}
  )

  target_compile_definitions(${CUB_TEST} PRIVATE TEST_ARCH=${CUB_TEST_ARCH})

  target_link_libraries(${CUB_TEST} CUB)

  target_include_directories(
    ${CUB_TEST} 
    PRIVATE ${PROJECT_SOURCE_DIR}/test
  )

  if (CUB_TEST_ADD_TO_CTEST)
    add_test(NAME ${CUB_TEST}
      COMMAND ${CMAKE_COMMAND}
        -DCUB_BINARY=$<TARGET_FILE:${CUB_TEST}>
        ${CUB_TEST_RUN_ARGUMENTS})
  endif ()

endforeach ()

# Handle examples.

list(APPEND CUB_EXAMPLE_GLOBS examples/example_*.cu)

if (CMAKE_VERSION VERSION_LESS 3.12)
  file(
    GLOB_RECURSE CUB_EXAMPLES
    RELATIVE ${PROJECT_SOURCE_DIR}/examples
    ${CUB_EXAMPLE_GLOBS}
    CONFIGURE_DEPENDS
  )
else ()
  file(
    GLOB_RECURSE CUB_EXAMPLES
    RELATIVE ${PROJECT_SOURCE_DIR}/examples
    ${CUB_EXAMPLE_GLOBS}
  )
endif ()

set(CUB_EXAMPLE_RUN_ARGUMENTS
  -DCUB_SOURCE=${CMAKE_SOURCE_DIR}
  -P "${CMAKE_SOURCE_DIR}/cmake/run_example.cmake")

foreach (CUB_EXAMPLE_SOURCE IN LISTS CUB_EXAMPLES)
  # TODO: Per-example flags.

  get_filename_component(CUB_EXAMPLE_CATEGORY ${CUB_EXAMPLE_SOURCE} DIRECTORY)
  if (NOT ("" STREQUAL "${CUB_EXAMPLE_CATEGORY}"))
    set(CUB_EXAMPLE_CATEGORY "${CUB_EXAMPLE_CATEGORY}.")
  endif ()

  get_filename_component(CUB_EXAMPLE_NAME ${CUB_EXAMPLE_SOURCE} NAME_WE)

  set(CUB_EXAMPLE "cub.example.${CUB_EXAMPLE_CATEGORY}${CUB_EXAMPLE_NAME}")

  add_executable(
    ${CUB_EXAMPLE}
    ${PROJECT_SOURCE_DIR}/examples/${CUB_EXAMPLE_SOURCE}
  )

  target_link_libraries(${CUB_EXAMPLE} CUB)

  target_include_directories(
    ${CUB_EXAMPLE}
    PRIVATE ${PROJECT_SOURCE_DIR}/examples
  )

  add_test(NAME ${CUB_EXAMPLE}
    COMMAND ${CMAKE_COMMAND}
      -DCUB_EXAMPLE=${CUB_EXAMPLE}
      -DCUB_BINARY=$<TARGET_FILE:${CUB_EXAMPLE}>
      ${CUB_EXAMPLE_RUN_ARGUMENTS})
endforeach ()
