# Check all source files for various issues that can be detected using pattern
# matching.
#
# This is run as a ctest test named `cub.test.cmake.check_namespace`, or
# manually with:
# cmake -D "CUB_SOURCE_DIR=<CUB project root>" -P check_namespace.cmake

cmake_minimum_required(VERSION 3.15)

function(count_substrings input search_regex output_var)
  string(REGEX MATCHALL "${search_regex}" matches "${input}")
  list(LENGTH matches num_matches)
  set(${output_var} ${num_matches} PARENT_SCOPE)
endfunction()

set(found_errors 0)
file(GLOB_RECURSE cub_srcs
  RELATIVE "${CUB_SOURCE_DIR}"
  "${CUB_SOURCE_DIR}/cub/*.cuh"
  "${CUB_SOURCE_DIR}/cub/*.cu"
  "${CUB_SOURCE_DIR}/cub/*.h"
  "${CUB_SOURCE_DIR}/cub/*.cpp"
)

################################################################################
# Namespace checks.
# Check all files in thrust to make sure that they use
# CUB_NAMESPACE_BEGIN/END instead of bare `namespace cub {}` declarations.
set(namespace_exclusions
  # This defines the macros and must have bare namespace declarations:
  cub/util_namespace.cuh
)

set(bare_ns_regex "namespace[ \n\r\t]+cub[ \n\r\t]*\\{")

# Validation check for the above regex:
count_substrings([=[
namespace cub{
namespace cub {
namespace  cub  {
 namespace cub {
namespace cub
{
namespace
cub
{
]=]
  ${bare_ns_regex} valid_count)
if (NOT valid_count EQUAL 6)
  message(FATAL_ERROR "Validation of bare namespace regex failed: "
                      "Matched ${valid_count} times, expected 6.")
endif()

################################################################################
# stdpar header checks.
# Check all files in CUB to make sure that they aren't including <algorithm>
# or <memory>, both of which will cause circular dependencies in nvc++'s
# stdpar library.
#
# The headers following headers should be used instead:
# <algorithm> -> <thrust/detail/algorithm_wrapper.h>
# <memory>    -> <thrust/detail/memory_wrapper.h>
#
set(stdpar_header_exclusions
  # Placeholder -- none yet.
)

set(algorithm_regex "#[ \t]*include[ \t]+<algorithm>")
set(memory_regex    "#[ \t]*include[ \t]+<memory>")
set(numeric_regex   "#[ \t]*include[ \t]+<numeric>")

# Validation check for the above regex pattern:
count_substrings([=[
#include <algorithm>
# include <algorithm>
#include  <algorithm>
# include  <algorithm>
# include  <algorithm> // ...
]=]
  ${algorithm_regex} valid_count)
if (NOT valid_count EQUAL 5)
  message(FATAL_ERROR "Validation of stdpar header regex failed: "
    "Matched ${valid_count} times, expected 5.")
endif()

################################################################################
# Legacy macro checks.
# Check all files in CUB to make sure that they aren't using the legacy
# CUB_RUNTIME_ENABLED and __THRUST_HAS_CUDART__ macros.
#
# These macros depend on __CUDA_ARCH__ and are not compatible with NV_IF_TARGET.
# They are provided for legacy purposes and should be replaced with
# [THRUST|CUB]_RDC_ENABLED and NV_IF_TARGET in Thrust/CUB code.
#
#
set(legacy_macro_header_exclusions
  # This header defines a legacy CUDART macro:
  cub/detail/detect_cuda_runtime.cuh
)

set(cub_legacy_macro_regex "CUB_RUNTIME_ENABLED")
set(thrust_legacy_macro_regex "__THRUST_HAS_CUDART__")

################################################################################
# Read source files:
foreach(src ${cub_srcs})
  file(READ "${CUB_SOURCE_DIR}/${src}" src_contents)

  if (NOT ${src} IN_LIST namespace_exclusions)
    count_substrings("${src_contents}" "${bare_ns_regex}" bare_ns_count)
    count_substrings("${src_contents}" CUB_NS_PREFIX prefix_count)
    count_substrings("${src_contents}" CUB_NS_POSTFIX postfix_count)
    count_substrings("${src_contents}" CUB_NAMESPACE_BEGIN begin_count)
    count_substrings("${src_contents}" CUB_NAMESPACE_END end_count)

    if (NOT bare_ns_count EQUAL 0)
      message("'${src}' contains 'namespace cub {...}'. Replace with CUB_NAMESPACE macros.")
      set(found_errors 1)
    endif()

    if (NOT prefix_count EQUAL 0)
      message("'${src}' contains 'CUB_NS_PREFIX'. Replace with CUB_NAMESPACE macros.")
      set(found_errors 1)
    endif()

    if (NOT postfix_count EQUAL 0)
      message("'${src}' contains 'CUB_NS_POSTFIX'. Replace with CUB_NAMESPACE macros.")
      set(found_errors 1)
    endif()

    if (NOT begin_count EQUAL end_count)
      message("'${src}' namespace macros are unbalanced:")
      message(" - CUB_NAMESPACE_BEGIN occurs ${begin_count} times.")
      message(" - CUB_NAMESPACE_END   occurs ${end_count} times.")
      set(found_errors 1)
    endif()
  endif()

  if (NOT ${src} IN_LIST stdpar_header_exclusions)
    count_substrings("${src_contents}" "${algorithm_regex}" algorithm_count)
    count_substrings("${src_contents}" "${memory_regex}" memory_count)
    count_substrings("${src_contents}" "${numeric_regex}" numeric_count)

    if (NOT algorithm_count EQUAL 0)
      message("'${src}' includes the <algorithm> header. Replace with <thrust/detail/algorithm_wrapper.h>.")
      set(found_errors 1)
    endif()

    if (NOT memory_count EQUAL 0)
      message("'${src}' includes the <memory> header. Replace with <thrust/detail/memory_wrapper.h>.")
      set(found_errors 1)
    endif()

    if (NOT numeric_count EQUAL 0)
      message("'${src}' includes the <numeric> header. Replace with <thrust/detail/numeric_wrapper.h>.")
      set(found_errors 1)
    endif()
  endif()

  if (NOT ${src} IN_LIST legacy_macro_header_exclusions)
    count_substrings("${src_contents}" "${thrust_legacy_macro_regex}" thrust_count)
    count_substrings("${src_contents}" "${cub_legacy_macro_regex}" cub_count)

    if (NOT thrust_count EQUAL 0)
      message("'${src}' uses __THRUST_HAS_CUDART__. Replace with THRUST_RDC_ENABLED and NV_IF_TARGET.")
      set(found_errors 1)
    endif()

    if (NOT cub_count EQUAL 0)
      message("'${src}' uses CUB_RUNTIME_ENABLED. Replace with CUB_RDC_ENABLED and NV_IF_TARGET.")
      set(found_errors 1)
    endif()
  endif()
endforeach()

if (NOT found_errors EQUAL 0)
  message(FATAL_ERROR "Errors detected.")
endif()
