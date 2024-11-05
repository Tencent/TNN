/******************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#ifdef DOXYGEN_SHOULD_SKIP_THIS // Only parse this during doxygen passes:
//! When this macro is defined, no NVTX ranges are emitted by CCCL
#  define CCCL_DISABLE_NVTX
#endif // DOXYGEN_SHOULD_SKIP_THIS

// Enable the functionality of this header if:
// * The NVTX3 C API is available in CTK
// * NVTX is not explicitly disabled (via CCCL_DISABLE_NVTX or NVTX_DISABLE)
// * C++14 is availabl for cuda::std::optional
// * NVTX3 uses module as an identifier, which trips up NVHPC
#if __has_include(<nvtx3/nvToolsExt.h> ) && !defined(CCCL_DISABLE_NVTX) && !defined(NVTX_DISABLE) \
&& _CCCL_STD_VER >= 2014 &&(!defined(_CCCL_COMPILER_NVHPC) || _CCCL_STD_VER <= 2017)
// Include our NVTX3 C++ wrapper if not available from the CTK
#  if __has_include(<nvtx3/nvtx3.hpp>) // TODO(bgruber): replace by a check for the first CTK version shipping the header
#    include <nvtx3/nvtx3.hpp>
#  else // __has_include(<nvtx3/nvtx3.hpp>)
#    include "nvtx3.hpp"
#  endif // __has_include(<nvtx3/nvtx3.hpp>)

// We expect the NVTX3 V1 C++ API to be available when nvtx3.hpp is available. This should work, because newer versions
// of NVTX3 will continue to declare previous API versions. See also:
// https://github.com/NVIDIA/NVTX/blob/release-v3/c/include/nvtx3/nvtx3.hpp#L2835-L2841.
#  ifdef NVTX3_CPP_DEFINITIONS_V1_0
#    include <cuda/std/optional>

CUB_NAMESPACE_BEGIN
namespace detail
{
struct NVTXCCCLDomain
{
  static constexpr const char* name = "CCCL";
};
} // namespace detail
CUB_NAMESPACE_END

// Hook for the NestedNVTXRangeGuard from the unit tests
#    ifndef CUB_DETAIL_BEFORE_NVTX_RANGE_SCOPE
#      define CUB_DETAIL_BEFORE_NVTX_RANGE_SCOPE(name)
#    endif // !CUB_DETAIL_BEFORE_NVTX_RANGE_SCOPE

// Conditionally inserts a NVTX range starting here until the end of the current function scope in host code. Does
// nothing in device code.
// The optional is needed to defer the construction of an NVTX range (host-only code) and message string registration
// into a dispatch region running only on the host, while preserving the semantic scope where the range is declared.
#    define CUB_DETAIL_NVTX_RANGE_SCOPE_IF(condition, name)                                                             \
      CUB_DETAIL_BEFORE_NVTX_RANGE_SCOPE(name)                                                                          \
      ::cuda::std::optional<::nvtx3::v1::scoped_range_in<CUB_NS_QUALIFIER::detail::NVTXCCCLDomain>> __cub_nvtx3_range;  \
      NV_IF_TARGET(                                                                                                     \
        NV_IS_HOST,                                                                                                     \
        static const ::nvtx3::v1::registered_string_in<CUB_NS_QUALIFIER::detail::NVTXCCCLDomain> __cub_nvtx3_func_name{ \
          name};                                                                                                        \
        static const ::nvtx3::v1::event_attributes __cub_nvtx3_func_attr{__cub_nvtx3_func_name};                        \
        if (condition) __cub_nvtx3_range.emplace(__cub_nvtx3_func_attr);                                                \
        (void) __cub_nvtx3_range;)

#    define CUB_DETAIL_NVTX_RANGE_SCOPE(name) CUB_DETAIL_NVTX_RANGE_SCOPE_IF(true, name)
#  else // NVTX3_CPP_DEFINITIONS_V1_0
#    if defined(_CCCL_COMPILER_MSVC)
#      pragma message( \
        "warning: nvtx3.hpp is available but does not define the V1 API. This is odd. Please open a GitHub issue at: https://github.com/NVIDIA/cccl/issues.")
#    else
#      warning nvtx3.hpp is available but does not define the V1 API. This is odd. Please open a GitHub issue at: https://github.com/NVIDIA/cccl/issues.
#    endif
#    define CUB_DETAIL_NVTX_RANGE_SCOPE_IF(condition, name)
#    define CUB_DETAIL_NVTX_RANGE_SCOPE(name)
#  endif // NVTX3_CPP_DEFINITIONS_V1_0
#else // __has_include(<nvtx3/nvToolsExt.h> ) && !defined(CCCL_DISABLE_NVTX) && !defined(NVTX_DISABLE) && _CCCL_STD_VER
      // >= 2014
#  define CUB_DETAIL_NVTX_RANGE_SCOPE_IF(condition, name)
#  define CUB_DETAIL_NVTX_RANGE_SCOPE(name)
#endif // __has_include(<nvtx3/nvToolsExt.h> ) && !defined(CCCL_DISABLE_NVTX) && !defined(NVTX_DISABLE) && _CCCL_STD_VER
       // >= 2014
