/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2024, NVIDIA CORPORATION.  All rights reserved.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

/******************************************************************************
 * Common C/C++ macro utilities
 ******************************************************************************/

#pragma once

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/detail/detect_cuda_runtime.cuh> // IWYU pragma: export
#include <cub/util_namespace.cuh> // IWYU pragma: export

#include <cuda/std/utility>

CUB_NAMESPACE_BEGIN

#ifndef DOXYGEN_SHOULD_SKIP_THIS // Do not document
#  define CUB_PREVENT_MACRO_SUBSTITUTION
template <typename T, typename U>
constexpr _CCCL_HOST_DEVICE auto min CUB_PREVENT_MACRO_SUBSTITUTION(T&& t, U&& u)
  -> decltype(t < u ? ::cuda::std::forward<T>(t) : ::cuda::std::forward<U>(u))
{
  return t < u ? ::cuda::std::forward<T>(t) : ::cuda::std::forward<U>(u);
}

template <typename T, typename U>
constexpr _CCCL_HOST_DEVICE auto max CUB_PREVENT_MACRO_SUBSTITUTION(T&& t, U&& u)
  -> decltype(t < u ? ::cuda::std::forward<U>(u) : ::cuda::std::forward<T>(t))
{
  return t < u ? ::cuda::std::forward<U>(u) : ::cuda::std::forward<T>(t);
}
#  undef CUB_PREVENT_MACRO_SUBSTITUTION
#endif

#ifndef CUB_MAX
/// Select maximum(a, b)
#  define CUB_MAX(a, b) (((b) > (a)) ? (b) : (a))
#endif

#ifndef CUB_MIN
/// Select minimum(a, b)
#  define CUB_MIN(a, b) (((b) < (a)) ? (b) : (a))
#endif

#ifndef CUB_QUOTIENT_FLOOR
/// Quotient of x/y rounded down to nearest integer
#  define CUB_QUOTIENT_FLOOR(x, y) ((x) / (y))
#endif

#ifndef CUB_QUOTIENT_CEILING
/// Quotient of x/y rounded up to nearest integer
// FIXME(bgruber): the following computation can overflow, use cuda::ceil_div instead
#  define CUB_QUOTIENT_CEILING(x, y) (((x) + (y) - 1) / (y))
#endif

#ifndef CUB_ROUND_UP_NEAREST
/// x rounded up to the nearest multiple of y
#  define CUB_ROUND_UP_NEAREST(x, y) (CUB_QUOTIENT_CEILING(x, y) * y)
#endif

#ifndef CUB_ROUND_DOWN_NEAREST
/// x rounded down to the nearest multiple of y
#  define CUB_ROUND_DOWN_NEAREST(x, y) (((x) / (y)) * y)
#endif

#ifndef CUB_DETAIL_KERNEL_ATTRIBUTES
#  define CUB_DETAIL_KERNEL_ATTRIBUTES CCCL_DETAIL_KERNEL_ATTRIBUTES
#endif

/**
 * @def CUB_DISABLE_KERNEL_VISIBILITY_WARNING_SUPPRESSION
 * If defined, the default suppression of kernel visibility attribute warning is disabled.
 */
#if !defined(CUB_DISABLE_KERNEL_VISIBILITY_WARNING_SUPPRESSION)
_CCCL_DIAG_SUPPRESS_GCC("-Wattributes")
_CCCL_DIAG_SUPPRESS_CLANG("-Wattributes")
#  if !defined(_CCCL_CUDA_COMPILER_NVHPC)
_CCCL_DIAG_SUPPRESS_NVHPC(attribute_requires_external_linkage)
#  endif // !_CCCL_CUDA_COMPILER_NVHPC
#  if defined(_CCCL_COMPILER_ICC)
#    pragma nv_diag_suppress 1407 // the "__visibility__" attribute can only appear on functions and
                                  // variables with external linkage'
#    pragma warning(disable : 1890) // the "__visibility__" attribute can only appear on functions and
                                    // variables with external linkage'
#  endif // _CCCL_COMPILER_ICC
#endif // !CUB_DISABLE_KERNEL_VISIBILITY_WARNING_SUPPRESSION

#ifndef CUB_DEFINE_KERNEL_GETTER
#  define CUB_DEFINE_KERNEL_GETTER(name, ...)                           \
    CUB_RUNTIME_FUNCTION static constexpr decltype(&__VA_ARGS__) name() \
    {                                                                   \
      return &__VA_ARGS__;                                              \
    }
#endif

#ifndef CUB_DEFINE_SUB_POLICY_GETTER
#  define CUB_DEFINE_SUB_POLICY_GETTER(name)                                                         \
    CUB_RUNTIME_FUNCTION static constexpr PolicyWrapper<typename StaticPolicyT::name##Policy> name() \
    {                                                                                                \
      return MakePolicyWrapper(typename StaticPolicyT::name##Policy());                              \
    }
#endif

CUB_NAMESPACE_END
