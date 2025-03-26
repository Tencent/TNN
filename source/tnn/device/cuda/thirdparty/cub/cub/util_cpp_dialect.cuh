/******************************************************************************
 * Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
 *AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

//! @file
//! Detect the version of the C++ standard used by the compiler.

#pragma once

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/util_compiler.cuh> // IWYU pragma: export

#ifndef DOXYGEN_SHOULD_SKIP_THIS // Do not document

// Deprecation warnings may be silenced by defining the following macros. These
// may be combined.
// - CUB_IGNORE_DEPRECATED_CPP_DIALECT:
//   Ignore all deprecated C++ dialects and outdated compilers.
// - CUB_IGNORE_DEPRECATED_CPP_11:
//   Ignore deprecation warnings when compiling with C++11. C++03 and outdated
//   compilers will still issue warnings.
// - CUB_IGNORE_DEPRECATED_CPP_14:
//   Ignore deprecation warnings when compiling with C++14. C++03 and outdated
//   compilers will still issue warnings.
// - CUB_IGNORE_DEPRECATED_COMPILER
//   Ignore deprecation warnings when using deprecated compilers. Compiling
//   with C++03, C++11 and C++14 will still issue warnings.

// Check for the thrust opt-outs as well:
#  if !defined(CUB_IGNORE_DEPRECATED_CPP_DIALECT) && defined(THRUST_IGNORE_DEPRECATED_CPP_DIALECT)
#    define CUB_IGNORE_DEPRECATED_CPP_DIALECT
#  endif
#  if !defined(CUB_IGNORE_DEPRECATED_CPP_11) && defined(THRUST_IGNORE_DEPRECATED_CPP_11)
#    define CUB_IGNORE_DEPRECATED_CPP_11
#  endif
#  if !defined(CUB_IGNORE_DEPRECATED_COMPILER) && defined(THRUST_IGNORE_DEPRECATED_COMPILER)
#    define CUB_IGNORE_DEPRECATED_COMPILER
#  endif

#  ifdef CUB_IGNORE_DEPRECATED_CPP_DIALECT
#    define CUB_IGNORE_DEPRECATED_CPP_11
#    define CUB_IGNORE_DEPRECATED_CPP_14
#    ifndef CUB_IGNORE_DEPRECATED_COMPILER
#      define CUB_IGNORE_DEPRECATED_COMPILER
#    endif
#  endif

#  define CUB_CPP_DIALECT _CCCL_STD_VER

// Define CUB_COMPILER_DEPRECATION macro:
#  if defined(_CCCL_COMPILER_MSVC)
#    define CUB_COMP_DEPR_IMPL(msg) _CCCL_PRAGMA(message(__FILE__ ":" _CCCL_TO_STRING(__LINE__) ": warning: " #msg))
#  else // clang / gcc:
#    define CUB_COMP_DEPR_IMPL(msg) _CCCL_PRAGMA(GCC warning #msg)
#  endif

#  define CUB_COMPILER_DEPRECATION(REQ) \
    CUB_COMP_DEPR_IMPL(CUB requires at least REQ.Define CUB_IGNORE_DEPRECATED_COMPILER to suppress this message.)

#  define CUB_COMPILER_DEPRECATION_SOFT(REQ, CUR)                                                             \
    CUB_COMP_DEPR_IMPL(                                                                                       \
      CUB requires at least REQ.CUR is deprecated but still supported.CUR support will be removed in a future \
        release.Define CUB_IGNORE_DEPRECATED_CPP_DIALECT to suppress this message.)

#  ifndef CUB_IGNORE_DEPRECATED_COMPILER

// Compiler checks:
#    if defined(_CCCL_COMPILER_GCC) && _CCCL_GCC_VERSION < 50000
CUB_COMPILER_DEPRECATION(GCC 5.0);
#    elif defined(_CCCL_COMPILER_CLANG) && _CCCL_CLANG_VERSION < 70000
CUB_COMPILER_DEPRECATION(Clang 7.0);
#    elif defined(_CCCL_COMPILER_MSVC) && _CCCL_MSVC_VERSION < 1910
// <2017. Hard upgrade message:
CUB_COMPILER_DEPRECATION(MSVC 2019(19.20 / 16.0 / 14.20));
#    elif defined(_CCCL_COMPILER_MSVC) && _CCCL_MSVC_VERSION < 1920
// >=2017, <2019. Soft deprecation message:
CUB_COMPILER_DEPRECATION_SOFT(MSVC 2019(19.20 / 16.0 / 14.20), MSVC 2017);
#    endif

#  endif // CUB_IGNORE_DEPRECATED_COMPILER

#  ifndef CUB_IGNORE_DEPRECATED_DIALECT

// Dialect checks:
#    if _CCCL_STD_VER < 2011
// <C++11. Hard upgrade message:
CUB_COMPILER_DEPRECATION(C++ 17);
#    elif _CCCL_STD_VER == 2011 && !defined(CUB_IGNORE_DEPRECATED_CPP_11)
// =C++11. Soft upgrade message:
CUB_COMPILER_DEPRECATION_SOFT(C++ 17, C++ 11);
#    elif _CCCL_STD_VER == 2014 && !defined(CUB_IGNORE_DEPRECATED_CPP_14)
// =C++14. Soft upgrade message:
CUB_COMPILER_DEPRECATION_SOFT(C++ 17, C++ 14);
#    endif

#  endif // CUB_IGNORE_DEPRECATED_DIALECT

#  undef CUB_COMPILER_DEPRECATION_SOFT
#  undef CUB_COMPILER_DEPRECATION
#  undef CUB_COMP_DEPR_IMPL
#  undef CUB_COMP_DEPR_IMPL0
#  undef CUB_COMP_DEPR_IMPL1

#endif // !DOXYGEN_SHOULD_SKIP_THIS
