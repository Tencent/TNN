/******************************************************************************
 * Copyright (c) 2011-2022, NVIDIA CORPORATION.  All rights reserved.
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

/*! \file version.cuh
 *  \brief Compile-time macros encoding CUB release version
 *
 *         <cub/version.h> is the only CUB header that is guaranteed to
 *         change with every CUB release.
 *
 */

#pragma once

// For _CCCL_IMPLICIT_SYSTEM_HEADER
#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/version>

/*! \def CUB_VERSION
 *  \brief The preprocessor macro \p CUB_VERSION encodes the version
 *         number of the CUB library as MMMmmmpp.
 *
 *  \note CUB_VERSION is formatted as `MMMmmmpp`, which differs from `CCCL_VERSION` that uses `MMMmmmppp`.
 *
 *         <tt>CUB_VERSION % 100</tt> is the sub-minor version.
 *         <tt>CUB_VERSION / 100 % 1000</tt> is the minor version.
 *         <tt>CUB_VERSION / 100000</tt> is the major version.
 */
#define CUB_VERSION 200800 // macro expansion with ## requires this to be a single value

/*! \def CUB_MAJOR_VERSION
 *  \brief The preprocessor macro \p CUB_MAJOR_VERSION encodes the
 *         major version number of the CUB library.
 */
#define CUB_MAJOR_VERSION (CUB_VERSION / 100000)

/*! \def CUB_MINOR_VERSION
 *  \brief The preprocessor macro \p CUB_MINOR_VERSION encodes the
 *         minor version number of the CUB library.
 */
#define CUB_MINOR_VERSION (CUB_VERSION / 100 % 1000)

/*! \def CUB_SUBMINOR_VERSION
 *  \brief The preprocessor macro \p CUB_SUBMINOR_VERSION encodes the
 *         sub-minor version number of the CUB library.
 */
#define CUB_SUBMINOR_VERSION (CUB_VERSION % 100)

/*! \def CUB_PATCH_NUMBER
 *  \brief The preprocessor macro \p CUB_PATCH_NUMBER encodes the
 *         patch number of the CUB library.
 */
#define CUB_PATCH_NUMBER 0

static_assert(CUB_MAJOR_VERSION == CCCL_MAJOR_VERSION, "");
static_assert(CUB_MINOR_VERSION == CCCL_MINOR_VERSION, "");
static_assert(CUB_SUBMINOR_VERSION == CCCL_PATCH_VERSION, "");
