
/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
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

//! @file
//! cub::DeviceSpmv provides device-wide parallel operations for performing sparse-matrix * vector multiplication
//! (SpMV).

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/config.cuh>

#include <cub/detail/nvtx.cuh>
#include <cub/device/dispatch/dispatch_spmv_orig.cuh>
#include <cub/util_deprecated.cuh>

#include <iterator>
#include <limits>

#include <stdio.h>

CUB_NAMESPACE_BEGIN

//! @rst
//! DeviceSpmv provides device-wide parallel operations for performing
//! sparse-matrix * dense-vector multiplication (SpMV).
//!
//! Overview
//! +++++++++++++++++++++++++++++++++++++++++++++
//!
//! The `SpMV computation <http://en.wikipedia.org/wiki/Sparse_matrix-vector_multiplication>`_
//! performs the matrix-vector operation ``y = A * x + y``, where:
//!
//!  - ``A`` is an ``m * n`` sparse matrix whose non-zero structure is specified in
//!    `compressed-storage-row (CSR) format
//!    <https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format)>`_ (i.e., three
//!    arrays:
//!    ``values``, ``row_offsets``, and ``column_indices``)
//!  - ``x`` and ``y`` are dense vectors
//!
//! Usage Considerations
//! +++++++++++++++++++++++++++++++++++++++++++++
//!
//! @cdp_class{DeviceSpmv}
//!
//! @endrst
struct DeviceSpmv
{
  //! @name CSR matrix operations
  //! @{

  //! @rst
  //! This function performs the matrix-vector operation ``y = A*x``.
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates SpMV upon a 9x9 CSR matrix ``A`` representing a 3x3 lattice (24 non-zeros).
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>   // or equivalently <cub/device/device_spmv.cuh>
  //!
  //!    // Declare, allocate, and initialize device-accessible pointers for input matrix A, input
  //!    vector x,
  //!    // and output vector y
  //!    int    num_rows = 9;
  //!    int    num_cols = 9;
  //!    int    num_nonzeros = 24;
  //!
  //!    float* d_values;  // e.g., [1, 1, 1, 1, 1, 1, 1, 1,
  //!                      //        1, 1, 1, 1, 1, 1, 1, 1,
  //!                      //        1, 1, 1, 1, 1, 1, 1, 1]
  //!
  //!    int*   d_column_indices; // e.g., [1, 3, 0, 2, 4, 1, 5, 0,
  //!                             //        4, 6, 1, 3, 5, 7, 2, 4,
  //!                             //        8, 3, 7, 4, 6, 8, 5, 7]
  //!
  //!    int*   d_row_offsets;    // e.g., [0, 2, 5, 7, 10, 14, 17, 19, 22, 24]
  //!
  //!    float* d_vector_x;       // e.g., [1, 1, 1, 1, 1, 1, 1, 1, 1]
  //!    float* d_vector_y;       // e.g., [ ,  ,  ,  ,  ,  ,  ,  ,  ]
  //!    ...
  //!
  //!    // Determine temporary device storage requirements
  //!    void*    d_temp_storage = nullptr;
  //!    size_t   temp_storage_bytes = 0;
  //!    cub::DeviceSpmv::CsrMV(d_temp_storage, temp_storage_bytes, d_values,
  //!        d_row_offsets, d_column_indices, d_vector_x, d_vector_y,
  //!        num_rows, num_cols, num_nonzeros);
  //!
  //!    // Allocate temporary storage
  //!    cudaMalloc(&d_temp_storage, temp_storage_bytes);
  //!
  //!    // Run SpMV
  //!    cub::DeviceSpmv::CsrMV(d_temp_storage, temp_storage_bytes, d_values,
  //!        d_row_offsets, d_column_indices, d_vector_x, d_vector_y,
  //!        num_rows, num_cols, num_nonzeros);
  //!
  //!    // d_vector_y <-- [2, 3, 2, 3, 4, 3, 2, 3, 2]
  //!
  //! @endrst
  //!
  //! @tparam ValueT
  //!   **[inferred]** Matrix and vector value type (e.g., `float`, `double`, etc.)
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage.
  //!   When nullptr, the required allocation size is written to `temp_storage_bytes` and no work is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in] d_values
  //!   Pointer to the array of `num_nonzeros` values of the corresponding nonzero elements
  //!   of matrix `A`.
  //!
  //! @param[in] d_row_offsets
  //!   Pointer to the array of `m + 1` offsets demarcating the start of every row in
  //!   `d_column_indices` and `d_values` (with the final entry being equal to `num_nonzeros`)
  //!
  //! @param[in] d_column_indices
  //!   Pointer to the array of `num_nonzeros` column-indices of the corresponding nonzero
  //!   elements of matrix `A`. (Indices are zero-valued.)
  //!
  //! @param[in] d_vector_x
  //!   Pointer to the array of `num_cols` values corresponding to the dense input vector `x`
  //!
  //! @param[out] d_vector_y
  //!   Pointer to the array of `num_rows` values corresponding to the dense output vector `y`
  //!
  //! @param[in] num_rows
  //!   number of rows of matrix `A`.
  //!
  //! @param[in] num_cols
  //!   number of columns of matrix `A`.
  //!
  //! @param[in] num_nonzeros
  //!   number of nonzero elements of matrix `A`.
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename ValueT>
  CUB_RUNTIME_FUNCTION static cudaError_t CsrMV(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    const ValueT* d_values,
    const int* d_row_offsets,
    const int* d_column_indices,
    const ValueT* d_vector_x,
    ValueT* d_vector_y,
    int num_rows,
    int num_cols,
    int num_nonzeros,
    cudaStream_t stream = 0)
  {
    CUB_DETAIL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceSpmv::CsrMV");

    SpmvParams<ValueT, int> spmv_params;
    spmv_params.d_values          = d_values;
    spmv_params.d_row_end_offsets = d_row_offsets + 1;
    spmv_params.d_column_indices  = d_column_indices;
    spmv_params.d_vector_x        = d_vector_x;
    spmv_params.d_vector_y        = d_vector_y;
    spmv_params.num_rows          = num_rows;
    spmv_params.num_cols          = num_cols;
    spmv_params.num_nonzeros      = num_nonzeros;
    spmv_params.alpha             = ValueT{1};
    spmv_params.beta              = ValueT{0};

    return DispatchSpmv<ValueT, int>::Dispatch(d_temp_storage, temp_storage_bytes, spmv_params, stream);
  }

#ifndef DOXYGEN_SHOULD_SKIP_THIS // Do not document
  template <typename ValueT>
  CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED CUB_RUNTIME_FUNCTION static cudaError_t CsrMV(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    const ValueT* d_values,
    const int* d_row_offsets,
    const int* d_column_indices,
    const ValueT* d_vector_x,
    ValueT* d_vector_y,
    int num_rows,
    int num_cols,
    int num_nonzeros,
    cudaStream_t stream,
    bool debug_synchronous)
  {
    CUB_DETAIL_RUNTIME_DEBUG_SYNC_USAGE_LOG

    return CsrMV<ValueT>(
      d_temp_storage,
      temp_storage_bytes,
      d_values,
      d_row_offsets,
      d_column_indices,
      d_vector_x,
      d_vector_y,
      num_rows,
      num_cols,
      num_nonzeros,
      stream);
  }
#endif // DOXYGEN_SHOULD_SKIP_THIS

  //! @}  end member group
};

CUB_NAMESPACE_END
