/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
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
//! The cub::BlockShuffle class provides :ref:`collective <collective-primitives>` methods for shuffling data
//! partitioned across a CUDA thread block.

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/util_ptx.cuh>
#include <cub/util_type.cuh>

CUB_NAMESPACE_BEGIN

//! @rst
//! The BlockShuffle class provides :ref:`collective <collective-primitives>`
//! methods for shuffling data partitioned across a CUDA thread block.
//!
//! Overview
//! ++++++++++++++++
//!
//! It is commonplace for blocks of threads to rearrange data items between threads.
//! The BlockShuffle abstraction allows threads to efficiently shift items either
//! (a) up to their successor or
//! (b) down to their predecessor
//!
//! @endrst
//!
//! @tparam T
//!   The data type to be exchanged.
//!
//! @tparam BLOCK_DIM_X
//!   The thread block length in threads along the X dimension
//!
//! @tparam BLOCK_DIM_Y
//!   **[optional]** The thread block length in threads along the Y dimension (default: 1)
//!
//! @tparam BLOCK_DIM_Z
//!   **[optional]** The thread block length in threads along the Z dimension (default: 1)
//!
//! @tparam LEGACY_PTX_ARCH
//!   **[optional]** Unused
template <typename T, int BLOCK_DIM_X, int BLOCK_DIM_Y = 1, int BLOCK_DIM_Z = 1, int LEGACY_PTX_ARCH = 0>
class BlockShuffle
{
private:
  enum
  {
    BLOCK_THREADS = BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z,

    LOG_WARP_THREADS = CUB_LOG_WARP_THREADS(0),
    WARP_THREADS     = 1 << LOG_WARP_THREADS,
    WARPS            = (BLOCK_THREADS + WARP_THREADS - 1) / WARP_THREADS,
  };

  /// Shared memory storage layout type (last element from each thread's input)
  using _TempStorage = T[BLOCK_THREADS];

public:
  /// \smemstorage{BlockShuffle}
  struct TempStorage : Uninitialized<_TempStorage>
  {};

private:
  /// Shared storage reference
  _TempStorage& temp_storage;

  /// Linear thread-id
  unsigned int linear_tid;

  /// Internal storage allocator
  _CCCL_DEVICE _CCCL_FORCEINLINE _TempStorage& PrivateStorage()
  {
    __shared__ _TempStorage private_storage;
    return private_storage;
  }

public:
  //! @name Collective constructors
  //! @{

  //! @brief Collective constructor using a private static allocation of shared memory as temporary storage.
  _CCCL_DEVICE _CCCL_FORCEINLINE BlockShuffle()
      : temp_storage(PrivateStorage())
      , linear_tid(RowMajorTid(BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z))
  {}

  /**
   * @brief Collective constructor using the specified memory allocation
   *        as temporary storage.
   *
   * @param[in] temp_storage
   *   Reference to memory allocation having layout type TempStorage
   */
  _CCCL_DEVICE _CCCL_FORCEINLINE BlockShuffle(TempStorage& temp_storage)
      : temp_storage(temp_storage.Alias())
      , linear_tid(RowMajorTid(BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z))
  {}

  //! @}  end member group
  //! @name Shuffle movement
  //! @{

  //! @rst
  //!
  //! Each *thread*\ :sub:`i` obtains the ``input`` provided by *thread*\ :sub:`i + distance`.
  //! The offset ``distance`` may be negative.
  //!
  //! - @smemreuse
  //!
  //! @endrst
  //!
  //! @param[in] input
  //!   @rst
  //!   The input item from the calling thread (*thread*\ :sub:`i`)
  //!   @endrst
  //!
  //! @param[out] output
  //!   @rst
  //!   The ``input`` item from the successor (or predecessor) thread
  //!   *thread*\ :sub:`i + distance` (may be aliased to ``input``).
  //!   This value is only updated for for *thread*\ :sub:`i` when
  //!   ``0 <= (i + distance) < BLOCK_THREADS - 1``
  //!   @endrst
  //!
  //! @param[in] distance
  //!   Offset distance (may be negative)
  _CCCL_DEVICE _CCCL_FORCEINLINE void Offset(T input, T& output, int distance = 1)
  {
    temp_storage[linear_tid] = input;

    CTA_SYNC();

    const int offset_tid = static_cast<int>(linear_tid) + distance;
    if ((offset_tid >= 0) && (offset_tid < BLOCK_THREADS))
    {
      output = temp_storage[static_cast<size_t>(offset_tid)];
    }
  }

  //! @rst
  //! Each *thread*\ :sub:`i` obtains the ``input`` provided by *thread*\ :sub:`i + distance`.
  //!
  //! - @smemreuse
  //!
  //! @endrst
  //!
  //! @param[in] input
  //!   The calling thread's input item
  //!
  //! @param[out] output
  //!   @rst
  //!   The ``input`` item from thread
  //!   *thread*\ :sub:`(i + distance>) % BLOCK_THREADS` (may be aliased to ``input``).
  //!   This value is not updated for *thread*\ :sub:`BLOCK_THREADS - 1`.
  //!   @endrst
  //!
  //! @param[in] distance
  //!   Offset distance (`0 < distance < `BLOCK_THREADS`)
  _CCCL_DEVICE _CCCL_FORCEINLINE void Rotate(T input, T& output, unsigned int distance = 1)
  {
    temp_storage[linear_tid] = input;

    CTA_SYNC();

    unsigned int offset = linear_tid + distance;
    if (offset >= BLOCK_THREADS)
    {
      offset -= BLOCK_THREADS;
    }

    output = temp_storage[offset];
  }

  //! @rst
  //! The thread block rotates its :ref:`blocked arrangement <flexible-data-arrangement>` of
  //! ``input`` items, shifting it up by one item.
  //!
  //! - @blocked
  //! - @granularity
  //! - @smemreuse
  //!
  //! @endrst
  //!
  //! @param[in] input
  //!   The calling thread's input items
  //!
  //! @param[out] prev
  //!   @rst
  //!   The corresponding predecessor items (may be aliased to ``input``).
  //!   The item ``prev[0]`` is not updated for *thread*\ :sub:`0`.
  //!   @endrst
  template <int ITEMS_PER_THREAD>
  _CCCL_DEVICE _CCCL_FORCEINLINE void Up(T (&input)[ITEMS_PER_THREAD], T (&prev)[ITEMS_PER_THREAD])
  {
    temp_storage[linear_tid] = input[ITEMS_PER_THREAD - 1];

    CTA_SYNC();

#pragma unroll
    for (int ITEM = ITEMS_PER_THREAD - 1; ITEM > 0; --ITEM)
    {
      prev[ITEM] = input[ITEM - 1];
    }

    if (linear_tid > 0)
    {
      prev[0] = temp_storage[linear_tid - 1];
    }
  }

  //! @rst
  //! The thread block rotates its :ref:`blocked arrangement <flexible-data-arrangement>`
  //! of ``input`` items, shifting it up by one item. All threads receive the ``input`` provided by
  //! *thread*\ :sub:`BLOCK_THREADS - 1`.
  //!
  //! - @blocked
  //! - @granularity
  //! - @smemreuse
  //!
  //! @endrst
  //!
  //! @param[in] input
  //!   The calling thread's input items
  //!
  //! @param[out] prev
  //!   @rst
  //!   The corresponding predecessor items (may be aliased to ``input``).
  //!   The item ``prev[0]`` is not updated for *thread*\ :sub:`0`.
  //!   @endrst
  //!
  //! @param[out] block_suffix
  //!   @rst
  //!   The item ``input[ITEMS_PER_THREAD - 1]`` from *thread*\ :sub:`BLOCK_THREADS - 1`, provided to all threads
  //!   @endrst
  template <int ITEMS_PER_THREAD>
  _CCCL_DEVICE _CCCL_FORCEINLINE void Up(T (&input)[ITEMS_PER_THREAD], T (&prev)[ITEMS_PER_THREAD], T& block_suffix)
  {
    Up(input, prev);
    block_suffix = temp_storage[BLOCK_THREADS - 1];
  }

  //! @rst
  //! The thread block rotates its :ref:`blocked arrangement <flexible-data-arrangement>`
  //! of ``input`` items, shifting it down by one item.
  //!
  //! - @blocked
  //! - @granularity
  //! - @smemreuse
  //!
  //! @endrst
  //!
  //! @param[in] input
  //!   The calling thread's input items
  //!
  //! @param[out] prev
  //!   @rst
  //!   The corresponding predecessor items (may be aliased to ``input``).
  //!   The value ``prev[0]`` is not updated for *thread*\ :sub:`BLOCK_THREADS - 1`.
  //!   @endrst
  template <int ITEMS_PER_THREAD>
  _CCCL_DEVICE _CCCL_FORCEINLINE void Down(T (&input)[ITEMS_PER_THREAD], T (&prev)[ITEMS_PER_THREAD])
  {
    temp_storage[linear_tid] = input[0];

    CTA_SYNC();

#pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD - 1; ITEM++)
    {
      prev[ITEM] = input[ITEM + 1];
    }

    if (linear_tid < BLOCK_THREADS - 1)
    {
      prev[ITEMS_PER_THREAD - 1] = temp_storage[linear_tid + 1];
    }
  }

  //! @rst
  //! The thread block rotates its :ref:`blocked arrangement <flexible-data-arrangement>` of input items,
  //! shifting it down by one item. All threads receive ``input[0]`` provided by *thread*\ :sub:`0`.
  //!
  //! - @blocked
  //! - @granularity
  //! - @smemreuse
  //!
  //! @endrst
  //!
  //! @param[in] input
  //!   The calling thread's input items
  //!
  //! @param[out] prev
  //!   @rst
  //!   The corresponding predecessor items (may be aliased to ``input``).
  //!   The value ``prev[0]`` is not updated for *thread*\ :sub:`BLOCK_THREADS - 1`.
  //!   @endrst
  //!
  //! @param[out] block_prefix
  //!   @rst
  //!   The item ``input[0]`` from *thread*\ :sub:`0`, provided to all threads
  //!   @endrst
  template <int ITEMS_PER_THREAD>
  _CCCL_DEVICE _CCCL_FORCEINLINE void Down(T (&input)[ITEMS_PER_THREAD], T (&prev)[ITEMS_PER_THREAD], T& block_prefix)
  {
    Down(input, prev);
    block_prefix = temp_storage[0];
  }

  //! @} end member group
};

CUB_NAMESPACE_END
