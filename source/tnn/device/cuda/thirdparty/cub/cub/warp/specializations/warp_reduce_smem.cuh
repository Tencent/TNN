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

/**
 * @file
 * cub::WarpReduceSmem provides smem-based variants of parallel reduction of items partitioned
 * across a CUDA thread warp.
 */

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/thread/thread_load.cuh>
#include <cub/thread/thread_operators.cuh>
#include <cub/thread/thread_store.cuh>
#include <cub/util_type.cuh>

CUB_NAMESPACE_BEGIN

/**
 * @brief WarpReduceSmem provides smem-based variants of parallel reduction of items partitioned
 *        across a CUDA thread warp.
 *
 * @tparam T
 *   Data type being reduced
 *
 * @tparam LOGICAL_WARP_THREADS
 *   Number of threads per logical warp
 *
 * @tparam LEGACY_PTX_ARCH
 *   The PTX compute capability for which to to specialize this collective
 */
template <typename T, int LOGICAL_WARP_THREADS, int LEGACY_PTX_ARCH = 0>
struct WarpReduceSmem
{
  /******************************************************************************
   * Constants and type definitions
   ******************************************************************************/

  enum
  {
    /// Whether the logical warp size and the PTX warp size coincide
    IS_ARCH_WARP = (LOGICAL_WARP_THREADS == CUB_WARP_THREADS(0)),

    /// Whether the logical warp size is a power-of-two
    IS_POW_OF_TWO = PowerOfTwo<LOGICAL_WARP_THREADS>::VALUE,

    /// The number of warp reduction steps
    STEPS = Log2<LOGICAL_WARP_THREADS>::VALUE,

    /// The number of threads in half a warp
    HALF_WARP_THREADS = 1 << (STEPS - 1),

    /// The number of shared memory elements per warp
    WARP_SMEM_ELEMENTS = LOGICAL_WARP_THREADS + HALF_WARP_THREADS,

    /// FlagT status (when not using ballot)
    UNSET = 0x0, // Is initially unset
    SET   = 0x1, // Is initially set
    SEEN  = 0x2, // Has seen another head flag from a successor peer
  };

  /// Shared memory flag type
  using SmemFlag = unsigned char;

  /// Shared memory storage layout type (1.5 warps-worth of elements for each warp)
  struct _TempStorage
  {
    T reduce[WARP_SMEM_ELEMENTS];
    SmemFlag flags[WARP_SMEM_ELEMENTS];
  };

  // Alias wrapper allowing storage to be unioned
  struct TempStorage : Uninitialized<_TempStorage>
  {};

  /******************************************************************************
   * Thread fields
   ******************************************************************************/

  _TempStorage& temp_storage;
  unsigned int lane_id;
  unsigned int member_mask;

  /******************************************************************************
   * Construction
   ******************************************************************************/

  /// Constructor
  explicit _CCCL_DEVICE _CCCL_FORCEINLINE WarpReduceSmem(TempStorage& temp_storage)
      : temp_storage(temp_storage.Alias())
      , lane_id(IS_ARCH_WARP ? LaneId() : LaneId() % LOGICAL_WARP_THREADS)
      , member_mask(WarpMask<LOGICAL_WARP_THREADS>(LaneId() / LOGICAL_WARP_THREADS))
  {}

  /******************************************************************************
   * Utility methods
   ******************************************************************************/

  //---------------------------------------------------------------------
  // Regular reduction
  //---------------------------------------------------------------------

  /**
   * @brief Reduction step
   *
   * @tparam ALL_LANES_VALID
   *   Whether all lanes in each warp are contributing a valid fold of items
   *
   * @param[in] input
   *   Calling thread's input
   *
   * @param[in] valid_items
   *   Total number of valid items across the logical warp
   *
   * @param[in] reduction_op
   *   Reduction operator
   */
  template <bool ALL_LANES_VALID, typename ReductionOp, int STEP>
  _CCCL_DEVICE _CCCL_FORCEINLINE T
  ReduceStep(T input, int valid_items, ReductionOp reduction_op, Int2Type<STEP> /*step*/)
  {
    constexpr int OFFSET = 1 << STEP;

    // Share input through buffer
    ThreadStore<STORE_VOLATILE>(&temp_storage.reduce[lane_id], input);

    WARP_SYNC(member_mask);

    // Update input if peer_addend is in range
    if ((ALL_LANES_VALID && IS_POW_OF_TWO) || ((lane_id + OFFSET) < valid_items))
    {
      T peer_addend = ThreadLoad<LOAD_VOLATILE>(&temp_storage.reduce[lane_id + OFFSET]);
      input         = reduction_op(input, peer_addend);
    }

    WARP_SYNC(member_mask);

    return ReduceStep<ALL_LANES_VALID>(input, valid_items, reduction_op, Int2Type<STEP + 1>());
  }

  /**
   * @brief Reduction step (terminate)
   *
   * @tparam ALL_LANES_VALID
   *   Whether all lanes in each warp are contributing a valid fold of items
   *
   * @param[in] input
   *   Calling thread's input
   *
   * @param[in] valid_items
   *   Total number of valid items across the logical warp
   *
   * @param[in] reduction_op
   *   Reduction operator
   */
  template <bool ALL_LANES_VALID, typename ReductionOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE T
  ReduceStep(T input, int valid_items, ReductionOp /*reduction_op*/, Int2Type<STEPS> /*step*/)
  {
    return input;
  }

  //---------------------------------------------------------------------
  // Segmented reduction
  //---------------------------------------------------------------------

  /**
   * @brief Ballot-based segmented reduce
   *
   * @tparam HEAD_SEGMENTED
   *   Whether flags indicate a segment-head or a segment-tail
   *
   * @param[in] input
   *   Calling thread's input
   *
   * @param[in] flag
   *   Whether or not the current lane is a segment head/tail
   *
   * @param[in] reduction_op
   *   Reduction operator
   *
   * @param[in] has_ballot
   *   Marker type for whether the target arch has ballot functionality
   */
  template <bool HEAD_SEGMENTED, typename FlagT, typename ReductionOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE T
  SegmentedReduce(T input, FlagT flag, ReductionOp reduction_op, Int2Type<true> /*has_ballot*/)
  {
    // Get the start flags for each thread in the warp.
    int warp_flags = WARP_BALLOT(flag, member_mask);

    if (!HEAD_SEGMENTED)
    {
      warp_flags <<= 1;
    }

    // Keep bits above the current thread.
    warp_flags &= LaneMaskGt();

    // Accommodate packing of multiple logical warps in a single physical warp
    if (!IS_ARCH_WARP)
    {
      warp_flags >>= (LaneId() / LOGICAL_WARP_THREADS) * LOGICAL_WARP_THREADS;
    }

    // Find next flag
    int next_flag = __clz(__brev(warp_flags));

    // Clip the next segment at the warp boundary if necessary
    if (LOGICAL_WARP_THREADS != 32)
    {
      next_flag = CUB_MIN(next_flag, LOGICAL_WARP_THREADS);
    }

#pragma unroll
    for (int STEP = 0; STEP < STEPS; STEP++)
    {
      const int OFFSET = 1 << STEP;

      // Share input into buffer
      ThreadStore<STORE_VOLATILE>(&temp_storage.reduce[lane_id], input);

      WARP_SYNC(member_mask);

      // Update input if peer_addend is in range
      if (OFFSET + lane_id < next_flag)
      {
        T peer_addend = ThreadLoad<LOAD_VOLATILE>(&temp_storage.reduce[lane_id + OFFSET]);
        input         = reduction_op(input, peer_addend);
      }

      WARP_SYNC(member_mask);
    }

    return input;
  }

  /**
   * @brief Smem-based segmented reduce
   *
   * @tparam HEAD_SEGMENTED
   *   Whether flags indicate a segment-head or a segment-tail
   *
   * @param[in] input
   *   Calling thread's input
   *
   * @param[in] flag
   *   Whether or not the current lane is a segment head/tail
   *
   * @param[in] reduction_op
   *   Reduction operator
   *
   * @param[in] has_ballot
   *   Marker type for whether the target arch has ballot functionality
   */
  template <bool HEAD_SEGMENTED, typename FlagT, typename ReductionOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE T
  SegmentedReduce(T input, FlagT flag, ReductionOp reduction_op, Int2Type<false> /*has_ballot*/)
  {
    enum
    {
      UNSET = 0x0, // Is initially unset
      SET   = 0x1, // Is initially set
      SEEN  = 0x2, // Has seen another head flag from a successor peer
    };

    // Alias flags onto shared data storage
    volatile SmemFlag* flag_storage = temp_storage.flags;

    SmemFlag flag_status = (flag) ? SET : UNSET;

    for (int STEP = 0; STEP < STEPS; STEP++)
    {
      const int OFFSET = 1 << STEP;

      // Share input through buffer
      ThreadStore<STORE_VOLATILE>(&temp_storage.reduce[lane_id], input);

      WARP_SYNC(member_mask);

      // Get peer from buffer
      T peer_addend = ThreadLoad<LOAD_VOLATILE>(&temp_storage.reduce[lane_id + OFFSET]);

      WARP_SYNC(member_mask);

      // Share flag through buffer
      flag_storage[lane_id] = flag_status;

      // Get peer flag from buffer
      SmemFlag peer_flag_status = flag_storage[lane_id + OFFSET];

      // Update input if peer was in range
      if (lane_id < LOGICAL_WARP_THREADS - OFFSET)
      {
        if (HEAD_SEGMENTED)
        {
          // Head-segmented
          if ((flag_status & SEEN) == 0)
          {
            // Has not seen a more distant head flag
            if (peer_flag_status & SET)
            {
              // Has now seen a head flag
              flag_status |= SEEN;
            }
            else
            {
              // Peer is not a head flag: grab its count
              input = reduction_op(input, peer_addend);
            }

            // Update seen status to include that of peer
            flag_status |= (peer_flag_status & SEEN);
          }
        }
        else
        {
          // Tail-segmented.  Simply propagate flag status
          if (!flag_status)
          {
            input = reduction_op(input, peer_addend);
            flag_status |= peer_flag_status;
          }
        }
      }
    }

    return input;
  }

  /******************************************************************************
   * Interface
   ******************************************************************************/

  /**
   * @brief Reduction
   *
   * @tparam ALL_LANES_VALID
   *   Whether all lanes in each warp are contributing a valid fold of items
   *
   * @param[in] input
   *   Calling thread's input
   *
   * @param[in] valid_items
   *   Total number of valid items across the logical warp
   *
   * @param[in] reduction_op
   *   Reduction operator
   */
  template <bool ALL_LANES_VALID, typename ReductionOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE T Reduce(T input, int valid_items, ReductionOp reduction_op)
  {
    return ReduceStep<ALL_LANES_VALID>(input, valid_items, reduction_op, Int2Type<0>());
  }

  /**
   * @brief Segmented reduction
   *
   * @tparam HEAD_SEGMENTED
   *   Whether flags indicate a segment-head or a segment-tail
   *
   * @param[in] input
   *   Calling thread's input
   *
   * @param[in] flag
   *   Whether or not the current lane is a segment head/tail
   *
   * @param[in] reduction_op
   *   Reduction operator
   */
  template <bool HEAD_SEGMENTED, typename FlagT, typename ReductionOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE T SegmentedReduce(T input, FlagT flag, ReductionOp reduction_op)
  {
    return SegmentedReduce<HEAD_SEGMENTED>(input, flag, reduction_op, Int2Type<true>());
  }
};

CUB_NAMESPACE_END
