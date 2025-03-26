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
 * cub::WarpReduceShfl provides SHFL-based variants of parallel reduction of items partitioned across a CUDA thread
 * warp.
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

#include <cub/thread/thread_operators.cuh>
#include <cub/util_ptx.cuh>
#include <cub/util_type.cuh>

#include <cuda/std/cstdint>
#include <cuda/std/type_traits>

#include <nv/target>

CUB_NAMESPACE_BEGIN

namespace detail
{

template <class A = int, class = A>
struct reduce_add_exists : ::cuda::std::false_type
{};

template <class T>
struct reduce_add_exists<T, decltype(__reduce_add_sync(0xFFFFFFFF, T{}))> : ::cuda::std::true_type
{};

template <class T = int, class = T>
struct reduce_min_exists : ::cuda::std::false_type
{};

template <class T>
struct reduce_min_exists<T, decltype(__reduce_min_sync(0xFFFFFFFF, T{}))> : ::cuda::std::true_type
{};

template <class T = int, class = T>
struct reduce_max_exists : ::cuda::std::false_type
{};

template <class T>
struct reduce_max_exists<T, decltype(__reduce_max_sync(0xFFFFFFFF, T{}))> : ::cuda::std::true_type
{};

} // namespace detail

/**
 * @brief WarpReduceShfl provides SHFL-based variants of parallel reduction of items partitioned
 *        across a CUDA thread warp.
 *
 * @tparam T
 *   Data type being reduced
 *
 * @tparam LOGICAL_WARP_THREADS
 *   Number of threads per logical warp (must be a power-of-two)
 *
 * @tparam LEGACY_PTX_ARCH
 *   The PTX compute capability for which to to specialize this collective
 */
template <typename T, int LOGICAL_WARP_THREADS, int LEGACY_PTX_ARCH = 0>
struct WarpReduceShfl
{
  static_assert(PowerOfTwo<LOGICAL_WARP_THREADS>::VALUE, "LOGICAL_WARP_THREADS must be a power of two");

  //---------------------------------------------------------------------
  // Constants and type definitions
  //---------------------------------------------------------------------

  enum
  {
    /// Whether the logical warp size and the PTX warp size coincide
    IS_ARCH_WARP = (LOGICAL_WARP_THREADS == CUB_WARP_THREADS(0)),

    /// The number of warp reduction steps
    STEPS = Log2<LOGICAL_WARP_THREADS>::VALUE,

    /// Number of logical warps in a PTX warp
    LOGICAL_WARPS = CUB_WARP_THREADS(0) / LOGICAL_WARP_THREADS,

    /// The 5-bit SHFL mask for logically splitting warps into sub-segments starts 8-bits up
    SHFL_C = (CUB_WARP_THREADS(0) - LOGICAL_WARP_THREADS) << 8

  };

  template <typename S>
  struct IsInteger
  {
    enum
    {
      /// Whether the data type is a small (32b or less) integer for which we can use a single SHFL instruction per
      /// exchange
      IS_SMALL_UNSIGNED = (Traits<S>::CATEGORY == UNSIGNED_INTEGER) && (sizeof(S) <= sizeof(unsigned int))
    };
  };

  /// Shared memory storage layout type
  using TempStorage = NullType;

  //---------------------------------------------------------------------
  // Thread fields
  //---------------------------------------------------------------------

  /// Lane index in logical warp
  int lane_id;

  /// Logical warp index in 32-thread physical warp
  int warp_id;

  /// 32-thread physical warp member mask of logical warp
  ::cuda::std::uint32_t member_mask;

  //---------------------------------------------------------------------
  // Construction
  //---------------------------------------------------------------------

  /// Constructor
  _CCCL_DEVICE _CCCL_FORCEINLINE WarpReduceShfl(TempStorage& /*temp_storage*/)
      : lane_id(static_cast<int>(LaneId()))
      , warp_id(IS_ARCH_WARP ? 0 : (lane_id / LOGICAL_WARP_THREADS))
      , member_mask(WarpMask<LOGICAL_WARP_THREADS>(warp_id))
  {
    if (!IS_ARCH_WARP)
    {
      lane_id = lane_id % LOGICAL_WARP_THREADS;
    }
  }

  //---------------------------------------------------------------------
  // Reduction steps
  //---------------------------------------------------------------------

  /**
   * @brief Reduction (specialized for summation across uint32 types)
   *
   * @param[in] input
   *   Calling thread's input item.
   *
   * @param[in] reduction_op
   *   Binary reduction operator
   *
   * @param[in] last_lane
   *   Index of last lane in segment
   *
   * @param[in] offset
   *   Up-offset to pull from
   */
  _CCCL_DEVICE _CCCL_FORCEINLINE unsigned int
  ReduceStep(unsigned int input, cub::Sum /*reduction_op*/, int last_lane, int offset)
  {
    unsigned int output;
    int shfl_c = last_lane | SHFL_C; // Shuffle control (mask and last_lane)

    // Use predicate set from SHFL to guard against invalid peers
    asm volatile(
      "{"
      "  .reg .u32 r0;"
      "  .reg .pred p;"
      "  shfl.sync.down.b32 r0|p, %1, %2, %3, %5;"
      "  @p add.u32 r0, r0, %4;"
      "  mov.u32 %0, r0;"
      "}"
      : "=r"(output)
      : "r"(input), "r"(offset), "r"(shfl_c), "r"(input), "r"(member_mask));

    return output;
  }

  /**
   * @brief Reduction (specialized for summation across fp32 types)
   *
   * @param[in] input
   *   Calling thread's input item.
   *
   * @param[in] reduction_op
   *   Binary reduction operator
   *
   * @param[in] last_lane
   *   Index of last lane in segment
   *
   * @param[in] offset
   *   Up-offset to pull from
   */
  _CCCL_DEVICE _CCCL_FORCEINLINE float ReduceStep(float input, cub::Sum /*reduction_op*/, int last_lane, int offset)
  {
    float output;
    int shfl_c = last_lane | SHFL_C; // Shuffle control (mask and last_lane)

    // Use predicate set from SHFL to guard against invalid peers
    asm volatile(
      "{"
      "  .reg .f32 r0;"
      "  .reg .pred p;"
      "  shfl.sync.down.b32 r0|p, %1, %2, %3, %5;"
      "  @p add.f32 r0, r0, %4;"
      "  mov.f32 %0, r0;"
      "}"
      : "=f"(output)
      : "f"(input), "r"(offset), "r"(shfl_c), "f"(input), "r"(member_mask));

    return output;
  }

  /**
   * @brief Reduction (specialized for summation across unsigned long long types)
   *
   * @param[in] input
   *   Calling thread's input item
   *
   * @param[in] reduction_op
   *   Binary reduction operator
   *
   * @param[in] last_lane
   *   Index of last lane in segment
   *
   * @param[in] offset
   *   Up-offset to pull from
   */
  _CCCL_DEVICE _CCCL_FORCEINLINE unsigned long long
  ReduceStep(unsigned long long input, cub::Sum /*reduction_op*/, int last_lane, int offset)
  {
    unsigned long long output;
    int shfl_c = last_lane | SHFL_C; // Shuffle control (mask and last_lane)

    asm volatile(
      "{"
      "  .reg .u32 lo;"
      "  .reg .u32 hi;"
      "  .reg .pred p;"
      "  mov.b64 {lo, hi}, %1;"
      "  shfl.sync.down.b32 lo|p, lo, %2, %3, %4;"
      "  shfl.sync.down.b32 hi|p, hi, %2, %3, %4;"
      "  mov.b64 %0, {lo, hi};"
      "  @p add.u64 %0, %0, %1;"
      "}"
      : "=l"(output)
      : "l"(input), "r"(offset), "r"(shfl_c), "r"(member_mask));

    return output;
  }

  /**
   * @brief Reduction (specialized for summation across long long types)
   *
   * @param[in] input
   *   Calling thread's input item
   *
   * @param[in] reduction_op
   *   Binary reduction operator
   *
   * @param[in] last_lane
   *   Index of last lane in segment
   *
   * @param[in] offset
   *   Up-offset to pull from
   */
  _CCCL_DEVICE _CCCL_FORCEINLINE long long
  ReduceStep(long long input, cub::Sum /*reduction_op*/, int last_lane, int offset)
  {
    long long output;
    int shfl_c = last_lane | SHFL_C; // Shuffle control (mask and last_lane)

    // Use predicate set from SHFL to guard against invalid peers
    asm volatile(
      "{"
      "  .reg .u32 lo;"
      "  .reg .u32 hi;"
      "  .reg .pred p;"
      "  mov.b64 {lo, hi}, %1;"
      "  shfl.sync.down.b32 lo|p, lo, %2, %3, %4;"
      "  shfl.sync.down.b32 hi|p, hi, %2, %3, %4;"
      "  mov.b64 %0, {lo, hi};"
      "  @p add.s64 %0, %0, %1;"
      "}"
      : "=l"(output)
      : "l"(input), "r"(offset), "r"(shfl_c), "r"(member_mask));

    return output;
  }

  /**
   * @brief Reduction (specialized for summation across double types)
   *
   * @param[in] input
   *   Calling thread's input item.
   *
   * @param[in] reduction_op
   *   Binary reduction operator
   *
   * @param[in] last_lane
   *   Index of last lane in segment
   *
   * @param[in] offset
   *   Up-offset to pull from
   */
  _CCCL_DEVICE _CCCL_FORCEINLINE double ReduceStep(double input, cub::Sum /*reduction_op*/, int last_lane, int offset)
  {
    double output;
    int shfl_c = last_lane | SHFL_C; // Shuffle control (mask and last_lane)

    // Use predicate set from SHFL to guard against invalid peers
    asm volatile(
      "{"
      "  .reg .u32 lo;"
      "  .reg .u32 hi;"
      "  .reg .pred p;"
      "  .reg .f64 r0;"
      "  mov.b64 %0, %1;"
      "  mov.b64 {lo, hi}, %1;"
      "  shfl.sync.down.b32 lo|p, lo, %2, %3, %4;"
      "  shfl.sync.down.b32 hi|p, hi, %2, %3, %4;"
      "  mov.b64 r0, {lo, hi};"
      "  @p add.f64 %0, %0, r0;"
      "}"
      : "=d"(output)
      : "d"(input), "r"(offset), "r"(shfl_c), "r"(member_mask));

    return output;
  }

  /**
   * @brief Reduction (specialized for swizzled ReduceByKeyOp<cub::Sum> across
   *        KeyValuePair<KeyT, ValueT> types)
   *
   * @param[in] input
   *   Calling thread's input item
   *
   * @param[in] reduction_op
   *   Binary reduction operator
   *
   * @param[in] last_lane
   *   Index of last lane in segment
   *
   * @param[in] offset
   *   Up-offset to pull from
   */
  template <typename ValueT, typename KeyT>
  _CCCL_DEVICE _CCCL_FORCEINLINE KeyValuePair<KeyT, ValueT> ReduceStep(
    KeyValuePair<KeyT, ValueT> input, SwizzleScanOp<ReduceByKeyOp<cub::Sum>> /*reduction_op*/, int last_lane, int offset)
  {
    KeyValuePair<KeyT, ValueT> output;

    KeyT other_key = ShuffleDown<LOGICAL_WARP_THREADS>(input.key, offset, last_lane, member_mask);

    output.key = input.key;
    output.value =
      ReduceStep(input.value, cub::Sum(), last_lane, offset, Int2Type<IsInteger<ValueT>::IS_SMALL_UNSIGNED>());

    if (input.key != other_key)
    {
      output.value = input.value;
    }

    return output;
  }

  /**
   * @brief Reduction (specialized for swizzled ReduceBySegmentOp<cub::Sum> across
   *        KeyValuePair<OffsetT, ValueT> types)
   *
   * @param[in] input
   *   Calling thread's input item.
   *
   * @param[in] reduction_op
   *   Binary reduction operator
   *
   * @param[in] last_lane
   *   Index of last lane in segment
   *
   * @param[in] offset
   *   Up-offset to pull from
   */
  template <typename ValueT, typename OffsetT>
  _CCCL_DEVICE _CCCL_FORCEINLINE KeyValuePair<OffsetT, ValueT> ReduceStep(
    KeyValuePair<OffsetT, ValueT> input,
    SwizzleScanOp<ReduceBySegmentOp<cub::Sum>> /*reduction_op*/,
    int last_lane,
    int offset)
  {
    KeyValuePair<OffsetT, ValueT> output;

    output.value =
      ReduceStep(input.value, cub::Sum(), last_lane, offset, Int2Type<IsInteger<ValueT>::IS_SMALL_UNSIGNED>());
    output.key =
      ReduceStep(input.key, cub::Sum(), last_lane, offset, Int2Type<IsInteger<OffsetT>::IS_SMALL_UNSIGNED>());

    if (input.key > 0)
    {
      output.value = input.value;
    }

    return output;
  }

  /**
   * @brief Reduction step (generic)
   *
   * @param[in] input
   *   Calling thread's input item
   *
   * @param[in] reduction_op
   *   Binary reduction operator
   *
   * @param[in] last_lane
   *   Index of last lane in segment
   *
   * @param[in] offset
   *   Up-offset to pull from
   */
  template <typename _T, typename ReductionOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE _T ReduceStep(_T input, ReductionOp reduction_op, int last_lane, int offset)
  {
    _T output = input;

    _T temp = ShuffleDown<LOGICAL_WARP_THREADS>(output, offset, last_lane, member_mask);

    // Perform reduction op if valid
    if (offset + lane_id <= last_lane)
    {
      output = reduction_op(input, temp);
    }

    return output;
  }

  /**
   * @brief Reduction step (specialized for small unsigned integers size 32b or less)
   *
   * @param[in] input
   *   Calling thread's input item.
   *
   * @param[in] reduction_op
   *   Binary reduction operator
   *
   * @param[in] last_lane
   *   Index of last lane in segment
   *
   * @param[in] offset
   *   Up-offset to pull from
   *
   * @param[in] is_small_unsigned
   *   Marker type indicating whether T is a small unsigned integer
   */
  template <typename _T, typename ReductionOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE _T
  ReduceStep(_T input, ReductionOp reduction_op, int last_lane, int offset, Int2Type<true> /*is_small_unsigned*/)
  {
    return ReduceStep(input, reduction_op, last_lane, offset);
  }

  /**
   * @brief Reduction step (specialized for types other than small unsigned integers size
   *        32b or less)
   *
   * @param[in] input
   *   Calling thread's input item.
   *
   * @param[in] reduction_op
   *   Binary reduction operator
   *
   * @param[in] last_lane
   *   Index of last lane in segment
   *
   * @param[in] offset
   *   Up-offset to pull from
   *
   * @param[in] is_small_unsigned
   *   Marker type indicating whether T is a small unsigned integer
   */
  template <typename _T, typename ReductionOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE _T
  ReduceStep(_T input, ReductionOp reduction_op, int last_lane, int offset, Int2Type<false> /*is_small_unsigned*/)
  {
    return ReduceStep(input, reduction_op, last_lane, offset);
  }

  //---------------------------------------------------------------------
  // Templated reduction iteration
  //---------------------------------------------------------------------

  /**
   * @param[in] input
   *   Calling thread's input item.
   *
   * @param[in] reduction_op
   *   Binary reduction operator
   *
   * @param[in] last_lane
   *   Index of last lane in segment
   */
  template <typename ReductionOp, int STEP>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  ReduceStep(T& input, ReductionOp reduction_op, int last_lane, Int2Type<STEP> /*step*/)
  {
    input = ReduceStep(input, reduction_op, last_lane, 1 << STEP, Int2Type<IsInteger<T>::IS_SMALL_UNSIGNED>());

    ReduceStep(input, reduction_op, last_lane, Int2Type<STEP + 1>());
  }

  /**
   * @param[in] input
   *   Calling thread's input item.
   *
   * @param[in] reduction_op
   *   Binary reduction operator
   *
   * @param[in] last_lane
   *   Index of last lane in segment
   */
  template <typename ReductionOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  ReduceStep(T& /*input*/, ReductionOp /*reduction_op*/, int /*last_lane*/, Int2Type<STEPS> /*step*/)
  {}

  //---------------------------------------------------------------------
  // Reduction operations
  //---------------------------------------------------------------------

  /**
   * @param[in] input
   *   Calling thread's input
   *
   * @param[in] valid_items
   *   Total number of valid items across the logical warp
   *
   * @param[in] reduction_op
   *   Binary reduction operator
   */
  template <typename ReductionOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE T
  ReduceImpl(Int2Type<0> /* all_lanes_valid */, T input, int valid_items, ReductionOp reduction_op)
  {
    int last_lane = valid_items - 1;

    T output = input;

    // Template-iterate reduction steps
    ReduceStep(output, reduction_op, last_lane, Int2Type<0>());

    return output;
  }

  /**
   * @param[in] input
   *   Calling thread's input
   *
   * @param[in] valid_items
   *   Total number of valid items across the logical warp
   *
   * @param[in] reduction_op
   *   Binary reduction operator
   */
  template <typename ReductionOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE T
  ReduceImpl(Int2Type<1> /* all_lanes_valid */, T input, int /* valid_items */, ReductionOp reduction_op)
  {
    int last_lane = LOGICAL_WARP_THREADS - 1;

    T output = input;

    // Template-iterate reduction steps
    ReduceStep(output, reduction_op, last_lane, Int2Type<0>());

    return output;
  }

  template <class U = T>
  _CCCL_DEVICE _CCCL_FORCEINLINE
  typename ::cuda::std::enable_if<(::cuda::std::is_same<int, U>::value || ::cuda::std::is_same<unsigned int, U>::value)
                                    && detail::reduce_add_exists<>::value,
                                  T>::type
  ReduceImpl(Int2Type<1> /* all_lanes_valid */, T input, int /* valid_items */, cub::Sum /* reduction_op */)
  {
    T output = input;

    NV_IF_TARGET(NV_PROVIDES_SM_80,
                 (output = __reduce_add_sync(member_mask, input);),
                 (output = ReduceImpl<cub::Sum>(Int2Type<1>{}, input, LOGICAL_WARP_THREADS, cub::Sum{});));

    return output;
  }

  template <class U = T>
  _CCCL_DEVICE _CCCL_FORCEINLINE
  typename ::cuda::std::enable_if<(::cuda::std::is_same<int, U>::value || ::cuda::std::is_same<unsigned int, U>::value)
                                    && detail::reduce_min_exists<>::value,
                                  T>::type
  ReduceImpl(Int2Type<1> /* all_lanes_valid */, T input, int /* valid_items */, cub::Min /* reduction_op */)
  {
    T output = input;

    NV_IF_TARGET(NV_PROVIDES_SM_80,
                 (output = __reduce_min_sync(member_mask, input);),
                 (output = ReduceImpl<cub::Min>(Int2Type<1>{}, input, LOGICAL_WARP_THREADS, cub::Min{});));

    return output;
  }

  template <class U = T>
  _CCCL_DEVICE _CCCL_FORCEINLINE
  typename ::cuda::std::enable_if<(::cuda::std::is_same<int, U>::value || ::cuda::std::is_same<unsigned int, U>::value)
                                    && detail::reduce_max_exists<>::value,
                                  T>::type
  ReduceImpl(Int2Type<1> /* all_lanes_valid */, T input, int /* valid_items */, cub::Max /* reduction_op */)
  {
    T output = input;

    NV_IF_TARGET(NV_PROVIDES_SM_80,
                 (output = __reduce_max_sync(member_mask, input);),
                 (output = ReduceImpl<cub::Max>(Int2Type<1>{}, input, LOGICAL_WARP_THREADS, cub::Max{});));

    return output;
  }

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
   *   Binary reduction operator
   */
  template <bool ALL_LANES_VALID, typename ReductionOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE T Reduce(T input, int valid_items, ReductionOp reduction_op)
  {
    return ReduceImpl(Int2Type<ALL_LANES_VALID>{}, input, valid_items, reduction_op);
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
   *   Binary reduction operator
   */
  template <bool HEAD_SEGMENTED, typename FlagT, typename ReductionOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE T SegmentedReduce(T input, FlagT flag, ReductionOp reduction_op)
  {
    // Get the start flags for each thread in the warp.
    int warp_flags = WARP_BALLOT(flag, member_mask);

    // Convert to tail-segmented
    if (HEAD_SEGMENTED)
    {
      warp_flags >>= 1;
    }

    // Mask out the bits below the current thread
    warp_flags &= LaneMaskGe();

    // Mask of physical lanes outside the logical warp and convert to logical lanemask
    if (!IS_ARCH_WARP)
    {
      warp_flags = (warp_flags & member_mask) >> (warp_id * LOGICAL_WARP_THREADS);
    }

    // Mask in the last lane of logical warp
    warp_flags |= 1u << (LOGICAL_WARP_THREADS - 1);

    // Find the next set flag
    int last_lane = __clz(__brev(warp_flags));

    T output = input;

    //        // Iterate reduction steps
    //        #pragma unroll
    //        for (int STEP = 0; STEP < STEPS; STEP++)
    //        {
    //            output = ReduceStep(output, reduction_op, last_lane, 1 << STEP,
    //            Int2Type<IsInteger<T>::IS_SMALL_UNSIGNED>());
    //        }

    // Template-iterate reduction steps
    ReduceStep(output, reduction_op, last_lane, Int2Type<0>());

    return output;
  }
};

CUB_NAMESPACE_END
