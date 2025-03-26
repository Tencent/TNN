/******************************************************************************
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/agent/agent_scan.cuh>
#include <cub/agent/single_pass_scan_operators.cuh>
#include <cub/block/block_load.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_store.cuh>
#include <cub/util_device.cuh>
#include <cub/util_type.cuh>

CUB_NAMESPACE_BEGIN

namespace detail
{
namespace scan
{

enum class keep_rejects
{
  no,
  yes
};
enum class primitive_accum
{
  no,
  yes
};
enum class primitive_op
{
  no,
  yes
};
enum class offset_size
{
  _4,
  _8,
  unknown
};
enum class accum_size
{
  _1,
  _2,
  _4,
  _8,
  _16,
  unknown
};

template <class AccumT>
constexpr primitive_accum is_primitive_accum()
{
  return Traits<AccumT>::PRIMITIVE ? primitive_accum::yes : primitive_accum::no;
}

template <class ScanOpT>
constexpr primitive_op is_primitive_op()
{
  return basic_binary_op_t<ScanOpT>::value ? primitive_op::yes : primitive_op::no;
}

template <class AccumT>
constexpr accum_size classify_accum_size()
{
  return sizeof(AccumT) == 1 ? accum_size::_1
       : sizeof(AccumT) == 2 ? accum_size::_2
       : sizeof(AccumT) == 4 ? accum_size::_4
       : sizeof(AccumT) == 8 ? accum_size::_8
       : sizeof(AccumT) == 16
         ? accum_size::_16
         : accum_size::unknown;
}

template <int Threads, int Items, int L2B, int L2W>
struct tuning
{
  static constexpr int threads = Threads;
  static constexpr int items   = Items;

  using delay_constructor = detail::fixed_delay_constructor_t<L2B, L2W>;
};

template <class AccumT,
          primitive_op PrimitiveOp,
          primitive_accum PrimitiveAccumulator = is_primitive_accum<AccumT>(),
          accum_size AccumSize                 = classify_accum_size<AccumT>()>
struct sm90_tuning
{
  static constexpr int threads = 128;
  static constexpr int items   = 15;

  using delay_constructor = detail::default_delay_constructor_t<AccumT>;
};

// clang-format off
template <class T> struct sm90_tuning<T, primitive_op::yes, primitive_accum::yes, accum_size::_1> : tuning<192, 22, 168, 1140> {};
template <class T> struct sm90_tuning<T, primitive_op::yes, primitive_accum::yes, accum_size::_2> : tuning<512, 12, 376, 1125> {};
template <class T> struct sm90_tuning<T, primitive_op::yes, primitive_accum::yes, accum_size::_4> : tuning<128, 24, 648, 1245> {};
template <class T> struct sm90_tuning<T, primitive_op::yes, primitive_accum::yes, accum_size::_8> : tuning<224, 24, 632, 1290> {};

template <> struct sm90_tuning<float,  primitive_op::yes, primitive_accum::yes, accum_size::_4> : tuning<128, 24, 688, 1140> {};
template <> struct sm90_tuning<double, primitive_op::yes, primitive_accum::yes, accum_size::_8> : tuning<224, 24, 576, 1215> {};

#if CUB_IS_INT128_ENABLED
template <> struct sm90_tuning< __int128_t, primitive_op::yes, primitive_accum::no, accum_size::_16> : tuning<576, 21, 860, 630> {};
template <> struct sm90_tuning<__uint128_t, primitive_op::yes, primitive_accum::no, accum_size::_16> : tuning<576, 21, 860, 630> {};
#endif
// clang-format on

template <class AccumT,
          primitive_op PrimitiveOp,
          primitive_accum PrimitiveAccumulator = is_primitive_accum<AccumT>(),
          accum_size AccumSize                 = classify_accum_size<AccumT>()>
struct sm80_tuning
{
  static constexpr int threads = 128;
  static constexpr int items   = 15;

  using delay_constructor = detail::default_delay_constructor_t<AccumT>;

  static constexpr bool LargeValues = sizeof(AccumT) > 128;

  static constexpr BlockLoadAlgorithm load_algorithm = //
    LargeValues ? BLOCK_LOAD_WARP_TRANSPOSE_TIMESLICED : BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm = //
    LargeValues ? BLOCK_STORE_WARP_TRANSPOSE_TIMESLICED : BLOCK_STORE_WARP_TRANSPOSE;
};

template <class T>
struct sm80_tuning<T, primitive_op::yes, primitive_accum::yes, accum_size::_1>
{
  static constexpr int threads = 320;
  static constexpr int items   = 14;

  using delay_constructor = detail::fixed_delay_constructor_t<368, 725>;

  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_WARP_TRANSPOSE;
};

template <class T>
struct sm80_tuning<T, primitive_op::yes, primitive_accum::yes, accum_size::_2>
{
  static constexpr int threads = 352;
  static constexpr int items   = 16;

  using delay_constructor = detail::fixed_delay_constructor_t<488, 1040>;

  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_WARP_TRANSPOSE;
};

template <class T>
struct sm80_tuning<T, primitive_op::yes, primitive_accum::yes, accum_size::_4>
{
  static constexpr int threads = 320;
  static constexpr int items   = 12;

  using delay_constructor = detail::fixed_delay_constructor_t<268, 1180>;

  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_WARP_TRANSPOSE;
};

template <class T>
struct sm80_tuning<T, primitive_op::yes, primitive_accum::yes, accum_size::_8>
{
  static constexpr int threads = 288;
  static constexpr int items   = 22;

  using delay_constructor = detail::fixed_delay_constructor_t<716, 785>;

  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_WARP_TRANSPOSE;
};

template <>
struct sm80_tuning<float, primitive_op::yes, primitive_accum::yes, accum_size::_4>
{
  static constexpr int threads = 288;
  static constexpr int items   = 8;

  using delay_constructor = detail::fixed_delay_constructor_t<724, 1050>;

  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_WARP_TRANSPOSE;
};

template <>
struct sm80_tuning<double, primitive_op::yes, primitive_accum::yes, accum_size::_8>
{
  static constexpr int threads = 384;
  static constexpr int items   = 12;

  using delay_constructor = detail::fixed_delay_constructor_t<388, 1100>;

  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_WARP_TRANSPOSE;
};

#if CUB_IS_INT128_ENABLED
template <>
struct sm80_tuning<__int128_t, primitive_op::yes, primitive_accum::no, accum_size::_16>
{
  static constexpr int threads = 640;
  static constexpr int items   = 24;

  using delay_constructor = detail::no_delay_constructor_t<1200>;

  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_DIRECT;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_DIRECT;
};

template <>
struct sm80_tuning<__uint128_t, primitive_op::yes, primitive_accum::no, accum_size::_16>
{
  static constexpr int threads = 640;
  static constexpr int items   = 24;

  using delay_constructor = detail::no_delay_constructor_t<1200>;

  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_DIRECT;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_DIRECT;
};
#endif

} // namespace scan
} // namespace detail

template <typename AccumT, typename ScanOpT = Sum>
struct DeviceScanPolicy
{
  // For large values, use timesliced loads/stores to fit shared memory.
  static constexpr bool LargeValues = sizeof(AccumT) > 128;
  static constexpr BlockLoadAlgorithm ScanTransposedLoad =
    LargeValues ? BLOCK_LOAD_WARP_TRANSPOSE_TIMESLICED : BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm ScanTransposedStore =
    LargeValues ? BLOCK_STORE_WARP_TRANSPOSE_TIMESLICED : BLOCK_STORE_WARP_TRANSPOSE;

  template <int NOMINAL_BLOCK_THREADS_4B,
            int NOMINAL_ITEMS_PER_THREAD_4B,
            typename ComputeT,
            BlockLoadAlgorithm LOAD_ALGORITHM,
            CacheLoadModifier LOAD_MODIFIER,
            BlockStoreAlgorithm STORE_ALGORITHM,
            BlockScanAlgorithm SCAN_ALGORITHM,
            typename DelayConstructorT>
  using policy_t =
    AgentScanPolicy<NOMINAL_BLOCK_THREADS_4B,
                    NOMINAL_ITEMS_PER_THREAD_4B,
                    ComputeT,
                    LOAD_ALGORITHM,
                    LOAD_MODIFIER,
                    STORE_ALGORITHM,
                    SCAN_ALGORITHM,
                    MemBoundScaling<NOMINAL_BLOCK_THREADS_4B, NOMINAL_ITEMS_PER_THREAD_4B, ComputeT>,
                    DelayConstructorT>;

  /// SM350
  struct Policy350 : ChainedPolicy<350, Policy350, Policy350>
  {
    // GTX Titan: 29.5B items/s (232.4 GB/s) @ 48M 32-bit T
    using ScanPolicyT =
      policy_t<128,
               12, ///< Threads per block, items per thread
               AccumT,
               BLOCK_LOAD_DIRECT,
               LOAD_CA,
               BLOCK_STORE_WARP_TRANSPOSE_TIMESLICED,
               BLOCK_SCAN_RAKING,
               detail::default_delay_constructor_t<AccumT>>;
  };

  /// SM520
  struct Policy520 : ChainedPolicy<520, Policy520, Policy350>
  {
    // Titan X: 32.47B items/s @ 48M 32-bit T
    using ScanPolicyT =
      policy_t<128,
               12, ///< Threads per block, items per thread
               AccumT,
               BLOCK_LOAD_DIRECT,
               LOAD_CA,
               ScanTransposedStore,
               BLOCK_SCAN_WARP_SCANS,
               detail::default_delay_constructor_t<AccumT>>;
  };

  /// SM600
  struct DefaultTuning
  {
    using ScanPolicyT =
      policy_t<128,
               15, ///< Threads per block, items per thread
               AccumT,
               ScanTransposedLoad,
               LOAD_DEFAULT,
               ScanTransposedStore,
               BLOCK_SCAN_WARP_SCANS,
               detail::default_delay_constructor_t<AccumT>>;
  };

  /// SM600
  struct Policy600
      : DefaultTuning
      , ChainedPolicy<600, Policy600, Policy520>
  {};

  /// SM800
  struct Policy800 : ChainedPolicy<800, Policy800, Policy600>
  {
    using tuning = detail::scan::sm80_tuning<AccumT, detail::scan::is_primitive_op<ScanOpT>()>;

    using ScanPolicyT =
      policy_t<tuning::threads,
               tuning::items,
               AccumT,
               tuning::load_algorithm,
               LOAD_DEFAULT,
               tuning::store_algorithm,
               BLOCK_SCAN_WARP_SCANS,
               typename tuning::delay_constructor>;
  };

  /// SM860
  struct Policy860
      : DefaultTuning
      , ChainedPolicy<860, Policy860, Policy800>
  {};

  /// SM900
  struct Policy900 : ChainedPolicy<900, Policy900, Policy860>
  {
    using tuning = detail::scan::sm90_tuning<AccumT, detail::scan::is_primitive_op<ScanOpT>()>;

    using ScanPolicyT =
      policy_t<tuning::threads,
               tuning::items,
               AccumT,
               ScanTransposedLoad,
               LOAD_DEFAULT,
               ScanTransposedStore,
               BLOCK_SCAN_WARP_SCANS,
               typename tuning::delay_constructor>;
  };

  using MaxPolicy = Policy900;
};

CUB_NAMESPACE_END
