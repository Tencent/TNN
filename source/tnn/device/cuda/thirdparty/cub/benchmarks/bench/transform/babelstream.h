// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include <cub/device/dispatch/dispatch_transform.cuh>
#include <cub/util_namespace.cuh>

#include <cuda/std/type_traits>

#include <stdexcept>

#include <nvbench_helper.cuh>

template <typename... RandomAccessIteratorsIn>
#if TUNE_BASE
using policy_hub_t = cub::detail::transform::policy_hub<false, ::cuda::std::tuple<RandomAccessIteratorsIn...>>;
#else
struct policy_hub_t
{
  struct max_policy : cub::ChainedPolicy<350, max_policy, max_policy>
  {
    static constexpr int min_bif    = cub::detail::transform::arch_to_min_bytes_in_flight(__CUDA_ARCH_LIST__);
    static constexpr auto algorithm = static_cast<cub::detail::transform::Algorithm>(TUNE_ALGORITHM);
    using algo_policy =
      ::cuda::std::_If<algorithm == cub::detail::transform::Algorithm::prefetch,
                       cub::detail::transform::prefetch_policy_t<TUNE_THREADS>,
                       cub::detail::transform::async_copy_policy_t<TUNE_THREADS>>;
  };
};
#endif

#ifdef TUNE_T
using element_types = nvbench::type_list<TUNE_T>;
#else
using element_types =
  nvbench::type_list<std::int8_t,
                     std::int16_t,
                     float,
                     double
#  ifdef NVBENCH_HELPER_HAS_I128
                     ,
                     __int128
#  endif
                     >;
#endif

// BabelStream uses 2^25, H200 can fit 2^31 int128s
// 2^20 chars / 2^16 int128 saturate V100 (min_bif =12 * SM count =80)
// 2^21 chars / 2^17 int128 saturate A100 (min_bif =16 * SM count =108)
// 2^23 chars / 2^19 int128 saturate H100/H200 HBM3 (min_bif =32or48 * SM count =132)
// inline auto array_size_powers = std::vector<nvbench::int64_t>{28};
inline auto array_size_powers = nvbench::range(16, 28, 4);

template <typename OffsetT,
          typename... RandomAccessIteratorsIn,
          typename RandomAccessIteratorOut,
          typename TransformOp,
          typename ExecTag = decltype(nvbench::exec_tag::no_batch)>
void bench_transform(
  nvbench::state& state,
  ::cuda::std::tuple<RandomAccessIteratorsIn...> inputs,
  RandomAccessIteratorOut output,
  OffsetT num_items,
  TransformOp transform_op,
  ExecTag exec_tag = nvbench::exec_tag::no_batch)
{
  state.exec(exec_tag, [&](const nvbench::launch& launch) {
    cub::detail::transform::dispatch_t<
      false,
      OffsetT,
      ::cuda::std::tuple<RandomAccessIteratorsIn...>,
      RandomAccessIteratorOut,
      TransformOp,
      policy_hub_t<RandomAccessIteratorsIn...>>::dispatch(inputs, output, num_items, transform_op, launch.get_stream());
  });
}

// Modified from BabelStream to also work for integers
inline constexpr auto startA      = 1; // BabelStream: 0.1
inline constexpr auto startB      = 2; // BabelStream: 0.2
inline constexpr auto startC      = 3; // BabelStream: 0.1
inline constexpr auto startScalar = 4; // BabelStream: 0.4

// TODO(bgruber): we should put those somewhere into libcu++:
// from C++ GSL
struct narrowing_error : std::runtime_error
{
  narrowing_error()
      : std::runtime_error("Narrowing error")
  {}
};

// from C++ GSL
// implementation insipired by: https://github.com/microsoft/GSL/blob/main/include/gsl/narrow
template <typename DstT, typename SrcT, ::cuda::std::__enable_if_t<::cuda::std::is_arithmetic<SrcT>::value, int> = 0>
constexpr DstT narrow(SrcT value)
{
  constexpr bool is_different_signedness = ::cuda::std::is_signed<SrcT>::value != ::cuda::std::is_signed<DstT>::value;
  const auto converted                   = static_cast<DstT>(value);
  if (static_cast<SrcT>(converted) != value || (is_different_signedness && ((converted < DstT{}) != (value < SrcT{}))))
  {
    throw narrowing_error{};
  }
  return converted;
}
