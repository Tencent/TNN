/******************************************************************************
 * Copyright (c) 2011-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include <cub/device/device_radix_sort.cuh>
#include <cub/util_arch.cuh>

#include <cuda/std/type_traits>

#include <nvbench_helper.cuh>

// %//RANGE//% TUNE_RADIX_BITS bits 8:9:1
#define TUNE_RADIX_BITS 8

// %RANGE% TUNE_ITEMS_PER_THREAD ipt 7:24:1
// %RANGE% TUNE_THREADS_PER_BLOCK tpb 128:1024:32

constexpr bool is_descending   = false;
constexpr bool is_overwrite_ok = false;

#if !TUNE_BASE
template <typename KeyT, typename ValueT, typename OffsetT>
struct policy_hub_t
{
  static constexpr bool KEYS_ONLY = std::is_same<ValueT, cub::NullType>::value;

  using DominantT = ::cuda::std::_If<(sizeof(ValueT) > sizeof(KeyT)), ValueT, KeyT>;

  struct policy_t : cub::ChainedPolicy<300, policy_t, policy_t>
  {
    static constexpr int ONESWEEP_RADIX_BITS = TUNE_RADIX_BITS;
    static constexpr bool ONESWEEP           = true;
    static constexpr bool OFFSET_64BIT       = sizeof(OffsetT) == 8;

    // Onesweep policy
    using OnesweepPolicy = cub::AgentRadixSortOnesweepPolicy<
      TUNE_THREADS_PER_BLOCK,
      TUNE_ITEMS_PER_THREAD,
      DominantT,
      1,
      cub::RADIX_RANK_MATCH_EARLY_COUNTS_ANY,
      cub::BLOCK_SCAN_RAKING_MEMOIZE,
      cub::RADIX_SORT_STORE_DIRECT,
      ONESWEEP_RADIX_BITS>;

    // These kernels are launched once, no point in tuning at the moment
    using HistogramPolicy    = cub::AgentRadixSortHistogramPolicy<128, 16, 1, KeyT, ONESWEEP_RADIX_BITS>;
    using ExclusiveSumPolicy = cub::AgentRadixSortExclusiveSumPolicy<256, ONESWEEP_RADIX_BITS>;
    using ScanPolicy =
      cub::AgentScanPolicy<512,
                           23,
                           OffsetT,
                           cub::BLOCK_LOAD_WARP_TRANSPOSE,
                           cub::LOAD_DEFAULT,
                           cub::BLOCK_STORE_WARP_TRANSPOSE,
                           cub::BLOCK_SCAN_RAKING_MEMOIZE>;

    // No point in tuning
    static constexpr int SINGLE_TILE_RADIX_BITS = (sizeof(KeyT) > 1) ? 6 : 5;

    // No point in tuning single-tile policy
    using SingleTilePolicy = cub::AgentRadixSortDownsweepPolicy<
      256,
      19,
      DominantT,
      cub::BLOCK_LOAD_DIRECT,
      cub::LOAD_LDG,
      cub::RADIX_RANK_MEMOIZE,
      cub::BLOCK_SCAN_WARP_SCANS,
      SINGLE_TILE_RADIX_BITS>;
  };

  using MaxPolicy = policy_t;
};

template <typename KeyT, typename ValueT, typename OffsetT>
constexpr std::size_t max_onesweep_temp_storage_size()
{
  using portion_offset  = int;
  using onesweep_policy = typename policy_hub_t<KeyT, ValueT, OffsetT>::policy_t::OnesweepPolicy;
  using agent_radix_sort_onesweep_t =
    cub::AgentRadixSortOnesweep<onesweep_policy, is_descending, KeyT, ValueT, OffsetT, portion_offset>;

  using hist_policy = typename policy_hub_t<KeyT, ValueT, OffsetT>::policy_t::HistogramPolicy;
  using hist_agent  = cub::AgentRadixSortHistogram<hist_policy, is_descending, KeyT, OffsetT>;

  return cub::max(sizeof(typename agent_radix_sort_onesweep_t::TempStorage), sizeof(typename hist_agent::TempStorage));
}

template <typename KeyT, typename ValueT, typename OffsetT>
constexpr std::size_t max_temp_storage_size()
{
  using policy_t = typename policy_hub_t<KeyT, ValueT, OffsetT>::policy_t;

  static_assert(policy_t::ONESWEEP);
  return max_onesweep_temp_storage_size<KeyT, ValueT, OffsetT>();
}

template <typename KeyT, typename ValueT, typename OffsetT>
constexpr bool fits_in_default_shared_memory()
{
  return max_temp_storage_size<KeyT, ValueT, OffsetT>() < cub::detail::max_smem_per_block;
}
#else // TUNE_BASE
template <typename, typename, typename>
constexpr bool fits_in_default_shared_memory()
{
  return true;
}
#endif // TUNE_BASE

template <typename KeyT, typename ValueT, typename OffsetT>
void radix_sort_values(
  std::integral_constant<bool, true>, nvbench::state& state, nvbench::type_list<KeyT, ValueT, OffsetT>)
{
  using offset_t = cub::detail::choose_offset_t<OffsetT>;

  using key_t   = KeyT;
  using value_t = ValueT;
#if !TUNE_BASE
  using policy_t   = policy_hub_t<key_t, value_t, offset_t>;
  using dispatch_t = cub::DispatchRadixSort<is_descending, key_t, value_t, offset_t, policy_t>;
#else // TUNE_BASE
  using dispatch_t = cub::DispatchRadixSort<is_descending, key_t, value_t, offset_t>;
#endif // TUNE_BASE

  constexpr int begin_bit = 0;
  constexpr int end_bit   = sizeof(key_t) * 8;

  // Retrieve axis parameters
  const auto elements       = static_cast<std::size_t>(state.get_int64("Elements{io}"));
  const bit_entropy entropy = str_to_entropy(state.get_string("Entropy"));

  thrust::device_vector<key_t> keys_buffer_1     = generate(elements, entropy);
  thrust::device_vector<value_t> values_buffer_1 = generate(elements);
  thrust::device_vector<key_t> keys_buffer_2(elements);
  thrust::device_vector<value_t> values_buffer_2(elements);

  key_t* d_keys_buffer_1     = thrust::raw_pointer_cast(keys_buffer_1.data());
  key_t* d_keys_buffer_2     = thrust::raw_pointer_cast(keys_buffer_2.data());
  value_t* d_values_buffer_1 = thrust::raw_pointer_cast(values_buffer_1.data());
  value_t* d_values_buffer_2 = thrust::raw_pointer_cast(values_buffer_2.data());

  cub::DoubleBuffer<key_t> d_keys(d_keys_buffer_1, d_keys_buffer_2);
  cub::DoubleBuffer<value_t> d_values(d_values_buffer_1, d_values_buffer_2);

  // Enable throughput calculations and add "Size" column to results.
  state.add_element_count(elements);
  state.add_global_memory_reads<KeyT>(elements);
  state.add_global_memory_reads<ValueT>(elements);
  state.add_global_memory_writes<KeyT>(elements);
  state.add_global_memory_writes<ValueT>(elements);

  // Allocate temporary storage:
  std::size_t temp_size{};
  dispatch_t::Dispatch(
    nullptr,
    temp_size,
    d_keys,
    d_values,
    static_cast<offset_t>(elements),
    begin_bit,
    end_bit,
    is_overwrite_ok,
    0 /* stream */);

  thrust::device_vector<nvbench::uint8_t> temp(temp_size);
  auto* temp_storage = thrust::raw_pointer_cast(temp.data());

  state.exec(nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    cub::DoubleBuffer<key_t> keys     = d_keys;
    cub::DoubleBuffer<value_t> values = d_values;

    dispatch_t::Dispatch(
      temp_storage,
      temp_size,
      keys,
      values,
      static_cast<offset_t>(elements),
      begin_bit,
      end_bit,
      is_overwrite_ok,
      launch.get_stream());
  });
}

template <typename KeyT, typename ValueT, typename OffsetT>
void radix_sort_values(std::integral_constant<bool, false>, nvbench::state&, nvbench::type_list<KeyT, ValueT, OffsetT>)
{
  (void) is_descending;
  (void) is_overwrite_ok;
}

template <typename KeyT, typename ValueT, typename OffsetT>
void radix_sort_values(nvbench::state& state, nvbench::type_list<KeyT, ValueT, OffsetT> tl)
{
  using offset_t = cub::detail::choose_offset_t<OffsetT>;

  radix_sort_values(std::integral_constant<bool, fits_in_default_shared_memory<KeyT, ValueT, offset_t>()>{}, state, tl);
}

#ifdef TUNE_KeyT
using key_types = nvbench::type_list<TUNE_KeyT>;
#else // !defined(TUNE_KeyT)
using key_types = integral_types;
#endif // TUNE_KeyT

#ifdef TUNE_ValueT
using value_types = nvbench::type_list<TUNE_ValueT>;
#else // !defined(Tune_ValueT)
using value_types =
  nvbench::type_list<int8_t,
                     int16_t,
                     int32_t,
                     int64_t
#  if NVBENCH_HELPER_HAS_I128
                     ,
                     int128_t
#  endif
                     >;
#endif // TUNE_ValueT

NVBENCH_BENCH_TYPES(radix_sort_values, NVBENCH_TYPE_AXES(key_types, value_types, offset_types))
  .set_name("base")
  .set_type_axes_names({"KeyT{ct}", "ValueT{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4))
  .add_string_axis("Entropy", {"1.000", "0.201"});
