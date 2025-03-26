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

#include <cub/device/device_segmented_sort.cuh>

#include <nvbench_helper.cuh>

// %RANGE% TUNE_L_ITEMS ipt 7:24:1
// %RANGE% TUNE_M_ITEMS ipmw 1:17:1
// %RANGE% TUNE_S_ITEMS ipsw 1:17:1
// %RANGE% TUNE_THREADS tpb 128:1024:32
// %RANGE% TUNE_SW_THREADS_POW2 tpsw 1:4:1
// %RANGE% TUNE_MW_THREADS_POW2 tpmw 1:5:1
// %RANGE% TUNE_RADIX_BITS bits 4:8:1
// %RANGE% TUNE_PARTITIONING_THRESHOLD pt 100:800:50
// %RANGE% TUNE_RANK_ALGORITHM ra 0:4:1
// %RANGE% TUNE_LOAD ld 0:2:1
// %RANGE% TUNE_TRANSPOSE trp 0:1:1
// %RANGE% TUNE_S_LOAD sld 0:2:1
// %RANGE% TUNE_S_TRANSPOSE strp 0:1:1
// %RANGE% TUNE_M_LOAD mld 0:2:1
// %RANGE% TUNE_M_TRANSPOSE mtrp 0:1:1

#if !TUNE_BASE

#  define TUNE_SW_THREADS (1 << TUNE_SW_THREADS_POW2)
#  define TUNE_MW_THREADS (1 << TUNE_MW_THREADS_POW2)

#  define SMALL_SEGMENT_SIZE  TUNE_S_ITEMS* TUNE_SW_THREADS
#  define MEDIUM_SEGMENT_SIZE TUNE_M_ITEMS* TUNE_MW_THREADS
#  define LARGE_SEGMENT_SIZE  TUNE_L_ITEMS* TUNE_THREADS

#  if (LARGE_SEGMENT_SIZE <= SMALL_SEGMENT_SIZE) || (LARGE_SEGMENT_SIZE <= MEDIUM_SEGMENT_SIZE)
#    error Large segment size must be larger than small and medium segment sizes
#  endif

#  if (MEDIUM_SEGMENT_SIZE <= SMALL_SEGMENT_SIZE)
#    error Medium segment size must be larger than small one
#  endif

#  if TUNE_LOAD == 0
#    define TUNE_LOAD_MODIFIER cub::LOAD_DEFAULT
#  elif TUNE_LOAD == 1
#    define TUNE_LOAD_MODIFIER cub::LOAD_LDG
#  else // TUNE_LOAD == 2
#    define TUNE_LOAD_MODIFIER cub::LOAD_CA
#  endif // TUNE_LOAD

#  if TUNE_S_LOAD == 0
#    define TUNE_S_LOAD_MODIFIER cub::LOAD_DEFAULT
#  elif TUNE_S_LOAD == 1
#    define TUNE_S_LOAD_MODIFIER cub::LOAD_LDG
#  else // TUNE_S_LOAD == 2
#    define TUNE_S_LOAD_MODIFIER cub::LOAD_CA
#  endif // TUNE_S_LOAD

#  if TUNE_M_LOAD == 0
#    define TUNE_M_LOAD_MODIFIER cub::LOAD_DEFAULT
#  elif TUNE_M_LOAD == 1
#    define TUNE_M_LOAD_MODIFIER cub::LOAD_LDG
#  else // TUNE_M_LOAD == 2
#    define TUNE_M_LOAD_MODIFIER cub::LOAD_CA
#  endif // TUNE_M_LOAD

#  if TUNE_TRANSPOSE == 0
#    define TUNE_LOAD_ALGORITHM cub::BLOCK_LOAD_DIRECT
#  else // TUNE_TRANSPOSE == 1
#    define TUNE_LOAD_ALGORITHM cub::BLOCK_LOAD_WARP_TRANSPOSE
#  endif // TUNE_TRANSPOSE

#  if TUNE_S_TRANSPOSE == 0
#    define TUNE_S_LOAD_ALGORITHM cub::WarpLoadAlgorithm::WARP_LOAD_DIRECT
#  else // TUNE_S_TRANSPOSE == 1
#    define TUNE_S_LOAD_ALGORITHM cub::WarpLoadAlgorithm::WARP_LOAD_TRANSPOSE
#  endif // TUNE_S_TRANSPOSE

#  if TUNE_M_TRANSPOSE == 0
#    define TUNE_M_LOAD_ALGORITHM cub::WarpLoadAlgorithm::WARP_LOAD_DIRECT
#  else // TUNE_M_TRANSPOSE == 1
#    define TUNE_M_LOAD_ALGORITHM cub::WarpLoadAlgorithm::WARP_LOAD_TRANSPOSE
#  endif // TUNE_M_TRANSPOSE

template <class KeyT>
struct device_seg_sort_policy_hub
{
  using DominantT = KeyT;

  struct Policy350 : cub::ChainedPolicy<350, Policy350, Policy350>
  {
    static constexpr int BLOCK_THREADS          = TUNE_THREADS;
    static constexpr int RADIX_BITS             = TUNE_RADIX_BITS;
    static constexpr int PARTITIONING_THRESHOLD = TUNE_PARTITIONING_THRESHOLD;

    using LargeSegmentPolicy = cub::AgentRadixSortDownsweepPolicy<
      BLOCK_THREADS,
      TUNE_L_ITEMS,
      DominantT,
      TUNE_LOAD_ALGORITHM,
      TUNE_LOAD_MODIFIER,
      static_cast<cub::RadixRankAlgorithm>(TUNE_RANK_ALGORITHM),
      cub::BLOCK_SCAN_WARP_SCANS,
      RADIX_BITS>;

    static constexpr int ITEMS_PER_SMALL_THREAD  = TUNE_S_ITEMS;
    static constexpr int ITEMS_PER_MEDIUM_THREAD = TUNE_M_ITEMS;

    using SmallAndMediumSegmentedSortPolicyT = cub::AgentSmallAndMediumSegmentedSortPolicy<

      BLOCK_THREADS,

      // Small policy
      cub::
        AgentSubWarpMergeSortPolicy<TUNE_SW_THREADS, ITEMS_PER_SMALL_THREAD, TUNE_S_LOAD_ALGORITHM, TUNE_S_LOAD_MODIFIER>,

      // Medium policy
      cub::AgentSubWarpMergeSortPolicy<TUNE_MW_THREADS,
                                       ITEMS_PER_MEDIUM_THREAD,
                                       TUNE_M_LOAD_ALGORITHM,
                                       TUNE_M_LOAD_MODIFIER>>;
  };

  using MaxPolicy = Policy350;
};
#endif // !TUNE_BASE

template <class T, typename OffsetT>
void seg_sort(nvbench::state& state,
              nvbench::type_list<T, OffsetT> ts,
              const thrust::device_vector<OffsetT>& offsets,
              bit_entropy entropy)
{
  constexpr bool is_descending   = false;
  constexpr bool is_overwrite_ok = false;

  using offset_t          = OffsetT;
  using begin_offset_it_t = const offset_t*;
  using end_offset_it_t   = const offset_t*;
  using key_t             = T;
  using value_t           = cub::NullType;

#if !TUNE_BASE
  using policy_t   = device_seg_sort_policy_hub<key_t>;
  using dispatch_t = //
    cub::DispatchSegmentedSort<is_descending, key_t, value_t, offset_t, begin_offset_it_t, end_offset_it_t, policy_t>;
#else
  using dispatch_t = //
    cub::DispatchSegmentedSort<is_descending, key_t, value_t, offset_t, begin_offset_it_t, end_offset_it_t>;
#endif

  const auto elements = static_cast<std::size_t>(state.get_int64("Elements{io}"));
  const auto segments = offsets.size() - 1;

  thrust::device_vector<key_t> buffer_1 = generate(elements, entropy);
  thrust::device_vector<key_t> buffer_2(elements);

  key_t* d_buffer_1 = thrust::raw_pointer_cast(buffer_1.data());
  key_t* d_buffer_2 = thrust::raw_pointer_cast(buffer_2.data());

  cub::DoubleBuffer<key_t> d_keys(d_buffer_1, d_buffer_2);
  cub::DoubleBuffer<value_t> d_values;

  begin_offset_it_t d_begin_offsets = thrust::raw_pointer_cast(offsets.data());
  end_offset_it_t d_end_offsets     = d_begin_offsets + 1;

  state.add_element_count(elements);
  state.add_global_memory_reads<key_t>(elements);
  state.add_global_memory_writes<key_t>(elements);
  state.add_global_memory_reads<offset_t>(segments + 1);

  std::size_t temp_storage_bytes{};
  std::uint8_t* d_temp_storage{};
  dispatch_t::Dispatch(
    d_temp_storage,
    temp_storage_bytes,
    d_keys,
    d_values,
    elements,
    segments,
    d_begin_offsets,
    d_end_offsets,
    is_overwrite_ok,
    0);

  thrust::device_vector<nvbench::uint8_t> temp_storage(temp_storage_bytes);
  d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

  state.exec(nvbench::exec_tag::no_batch | nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    cub::DoubleBuffer<key_t> keys     = d_keys;
    cub::DoubleBuffer<value_t> values = d_values;

    dispatch_t::Dispatch(
      d_temp_storage,
      temp_storage_bytes,
      keys,
      values,
      elements,
      segments,
      d_begin_offsets,
      d_end_offsets,
      is_overwrite_ok,
      launch.get_stream());
  });
}

using some_offset_types = nvbench::type_list<uint32_t>;

template <class T, typename OffsetT>
void power_law(nvbench::state& state, nvbench::type_list<T, OffsetT> ts)
{
  const auto elements                    = static_cast<std::size_t>(state.get_int64("Elements{io}"));
  const auto segments                    = static_cast<std::size_t>(state.get_int64("Segments{io}"));
  const bit_entropy entropy              = str_to_entropy(state.get_string("Entropy"));
  thrust::device_vector<OffsetT> offsets = generate.power_law.segment_offsets(elements, segments);

  seg_sort(state, ts, offsets, entropy);
}

NVBENCH_BENCH_TYPES(power_law, NVBENCH_TYPE_AXES(fundamental_types, some_offset_types))
  .set_name("power")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(22, 30, 4))
  .add_int64_power_of_two_axis("Segments{io}", nvbench::range(12, 20, 4))
  .add_string_axis("Entropy", {"1.000", "0.201"});

template <class T, typename OffsetT>
void uniform(nvbench::state& state, nvbench::type_list<T, OffsetT> ts)
{
  const auto elements         = static_cast<std::size_t>(state.get_int64("Elements{io}"));
  const auto max_segment_size = static_cast<std::size_t>(state.get_int64("MaxSegmentSize"));

  const auto max_segment_size_log = static_cast<OffsetT>(std::log2(max_segment_size));
  const auto min_segment_size     = 1 << (max_segment_size_log - 1);

  thrust::device_vector<OffsetT> offsets =
    generate.uniform.segment_offsets(elements, min_segment_size, max_segment_size);

  seg_sort(state, ts, offsets, bit_entropy::_1_000);
}

NVBENCH_BENCH_TYPES(uniform, NVBENCH_TYPE_AXES(fundamental_types, some_offset_types))
  .set_name("small")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(22, 30, 4))
  .add_int64_power_of_two_axis("MaxSegmentSize", nvbench::range(1, 8, 1));

NVBENCH_BENCH_TYPES(uniform, NVBENCH_TYPE_AXES(fundamental_types, some_offset_types))
  .set_name("large")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(22, 30, 4))
  .add_int64_power_of_two_axis("MaxSegmentSize", nvbench::range(10, 18, 2));
