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

#include "../histogram_common.cuh"
#include <nvbench_helper.cuh>

// %RANGE% TUNE_ITEMS ipt 7:24:1
// %RANGE% TUNE_THREADS tpb 128:1024:32
// %RANGE% TUNE_RLE_COMPRESS rle 0:1:1
// %RANGE% TUNE_WORK_STEALING ws 0:1:1
// %RANGE% TUNE_MEM_PREFERENCE mem 0:2:1
// %RANGE% TUNE_LOAD ld 0:2:1
// %RANGE% TUNE_LOAD_ALGORITHM_ID laid 0:2:1
// %RANGE% TUNE_VEC_SIZE_POW vec 0:2:1

template <typename SampleT, typename CounterT, typename OffsetT>
static void even(nvbench::state& state, nvbench::type_list<SampleT, CounterT, OffsetT>)
{
  constexpr int num_channels        = 4;
  constexpr int num_active_channels = 3;

  using sample_iterator_t = SampleT*;

#if !TUNE_BASE
  using policy_t = policy_hub_t<key_t, num_channels, num_active_channels>;
  using dispatch_t =
    cub::DispatchHistogram<num_channels, //
                           num_active_channels,
                           sample_iterator_t,
                           CounterT,
                           SampleT,
                           OffsetT,
                           policy_t>;
#else // TUNE_BASE
  using dispatch_t =
    cub::DispatchHistogram<num_channels, //
                           num_active_channels,
                           sample_iterator_t,
                           CounterT,
                           SampleT,
                           OffsetT>;
#endif // TUNE_BASE

  const auto entropy     = str_to_entropy(state.get_string("Entropy"));
  const auto elements    = state.get_int64("Elements{io}");
  const auto num_bins    = state.get_int64("Bins");
  const int num_levels_r = static_cast<int>(num_bins) + 1;
  const int num_levels_g = num_levels_r;
  const int num_levels_b = num_levels_g;

  const SampleT lower_level_r = 0;
  const SampleT upper_level_r = get_upper_level<SampleT>(num_bins, elements);
  const SampleT lower_level_g = lower_level_r;
  const SampleT upper_level_g = upper_level_r;
  const SampleT lower_level_b = lower_level_g;
  const SampleT upper_level_b = upper_level_g;

  thrust::device_vector<CounterT> hist_r(num_bins);
  thrust::device_vector<CounterT> hist_g(num_bins);
  thrust::device_vector<CounterT> hist_b(num_bins);
  thrust::device_vector<SampleT> input = generate(elements * num_channels, entropy, lower_level_r, upper_level_r);

  SampleT* d_input        = thrust::raw_pointer_cast(input.data());
  CounterT* d_histogram_r = thrust::raw_pointer_cast(hist_r.data());
  CounterT* d_histogram_g = thrust::raw_pointer_cast(hist_g.data());
  CounterT* d_histogram_b = thrust::raw_pointer_cast(hist_b.data());

  CounterT* d_histogram[num_active_channels] = {d_histogram_r, d_histogram_g, d_histogram_b};
  int num_levels[num_active_channels]        = {num_levels_r, num_levels_g, num_levels_b};
  SampleT lower_level[num_active_channels]   = {lower_level_r, lower_level_g, lower_level_b};
  SampleT upper_level[num_active_channels]   = {upper_level_r, upper_level_g, upper_level_b};

  std::uint8_t* d_temp_storage = nullptr;
  std::size_t temp_storage_bytes{};

  cub::Int2Type<sizeof(SampleT) == 1> is_byte_sample;
  OffsetT num_row_pixels     = static_cast<OffsetT>(elements);
  OffsetT num_rows           = 1;
  OffsetT row_stride_samples = num_row_pixels;

  state.add_element_count(elements);
  state.add_global_memory_reads<SampleT>(elements * num_active_channels);
  state.add_global_memory_writes<CounterT>(num_bins * num_active_channels);

  dispatch_t::DispatchEven(
    d_temp_storage,
    temp_storage_bytes,
    d_input,
    d_histogram,
    num_levels,
    lower_level,
    upper_level,
    num_row_pixels,
    num_rows,
    row_stride_samples,
    0,
    is_byte_sample);

  thrust::device_vector<nvbench::uint8_t> tmp(temp_storage_bytes);
  d_temp_storage = thrust::raw_pointer_cast(tmp.data());

  state.exec(nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    dispatch_t::DispatchEven(
      d_temp_storage,
      temp_storage_bytes,
      d_input,
      d_histogram,
      num_levels,
      lower_level,
      upper_level,
      num_row_pixels,
      num_rows,
      row_stride_samples,
      launch.get_stream(),
      is_byte_sample);
  });
}

using bin_types         = nvbench::type_list<int32_t>;
using some_offset_types = nvbench::type_list<int32_t>;

#ifdef TUNE_SampleT
using sample_types = nvbench::type_list<TUNE_SampleT>;
#else // !defined(TUNE_SampleT)
using sample_types = nvbench::type_list<int8_t, int16_t, int32_t, int64_t, float, double>;
#endif // TUNE_SampleT

NVBENCH_BENCH_TYPES(even, NVBENCH_TYPE_AXES(sample_types, bin_types, some_offset_types))
  .set_name("base")
  .set_type_axes_names({"SampleT{ct}", "BinT{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4))
  .add_int64_axis("Bins", {32, 128, 2048, 2097152})
  .add_string_axis("Entropy", {"0.201", "1.000"});
