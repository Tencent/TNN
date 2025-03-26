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

#include "histogram_common.cuh"
#include <nvbench_helper.cuh>

// %RANGE% TUNE_ITEMS ipt 4:28:1
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
  constexpr int num_channels        = 1;
  constexpr int num_active_channels = 1;

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

  const auto entropy   = str_to_entropy(state.get_string("Entropy"));
  const auto elements  = state.get_int64("Elements{io}");
  const auto num_bins  = state.get_int64("Bins");
  const int num_levels = static_cast<int>(num_bins) + 1;

  const SampleT lower_level = 0;
  const SampleT upper_level = get_upper_level<SampleT>(num_bins, elements);

  thrust::device_vector<SampleT> input = generate(elements, entropy, lower_level, upper_level);
  thrust::device_vector<CounterT> hist(num_bins);

  SampleT* d_input      = thrust::raw_pointer_cast(input.data());
  CounterT* d_histogram = thrust::raw_pointer_cast(hist.data());

  CounterT* d_histogram1[1] = {d_histogram};
  int num_levels1[1]        = {num_levels};
  SampleT lower_level1[1]   = {lower_level};
  SampleT upper_level1[1]   = {upper_level};

  std::uint8_t* d_temp_storage = nullptr;
  std::size_t temp_storage_bytes{};

  cub::Int2Type<sizeof(SampleT) == 1> is_byte_sample;
  OffsetT num_row_pixels     = static_cast<OffsetT>(elements);
  OffsetT num_rows           = 1;
  OffsetT row_stride_samples = num_row_pixels;

  state.add_element_count(elements);
  state.add_global_memory_reads<SampleT>(elements);
  state.add_global_memory_writes<CounterT>(num_bins);

  dispatch_t::DispatchEven(
    d_temp_storage,
    temp_storage_bytes,
    d_input,
    d_histogram1,
    num_levels1,
    lower_level1,
    upper_level1,
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
      d_histogram1,
      num_levels1,
      lower_level1,
      upper_level1,
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
