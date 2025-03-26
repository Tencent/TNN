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

#include <cub/device/device_run_length_encode.cuh>

#include <look_back_helper.cuh>
#include <nvbench_helper.cuh>

// %RANGE% TUNE_ITEMS ipt 7:24:1
// %RANGE% TUNE_THREADS tpb 128:1024:32
// %RANGE% TUNE_TRANSPOSE trp 0:1:1
// %RANGE% TUNE_TIME_SLICING ts 0:1:1
// %RANGE% TUNE_LOAD ld 0:1:1
// %RANGE% TUNE_MAGIC_NS ns 0:2048:4
// %RANGE% TUNE_DELAY_CONSTRUCTOR_ID dcid 0:7:1
// %RANGE% TUNE_L2_WRITE_LATENCY_NS l2w 0:1200:5

#if !TUNE_BASE
#  if TUNE_TRANSPOSE == 0
#    define TUNE_LOAD_ALGORITHM cub::BLOCK_LOAD_DIRECT
#  else // TUNE_TRANSPOSE == 1
#    define TUNE_LOAD_ALGORITHM cub::BLOCK_LOAD_WARP_TRANSPOSE
#  endif // TUNE_TRANSPOSE

#  if TUNE_LOAD == 0
#    define TUNE_LOAD_MODIFIER cub::LOAD_DEFAULT
#  else // TUNE_LOAD == 1
#    define TUNE_LOAD_MODIFIER cub::LOAD_CA
#  endif // TUNE_LOAD

struct device_rle_policy_hub
{
  struct Policy350 : cub::ChainedPolicy<350, Policy350, Policy350>
  {
    using RleSweepPolicyT =
      cub::AgentRlePolicy<TUNE_THREADS,
                          TUNE_ITEMS,
                          TUNE_LOAD_ALGORITHM,
                          TUNE_LOAD_MODIFIER,
                          TUNE_TIME_SLICING,
                          cub::BLOCK_SCAN_WARP_SCANS,
                          delay_constructor_t>;
  };

  using MaxPolicy = Policy350;
};
#endif // !TUNE_BASE

template <class T, class OffsetT>
static void rle(nvbench::state& state, nvbench::type_list<T, OffsetT>)
{
  using offset_t                   = OffsetT;
  using keys_input_it_t            = const T*;
  using offset_output_it_t         = offset_t*;
  using length_output_it_t         = offset_t*;
  using num_runs_output_iterator_t = offset_t*;
  using equality_op_t              = cub::Equality;
  using accum_t                    = offset_t;

#if !TUNE_BASE
  using dispatch_t =
    cub::DeviceRleDispatch<keys_input_it_t,
                           offset_output_it_t,
                           length_output_it_t,
                           num_runs_output_iterator_t,
                           equality_op_t,
                           offset_t,
                           device_rle_policy_hub>;
#else
  using dispatch_t =
    cub::DeviceRleDispatch<keys_input_it_t,
                           offset_output_it_t,
                           length_output_it_t,
                           num_runs_output_iterator_t,
                           equality_op_t,
                           offset_t>;
#endif

  const auto elements                    = static_cast<std::size_t>(state.get_int64("Elements{io}"));
  constexpr std::size_t min_segment_size = 1;
  const std::size_t max_segment_size     = static_cast<std::size_t>(state.get_int64("MaxSegSize"));

  thrust::device_vector<offset_t> num_runs_out(1);
  thrust::device_vector<offset_t> out_offsets(elements);
  thrust::device_vector<offset_t> out_lengths(elements);
  thrust::device_vector<T> in_keys = generate.uniform.key_segments(elements, min_segment_size, max_segment_size);

  T* d_in_keys             = thrust::raw_pointer_cast(in_keys.data());
  offset_t* d_out_offsets  = thrust::raw_pointer_cast(out_offsets.data());
  offset_t* d_out_lengths  = thrust::raw_pointer_cast(out_lengths.data());
  offset_t* d_num_runs_out = thrust::raw_pointer_cast(num_runs_out.data());

  std::uint8_t* d_temp_storage{};
  std::size_t temp_storage_bytes{};

  dispatch_t::Dispatch(
    d_temp_storage,
    temp_storage_bytes,
    d_in_keys,
    d_out_offsets,
    d_out_lengths,
    d_num_runs_out,
    equality_op_t{},
    elements,
    0);

  thrust::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
  d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

  dispatch_t::Dispatch(
    d_temp_storage,
    temp_storage_bytes,
    d_in_keys,
    d_out_offsets,
    d_out_lengths,
    d_num_runs_out,
    equality_op_t{},
    elements,
    0);
  cudaDeviceSynchronize();
  const OffsetT num_runs = num_runs_out[0];

  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements);
  state.add_global_memory_writes<OffsetT>(num_runs);
  state.add_global_memory_writes<OffsetT>(num_runs);
  state.add_global_memory_writes<OffsetT>(1);

  state.exec(nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    dispatch_t::Dispatch(
      d_temp_storage,
      temp_storage_bytes,
      d_in_keys,
      d_out_offsets,
      d_out_lengths,
      d_num_runs_out,
      equality_op_t{},
      elements,
      launch.get_stream());
  });
}

using some_offset_types = nvbench::type_list<nvbench::int32_t>;

NVBENCH_BENCH_TYPES(rle, NVBENCH_TYPE_AXES(all_types, some_offset_types))
  .set_name("base")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4))
  .add_int64_power_of_two_axis("MaxSegSize", {1, 4, 8});
