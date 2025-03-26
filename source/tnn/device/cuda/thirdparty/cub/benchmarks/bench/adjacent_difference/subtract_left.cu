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

#include <cub/device/device_adjacent_difference.cuh>

#include <nvbench_helper.cuh>

// %RANGE% TUNE_ITEMS_PER_THREAD ipt 7:24:1
// %RANGE% TUNE_THREADS_PER_BLOCK tpb 128:1024:32

#if !TUNE_BASE
struct policy_hub_t
{
  struct Policy350 : cub::ChainedPolicy<350, Policy350, Policy350>
  {
    using AdjacentDifferencePolicy =
      cub::AgentAdjacentDifferencePolicy<TUNE_THREADS_PER_BLOCK,
                                         TUNE_ITEMS_PER_THREAD,
                                         cub::BLOCK_LOAD_WARP_TRANSPOSE,
                                         cub::LOAD_CA,
                                         cub::BLOCK_STORE_WARP_TRANSPOSE>;
  };

  using MaxPolicy = Policy350;
};
#endif // !TUNE_BASE

template <class T, class OffsetT>
void left(nvbench::state& state, nvbench::type_list<T, OffsetT>)
{
  constexpr bool may_alias = false;
  constexpr bool read_left = true;

  using input_it_t      = const T*;
  using output_it_t     = T*;
  using difference_op_t = cub::Difference;
  using offset_t        = cub::detail::choose_offset_t<OffsetT>;

#if !TUNE_BASE
  using dispatch_t = cub::
    DispatchAdjacentDifference<input_it_t, output_it_t, difference_op_t, offset_t, may_alias, read_left, policy_hub_t>;
#else
  using dispatch_t =
    cub::DispatchAdjacentDifference<input_it_t, output_it_t, difference_op_t, offset_t, may_alias, read_left>;
#endif // TUNE_BASE

  const auto elements         = static_cast<std::size_t>(state.get_int64("Elements{io}"));
  thrust::device_vector<T> in = generate(elements);
  thrust::device_vector<T> out(elements);

  input_it_t d_in   = thrust::raw_pointer_cast(in.data());
  output_it_t d_out = thrust::raw_pointer_cast(out.data());

  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements);
  state.add_global_memory_writes<T>(elements);

  std::size_t temp_storage_bytes{};
  dispatch_t::Dispatch(nullptr, temp_storage_bytes, d_in, d_out, static_cast<offset_t>(elements), difference_op_t{}, 0);

  thrust::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
  std::uint8_t* d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

  state.exec(nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    dispatch_t::Dispatch(
      d_temp_storage,
      temp_storage_bytes,
      d_in,
      d_out,
      static_cast<offset_t>(elements),
      difference_op_t{},
      launch.get_stream());
  });
}

using types = nvbench::type_list<int32_t>;

NVBENCH_BENCH_TYPES(left, NVBENCH_TYPE_AXES(types, offset_types))
  .set_name("base")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4));
