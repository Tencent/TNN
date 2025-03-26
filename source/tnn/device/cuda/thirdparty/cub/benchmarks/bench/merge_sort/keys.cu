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

#include <cub/device/device_merge_sort.cuh>

#include <nvbench_helper.cuh>

// %RANGE% TUNE_TRANSPOSE trp 0:1:1
// %RANGE% TUNE_LOAD ld 0:2:1
// %RANGE% TUNE_ITEMS_PER_THREAD ipt 7:24:1
// %RANGE% TUNE_THREADS_PER_BLOCK_POW2 tpb 6:10:1

#ifndef TUNE_BASE
#  define TUNE_THREADS_PER_BLOCK (1 << TUNE_THREADS_PER_BLOCK_POW2)
#endif // TUNE_BASE

using value_t = cub::NullType;

#if !TUNE_BASE

#  if TUNE_TRANSPOSE == 0
#    define TUNE_LOAD_ALGORITHM  cub::BLOCK_LOAD_DIRECT
#    define TUNE_STORE_ALGORITHM cub::BLOCK_STORE_DIRECT
#  else // TUNE_TRANSPOSE == 1
#    define TUNE_LOAD_ALGORITHM  cub::BLOCK_LOAD_WARP_TRANSPOSE
#    define TUNE_STORE_ALGORITHM cub::BLOCK_STORE_WARP_TRANSPOSE
#  endif // TUNE_TRANSPOSE

#  if TUNE_LOAD == 0
#    define TUNE_LOAD_MODIFIER cub::LOAD_DEFAULT
#  elif TUNE_LOAD == 1
#    define TUNE_LOAD_MODIFIER cub::LOAD_LDG
#  else // TUNE_LOAD == 2
#    define TUNE_LOAD_MODIFIER cub::LOAD_CA
#  endif // TUNE_LOAD

template <typename KeyT>
struct policy_hub_t
{
  struct policy_t : cub::ChainedPolicy<300, policy_t, policy_t>
  {
    using MergeSortPolicy =
      cub::AgentMergeSortPolicy<TUNE_THREADS_PER_BLOCK,
                                cub::Nominal4BItemsToItems<KeyT>(TUNE_ITEMS_PER_THREAD),
                                TUNE_LOAD_ALGORITHM,
                                TUNE_LOAD_MODIFIER,
                                TUNE_STORE_ALGORITHM>;
  };

  using MaxPolicy = policy_t;
};
#endif // !TUNE_BASE

template <typename T, typename OffsetT>
void keys(nvbench::state& state, nvbench::type_list<T, OffsetT>)
{
  using key_t            = T;
  using value_t          = cub::NullType;
  using key_input_it_t   = key_t*;
  using value_input_it_t = value_t*;
  using key_it_t         = key_t*;
  using value_it_t       = value_t*;
  using offset_t         = OffsetT;
  using compare_op_t     = less_t;

#if !TUNE_BASE
  using policy_t = policy_hub_t<key_t>;
  using dispatch_t =
    cub::DispatchMergeSort<key_input_it_t, value_input_it_t, key_it_t, value_it_t, offset_t, compare_op_t, policy_t>;
#else // TUNE_BASE
  using dispatch_t =
    cub::DispatchMergeSort<key_input_it_t, value_input_it_t, key_it_t, value_it_t, offset_t, compare_op_t>;
#endif // TUNE_BASE

  // Retrieve axis parameters
  const auto elements       = static_cast<std::size_t>(state.get_int64("Elements{io}"));
  const bit_entropy entropy = str_to_entropy(state.get_string("Entropy"));

  thrust::device_vector<T> buffer_1 = generate(elements, entropy);
  thrust::device_vector<T> buffer_2(elements);

  key_t* d_buffer_1 = thrust::raw_pointer_cast(buffer_1.data());
  key_t* d_buffer_2 = thrust::raw_pointer_cast(buffer_2.data());

  // Enable throughput calculations and add "Size" column to results.
  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements, "Size");
  state.add_global_memory_writes<T>(elements);

  // Allocate temporary storage:
  std::size_t temp_size{};
  dispatch_t::Dispatch(
    nullptr,
    temp_size,
    d_buffer_1,
    nullptr,
    d_buffer_2,
    nullptr,
    static_cast<offset_t>(elements),
    compare_op_t{},
    0 /* stream */);

  thrust::device_vector<nvbench::uint8_t> temp(temp_size);
  auto* temp_storage = thrust::raw_pointer_cast(temp.data());

  state.exec(nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    dispatch_t::Dispatch(
      temp_storage,
      temp_size,
      d_buffer_1,
      nullptr,
      d_buffer_2,
      nullptr,
      static_cast<offset_t>(elements),
      compare_op_t{},
      launch.get_stream());
  });
}

NVBENCH_BENCH_TYPES(keys, NVBENCH_TYPE_AXES(all_types, offset_types))
  .set_name("base")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4))
  .add_string_axis("Entropy", {"1.000", "0.201"});
