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

#include <cub/device/device_partition.cuh>

#include <thrust/count.h>

#include <cuda/std/type_traits>

#include <look_back_helper.cuh>
#include <nvbench_helper.cuh>

// %RANGE% TUNE_TRANSPOSE trp 0:1:1
// %RANGE% TUNE_LOAD ld 0:1:1
// %RANGE% TUNE_ITEMS_PER_THREAD ipt 7:24:1
// %RANGE% TUNE_THREADS_PER_BLOCK tpb 128:1024:32
// %RANGE% TUNE_MAGIC_NS ns 0:2048:4
// %RANGE% TUNE_DELAY_CONSTRUCTOR_ID dcid 0:7:1
// %RANGE% TUNE_L2_WRITE_LATENCY_NS l2w 0:1200:5

constexpr bool keep_rejects = true;
constexpr bool may_alias    = false;

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

template <typename InputT>
struct policy_hub_t
{
  struct policy_t : cub::ChainedPolicy<300, policy_t, policy_t>
  {
    static constexpr int NOMINAL_4B_ITEMS_PER_THREAD = TUNE_ITEMS_PER_THREAD;

    static constexpr int ITEMS_PER_THREAD =
      CUB_MIN(NOMINAL_4B_ITEMS_PER_THREAD, CUB_MAX(1, (NOMINAL_4B_ITEMS_PER_THREAD * 4 / sizeof(InputT))));

    using SelectIfPolicyT =
      cub::AgentSelectIfPolicy<TUNE_THREADS_PER_BLOCK,
                               ITEMS_PER_THREAD,
                               TUNE_LOAD_ALGORITHM,
                               TUNE_LOAD_MODIFIER,
                               cub::BLOCK_SCAN_WARP_SCANS,
                               delay_constructor_t>;
  };

  using MaxPolicy = policy_t;
};
#endif // !TUNE_BASE

template <class T>
struct less_then_t
{
  T m_val;

  __device__ bool operator()(const T& val) const
  {
    return val < m_val;
  }
};

template <typename T>
T value_from_entropy(double percentage)
{
  if (percentage == 1)
  {
    return std::numeric_limits<T>::max();
  }

  const auto max_val = static_cast<double>(std::numeric_limits<T>::max());
  const auto min_val = static_cast<double>(std::numeric_limits<T>::lowest());
  const auto result  = min_val + percentage * max_val - percentage * min_val;
  return static_cast<T>(result);
}

template <typename InItT, typename T, typename OffsetT, typename SelectOpT>
void init_output_partition_buffer(
  InItT d_in,
  OffsetT num_items,
  T* d_out,
  SelectOpT select_op,
  cub::detail::partition_distinct_output_t<T*, T*>& d_partition_out_buffer)
{
  const auto selected_elements = thrust::count_if(d_in, d_in + num_items, select_op);
  d_partition_out_buffer       = cub::detail::partition_distinct_output_t<T*, T*>{d_out, d_out + selected_elements};
}

template <typename InItT, typename T, typename OffsetT, typename SelectOpT>
void init_output_partition_buffer(InItT, OffsetT, T* d_out, SelectOpT, T*& d_partition_out_buffer)
{
  d_partition_out_buffer = d_out;
}

template <typename T, typename OffsetT, typename UseDistinctPartitionT>
void partition(nvbench::state& state, nvbench::type_list<T, OffsetT, UseDistinctPartitionT>)
{
  using input_it_t                           = const T*;
  using flag_it_t                            = cub::NullType*;
  using num_selected_it_t                    = OffsetT*;
  using select_op_t                          = less_then_t<T>;
  using equality_op_t                        = cub::NullType;
  using offset_t                             = OffsetT;
  constexpr bool use_distinct_out_partitions = UseDistinctPartitionT::value;
  using output_it_t                          = typename ::cuda::std::
    conditional<use_distinct_out_partitions, cub::detail::partition_distinct_output_t<T*, T*>, T*>::type;

#if !TUNE_BASE
  using policy_t   = policy_hub_t<T>;
  using dispatch_t = cub::DispatchSelectIf<
    input_it_t,
    flag_it_t,
    output_it_t,
    num_selected_it_t,
    select_op_t,
    equality_op_t,
    offset_t,
    keep_rejects,
    may_alias,
    policy_t>;
#else // TUNE_BASE
  using dispatch_t = cub::DispatchSelectIf<
    input_it_t,
    flag_it_t,
    output_it_t,
    num_selected_it_t,
    select_op_t,
    equality_op_t,
    offset_t,
    keep_rejects,
    may_alias>;
#endif // !TUNE_BASE

  // Retrieve axis parameters
  const auto elements       = static_cast<std::size_t>(state.get_int64("Elements{io}"));
  const bit_entropy entropy = str_to_entropy(state.get_string("Entropy"));

  T val = value_from_entropy<T>(entropy_to_probability(entropy));
  select_op_t select_op{val};

  thrust::device_vector<T> in = generate(elements);
  thrust::device_vector<offset_t> num_selected(1);

  thrust::device_vector<T> out(elements);

  input_it_t d_in                  = thrust::raw_pointer_cast(in.data());
  flag_it_t d_flags                = nullptr;
  num_selected_it_t d_num_selected = thrust::raw_pointer_cast(num_selected.data());
  output_it_t d_out{};
  init_output_partition_buffer(in.cbegin(), elements, thrust::raw_pointer_cast(out.data()), select_op, d_out);

  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements);
  state.add_global_memory_writes<T>(elements);
  state.add_global_memory_writes<offset_t>(1);

  std::size_t temp_size{};
  dispatch_t::Dispatch(
    nullptr, temp_size, d_in, d_flags, d_out, d_num_selected, select_op, equality_op_t{}, elements, 0);

  thrust::device_vector<nvbench::uint8_t> temp(temp_size);
  auto* temp_storage = thrust::raw_pointer_cast(temp.data());

  state.exec(nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    dispatch_t::Dispatch(
      temp_storage,
      temp_size,
      d_in,
      d_flags,
      d_out,
      d_num_selected,
      select_op,
      equality_op_t{},
      elements,
      launch.get_stream());
  });
}

using distinct_partitions = nvbench::type_list<::cuda::std::false_type, ::cuda::std::true_type>;

NVBENCH_BENCH_TYPES(partition, NVBENCH_TYPE_AXES(fundamental_types, offset_types, distinct_partitions))
  .set_name("base")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}", "DistinctPartitions{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4))
  .add_string_axis("Entropy", {"1.000", "0.544", "0.000"});
