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

#include <cub/device/device_scan.cuh>

#include <cuda/std/__functional/invoke.h>

#include <look_back_helper.cuh>

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
#  else // TUNE_LOAD == 1
#    define TUNE_LOAD_MODIFIER cub::LOAD_CA
#  endif // TUNE_LOAD

template <typename AccumT>
struct policy_hub_t
{
  template <int NOMINAL_BLOCK_THREADS_4B,
            int NOMINAL_ITEMS_PER_THREAD_4B,
            typename ComputeT,
            cub::BlockLoadAlgorithm LOAD_ALGORITHM,
            cub::CacheLoadModifier LOAD_MODIFIER,
            cub::BlockStoreAlgorithm STORE_ALGORITHM,
            cub::BlockScanAlgorithm SCAN_ALGORITHM>
  using agent_policy_t = cub::AgentScanPolicy<
    NOMINAL_BLOCK_THREADS_4B,
    NOMINAL_ITEMS_PER_THREAD_4B,
    ComputeT,
    LOAD_ALGORITHM,
    LOAD_MODIFIER,
    STORE_ALGORITHM,
    SCAN_ALGORITHM,
    cub::MemBoundScaling<NOMINAL_BLOCK_THREADS_4B, NOMINAL_ITEMS_PER_THREAD_4B, ComputeT>,
    delay_constructor_t>;

  struct policy_t : cub::ChainedPolicy<300, policy_t, policy_t>
  {
    using ScanPolicyT =
      agent_policy_t<TUNE_THREADS,
                     TUNE_ITEMS,
                     AccumT,
                     TUNE_LOAD_ALGORITHM,
                     TUNE_LOAD_MODIFIER,
                     TUNE_STORE_ALGORITHM,
                     cub::BLOCK_SCAN_WARP_SCANS>;
  };

  using MaxPolicy = policy_t;
};
#endif // TUNE_BASE

template <typename T, typename OffsetT>
static void basic(nvbench::state& state, nvbench::type_list<T, OffsetT>)
{
  using init_t      = cub::detail::InputValue<T>;
  using accum_t     = ::cuda::std::__accumulator_t<op_t, T, T>;
  using input_it_t  = const T*;
  using output_it_t = T*;
  using offset_t    = OffsetT;

#if !TUNE_BASE
  using policy_t   = policy_hub_t<accum_t>;
  using dispatch_t = cub::DispatchScan<input_it_t, output_it_t, op_t, init_t, offset_t, accum_t, policy_t>;
#else
  using dispatch_t = cub::DispatchScan<input_it_t, output_it_t, op_t, init_t, offset_t, accum_t>;
#endif

  const auto elements = static_cast<std::size_t>(state.get_int64("Elements{io}"));

  thrust::device_vector<T> input = generate(elements);
  thrust::device_vector<T> output(elements);

  T* d_input  = thrust::raw_pointer_cast(input.data());
  T* d_output = thrust::raw_pointer_cast(output.data());

  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements, "Size");
  state.add_global_memory_writes<T>(elements);

  size_t tmp_size;
  dispatch_t::Dispatch(
    nullptr, tmp_size, d_input, d_output, op_t{}, init_t{T{}}, static_cast<int>(input.size()), 0 /* stream */);

  thrust::device_vector<nvbench::uint8_t> tmp(tmp_size);
  nvbench::uint8_t* d_tmp = thrust::raw_pointer_cast(tmp.data());

  state.exec(nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    dispatch_t::Dispatch(
      thrust::raw_pointer_cast(tmp.data()),
      tmp_size,
      d_input,
      d_output,
      op_t{},
      init_t{T{}},
      static_cast<int>(input.size()),
      launch.get_stream());
  });
}

using some_offset_types = nvbench::type_list<nvbench::uint32_t, nvbench::uint64_t>;

NVBENCH_BENCH_TYPES(basic, NVBENCH_TYPE_AXES(all_types, some_offset_types))
  .set_name("base")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4));
