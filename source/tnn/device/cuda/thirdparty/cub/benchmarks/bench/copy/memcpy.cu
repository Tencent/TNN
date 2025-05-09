/******************************************************************************
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

#include <cub/device/device_copy.cuh>

// %RANGE% TUNE_THREADS tpb 128:1024:32
// %RANGE% TUNE_BUFFERS_PER_THREAD bpt 1:18:1
// %RANGE% TUNE_TLEV_BYTES_PER_THREAD tlevbpt 2:16:2
// %RANGE% TUNE_LARGE_THREADS ltpb 128:1024:32
// %RANGE% TUNE_LARGE_BUFFER_BYTES_PER_THREAD lbbpt 4:128:4
// %RANGE% TUNE_PREFER_POW2_BITS ppb 0:1:1
// %RANGE% TUNE_WARP_LEVEL_THRESHOLD wlt 32:512:32
// %RANGE% TUNE_BLOCK_LEVEL_THRESHOLD blt 1024:16384:512
// %RANGE% TUNE_BLOCK_MAGIC_NS blns 0:2048:4
// %RANGE% TUNE_BLOCK_DELAY_CONSTRUCTOR_ID bldcid 0:7:1
// %RANGE% TUNE_BLOCK_L2_WRITE_LATENCY_NS bll2w 0:1200:5
// %RANGE% TUNE_BUFF_MAGIC_NS buns 0:2048:4
// %RANGE% TUNE_BUFF_DELAY_CONSTRUCTOR_ID budcid 0:7:1
// %RANGE% TUNE_BUFF_L2_WRITE_LATENCY_NS bul2w 0:1200:5

#include <thrust/random.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <thrust/sequence.h>
#include <thrust/shuffle.h>
#include <thrust/tabulate.h>

#include <nvbench_helper.cuh>

template <class T, class OffsetT>
struct offset_to_ptr_t
{
  T* d_ptr;
  OffsetT* d_offsets;

  __device__ T* operator()(OffsetT i) const
  {
    return d_ptr + d_offsets[i];
  }
};

template <class T, class OffsetT>
struct reordered_offset_to_ptr_t
{
  T* d_ptr;
  OffsetT* d_map;
  OffsetT* d_offsets;

  __device__ T* operator()(OffsetT i) const
  {
    return d_ptr + d_offsets[d_map[i]];
  }
};

template <class T, class OffsetT>
struct offset_to_bytes_t
{
  OffsetT* d_offsets;

  __device__ OffsetT operator()(OffsetT i) const
  {
    return (d_offsets[i + 1] - d_offsets[i]) * sizeof(T);
  }
};

template <class T, class OffsetT>
struct offset_to_size_t
{
  OffsetT* d_offsets;

  __device__ OffsetT operator()(OffsetT i) const
  {
    return d_offsets[i + 1] - d_offsets[i];
  }
};

#if !TUNE_BASE
template <unsigned int MagicNs, unsigned int L2W, unsigned int DCID>
using delay_constructor_t =
  nvbench::tl::get<DCID,
                   nvbench::type_list<cub::detail::no_delay_constructor_t<L2W>,
                                      cub::detail::fixed_delay_constructor_t<MagicNs, L2W>,
                                      cub::detail::exponential_backoff_constructor_t<MagicNs, L2W>,
                                      cub::detail::exponential_backoff_jitter_constructor_t<MagicNs, L2W>,
                                      cub::detail::exponential_backoff_jitter_window_constructor_t<MagicNs, L2W>,
                                      cub::detail::exponential_backon_jitter_window_constructor_t<MagicNs, L2W>,
                                      cub::detail::exponential_backon_jitter_constructor_t<MagicNs, L2W>,
                                      cub::detail::exponential_backon_constructor_t<MagicNs, L2W>>>;

using buff_delay_constructor_t =
  delay_constructor_t<TUNE_BUFF_MAGIC_NS, TUNE_BUFF_L2_WRITE_LATENCY_NS, TUNE_BUFF_DELAY_CONSTRUCTOR_ID>;
using block_delay_constructor_t =
  delay_constructor_t<TUNE_BLOCK_MAGIC_NS, TUNE_BLOCK_L2_WRITE_LATENCY_NS, TUNE_BLOCK_DELAY_CONSTRUCTOR_ID>;

struct policy_hub_t
{
  struct policy_t : cub::ChainedPolicy<350, policy_t, policy_t>
  {
    using AgentSmallBufferPolicyT = cub::detail::AgentBatchMemcpyPolicy<
      TUNE_THREADS,
      TUNE_BUFFERS_PER_THREAD,
      TUNE_TLEV_BYTES_PER_THREAD,
      TUNE_PREFER_POW2_BITS,
      TUNE_LARGE_THREADS * TUNE_LARGE_BUFFER_BYTES_PER_THREAD,
      TUNE_WARP_LEVEL_THRESHOLD,
      TUNE_BLOCK_LEVEL_THRESHOLD,
      buff_delay_constructor_t,
      block_delay_constructor_t>;

    using AgentLargeBufferPolicyT =
      cub::detail::AgentBatchMemcpyLargeBuffersPolicy<TUNE_LARGE_THREADS, TUNE_LARGE_BUFFER_BYTES_PER_THREAD>;
  };

  using MaxPolicy = policy_t;
};
#endif

template <class T, class OffsetT>
void gen_it(T* d_buffer,
            thrust::device_vector<T*>& output,
            thrust::device_vector<OffsetT> offsets,
            bool randomize,
            thrust::default_random_engine& rne)
{
  OffsetT* d_offsets = thrust::raw_pointer_cast(offsets.data());

  if (randomize)
  {
    const auto buffers = output.size();
    thrust::device_vector<OffsetT> map(buffers);
    thrust::sequence(map.begin(), map.end());
    thrust::shuffle(map.begin(), map.end(), rne);
    thrust::device_vector<OffsetT> sizes(buffers);
    thrust::tabulate(sizes.begin(), sizes.end(), offset_to_size_t<T, OffsetT>{d_offsets});
    thrust::scatter(sizes.begin(), sizes.end(), map.begin(), offsets.begin());
    thrust::exclusive_scan(offsets.begin(), offsets.end(), offsets.begin());
    OffsetT* d_map = thrust::raw_pointer_cast(map.data());
    thrust::tabulate(output.begin(), output.end(), reordered_offset_to_ptr_t<T, OffsetT>{d_buffer, d_map, d_offsets});
  }
  else
  {
    thrust::tabulate(output.begin(), output.end(), offset_to_ptr_t<T, OffsetT>{d_buffer, d_offsets});
  }
}

template <class T, class OffsetT>
void copy(nvbench::state& state,
          nvbench::type_list<T, OffsetT>,
          std::size_t elements,
          std::size_t min_buffer_size,
          std::size_t max_buffer_size,
          bool randomize_input,
          bool randomize_output)
{
  using offset_t           = OffsetT;
  using it_t               = T*;
  using input_buffer_it_t  = it_t*;
  using output_buffer_it_t = it_t*;
  using buffer_size_it_t   = offset_t*;
  using buffer_offset_t    = std::uint32_t;
  using block_offset_t     = std::uint32_t;

  constexpr bool is_memcpy = true;

#if !TUNE_BASE
  using policy_t = policy_hub_t;
#else
  using policy_t = cub::detail::DeviceBatchMemcpyPolicy<buffer_offset_t, block_offset_t>;
#endif

  using dispatch_t = cub::detail::DispatchBatchMemcpy<
    input_buffer_it_t,
    output_buffer_it_t,
    buffer_size_it_t,
    buffer_offset_t,
    block_offset_t,
    policy_t,
    is_memcpy>;

  thrust::device_vector<T> input_buffer = generate(elements);
  thrust::device_vector<T> output_buffer(elements);
  thrust::device_vector<offset_t> offsets =
    generate.uniform.segment_offsets(elements, min_buffer_size, max_buffer_size);

  T* d_input_buffer   = thrust::raw_pointer_cast(input_buffer.data());
  T* d_output_buffer  = thrust::raw_pointer_cast(output_buffer.data());
  offset_t* d_offsets = thrust::raw_pointer_cast(offsets.data());

  const auto buffers = offsets.size() - 1;

  thrust::device_vector<it_t> input_buffers(buffers);
  thrust::device_vector<it_t> output_buffers(buffers);
  thrust::device_vector<offset_t> buffer_sizes(buffers);
  thrust::tabulate(buffer_sizes.begin(), buffer_sizes.end(), offset_to_bytes_t<T, offset_t>{d_offsets});

  thrust::default_random_engine rne;
  gen_it(d_input_buffer, input_buffers, offsets, randomize_input, rne);
  gen_it(d_output_buffer, output_buffers, offsets, randomize_output, rne);

  // Clear the offsets vector to free memory
  offsets.clear();
  offsets.shrink_to_fit();
  d_offsets = nullptr;

  input_buffer_it_t d_input_buffers   = thrust::raw_pointer_cast(input_buffers.data());
  output_buffer_it_t d_output_buffers = thrust::raw_pointer_cast(output_buffers.data());
  buffer_size_it_t d_buffer_sizes     = thrust::raw_pointer_cast(buffer_sizes.data());

  state.add_element_count(elements);
  state.add_global_memory_writes<T>(elements);
  state.add_global_memory_reads<T>(elements);
  state.add_global_memory_reads<it_t>(buffers);
  state.add_global_memory_reads<it_t>(buffers);
  state.add_global_memory_reads<offset_t>(buffers);

  std::size_t temp_storage_bytes{};
  std::uint8_t* d_temp_storage{};
  dispatch_t::Dispatch(
    d_temp_storage, temp_storage_bytes, d_input_buffers, d_output_buffers, d_buffer_sizes, buffers, 0);

  thrust::device_vector<nvbench::uint8_t> temp_storage(temp_storage_bytes);
  d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

  state.exec(nvbench::exec_tag::no_batch | nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    dispatch_t::Dispatch(
      d_temp_storage,
      temp_storage_bytes,
      d_input_buffers,
      d_output_buffers,
      d_buffer_sizes,
      buffers,
      launch.get_stream());
  });
}

template <class T, class OffsetT>
void uniform(nvbench::state& state, nvbench::type_list<T, OffsetT> tl)
{
  const auto elements              = static_cast<std::size_t>(state.get_int64("Elements{io}"));
  const auto max_buffer_size       = static_cast<std::size_t>(state.get_int64("MaxBufferSize"));
  const auto min_buffer_size_ratio = static_cast<std::size_t>(state.get_int64("MinBufferSizeRatio"));
  const auto min_buffer_size =
    static_cast<std::size_t>(static_cast<double>(max_buffer_size) / 100.0) * min_buffer_size_ratio;

  copy(
    state, tl, elements, min_buffer_size, max_buffer_size, state.get_int64("Randomize"), state.get_int64("Randomize"));
}

template <class T, class OffsetT>
void large(nvbench::state& state, nvbench::type_list<T, OffsetT> tl)
{
  const auto elements                  = static_cast<std::size_t>(state.get_int64("Elements{io}"));
  const auto max_buffer_size           = elements;
  constexpr auto min_buffer_size_ratio = 99;
  const auto min_buffer_size =
    static_cast<std::size_t>(static_cast<double>(max_buffer_size) / 100.0) * min_buffer_size_ratio;

  // No need to randomize large buffers
  constexpr bool randomize_input  = false;
  constexpr bool randomize_output = false;

  copy(state, tl, elements, min_buffer_size, max_buffer_size, randomize_input, randomize_output);
}

using types = nvbench::type_list<nvbench::uint8_t, nvbench::uint32_t>;

#ifdef TUNE_OffsetT
using u_offset_types = nvbench::type_list<TUNE_OffsetT>;
#else
using u_offset_types = nvbench::type_list<uint32_t, uint64_t>;
#endif

NVBENCH_BENCH_TYPES(uniform, NVBENCH_TYPE_AXES(types, u_offset_types))
  .set_name("uniform")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(25, 29, 2))
  .add_int64_axis("MinBufferSizeRatio", {1, 99})
  .add_int64_axis("MaxBufferSize", {8, 64, 256, 1024, 64 * 1024})
  .add_int64_axis("Randomize", {0, 1});

NVBENCH_BENCH_TYPES(large, NVBENCH_TYPE_AXES(types, u_offset_types))
  .set_name("large")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", {28, 29});
