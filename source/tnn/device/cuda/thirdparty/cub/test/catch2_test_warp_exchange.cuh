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

#pragma once

#include <cub/util_macro.cuh>
#include <cub/warp/warp_exchange.cuh>

#include <thrust/reverse.h>
#include <thrust/sequence.h>

#include <type_traits>

#include <c2h/catch2_test_helper.cuh>
#include <c2h/fill_striped.cuh>

template <typename InputT, typename OutputT, int ItemsPerThread, cub::WarpExchangeAlgorithm Alg, typename = void>
struct exchange_data_t;

template <typename InputT, typename OutputT, int ItemsPerThread, cub::WarpExchangeAlgorithm Alg>
struct exchange_data_t<InputT,
                       OutputT,
                       ItemsPerThread,
                       Alg,
                       typename std::enable_if<std::is_same<InputT, OutputT>::value>::type>
{
  InputT input[ItemsPerThread];
  OutputT (&output)[ItemsPerThread] = input;

  template <int LogicalWarpThreads>
  inline __device__ void
  scatter(cub::WarpExchange<InputT, ItemsPerThread, LogicalWarpThreads, 0, Alg>& exchange, int (&ranks)[ItemsPerThread])
  {
    exchange.ScatterToStriped(input, ranks);
  }
};

template <typename InputT, typename OutputT, int ItemsPerThread, cub::WarpExchangeAlgorithm Alg>
struct exchange_data_t<InputT,
                       OutputT,
                       ItemsPerThread,
                       Alg,
                       typename std::enable_if<!std::is_same<InputT, OutputT>::value>::type>
{
  InputT input[ItemsPerThread];
  OutputT output[ItemsPerThread];

  template <int LogicalWarpThreads>
  inline __device__ void
  scatter(cub::WarpExchange<InputT, ItemsPerThread, LogicalWarpThreads, 0, Alg>& exchange, int (&ranks)[ItemsPerThread])
  {
    exchange.ScatterToStriped(input, output, ranks);
  }
};

template <int LOGICAL_WARP_THREADS,
          int ITEMS_PER_THREAD,
          int TOTAL_WARPS,
          cub::WarpExchangeAlgorithm Alg,
          typename InputT,
          typename OutputT>
__global__ void scatter_kernel(const InputT* input_data, OutputT* output_data)
{
  using warp_exchange_t = cub::WarpExchange<InputT, ITEMS_PER_THREAD, LOGICAL_WARP_THREADS, 0, Alg>;
  using storage_t       = typename warp_exchange_t::TempStorage;

  constexpr int tile_size = ITEMS_PER_THREAD * LOGICAL_WARP_THREADS;
  __shared__ storage_t temp_storage[TOTAL_WARPS];

  const int tid = cub::RowMajorTid(blockDim.x, blockDim.y, blockDim.z);

  // Get warp index
  const int warp_id = tid / LOGICAL_WARP_THREADS;
  const int lane_id = tid % LOGICAL_WARP_THREADS;

  warp_exchange_t exchange(temp_storage[warp_id]);

  exchange_data_t<InputT, OutputT, ITEMS_PER_THREAD, Alg> exchange_data;

  // Reverse data
  int ranks[ITEMS_PER_THREAD];

  input_data += warp_id * tile_size;
  output_data += warp_id * tile_size;

  for (int item = 0; item < ITEMS_PER_THREAD; item++)
  {
    const auto item_idx       = lane_id * ITEMS_PER_THREAD + item;
    exchange_data.input[item] = input_data[item_idx];
    ranks[item]               = tile_size - 1 - item_idx;
  }

  exchange_data.scatter(exchange, ranks);

  // Striped to blocked
  for (int item = 0; item < ITEMS_PER_THREAD; item++)
  {
    output_data[item * LOGICAL_WARP_THREADS + lane_id] = exchange_data.output[item];
  }
}

template <int LOGICAL_WARP_THREADS,
          int ITEMS_PER_THREAD,
          int TOTAL_WARPS,
          cub::WarpExchangeAlgorithm Alg,
          typename InputT,
          typename OutputT>
void warp_scatter_strided(c2h::device_vector<InputT>& in, c2h::device_vector<OutputT>& out)
{
  scatter_kernel<LOGICAL_WARP_THREADS, ITEMS_PER_THREAD, TOTAL_WARPS, Alg, InputT, OutputT>
    <<<1, LOGICAL_WARP_THREADS * TOTAL_WARPS>>>(
      thrust::raw_pointer_cast(in.data()), thrust::raw_pointer_cast(out.data()));
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
}

template <int LOGICAL_WARP_THREADS,
          int ITEMS_PER_THREAD,
          int TOTAL_WARPS,
          cub::WarpExchangeAlgorithm Alg,
          typename InputT,
          typename OutputT,
          typename ActionT>
__global__ void kernel(const InputT* input_data, OutputT* output_data, ActionT action)
{
  using warp_exchange_t = cub::WarpExchange<InputT, ITEMS_PER_THREAD, LOGICAL_WARP_THREADS, 0, Alg>;
  using storage_t       = typename warp_exchange_t::TempStorage;

  constexpr int tile_size = ITEMS_PER_THREAD * LOGICAL_WARP_THREADS;
  __shared__ storage_t temp_storage[TOTAL_WARPS];

  const int tid = cub::RowMajorTid(blockDim.x, blockDim.y, blockDim.z);

  // Get warp index
  const int warp_id = tid / LOGICAL_WARP_THREADS;
  const int lane_id = tid % LOGICAL_WARP_THREADS;

  warp_exchange_t exchange(temp_storage[warp_id]);

  exchange_data_t<InputT, OutputT, ITEMS_PER_THREAD, Alg> exchange_data;

  input_data += warp_id * tile_size;
  output_data += warp_id * tile_size;

  for (int item = 0; item < ITEMS_PER_THREAD; item++)
  {
    exchange_data.input[item] = input_data[lane_id * ITEMS_PER_THREAD + item];
  }

  action(exchange_data.input, exchange_data.output, exchange);

  for (int item = 0; item < ITEMS_PER_THREAD; item++)
  {
    output_data[lane_id * ITEMS_PER_THREAD + item] = exchange_data.output[item];
  }
}

template <int LOGICAL_WARP_THREADS,
          int ITEMS_PER_THREAD,
          int TOTAL_WARPS,
          cub::WarpExchangeAlgorithm Alg,
          typename InputT,
          typename OutputT,
          typename ActionT>
void warp_exchange(c2h::device_vector<InputT>& in, c2h::device_vector<OutputT>& out, ActionT action)
{
  kernel<LOGICAL_WARP_THREADS, ITEMS_PER_THREAD, TOTAL_WARPS, Alg, InputT, OutputT, ActionT>
    <<<1, LOGICAL_WARP_THREADS * TOTAL_WARPS>>>(
      thrust::raw_pointer_cast(in.data()), thrust::raw_pointer_cast(out.data()), action);
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
}

struct blocked_to_striped
{
  template <typename InputT,
            typename OutputT,
            int LogicalWarpThreads,
            int ItemsPerThread,
            int ITEMS_PER_THREAD,
            cub::WarpExchangeAlgorithm Alg>
  __device__ void operator()(InputT (&input)[ITEMS_PER_THREAD],
                             OutputT (&output)[ITEMS_PER_THREAD],
                             cub::WarpExchange<InputT, ItemsPerThread, LogicalWarpThreads, 0, Alg>& exchange)
  {
    exchange.BlockedToStriped(input, output);
  }
};

struct striped_to_blocked
{
  template <typename InputT,
            typename OutputT,
            int LogicalWarpThreads,
            int ItemsPerThread,
            int ITEMS_PER_THREAD,
            cub::WarpExchangeAlgorithm Alg>
  __device__ void operator()(InputT (&input)[ITEMS_PER_THREAD],
                             OutputT (&output)[ITEMS_PER_THREAD],
                             cub::WarpExchange<InputT, ItemsPerThread, LogicalWarpThreads, 0, Alg>& exchange)
  {
    exchange.StripedToBlocked(input, output);
  }
};

template <typename T>
c2h::host_vector<T> compute_host_reference(const c2h::device_vector<T>& d_input, int tile_size)
{
  c2h::host_vector<T> input = d_input;

  int num_warps = CUB_QUOTIENT_CEILING(static_cast<int>(d_input.size()), tile_size);
  for (int warp_id = 0; warp_id < num_warps; warp_id++)
  {
    const int warp_data_begin = tile_size * warp_id;
    const int warp_data_end   = warp_data_begin + tile_size;
    thrust::reverse(input.begin() + warp_data_begin, input.begin() + warp_data_end);
  }
  return input;
}

template <int logical_warp_threads>
struct total_warps_t
{
private:
  static constexpr int max_warps      = 2;
  static constexpr bool is_arch_warp  = (logical_warp_threads == CUB_WARP_THREADS(0));
  static constexpr bool is_pow_of_two = ((logical_warp_threads & (logical_warp_threads - 1)) == 0);
  static constexpr int total_warps    = (is_arch_warp || is_pow_of_two) ? max_warps : 1;

public:
  static constexpr int value()
  {
    return total_warps;
  }
};
