/******************************************************************************
 * Copyright (c) 2011-2022, NVIDIA CORPORATION.  All rights reserved.
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

#include <cub/block/block_radix_sort.cuh>

#include "catch2_radix_sort_helper.cuh"
#include <c2h/catch2_test_helper.cuh>

template <typename InputIteratorT,
          typename OutputIteratorT,
          typename ActionT,
          int ItemsPerThread,
          int ThreadsInBlock,
          int RadixBits,
          bool Memoize,
          cub::BlockScanAlgorithm Algorithm,
          cudaSharedMemConfig ShmemConfig>
__global__ void
kernel(ActionT action, InputIteratorT input, OutputIteratorT output, int begin_bit, int end_bit, bool striped)
{
  using key_t = cub::detail::value_t<InputIteratorT>;
  using block_radix_sort_t =
    cub::BlockRadixSort<key_t, ThreadsInBlock, ItemsPerThread, cub::NullType, RadixBits, Memoize, Algorithm, ShmemConfig>;

  using storage_t = typename block_radix_sort_t::TempStorage;

  __shared__ storage_t storage;

  key_t keys[ItemsPerThread];

  for (int i = 0; i < ItemsPerThread; i++)
  {
    keys[i] = input[threadIdx.x * ItemsPerThread + i];
  }

  block_radix_sort_t block_radix_sort(storage);

  if (striped)
  {
    action(block_radix_sort, keys, begin_bit, end_bit, cub::Int2Type<1>{});

    for (int i = 0; i < ItemsPerThread; i++)
    {
      output[threadIdx.x + ThreadsInBlock * i] = keys[i];
    }
  }
  else
  {
    action(block_radix_sort, keys, begin_bit, end_bit, cub::Int2Type<0>{});

    for (int i = 0; i < ItemsPerThread; i++)
    {
      output[threadIdx.x * ItemsPerThread + i] = keys[i];
    }
  }
}

template <int ItemsPerThread,
          int ThreadsInBlock,
          int RadixBits,
          bool Memoize,
          cub::BlockScanAlgorithm Algorithm,
          cudaSharedMemConfig ShmemConfig,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename ActionT>
void block_radix_sort(
  ActionT action, InputIteratorT input, OutputIteratorT output, int begin_bit, int end_bit, bool striped)
{
  kernel<InputIteratorT, OutputIteratorT, ActionT, ItemsPerThread, ThreadsInBlock, RadixBits, Memoize, Algorithm, ShmemConfig>
    <<<1, ThreadsInBlock>>>(action, input, output, begin_bit, end_bit, striped);

  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
}

template <typename InputKeyIteratorT,
          typename InputValueIteratorT,
          typename OutputKeyIteratorT,
          typename OutputValueIteratorT,
          typename ActionT,
          int ItemsPerThread,
          int ThreadsInBlock,
          int RadixBits,
          bool Memoize,
          cub::BlockScanAlgorithm Algorithm,
          cudaSharedMemConfig ShmemConfig>
__global__ void kernel(
  ActionT action,
  InputKeyIteratorT input_keys,
  InputValueIteratorT input_values,
  OutputKeyIteratorT output_keys,
  OutputValueIteratorT output_values,
  int begin_bit,
  int end_bit,
  bool striped)
{
  using key_t   = cub::detail::value_t<InputKeyIteratorT>;
  using value_t = cub::detail::value_t<InputValueIteratorT>;
  using block_radix_sort_t =
    cub::BlockRadixSort<key_t, ThreadsInBlock, ItemsPerThread, value_t, RadixBits, Memoize, Algorithm, ShmemConfig>;

  using storage_t = typename block_radix_sort_t::TempStorage;
  __shared__ storage_t storage;

  key_t keys[ItemsPerThread];
  value_t values[ItemsPerThread];

  for (int i = 0; i < ItemsPerThread; i++)
  {
    keys[i]   = input_keys[threadIdx.x * ItemsPerThread + i];
    values[i] = input_values[threadIdx.x * ItemsPerThread + i];
  }

  block_radix_sort_t block_radix_sort(storage);

  if (striped)
  {
    action(block_radix_sort, keys, values, begin_bit, end_bit, cub::Int2Type<1>{});

    for (int i = 0; i < ItemsPerThread; i++)
    {
      output_keys[threadIdx.x + ThreadsInBlock * i]   = keys[i];
      output_values[threadIdx.x + ThreadsInBlock * i] = values[i];
    }
  }
  else
  {
    action(block_radix_sort, keys, values, begin_bit, end_bit, cub::Int2Type<0>{});

    for (int i = 0; i < ItemsPerThread; i++)
    {
      output_keys[threadIdx.x * ItemsPerThread + i]   = keys[i];
      output_values[threadIdx.x * ItemsPerThread + i] = values[i];
    }
  }
}

template <int ItemsPerThread,
          int ThreadsInBlock,
          int RadixBits,
          bool Memoize,
          cub::BlockScanAlgorithm Algorithm,
          cudaSharedMemConfig ShmemConfig,
          typename InputKeyIteratorT,
          typename InputValueIteratorT,
          typename OutputKeyIteratorT,
          typename OutputValueIteratorT,
          typename ActionT>
void block_radix_sort(
  ActionT action,
  InputKeyIteratorT input_keys,
  InputValueIteratorT input_values,
  OutputKeyIteratorT output_keys,
  OutputValueIteratorT output_values,
  int begin_bit,
  int end_bit,
  bool striped)
{
  kernel<InputKeyIteratorT,
         InputValueIteratorT,
         OutputKeyIteratorT,
         OutputValueIteratorT,
         ActionT,
         ItemsPerThread,
         ThreadsInBlock,
         RadixBits,
         Memoize,
         Algorithm,
         ShmemConfig>
    <<<1, ThreadsInBlock>>>(action, input_keys, input_values, output_keys, output_values, begin_bit, end_bit, striped);

  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
}

struct sort_op_t
{
  template <class BlockRadixSortT, class KeysT>
  __device__ void
  operator()(BlockRadixSortT& block_radix_sort, KeysT& keys, int begin_bit, int end_bit, cub::Int2Type<0> /* striped */)
  {
    block_radix_sort.Sort(keys, begin_bit, end_bit);
  }

  template <class BlockRadixSortT, class KeysT>
  __device__ void
  operator()(BlockRadixSortT& block_radix_sort, KeysT& keys, int begin_bit, int end_bit, cub::Int2Type<1> /* striped */)
  {
    block_radix_sort.SortBlockedToStriped(keys, begin_bit, end_bit);
  }
};

struct descending_sort_op_t
{
  template <class BlockRadixSortT, class KeysT>
  __device__ void
  operator()(BlockRadixSortT& block_radix_sort, KeysT& keys, int begin_bit, int end_bit, cub::Int2Type<0> /* striped */)
  {
    block_radix_sort.SortDescending(keys, begin_bit, end_bit);
  }

  template <class BlockRadixSortT, class KeysT>
  __device__ void
  operator()(BlockRadixSortT& block_radix_sort, KeysT& keys, int begin_bit, int end_bit, cub::Int2Type<1> /* striped */)
  {
    block_radix_sort.SortDescendingBlockedToStriped(keys, begin_bit, end_bit);
  }
};

struct sort_pairs_op_t
{
  template <class BlockRadixSortT, class KeysT, class ValuesT>
  __device__ void operator()(
    BlockRadixSortT& block_radix_sort,
    KeysT& keys,
    ValuesT& values,
    int begin_bit,
    int end_bit,
    cub::Int2Type<0> /* striped */)
  {
    block_radix_sort.Sort(keys, values, begin_bit, end_bit);
  }

  template <class BlockRadixSortT, class KeysT, class ValuesT>
  __device__ void operator()(
    BlockRadixSortT& block_radix_sort,
    KeysT& keys,
    ValuesT& values,
    int begin_bit,
    int end_bit,
    cub::Int2Type<1> /* striped */)
  {
    block_radix_sort.SortBlockedToStriped(keys, values, begin_bit, end_bit);
  }
};

struct descending_sort_pairs_op_t
{
  template <class BlockRadixSortT, class KeysT, class ValuesT>
  __device__ void operator()(
    BlockRadixSortT& block_radix_sort,
    KeysT& keys,
    ValuesT& values,
    int begin_bit,
    int end_bit,
    cub::Int2Type<0> /* striped */)
  {
    block_radix_sort.SortDescending(keys, values, begin_bit, end_bit);
  }

  template <class BlockRadixSortT, class KeysT, class ValuesT>
  __device__ void operator()(
    BlockRadixSortT& block_radix_sort,
    KeysT& keys,
    ValuesT& values,
    int begin_bit,
    int end_bit,
    cub::Int2Type<1> /* striped */)
  {
    block_radix_sort.SortDescendingBlockedToStriped(keys, values, begin_bit, end_bit);
  }
};
