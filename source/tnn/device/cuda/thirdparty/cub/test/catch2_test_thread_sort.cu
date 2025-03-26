/*******************************************************************************
 * Copyright (c) 2011-2021, NVIDIA CORPORATION.  All rights reserved.
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

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>
#include <thrust/sequence.h>
#include <thrust/shuffle.h>
#include <thrust/sort.h>

#include "cub/thread/thread_sort.cuh"
#include <c2h/catch2_test_helper.cuh>

struct CustomLess
{
  template <typename DataType>
  __host__ __device__ bool operator()(DataType& lhs, DataType& rhs)
  {
    return lhs < rhs;
  }
};

template <typename KeyT, typename ValueT, int ItemsPerThread>
__global__ void kernel(const KeyT* keys_in, KeyT* keys_out, const ValueT* values_in, ValueT* values_out)
{
  KeyT thread_keys[ItemsPerThread];
  KeyT thread_values[ItemsPerThread];

  const auto thread_offset = ItemsPerThread * threadIdx.x;
  keys_in += thread_offset;
  keys_out += thread_offset;
  values_in += thread_offset;
  values_out += thread_offset;

  for (int item = 0; item < ItemsPerThread; item++)
  {
    thread_keys[item]   = keys_in[item];
    thread_values[item] = values_in[item];
  }

  cub::StableOddEvenSort(thread_keys, thread_values, CustomLess{});

  for (int item = 0; item < ItemsPerThread; item++)
  {
    keys_out[item]   = thread_keys[item];
    values_out[item] = thread_values[item];
  }
}

using value_types           = c2h::type_list<std::uint32_t, std::uint64_t>;
using items_per_thread_list = c2h::enum_type_list<int, 2, 3, 4, 5, 7, 8, 9, 11>;

C2H_TEST("Test", "[thread_sort]", value_types, items_per_thread_list)
{
  using key_t                             = std::uint32_t;
  using value_t                           = c2h::get<0, TestType>;
  constexpr int items_per_thread          = c2h::get<1, TestType>::value;
  constexpr unsigned int threads_in_block = 1024;
  constexpr unsigned int elements         = threads_in_block * items_per_thread;

  thrust::default_random_engine re;
  c2h::device_vector<std::uint8_t> data_source(elements);

  for (int iteration = 0; iteration < 10; iteration++)
  {
    c2h::gen(C2H_SEED(2), data_source);
    c2h::device_vector<key_t> in_keys(data_source);
    c2h::device_vector<key_t> out_keys(elements);

    thrust::shuffle(data_source.begin(), data_source.end(), re);
    c2h::device_vector<value_t> in_values(data_source);
    c2h::device_vector<value_t> out_values(elements);

    c2h::host_vector<key_t> host_keys(in_keys);
    c2h::host_vector<value_t> host_values(in_values);

    kernel<key_t, value_t, items_per_thread><<<1, threads_in_block>>>(
      thrust::raw_pointer_cast(in_keys.data()),
      thrust::raw_pointer_cast(out_keys.data()),
      thrust::raw_pointer_cast(in_values.data()),
      thrust::raw_pointer_cast(out_values.data()));

    for (unsigned int tid = 0; tid < threads_in_block; tid++)
    {
      const auto thread_begin = tid * items_per_thread;
      const auto thread_end   = thread_begin + items_per_thread;

      thrust::sort_by_key(host_keys.begin() + thread_begin,
                          host_keys.begin() + thread_end,
                          host_values.begin() + thread_begin,
                          CustomLess{});
    }

    CHECK(host_keys == out_keys);
    CHECK(host_values == out_values);
  }
}
