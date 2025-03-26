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

#include <cub/block/block_adjacent_difference.cuh>

#include <c2h/catch2_test_helper.cuh>

template <int ThreadsInBlock, int ItemsPerThread, class T, class ActionT>
__global__ void block_adj_diff_kernel(T* data, ActionT action, bool in_place)
{
  using block_adjacent_differencet_t = cub::BlockAdjacentDifference<T, ThreadsInBlock>;
  using temp_storage_t               = typename block_adjacent_differencet_t::TempStorage;

  __shared__ temp_storage_t temp_storage;

  T thread_in[ItemsPerThread];
  T thread_out[ItemsPerThread];

  const int thread_offset = static_cast<int>(threadIdx.x) * ItemsPerThread;

  for (int item = 0; item < ItemsPerThread; item++)
  {
    thread_in[item] = data[thread_offset + item];
  }
  __syncthreads();

  block_adjacent_differencet_t adj_diff(temp_storage);

  if (in_place)
  {
    action(adj_diff, thread_in, thread_in);

    for (unsigned int item = 0; item < ItemsPerThread; item++)
    {
      data[thread_offset + item] = thread_in[item];
    }
  }
  else
  {
    action(adj_diff, thread_in, thread_out);

    for (unsigned int item = 0; item < ItemsPerThread; item++)
    {
      data[thread_offset + item] = thread_out[item];
    }
  }
}

template <class T>
struct custom_difference_t
{
  __host__ __device__ T operator()(const T& lhs, const T& rhs)
  {
    return lhs - rhs;
  }
};

template <bool ReadLeft>
struct base_op_t
{
  template <int ItemsPerThread, typename T, typename BlockAdjDiff>
  __device__ void operator()(BlockAdjDiff& block_adj_diff, T (&input)[ItemsPerThread], T (&output)[ItemsPerThread]) const
  {
    if (ReadLeft)
    {
      block_adj_diff.SubtractLeft(input, output, custom_difference_t<T>{});
    }
    else
    {
      block_adj_diff.SubtractRight(input, output, custom_difference_t<T>{});
    }
  }
};

template <bool ReadLeft>
struct last_tile_op_t
{
  int m_valid_items{};

  __host__ last_tile_op_t(int valid_items)
      : m_valid_items(valid_items)
  {}

  template <int ITEMS_PER_THREAD, typename T, typename BlockAdjDiff>
  __device__ void
  operator()(BlockAdjDiff& block_adj_diff, T (&input)[ITEMS_PER_THREAD], T (&output)[ITEMS_PER_THREAD]) const
  {
    custom_difference_t<T> diff{};

    if (ReadLeft)
    {
      block_adj_diff.SubtractLeftPartialTile(input, output, diff, m_valid_items);
    }
    else
    {
      block_adj_diff.SubtractRightPartialTile(input, output, diff, m_valid_items);
    }
  }
};

template <class T, bool ReadLeft>
struct middle_tile_op_t
{
  T m_neighbour_tile_value;

  __host__ middle_tile_op_t(T neighbour_tile_value)
      : m_neighbour_tile_value(neighbour_tile_value)
  {}

  template <int ITEMS_PER_THREAD, typename BlockAdjDiff>
  __device__ void
  operator()(BlockAdjDiff& block_adj_diff, T (&input)[ITEMS_PER_THREAD], T (&output)[ITEMS_PER_THREAD]) const
  {
    custom_difference_t<T> diff{};

    if (ReadLeft)
    {
      block_adj_diff.SubtractLeft(input, output, diff, m_neighbour_tile_value);
    }
    else
    {
      block_adj_diff.SubtractRight(input, output, diff, m_neighbour_tile_value);
    }
  }
};

template <typename T>
struct last_tile_with_pred_op_t
{
  int m_valid_items;
  T m_neighbour_tile_value;

  __host__ last_tile_with_pred_op_t(int valid_items, T neighbour_tile_value)
      : m_valid_items(valid_items)
      , m_neighbour_tile_value(neighbour_tile_value)
  {}

  template <int ITEMS_PER_THREAD, typename BlockAdjDiff>
  __device__ void
  operator()(BlockAdjDiff& block_adj_diff, T (&input)[ITEMS_PER_THREAD], T (&output)[ITEMS_PER_THREAD]) const
  {
    custom_difference_t<T> diff{};
    block_adj_diff.SubtractLeftPartialTile(input, output, diff, m_valid_items, m_neighbour_tile_value);
  }
};

template <int ItemsPerThread, int ThreadsInBlock, class T, class ActionT>
void block_adj_diff(c2h::device_vector<T>& data, bool in_place, ActionT action)
{
  block_adj_diff_kernel<ThreadsInBlock, ItemsPerThread, T, ActionT>
    <<<1, ThreadsInBlock>>>(thrust::raw_pointer_cast(data.data()), action, in_place);

  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
}

template <bool ReadLeft, class T>
void host_adj_diff(c2h::host_vector<T>& h_data, int valid_items)
{
  custom_difference_t<T> diff{};

  if (ReadLeft)
  {
    for (int i = valid_items - 1; i > 0; i--)
    {
      h_data[i] = diff(h_data[i], h_data[i - 1]);
    }
  }
  else
  {
    for (int i = 0; i < valid_items - 1; i++)
    {
      h_data[i] = diff(h_data[i], h_data[i + 1]);
    }
  }
}

template <bool ReadLeft, class T>
void host_adj_diff(c2h::host_vector<T>& h_data, int valid_items, T neighbour_value)
{
  custom_difference_t<T> diff{};

  host_adj_diff<ReadLeft>(h_data, valid_items);

  if (valid_items == 0)
  {
    return;
  }

  if (ReadLeft)
  {
    h_data[0] = diff(h_data[0], neighbour_value);
  }
  else
  {
    h_data[valid_items - 1] = diff(h_data[valid_items - 1], neighbour_value);
  }
}

// %PARAM% THREADS_IN_BLOCK bs 64:256

using key_types = c2h::type_list<std::uint16_t, std::int32_t, std::int64_t>;

using threads_in_block = c2h::enum_type_list<int, THREADS_IN_BLOCK>;
using items_per_thread = c2h::enum_type_list<int, 1, 2, 10, 15>;
using directions       = c2h::enum_type_list<bool, false, true>;
using left_only        = c2h::enum_type_list<bool, true>;

template <class TestType>
struct params_t
{
  using key_t = typename c2h::get<0, TestType>;

  static constexpr int items_per_thread = c2h::get<1, TestType>::value;
  static constexpr int threads_in_block = c2h::get<2, TestType>::value;
  static constexpr int tile_size        = items_per_thread * threads_in_block;
  static constexpr bool read_left       = c2h::get<3, TestType>::value;
};

C2H_TEST("Block adjacent difference works with full tiles",
         "[adjacent difference][block]",
         key_types,
         items_per_thread,
         threads_in_block,
         directions)
{
  using params = params_t<TestType>;
  using key_t  = typename params::key_t;

  c2h::device_vector<key_t> d_data(params::tile_size);
  c2h::gen(C2H_SEED(10), d_data);

  const bool in_place = GENERATE(false, true);

  c2h::host_vector<key_t> h_data = d_data;
  host_adj_diff<params::read_left>(h_data, params::tile_size);

  block_adj_diff<params::items_per_thread, params::threads_in_block>(d_data, in_place, base_op_t<params::read_left>{});

  REQUIRE(h_data == d_data);
}

C2H_TEST("Block adjacent difference works with last tiles",
         "[adjacent difference][block]",
         key_types,
         items_per_thread,
         threads_in_block,
         directions)
{
  using params = params_t<TestType>;
  using key_t  = typename params::key_t;

  c2h::device_vector<key_t> d_data(params::tile_size);
  c2h::gen(C2H_SEED(10), d_data);

  const bool in_place   = GENERATE(false, true);
  const int valid_items = GENERATE_COPY(take(10, random(0, params::tile_size)));

  c2h::host_vector<key_t> h_data = d_data;
  host_adj_diff<params::read_left>(h_data, valid_items);

  block_adj_diff<params::items_per_thread, params::threads_in_block>(
    d_data, in_place, last_tile_op_t<params::read_left>{valid_items});

  REQUIRE(h_data == d_data);
}

C2H_TEST("Block adjacent difference works with single tiles",
         "[adjacent difference][block]",
         key_types,
         items_per_thread,
         threads_in_block,
         left_only)
{
  using params = params_t<TestType>;
  using key_t  = typename params::key_t;

  c2h::device_vector<key_t> d_data(params::tile_size);
  c2h::gen(C2H_SEED(10), d_data);

  const bool in_place      = GENERATE(false, true);
  const int valid_items    = GENERATE_COPY(take(10, random(0, params::tile_size)));
  constexpr bool read_left = true;

  c2h::host_vector<key_t> h_data = d_data;
  key_t neighbour_value          = h_data[h_data.size() / 2];

  host_adj_diff<read_left>(h_data, valid_items, neighbour_value);

  block_adj_diff<params::items_per_thread, params::threads_in_block>(
    d_data, in_place, last_tile_with_pred_op_t<key_t>{valid_items, neighbour_value});

  REQUIRE(h_data == d_data);
}

C2H_TEST("Block adjacent difference works with middle tiles",
         "[adjacent difference][block]",
         key_types,
         items_per_thread,
         threads_in_block,
         directions)
{
  using params = params_t<TestType>;
  using key_t  = typename params::key_t;

  c2h::device_vector<key_t> d_data(params::tile_size);
  c2h::gen(C2H_SEED(10), d_data);

  const bool in_place = GENERATE(false, true);

  c2h::host_vector<key_t> h_data = d_data;
  key_t neighbour_value          = h_data[h_data.size() / 2];

  host_adj_diff<params::read_left>(h_data, params::tile_size, neighbour_value);

  block_adj_diff<params::items_per_thread, params::threads_in_block>(
    d_data, in_place, middle_tile_op_t<key_t, params::read_left>{neighbour_value});

  REQUIRE(h_data == d_data);
}

C2H_TEST("Block adjacent difference supports custom types", "[adjacent difference][block]", threads_in_block)
{
  using key_t = c2h::custom_type_t<c2h::equal_comparable_t, c2h::subtractable_t>;

  constexpr int items_per_thread = 2;
  constexpr int threads_in_block = c2h::get<0, TestType>::value;
  constexpr int tile_size        = threads_in_block * items_per_thread;
  constexpr bool read_left       = true;
  constexpr bool in_place        = true;

  c2h::device_vector<key_t> d_data(tile_size);
  c2h::gen(C2H_SEED(10), d_data);

  c2h::host_vector<key_t> h_data = d_data;
  host_adj_diff<read_left>(h_data, tile_size);

  block_adj_diff<items_per_thread, threads_in_block>(d_data, in_place, base_op_t<read_left>{});

  REQUIRE(h_data == d_data);
}

// TODO Test different input/output types
