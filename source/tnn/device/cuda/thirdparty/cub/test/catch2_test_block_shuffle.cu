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

/******************************************************************************
 * Test of BlockMergeSort utilities
 ******************************************************************************/

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <cub/block/block_shuffle.cuh>

#include <thrust/sort.h>

#include <algorithm>

#include <c2h/catch2_test_helper.cuh>

template <int BlockDimX, int BlockDimY, int BlockDimZ, int ItemsPerThread, class T, class ActionT>
__global__ void block_shuffle_kernel(T* data, ActionT action)
{
  using block_shuffle_t = cub::BlockShuffle<T, BlockDimX, BlockDimY, BlockDimZ>;
  using temp_storage_t  = typename block_shuffle_t::TempStorage;

  __shared__ temp_storage_t temp_storage;

  T thread_data[ItemsPerThread];

  data += cub::RowMajorTid(BlockDimX, BlockDimY, BlockDimZ) * ItemsPerThread;
  for (int item = 0; item < ItemsPerThread; item++)
  {
    thread_data[item] = data[item];
  }
  __syncthreads();

  block_shuffle_t block_shuffle(temp_storage);
  action(block_shuffle, thread_data);

  for (int item = 0; item < ItemsPerThread; item++)
  {
    data[item] = thread_data[item];
  }
}

struct up_op_t
{
  template <class BlockShuffleT, class T, int ItemsPerThread>
  __device__ void operator()(BlockShuffleT& block_shuffle, T (&thread_data)[ItemsPerThread]) const
  {
    block_shuffle.Up(thread_data, thread_data);
  }
};

struct offset_op_t
{
  int m_distance;

  __host__ offset_op_t(int distance)
      : m_distance(distance)
  {}

  template <class BlockShuffleT, class T, int ItemsPerThread>
  __device__ void operator()(BlockShuffleT& block_shuffle, T (&thread_data)[ItemsPerThread]) const
  {
    block_shuffle.Offset(thread_data[0], thread_data[0], m_distance);
  }
};

struct rotate_op_t
{
  unsigned int m_distance;

  __host__ rotate_op_t(unsigned int distance)
      : m_distance(distance)
  {}

  template <class BlockShuffleT, class T, int ItemsPerThread>
  __device__ void operator()(BlockShuffleT& block_shuffle, T (&thread_data)[ItemsPerThread]) const
  {
    block_shuffle.Rotate(thread_data[0], thread_data[0], m_distance);
  }
};

template <class T>
struct up_with_suffix_op_t
{
  int m_target_thread_id;
  T* m_d_suffix_ptr;

  __host__ up_with_suffix_op_t(int target_thread_id, T* d_suffix_ptr)
      : m_target_thread_id(target_thread_id)
      , m_d_suffix_ptr(d_suffix_ptr)
  {}

  template <class BlockShuffleT, int ItemsPerThread>
  __device__ void operator()(BlockShuffleT& block_shuffle, T (&thread_data)[ItemsPerThread]) const
  {
    T suffix{};

    block_shuffle.Up(thread_data, thread_data, suffix);

    if (cub::RowMajorTid(blockDim.x, blockDim.y, blockDim.z) == m_target_thread_id)
    {
      m_d_suffix_ptr[0] = suffix;
    }
  }
};

struct down_op_t
{
  template <class BlockShuffleT, class T, int ItemsPerThread>
  __device__ void operator()(BlockShuffleT& block_shuffle, T (&thread_data)[ItemsPerThread]) const
  {
    block_shuffle.Down(thread_data, thread_data);
  }
};

template <class T>
struct down_with_prefix_op_t
{
  int m_target_thread_id;
  T* m_d_prefix_ptr;

  __host__ down_with_prefix_op_t(int target_thread_id, T* d_prefix_ptr)
      : m_target_thread_id(target_thread_id)
      , m_d_prefix_ptr(d_prefix_ptr)
  {}

  template <class BlockShuffleT, int ItemsPerThread>
  __device__ void operator()(BlockShuffleT& block_shuffle, T (&thread_data)[ItemsPerThread]) const
  {
    T prefix{};

    block_shuffle.Down(thread_data, thread_data, prefix);

    if (cub::RowMajorTid(blockDim.x, blockDim.y, blockDim.z) == m_target_thread_id)
    {
      m_d_prefix_ptr[0] = prefix;
    }
  }
};

template <int ItemsPerThread, int BlockDimX, int BlockDimY, int BlockDimZ, class T, class ActionT>
void block_shuffle(c2h::device_vector<T>& data, ActionT action)
{
  dim3 block(BlockDimX, BlockDimY, BlockDimZ);
  block_shuffle_kernel<BlockDimX, BlockDimY, BlockDimZ, ItemsPerThread>
    <<<1, block>>>(thrust::raw_pointer_cast(data.data()), action);

  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
}

// %PARAM% MULTI_DIM mdim 0:1
// %PARAM% DIM_IDX dim_idx 0:1:2

#if MULTI_DIM
using block_dim_xs = c2h::enum_type_list<int, 7, 32, 64>;
using block_dim_yz = c2h::enum_type_list<int, 2>;
#else
using block_dim_xs = c2h::enum_type_list<int, 64, 512, 1024>;
using block_dim_yz = c2h::enum_type_list<int, 1>;
#endif

using block_dim_x = c2h::enum_type_list<int, c2h::get<DIM_IDX, block_dim_xs>::value>;

using types                  = c2h::type_list<std::int32_t, std::int64_t>;
using items_per_thread       = c2h::enum_type_list<int, 1, 2, 15>;
using single_item_per_thread = c2h::enum_type_list<int, 1>;

template <class TestType>
struct params_t
{
  using type = typename c2h::get<0, TestType>;

  static constexpr int items_per_thread = c2h::get<1, TestType>::value;
  static constexpr int block_dim_x      = c2h::get<2, TestType>::value;
  static constexpr int block_dim_y      = c2h::get<3, TestType>::value;
  static constexpr int block_dim_z      = block_dim_y;
  static constexpr int threads_in_block = block_dim_x * block_dim_y * block_dim_z;
  static constexpr int tile_size        = items_per_thread * threads_in_block;
};

C2H_TEST("Block shuffle offset works", "[shuffle][block]", types, single_item_per_thread, block_dim_x, block_dim_yz)
{
  using params = params_t<TestType>;
  using type   = typename params::type;

  c2h::device_vector<type> d_data(params::tile_size);
  c2h::gen(C2H_SEED(10), d_data);

  const int distance = GENERATE_COPY(take(4, random(1 - params::tile_size, params::tile_size - 1)));

  c2h::host_vector<type> h_data = d_data;
  c2h::host_vector<type> h_ref(params::tile_size);

  for (int i = 0; i < static_cast<int>(h_data.size()); i++)
  {
    const int source = i + distance;
    h_ref[i]         = (source >= 0) && (source < params::tile_size) ? h_data[source] : h_data[i];
  }

  block_shuffle<params::items_per_thread, params::block_dim_x, params::block_dim_y, params::block_dim_z>(
    d_data, offset_op_t{distance});

  REQUIRE(h_ref == d_data);
}

C2H_TEST("Block shuffle rotate works", "[shuffle][block]", types, single_item_per_thread, block_dim_x, block_dim_yz)
{
  using params = params_t<TestType>;
  using type   = typename params::type;

  c2h::device_vector<type> d_data(params::tile_size);
  c2h::gen(C2H_SEED(10), d_data);

  c2h::device_vector<type> d_ref = d_data;

  const unsigned int distance = GENERATE_COPY(take(4, random(0, params::tile_size - 1)));

  c2h::host_vector<type> h_ref = d_data;
  std::rotate(h_ref.begin(), h_ref.begin() + distance, h_ref.end());

  block_shuffle<params::items_per_thread, params::block_dim_x, params::block_dim_y, params::block_dim_z>(
    d_data, rotate_op_t{distance});

  REQUIRE(h_ref == d_data);
}

C2H_TEST("Block shuffle up works", "[shuffle][block]", types, items_per_thread, block_dim_x, block_dim_yz)
{
  using params = params_t<TestType>;
  using type   = typename params::type;

  c2h::device_vector<type> d_data(params::tile_size);
  c2h::gen(C2H_SEED(10), d_data);

  c2h::device_vector<type> d_ref(params::tile_size);
  thrust::copy(d_data.begin(), d_data.end() - 1, d_ref.begin() + 1);
  thrust::copy(d_data.begin(), d_data.begin() + 1, d_ref.begin());

  block_shuffle<params::items_per_thread, params::block_dim_x, params::block_dim_y, params::block_dim_z>(
    d_data, up_op_t{});

  REQUIRE(d_ref == d_data);
}

C2H_TEST("Block shuffle up works when suffix is required",
         "[shuffle][block]",
         types,
         items_per_thread,
         block_dim_x,
         block_dim_yz)
{
  using params = params_t<TestType>;
  using type   = typename params::type;

  c2h::device_vector<type> d_data(params::tile_size);
  c2h::gen(C2H_SEED(10), d_data);

  const int target_thread_id = GENERATE_COPY(take(2, random(0, params::threads_in_block - 1)));

  c2h::device_vector<type> d_ref(params::tile_size);
  thrust::copy(d_data.begin(), d_data.end() - 1, d_ref.begin() + 1);
  thrust::copy(d_data.begin(), d_data.begin() + 1, d_ref.begin());

  c2h::device_vector<type> d_suffix(1);
  c2h::device_vector<type> d_suffix_ref(1);
  thrust::copy(d_data.end() - 1, d_data.end(), d_suffix_ref.begin());

  block_shuffle<params::items_per_thread, params::block_dim_x, params::block_dim_y, params::block_dim_z>(
    d_data, up_with_suffix_op_t<type>{target_thread_id, thrust::raw_pointer_cast(d_suffix.data())});

  REQUIRE(d_ref == d_data);
  REQUIRE(d_suffix_ref == d_suffix);
}

C2H_TEST("Block shuffle down works", "[shuffle][block]", types, items_per_thread, block_dim_x, block_dim_yz)
{
  using params = params_t<TestType>;
  using type   = typename params::type;

  c2h::device_vector<type> d_data(params::tile_size);
  c2h::gen(C2H_SEED(10), d_data);

  c2h::device_vector<type> d_ref(params::tile_size);
  thrust::copy(d_data.begin() + 1, d_data.end(), d_ref.begin());
  thrust::copy(d_data.end() - 1, d_data.end(), d_ref.end() - 1);

  block_shuffle<params::items_per_thread, params::block_dim_x, params::block_dim_y, params::block_dim_z>(
    d_data, down_op_t{});

  REQUIRE(d_ref == d_data);
}

C2H_TEST("Block shuffle down works when prefix is required",
         "[shuffle][block]",
         types,
         items_per_thread,
         block_dim_x,
         block_dim_yz)
{
  using params = params_t<TestType>;
  using type   = typename params::type;

  c2h::device_vector<type> d_data(params::tile_size);
  c2h::gen(C2H_SEED(10), d_data);

  const int target_thread_id = GENERATE_COPY(take(2, random(0, params::threads_in_block - 1)));

  c2h::device_vector<type> d_ref(params::tile_size);
  thrust::copy(d_data.begin() + 1, d_data.end(), d_ref.begin());
  thrust::copy(d_data.end() - 1, d_data.end(), d_ref.end() - 1);

  c2h::device_vector<type> d_prefix(1);
  c2h::device_vector<type> d_prefix_ref(1);
  thrust::copy(d_data.begin(), d_data.begin() + 1, d_prefix_ref.begin());

  block_shuffle<params::items_per_thread, params::block_dim_x, params::block_dim_y, params::block_dim_z>(
    d_data, down_with_prefix_op_t<type>{target_thread_id, thrust::raw_pointer_cast(d_prefix.data())});

  REQUIRE(d_ref == d_data);
  REQUIRE(d_prefix_ref == d_prefix);
}
