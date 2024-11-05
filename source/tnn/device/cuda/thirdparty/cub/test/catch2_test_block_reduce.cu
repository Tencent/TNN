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

#include <cub/block/block_reduce.cuh>

#include <limits>
#include <numeric>

#include <c2h/catch2_test_helper.cuh>

template <cub::BlockReduceAlgorithm Algorithm,
          int ItemsPerThread,
          int BlockDimX,
          int BlockDimY,
          int BlockDimZ,
          class T,
          class ActionT>
__global__ void block_reduce_kernel(T* in, T* out, int valid_items, ActionT action)
{
  using block_reduce_t = cub::BlockReduce<T, BlockDimX, Algorithm, BlockDimY, BlockDimZ>;
  using storage_t      = typename block_reduce_t::TempStorage;

  __shared__ storage_t storage;

  T thread_data[ItemsPerThread];

  const int tid           = static_cast<int>(cub::RowMajorTid(BlockDimX, BlockDimY, BlockDimZ));
  const int thread_offset = tid * ItemsPerThread;

  for (int item = 0; item < ItemsPerThread; item++)
  {
    const int idx     = thread_offset + item;
    thread_data[item] = idx < valid_items ? in[idx] : T();
  }
  __syncthreads();

  block_reduce_t reduce(storage);

  T aggregate = action(reduce, thread_data, valid_items);

  if (tid == 0)
  {
    out[0] = aggregate;
  }
}

template <cub::BlockReduceAlgorithm Algorithm,
          int ItemsPerThread,
          int BlockDimX,
          int BlockDimY,
          int BlockDimZ,
          class T,
          class ActionT>
void block_reduce(c2h::device_vector<T>& in, c2h::device_vector<T>& out, ActionT action)
{
  dim3 block_dims(BlockDimX, BlockDimY, BlockDimZ);

  block_reduce_kernel<Algorithm, ItemsPerThread, BlockDimX, BlockDimY, BlockDimZ, T, ActionT><<<1, block_dims>>>(
    thrust::raw_pointer_cast(in.data()), thrust::raw_pointer_cast(out.data()), static_cast<int>(in.size()), action);

  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
}

struct sum_partial_tile_op_t
{
  template <int ItemsPerThread, class BlockReduceT, class T>
  __device__ T operator()(BlockReduceT& reduce, T (&thread_data)[ItemsPerThread], int valid_items) const
  {
    return reduce.Sum(thread_data[0], valid_items);
  }
};

struct sum_full_tile_op_t
{
  template <int ItemsPerThread, class BlockReduceT, class T>
  __device__ T operator()(BlockReduceT& reduce, T (&thread_data)[ItemsPerThread], int /* valid_items */) const
  {
    return reduce.Sum(thread_data);
  }
};

struct max_partial_tile_op_t
{
  template <int ItemsPerThread, class BlockReduceT, class T>
  __device__ T operator()(BlockReduceT& reduce, T (&thread_data)[ItemsPerThread], int valid_items) const
  {
    return reduce.Reduce(thread_data[0], cub::Max{}, valid_items);
  }
};

struct max_full_tile_op_t
{
  template <int ItemsPerThread, class BlockReduceT, class T>
  __device__ T operator()(BlockReduceT& reduce, T (&thread_data)[ItemsPerThread], int /* valid_items */) const
  {
    return reduce.Reduce(thread_data, cub::Max{});
  }
};

using types     = c2h::type_list<std::uint8_t, std::uint16_t, std::int32_t, std::int64_t, float, double>;
using vec_types = c2h::type_list<ulonglong4, uchar3, short2>;

// %PARAM% TEST_DIM_X dimx 1:7:32:65:128
// %PARAM% TEST_DIM_YZ dimyz 1:2

using block_dim_xs           = c2h::enum_type_list<int, TEST_DIM_X>;
using block_dim_yzs          = c2h::enum_type_list<int, TEST_DIM_YZ>;
using items_per_thread       = c2h::enum_type_list<int, 1, 4>;
using single_item_per_thread = c2h::enum_type_list<int, 1>;
using algorithm =
  c2h::enum_type_list<cub::BlockReduceAlgorithm,
                      cub::BlockReduceAlgorithm::BLOCK_REDUCE_RAKING,
                      cub::BlockReduceAlgorithm::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY,
                      cub::BlockReduceAlgorithm::BLOCK_REDUCE_WARP_REDUCTIONS>;

template <class TestType>
struct params_t
{
  using type = typename c2h::get<0, TestType>;

  static constexpr int items_per_thread = c2h::get<1, TestType>::value;
  static constexpr int block_dim_x      = c2h::get<2, TestType>::value;
  static constexpr int block_dim_y      = c2h::get<3, TestType>::value;
  static constexpr int block_dim_z      = block_dim_y;
  static constexpr int tile_size        = items_per_thread * block_dim_x * block_dim_y * block_dim_z;

  static constexpr cub::BlockReduceAlgorithm algorithm = c2h::get<4, TestType>::value;
};

C2H_TEST(
  "Block reduce works with sum", "[reduce][block]", types, items_per_thread, block_dim_xs, block_dim_yzs, algorithm)
{
  using params = params_t<TestType>;
  using type   = typename params::type;

  c2h::device_vector<type> d_out(1);
  c2h::device_vector<type> d_in(params::tile_size);
  c2h::gen(C2H_SEED(10), d_in, std::numeric_limits<type>::min());

  c2h::host_vector<type> h_in = d_in;
  c2h::host_vector<type> h_reference(
    1, std::accumulate(h_in.begin() + 1, h_in.end(), h_in[0], [](const type& lhs, const type& rhs) {
      return static_cast<type>(lhs + rhs);
    }));

  block_reduce<params::algorithm,
               params::items_per_thread,
               params::block_dim_x,
               params::block_dim_y,
               params::block_dim_z,
               type>(d_in, d_out, sum_full_tile_op_t{});

  REQUIRE_APPROX_EQ(h_reference, d_out);
}

C2H_TEST("Block reduce works with sum in partial tiles",
         "[reduce][block]",
         types,
         single_item_per_thread,
         block_dim_xs,
         block_dim_yzs,
         algorithm)
{
  using params = params_t<TestType>;
  using type   = typename params::type;

  c2h::device_vector<type> d_out(1);
  c2h::device_vector<type> d_in(GENERATE_COPY(take(2, random(1, params::tile_size))));
  c2h::gen(C2H_SEED(10), d_in, std::numeric_limits<type>::min());

  c2h::host_vector<type> h_in = d_in;
  std::vector<type> h_reference(
    1, std::accumulate(h_in.begin() + 1, h_in.end(), h_in[0], [](const type& lhs, const type& rhs) {
      return static_cast<type>(lhs + rhs);
    }));

  block_reduce<params::algorithm,
               params::items_per_thread,
               params::block_dim_x,
               params::block_dim_y,
               params::block_dim_z,
               type>(d_in, d_out, sum_partial_tile_op_t{});

  REQUIRE_APPROX_EQ(h_reference, d_out);
}

C2H_TEST("Block reduce works with custom op",
         "[reduce][block]",
         types,
         items_per_thread,
         block_dim_xs,
         block_dim_yzs,
         algorithm)
{
  using params = params_t<TestType>;
  using type   = typename params::type;

  c2h::device_vector<type> d_out(1);
  c2h::device_vector<type> d_in(params::tile_size);
  c2h::gen(C2H_SEED(10), d_in, std::numeric_limits<type>::min());

  c2h::host_vector<type> h_in = d_in;
  c2h::host_vector<type> h_reference(
    1, std::accumulate(h_in.begin() + 1, h_in.end(), h_in[0], [](const type& lhs, const type& rhs) {
      return std::max(lhs, rhs);
    }));

  block_reduce<params::algorithm,
               params::items_per_thread,
               params::block_dim_x,
               params::block_dim_y,
               params::block_dim_z,
               type>(d_in, d_out, max_full_tile_op_t{});

  REQUIRE_APPROX_EQ(h_reference, d_out);
}

C2H_TEST("Block reduce works with custom op in partial tiles",
         "[reduce][block]",
         types,
         single_item_per_thread,
         block_dim_xs,
         block_dim_yzs,
         algorithm)
{
  using params = params_t<TestType>;
  using type   = typename params::type;

  c2h::device_vector<type> d_out(1);
  c2h::device_vector<type> d_in(GENERATE_COPY(take(2, random(1, params::tile_size))));
  c2h::gen(C2H_SEED(10), d_in, std::numeric_limits<type>::min());

  c2h::host_vector<type> h_in = d_in;
  c2h::host_vector<type> h_reference(
    1, std::accumulate(h_in.begin() + 1, h_in.end(), h_in[0], [](const type& lhs, const type& rhs) {
      return std::max(lhs, rhs);
    }));

  block_reduce<params::algorithm,
               params::items_per_thread,
               params::block_dim_x,
               params::block_dim_y,
               params::block_dim_z,
               type>(d_in, d_out, max_partial_tile_op_t{});

  REQUIRE_APPROX_EQ(h_reference, d_out);
}

C2H_TEST("Block reduce works with custom types", "[reduce][block]", block_dim_xs, block_dim_yzs, algorithm)
{
  using type = c2h::custom_type_t<c2h::accumulateable_t, c2h::equal_comparable_t>;

  constexpr int items_per_thread                = 1;
  constexpr int block_dim_x                     = c2h::get<0, TestType>::value;
  constexpr int block_dim_y                     = c2h::get<1, TestType>::value;
  constexpr int block_dim_z                     = block_dim_y;
  constexpr cub::BlockReduceAlgorithm algorithm = c2h::get<2, TestType>::value;

  constexpr int tile_size = block_dim_x * block_dim_y * block_dim_z * items_per_thread;

  c2h::device_vector<type> d_out(1);
  c2h::device_vector<type> d_in(GENERATE_COPY(take(2, random(1, tile_size))));
  c2h::gen(C2H_SEED(10), d_in, std::numeric_limits<type>::min());

  c2h::host_vector<type> h_in = d_in;
  c2h::host_vector<type> h_reference(
    1, std::accumulate(h_in.begin() + 1, h_in.end(), h_in[0], [](const type& lhs, const type& rhs) {
      return static_cast<type>(lhs + rhs);
    }));

  block_reduce<algorithm, items_per_thread, block_dim_x, block_dim_y, block_dim_z, type>(
    d_in, d_out, sum_partial_tile_op_t{});

  REQUIRE(h_reference == d_out);
}

C2H_TEST("Block reduce works with vec types", "[reduce][block]", vec_types, block_dim_xs, block_dim_yzs, algorithm)
{
  using type = c2h::get<0, TestType>;

  constexpr int items_per_thread                = 1;
  constexpr int block_dim_x                     = c2h::get<1, TestType>::value;
  constexpr int block_dim_y                     = c2h::get<2, TestType>::value;
  constexpr int block_dim_z                     = block_dim_y;
  constexpr cub::BlockReduceAlgorithm algorithm = c2h::get<3, TestType>::value;

  constexpr int tile_size = block_dim_x * block_dim_y * block_dim_z * items_per_thread;

  c2h::device_vector<type> d_out(1);
  c2h::device_vector<type> d_in(GENERATE_COPY(take(2, random(1, tile_size))));
  c2h::gen(C2H_SEED(10), d_in);

  c2h::host_vector<type> h_in = d_in;
  c2h::host_vector<type> h_reference(
    1, std::accumulate(h_in.begin() + 1, h_in.end(), h_in[0], [](const type& lhs, const type& rhs) {
      return static_cast<type>(lhs + rhs);
    }));

  block_reduce<algorithm, items_per_thread, block_dim_x, block_dim_y, block_dim_z, type>(
    d_in, d_out, sum_partial_tile_op_t{});

  REQUIRE(h_reference == d_out);
}
