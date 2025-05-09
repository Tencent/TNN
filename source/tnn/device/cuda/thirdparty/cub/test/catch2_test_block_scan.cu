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

#include <cub/block/block_scan.cuh>

#include <climits>

#include <c2h/catch2_test_helper.cuh>

template <cub::BlockScanAlgorithm Algorithm,
          int ItemsPerThread,
          int BlockDimX,
          int BlockDimY,
          int BlockDimZ,
          class T,
          class ActionT>
__global__ void block_scan_kernel(T* in, T* out, ActionT action)
{
  using block_scan_t = cub::BlockScan<T, BlockDimX, Algorithm, BlockDimY, BlockDimZ>;
  using storage_t    = typename block_scan_t::TempStorage;

  __shared__ storage_t storage;

  T thread_data[ItemsPerThread];

  const int tid           = static_cast<int>(cub::RowMajorTid(BlockDimX, BlockDimY, BlockDimZ));
  const int thread_offset = tid * ItemsPerThread;

  for (int item = 0; item < ItemsPerThread; item++)
  {
    const int idx     = thread_offset + item;
    thread_data[item] = in[idx];
  }
  __syncthreads();

  block_scan_t scan(storage);

  action(scan, thread_data);

  for (int item = 0; item < ItemsPerThread; item++)
  {
    const int idx = thread_offset + item;
    out[idx]      = thread_data[item];
  }
}

template <cub::BlockScanAlgorithm Algorithm,
          int ItemsPerThread,
          int BlockDimX,
          int BlockDimY,
          int BlockDimZ,
          class T,
          class ActionT>
void block_scan(c2h::device_vector<T>& in, c2h::device_vector<T>& out, ActionT action)
{
  dim3 block_dims(BlockDimX, BlockDimY, BlockDimZ);

  block_scan_kernel<Algorithm, ItemsPerThread, BlockDimX, BlockDimY, BlockDimZ, T, ActionT>
    <<<1, block_dims>>>(thrust::raw_pointer_cast(in.data()), thrust::raw_pointer_cast(out.data()), action);

  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
}

enum class scan_mode
{
  exclusive,
  inclusive
};

template <scan_mode Mode>
struct sum_op_t
{
  template <int ItemsPerThread, class BlockScanT, class T>
  __device__ void operator()(BlockScanT& scan, T (&thread_data)[ItemsPerThread]) const
  {
    if (Mode == scan_mode::exclusive)
    {
      scan.ExclusiveSum(thread_data, thread_data);
    }
    else
    {
      scan.InclusiveSum(thread_data, thread_data);
    }
  }
};

template <class T, scan_mode Mode>
struct min_init_value_op_t
{
  T initial_value;
  template <int ItemsPerThread, class BlockScanT>
  __device__ void operator()(BlockScanT& scan, T (&thread_data)[ItemsPerThread]) const
  {
    _CCCL_IF_CONSTEXPR (Mode == scan_mode::exclusive)
    {
      scan.ExclusiveScan(thread_data, thread_data, initial_value, cub::Min{});
    }
    else
    {
      scan.InclusiveScan(thread_data, thread_data, initial_value, cub::Min{});
    }
  }
};

template <scan_mode Mode>
struct min_op_t
{
  template <int ItemsPerThread, class BlockScanT>
  __device__ void operator()(BlockScanT& scan, int (&thread_data)[ItemsPerThread]) const
  {
    if (Mode == scan_mode::exclusive)
    {
      scan.ExclusiveScan(thread_data, thread_data, cub::Min{});
    }
    else
    {
      scan.InclusiveScan(thread_data, thread_data, cub::Min{});
    }
  }
};

template <class T, scan_mode Mode>
struct min_init_value_aggregate_op_t
{
  int m_target_thread_id;
  T initial_value;
  T* m_d_block_aggregate;

  template <int ItemsPerThread, class BlockScanT>
  __device__ void operator()(BlockScanT& scan, T (&thread_data)[ItemsPerThread]) const
  {
    T block_aggregate{};

    _CCCL_IF_CONSTEXPR (Mode == scan_mode::exclusive)
    {
      scan.ExclusiveScan(thread_data, thread_data, initial_value, cub::Min{}, block_aggregate);
    }
    else
    {
      scan.InclusiveScan(thread_data, thread_data, initial_value, cub::Min{}, block_aggregate);
    }

    const int tid = cub::RowMajorTid(blockDim.x, blockDim.y, blockDim.z);

    if (tid == m_target_thread_id)
    {
      *m_d_block_aggregate = block_aggregate;
    }
  }
};

template <class T, scan_mode Mode>
struct sum_aggregate_op_t
{
  int m_target_thread_id;
  T* m_d_block_aggregate;

  template <int ItemsPerThread, class BlockScanT>
  __device__ void operator()(BlockScanT& scan, T (&thread_data)[ItemsPerThread]) const
  {
    T block_aggregate{};

    if (Mode == scan_mode::exclusive)
    {
      scan.ExclusiveSum(thread_data, thread_data, block_aggregate);
    }
    else
    {
      scan.InclusiveSum(thread_data, thread_data, block_aggregate);
    }

    const int tid = static_cast<int>(cub::RowMajorTid(blockDim.x, blockDim.y, blockDim.z));

    if (tid == m_target_thread_id)
    {
      *m_d_block_aggregate = block_aggregate;
    }
  }
};

template <class T, scan_mode Mode>
struct sum_prefix_op_t
{
  T m_prefix;

  struct block_prefix_op_t
  {
    int linear_tid;
    T prefix;

    __device__ block_prefix_op_t(int linear_tid, T prefix)
        : linear_tid(linear_tid)
        , prefix(prefix)
    {}

    __device__ T operator()(T block_aggregate)
    {
      T retval = (linear_tid == 0) ? prefix : T{};
      prefix   = prefix + block_aggregate;
      return retval;
    }
  };

  template <int ItemsPerThread, class BlockScanT>
  __device__ void operator()(BlockScanT& scan, T (&thread_data)[ItemsPerThread]) const
  {
    const int tid = static_cast<int>(cub::RowMajorTid(blockDim.x, blockDim.y, blockDim.z));
    block_prefix_op_t prefix_op{tid, m_prefix};

    if (Mode == scan_mode::exclusive)
    {
      scan.ExclusiveSum(thread_data, thread_data, prefix_op);
    }
    else
    {
      scan.InclusiveSum(thread_data, thread_data, prefix_op);
    }
  }
};

template <class T, scan_mode Mode>
struct min_prefix_op_t
{
  T m_prefix;
  static constexpr T min_identity = std::numeric_limits<T>::max();

  struct block_prefix_op_t
  {
    int linear_tid;
    T prefix;

    __device__ block_prefix_op_t(int linear_tid, T prefix)
        : linear_tid(linear_tid)
        , prefix(prefix)
    {}

    __device__ T operator()(T block_aggregate)
    {
      T retval = (linear_tid == 0) ? prefix : min_identity;
      prefix   = cub::Min{}(prefix, block_aggregate);
      return retval;
    }
  };

  template <int ItemsPerThread, class BlockScanT>
  __device__ void operator()(BlockScanT& scan, T (&thread_data)[ItemsPerThread]) const
  {
    const int tid = static_cast<int>(cub::RowMajorTid(blockDim.x, blockDim.y, blockDim.z));
    block_prefix_op_t prefix_op{tid, m_prefix};

    if (Mode == scan_mode::exclusive)
    {
      scan.ExclusiveScan(thread_data, thread_data, cub::Min{}, prefix_op);
    }
    else
    {
      scan.InclusiveScan(thread_data, thread_data, cub::Min{}, prefix_op);
    }
  }
};

template <class T, class ScanOpT>
T host_scan(scan_mode mode, c2h::host_vector<T>& result, ScanOpT scan_op, T initial_value = T{})
{
  if (result.empty())
  {
    return {};
  }

  T accumulator       = static_cast<T>(scan_op(initial_value, result[0]));
  T block_accumulator = result[0];

  if (mode == scan_mode::exclusive)
  {
    result[0] = initial_value;

    for (std::size_t i = 1; i < result.size(); i++)
    {
      T tmp             = result[i];
      result[i]         = accumulator;
      accumulator       = static_cast<T>(scan_op(accumulator, tmp));
      block_accumulator = static_cast<T>(scan_op(block_accumulator, tmp));
    }
  }
  else
  {
    result[0] = accumulator;

    for (std::size_t i = 1; i < result.size(); i++)
    {
      accumulator       = static_cast<T>(scan_op(accumulator, result[i]));
      block_accumulator = static_cast<T>(scan_op(block_accumulator, result[i]));
      result[i]         = accumulator;
    }
  }

  return block_accumulator;
}

// %PARAM% ALGO_TYPE alg 0:1:2
// %PARAM% TEST_MODE mode 0:1

using types            = c2h::type_list<std::uint8_t, std::uint16_t, std::int32_t, std::int64_t>;
using vec_types        = c2h::type_list<ulonglong4, uchar3, short2>;
using block_dim_x      = c2h::enum_type_list<int, 17, 32, 65, 96>;
using block_dim_yz     = c2h::enum_type_list<int, 1, 2>;
using items_per_thread = c2h::enum_type_list<int, 1, 9>;
using algorithms =
  c2h::enum_type_list<cub::BlockScanAlgorithm,
                      cub::BlockScanAlgorithm::BLOCK_SCAN_RAKING,
                      cub::BlockScanAlgorithm::BLOCK_SCAN_WARP_SCANS,
                      cub::BlockScanAlgorithm::BLOCK_SCAN_RAKING_MEMOIZE>;
using algorithm = c2h::enum_type_list<cub::BlockScanAlgorithm, c2h::get<ALGO_TYPE, algorithms>::value>;

#if TEST_MODE == 0
using modes = c2h::enum_type_list<scan_mode, scan_mode::inclusive>;
#else
using modes = c2h::enum_type_list<scan_mode, scan_mode::exclusive>;
#endif

template <class TestType>
struct params_t
{
  using type = typename c2h::get<0, TestType>;

  static constexpr int block_dim_x      = c2h::get<1, TestType>::value;
  static constexpr int block_dim_y      = c2h::get<2, TestType>::value;
  static constexpr int block_dim_z      = block_dim_y;
  static constexpr int items_per_thread = c2h::get<3, TestType>::value;
  static constexpr int tile_size        = items_per_thread * block_dim_x * block_dim_y * block_dim_z;

  static constexpr cub::BlockScanAlgorithm algorithm = c2h::get<4, TestType>::value;
  static constexpr scan_mode mode                    = c2h::get<5, TestType>::value;
};

C2H_TEST(
  "Block scan works with sum", "[scan][block]", types, block_dim_x, block_dim_yz, items_per_thread, algorithm, modes)
{
  using params = params_t<TestType>;
  using type   = typename params::type;

  c2h::device_vector<type> d_out(params::tile_size);
  c2h::device_vector<type> d_in(params::tile_size);
  c2h::gen(C2H_SEED(10), d_in);

  block_scan<params::algorithm, params::items_per_thread, params::block_dim_x, params::block_dim_y, params::block_dim_z>(
    d_in, d_out, sum_op_t<params::mode>{});

  c2h::host_vector<type> h_out = d_in;
  host_scan(params::mode, h_out, std::plus<type>{});

  REQUIRE_APPROX_EQ(h_out, d_out);
}

C2H_TEST("Block scan works with vec types", "[scan][block]", vec_types, algorithm, modes)
{
  constexpr int items_per_thread              = 3;
  constexpr int block_dim_x                   = 256;
  constexpr int block_dim_y                   = 1;
  constexpr int block_dim_z                   = 1;
  constexpr int tile_size                     = items_per_thread * block_dim_x * block_dim_y * block_dim_z;
  constexpr cub::BlockScanAlgorithm algorithm = c2h::get<1, TestType>::value;
  constexpr scan_mode mode                    = c2h::get<2, TestType>::value;

  using type = typename c2h::get<0, TestType>;

  c2h::device_vector<type> d_out(tile_size);
  c2h::device_vector<type> d_in(tile_size);
  c2h::gen(C2H_SEED(10), d_in);

  block_scan<algorithm, items_per_thread, block_dim_x, block_dim_y, block_dim_z>(d_in, d_out, sum_op_t<mode>{});

  c2h::host_vector<type> h_out = d_in;
  host_scan(mode, h_out, std::plus<type>{});

  REQUIRE(h_out == d_out);
}

C2H_TEST("Block scan works with custom types", "[scan][block]", algorithm, modes)
{
  constexpr int items_per_thread              = 3;
  constexpr int block_dim_x                   = 256;
  constexpr int block_dim_y                   = 1;
  constexpr int block_dim_z                   = 1;
  constexpr int tile_size                     = items_per_thread * block_dim_x * block_dim_y * block_dim_z;
  constexpr cub::BlockScanAlgorithm algorithm = c2h::get<0, TestType>::value;
  constexpr scan_mode mode                    = c2h::get<1, TestType>::value;

  using type = c2h::custom_type_t<c2h::accumulateable_t, c2h::equal_comparable_t>;

  c2h::device_vector<type> d_out(tile_size);
  c2h::device_vector<type> d_in(tile_size);
  c2h::gen(C2H_SEED(10), d_in);

  block_scan<algorithm, items_per_thread, block_dim_x, block_dim_y, block_dim_z>(d_in, d_out, sum_op_t<mode>{});

  c2h::host_vector<type> h_out = d_in;
  host_scan(mode, h_out, std::plus<type>{});

  REQUIRE(h_out == d_out);
}

C2H_TEST("Block scan returns valid block aggregate", "[scan][block]", algorithm, modes, block_dim_yz)
{
  constexpr int items_per_thread              = 3;
  constexpr int block_dim_x                   = 64;
  constexpr int block_dim_y                   = c2h::get<2, TestType>::value;
  constexpr int block_dim_z                   = block_dim_y;
  constexpr int threads_in_block              = block_dim_x * block_dim_y * block_dim_z;
  constexpr int tile_size                     = items_per_thread * threads_in_block;
  constexpr cub::BlockScanAlgorithm algorithm = c2h::get<0, TestType>::value;
  constexpr scan_mode mode                    = c2h::get<1, TestType>::value;

  using type = c2h::custom_type_t<c2h::accumulateable_t, c2h::equal_comparable_t>;

  const int target_thread_id = GENERATE_COPY(take(2, random(0, threads_in_block - 1)));

  c2h::device_vector<type> d_block_aggregate(1);
  c2h::device_vector<type> d_out(tile_size);
  c2h::device_vector<type> d_in(tile_size);
  c2h::gen(C2H_SEED(10), d_in);

  block_scan<algorithm, items_per_thread, block_dim_x, block_dim_y, block_dim_z>(
    d_in, d_out, sum_aggregate_op_t<type, mode>{target_thread_id, thrust::raw_pointer_cast(d_block_aggregate.data())});

  c2h::host_vector<type> h_out = d_in;
  type block_aggregate         = host_scan(mode, h_out, std::plus<type>{});

  REQUIRE(h_out == d_out);
  REQUIRE(block_aggregate == d_block_aggregate[0]);
}

C2H_TEST("Block scan supports prefix op", "[scan][block]", algorithm, modes, block_dim_yz)
{
  constexpr int items_per_thread              = 3;
  constexpr int block_dim_x                   = 64;
  constexpr int block_dim_y                   = c2h::get<2, TestType>::value;
  constexpr int block_dim_z                   = block_dim_y;
  constexpr int threads_in_block              = block_dim_x * block_dim_y * block_dim_z;
  constexpr int tile_size                     = items_per_thread * threads_in_block;
  constexpr cub::BlockScanAlgorithm algorithm = c2h::get<0, TestType>::value;
  constexpr scan_mode mode                    = c2h::get<1, TestType>::value;

  using type = int;

  const type prefix = GENERATE_COPY(take(2, random(0, tile_size)));

  c2h::device_vector<type> d_out(tile_size);
  c2h::device_vector<type> d_in(tile_size);
  c2h::gen(C2H_SEED(10), d_in);

  block_scan<algorithm, items_per_thread, block_dim_x, block_dim_y, block_dim_z>(
    d_in, d_out, sum_prefix_op_t<type, mode>{prefix});

  c2h::host_vector<type> h_out = d_in;
  host_scan(mode, h_out, std::plus<type>{}, prefix);

  REQUIRE(h_out == d_out);
}

C2H_TEST("Block scan supports custom scan op", "[scan][block]", algorithm, modes, block_dim_yz)
{
  constexpr int items_per_thread              = 3;
  constexpr int block_dim_x                   = 64;
  constexpr int block_dim_y                   = c2h::get<2, TestType>::value;
  constexpr int block_dim_z                   = block_dim_y;
  constexpr int threads_in_block              = block_dim_x * block_dim_y * block_dim_z;
  constexpr int tile_size                     = items_per_thread * threads_in_block;
  constexpr cub::BlockScanAlgorithm algorithm = c2h::get<0, TestType>::value;
  constexpr scan_mode mode                    = c2h::get<1, TestType>::value;

  using type = int;

  c2h::device_vector<type> d_out(tile_size);
  c2h::device_vector<type> d_in(tile_size);
  c2h::gen(C2H_SEED(10), d_in);

  block_scan<algorithm, items_per_thread, block_dim_x, block_dim_y, block_dim_z>(d_in, d_out, min_op_t<mode>{});

  c2h::host_vector<type> h_out = d_in;
  host_scan(
    mode,
    h_out,
    [](type l, type r) {
      return std::min(l, r);
    },
    INT_MAX);

  _CCCL_IF_CONSTEXPR (mode == scan_mode::exclusive)
  {
    //! With no initial value, the output computed for *thread*\ :sub:`0` is undefined.
    d_out.erase(d_out.begin());
    h_out.erase(h_out.begin());
  }

  REQUIRE(h_out == d_out);
}

C2H_TEST("Block custom op scan works with initial value", "[scan][block]", algorithm, modes, block_dim_yz)
{
  constexpr int items_per_thread              = 3;
  constexpr int block_dim_x                   = 64;
  constexpr int block_dim_y                   = c2h::get<2, TestType>::value;
  constexpr int block_dim_z                   = block_dim_y;
  constexpr int threads_in_block              = block_dim_x * block_dim_y * block_dim_z;
  constexpr int tile_size                     = items_per_thread * threads_in_block;
  constexpr cub::BlockScanAlgorithm algorithm = c2h::get<0, TestType>::value;
  constexpr scan_mode mode                    = c2h::get<1, TestType>::value;

  using type = int;

  c2h::device_vector<type> d_out(tile_size);
  c2h::device_vector<type> d_in(tile_size);
  c2h::gen(C2H_SEED(10), d_in);

  const type initial_value = static_cast<type>(GENERATE_COPY(take(2, random(0, tile_size))));

  block_scan<algorithm, items_per_thread, block_dim_x, block_dim_y, block_dim_z>(
    d_in, d_out, min_init_value_op_t<type, mode>{initial_value});

  c2h::host_vector<type> h_out = d_in;
  host_scan(
    mode,
    h_out,
    [](type l, type r) {
      return std::min(l, r);
    },
    initial_value);

  REQUIRE(h_out == d_out);
}

C2H_TEST("Block custom op scan with initial value returns valid block aggregate",
         "[scan][block]",
         algorithm,
         modes,
         block_dim_yz)
{
  constexpr int items_per_thread              = 3;
  constexpr int block_dim_x                   = 64;
  constexpr int block_dim_y                   = c2h::get<2, TestType>::value;
  constexpr int block_dim_z                   = block_dim_y;
  constexpr int threads_in_block              = block_dim_x * block_dim_y * block_dim_z;
  constexpr int tile_size                     = items_per_thread * threads_in_block;
  constexpr cub::BlockScanAlgorithm algorithm = c2h::get<0, TestType>::value;
  constexpr scan_mode mode                    = c2h::get<1, TestType>::value;

  using type = int;

  c2h::device_vector<type> d_out(tile_size);
  c2h::device_vector<type> d_in(tile_size);
  c2h::gen(C2H_SEED(10), d_in);

  const type initial_value = static_cast<type>(GENERATE_COPY(take(2, random(0, tile_size))));

  const int target_thread_id = GENERATE_COPY(take(2, random(0, threads_in_block - 1)));

  c2h::device_vector<type> d_block_aggregate(1);

  block_scan<algorithm, items_per_thread, block_dim_x, block_dim_y, block_dim_z>(
    d_in,
    d_out,
    min_init_value_aggregate_op_t<type, mode>{
      target_thread_id, initial_value, thrust::raw_pointer_cast(d_block_aggregate.data())});

  c2h::host_vector<type> h_out = d_in;
  type h_block_aggregate       = host_scan(
    mode,
    h_out,
    [](type l, type r) {
      return std::min(l, r);
    },
    initial_value);

  REQUIRE(h_out == d_out);
  REQUIRE(h_block_aggregate == d_block_aggregate[0]);
}

C2H_TEST("Block scan supports prefix op and custom scan op", "[scan][block]", algorithm, modes, block_dim_yz)
{
  constexpr int items_per_thread              = 3;
  constexpr int block_dim_x                   = 64;
  constexpr int block_dim_y                   = c2h::get<2, TestType>::value;
  constexpr int block_dim_z                   = block_dim_y;
  constexpr int threads_in_block              = block_dim_x * block_dim_y * block_dim_z;
  constexpr int tile_size                     = items_per_thread * threads_in_block;
  constexpr cub::BlockScanAlgorithm algorithm = c2h::get<0, TestType>::value;
  constexpr scan_mode mode                    = c2h::get<1, TestType>::value;

  using type = int;

  const type prefix = GENERATE_COPY(take(2, random(0, tile_size)));

  c2h::device_vector<type> d_out(tile_size);
  c2h::device_vector<type> d_in(tile_size);
  c2h::gen(C2H_SEED(10), d_in);

  block_scan<algorithm, items_per_thread, block_dim_x, block_dim_y, block_dim_z>(
    d_in, d_out, min_prefix_op_t<type, mode>{prefix});

  c2h::host_vector<type> h_out = d_in;
  host_scan(
    mode,
    h_out,
    [](type a, type b) {
      return std::min(a, b);
    },
    prefix);

  REQUIRE(h_out == d_out);
}
