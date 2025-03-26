/******************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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

#include <cub/warp/warp_scan.cuh>

#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "cuda/std/__algorithm/fill.h"
#include "cuda/std/__algorithm/max.h"
#include "cuda/std/__numeric/inclusive_scan.h"
#include "cuda/std/__numeric/iota.h"
#include <c2h/catch2_test_helper.cuh>

constexpr int num_warps = 4;

struct max_op
{
  __host__ __device__ int operator()(int i, int j)
  {
    return cuda::std::max(i, j);
  }
};

struct sum_op
{
  __host__ __device__ int operator()(int i, int j)
  {
    return i + j;
  }
};

// example-begin inclusive-warp-scan-init-value
__global__ void InclusiveWarpScanKernel(int* output)
{
  // Specialize WarpScan for type int
  using warp_scan_t = cub::WarpScan<int>;
  // Allocate WarpScan shared memory for 4 warps
  __shared__ typename warp_scan_t::TempStorage temp_storage[num_warps];

  int warp_id       = threadIdx.x / 32;
  int initial_value = 3;
  int thread_data   = threadIdx.x % 32 + warp_id;

  // warp #0 input: {0, 1, 2, 3, ..., 31}
  // warp #1 input: {1, 2, 3, 4, ..., 32}
  // warp #2 input: {2, 3, 4, 5, ..., 33}
  // warp #4 input: {3, 4, 5, 6, ..., 34}

  // Collectively compute the warp-wide inclusive prefix max scan
  warp_scan_t(temp_storage[warp_id]).InclusiveScan(thread_data, thread_data, initial_value, cub::Max());

  // initial value = 3 (for each warp)
  // warp #0 output: {3, 3, 3, 3, ..., 31}
  // warp #1 output: {3, 3, 3, 4, ..., 32}
  // warp #2 output: {3, 3, 4, 5, ..., 33}
  // warp #3 output: {3, 4, 5, 6, ..., 34}
  output[threadIdx.x] = thread_data;

  // example-end inclusive-warp-scan-init-value
}

C2H_TEST("Warp array-based inclusive scan works with initial value", "[scan][warp]")
{
  thrust::device_vector<int> d_out(num_warps * 32);

  InclusiveWarpScanKernel<<<1, num_warps * 32>>>(thrust::raw_pointer_cast(d_out.data()));
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  c2h::host_vector<int> expected(d_out.size());

  for (int i = 0; i < num_warps; ++i)
  {
    auto start = expected.begin() + i * 32;
    auto end   = start + 32;

    cuda::std::iota(start, end, i); // initialize host input for every warp

    cuda::std::inclusive_scan(start, end, start, max_op{}, 3);
  }

  REQUIRE(expected == d_out);
}

// example-begin inclusive-warp-scan-init-value-aggregate
__global__ void InclusiveWarpScanKernelAggr(int* output, int* d_warp_aggregate)
{
  // Specialize WarpScan for type int
  using warp_scan_t = cub::WarpScan<int>;
  // Allocate WarpScan shared memory for 4 warps
  __shared__ typename warp_scan_t::TempStorage temp_storage[num_warps];

  int warp_id       = threadIdx.x / 32;
  int initial_value = 3; // for each warp
  int thread_data   = 1;
  int warp_aggregate;

  // warp #0 input: {1, 1, 1, 1, ..., 1}
  // warp #1 input: {1, 1, 1, 1, ..., 1}
  // warp #2 input: {1, 1, 1, 1, ..., 1}
  // warp #4 input: {1, 1, 1, 1, ..., 1}

  // Collectively compute the warp-wide inclusive prefix max scan
  warp_scan_t(temp_storage[warp_id]).InclusiveScan(thread_data, thread_data, initial_value, cub::Sum(), warp_aggregate);

  // warp #1 output: {4, 5, 6, 7, ..., 35} - warp aggregate: 32
  // warp #2 output: {4, 5, 6, 7, ..., 35} - warp aggregate: 32
  // warp #0 output: {4, 5, 6, 7, ..., 35} - warp aggregate: 32
  // warp #3 output: {4, 5, 6, 7, ..., 35} - warp aggregate: 32

  // example-end inclusive-warp-scan-init-value-aggregate
  output[threadIdx.x]       = thread_data;
  d_warp_aggregate[warp_id] = warp_aggregate;
}

C2H_TEST("Warp array-based inclusive scan aggregate works with initial value", "[scan][warp]")
{
  thrust::device_vector<int> d_out(num_warps * 32);
  c2h::device_vector<int> d_warp_aggregate(num_warps);

  InclusiveWarpScanKernelAggr<<<1, num_warps * 32>>>(
    thrust::raw_pointer_cast(d_out.data()), thrust::raw_pointer_cast(d_warp_aggregate.data()));
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  c2h::host_vector<int> expected(d_out.size());
  c2h::host_vector<int> expected_aggr{};

  for (int i = 0; i < num_warps; ++i)
  {
    auto start   = expected.begin() + i * 32;
    auto end     = start + 32;
    int init_val = 3;

    cuda::std::fill(start, end, 1); // initialize host input for every warp

    cuda::std::inclusive_scan(start, end, start, sum_op{}, init_val);

    expected_aggr.push_back(expected[i * 32 + 31] - init_val); // warp aggregate does not take
                                                               // initial value into account
  }

  REQUIRE(expected == d_out);
  REQUIRE(expected_aggr == d_warp_aggregate);
}
