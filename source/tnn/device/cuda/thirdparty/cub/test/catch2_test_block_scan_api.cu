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

#include <cub/block/block_scan.cuh>

#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cuda/std/numeric>

#include <c2h/catch2_test_helper.cuh>

constexpr int num_items_per_thread = 2;
constexpr int block_num_threads    = 64;

// example-begin inclusive-scan-array-init-value
__global__ void InclusiveBlockScanKernel(int* output)
{
  // Specialize BlockScan for a 1D block of 64 threads of type int
  using block_scan_t   = cub::BlockScan<int, 64>;
  using temp_storage_t = block_scan_t::TempStorage;

  // Allocate shared memory for BlockScan
  __shared__ temp_storage_t temp_storage;

  int initial_value = 1;
  int thread_data[] = {
    +1 * ((int) threadIdx.x * num_items_per_thread), // item 0
    -1 * ((int) threadIdx.x * num_items_per_thread + 1) // item 1
  };
  //  input: {[0, -1], [2, -3],[4, -5], ... [126, -127]}

  // Collectively compute the block-wide inclusive scan max
  block_scan_t(temp_storage).InclusiveScan(thread_data, thread_data, initial_value, cub::Max());

  // output: {[1, 1], [2, 2],[3, 3], ... [126, 126]}
  // ...
  // example-end inclusive-scan-array-init-value
  output[threadIdx.x * 2]     = thread_data[0];
  output[threadIdx.x * 2 + 1] = thread_data[1];
}

C2H_TEST("Block array-based inclusive scan works with initial value", "[scan][block]")
{
  thrust::device_vector<int> d_out(block_num_threads * num_items_per_thread);

  InclusiveBlockScanKernel<<<1, block_num_threads>>>(thrust::raw_pointer_cast(d_out.data()));
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  c2h::host_vector<int> expected(d_out.size());
  for (size_t i = 0; i < expected.size() - 1; i += 2)
  {
    expected[i]     = static_cast<int>(i);
    expected[i + 1] = static_cast<int>(i);
  }

  // When initial value = 1 for the given input the first two
  // elements of the result are equal to 1.
  expected[0] = 1;
  expected[1] = 1;

  REQUIRE(expected == d_out);
}

// example-begin inclusive-scan-array-aggregate-init-value
__global__ void InclusiveBlockScanKernelAggregate(int* output, int* d_block_aggregate)
{
  // Specialize BlockScan for a 1D block of 64 threads of type int
  using block_scan_t   = cub::BlockScan<int, 64>;
  using temp_storage_t = block_scan_t::TempStorage;

  // Allocate shared memory for BlockScan
  __shared__ temp_storage_t temp_storage;

  int initial_value = 1;
  int thread_data[] = {
    +1 * ((int) threadIdx.x * num_items_per_thread), // item 0
    -1 * ((int) threadIdx.x * num_items_per_thread + 1) // item 1
  };
  //  input: {[0, -1], [2, -3],[4, -5], ... [126, -127]}

  // Collectively compute the block-wide inclusive scan max
  int block_aggregate;
  block_scan_t(temp_storage).InclusiveScan(thread_data, thread_data, initial_value, cub::Max(), block_aggregate);

  // output: {[1, 1], [2, 2],[3, 3], ... [126, 126]}
  // block_aggregate = 126;
  // ...
  // example-end inclusive-scan-array-aggregate-init-value

  *d_block_aggregate          = block_aggregate;
  output[threadIdx.x * 2]     = thread_data[0];
  output[threadIdx.x * 2 + 1] = thread_data[1];
}

C2H_TEST("Block array-based inclusive scan with block aggregate works with initial value", "[scan][block]")
{
  thrust::device_vector<int> d_out(block_num_threads * num_items_per_thread);

  c2h::device_vector<int> d_block_aggregate(1);
  InclusiveBlockScanKernelAggregate<<<1, block_num_threads>>>(
    thrust::raw_pointer_cast(d_out.data()), thrust::raw_pointer_cast(d_block_aggregate.data()));
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  c2h::host_vector<int> expected(d_out.size());
  for (size_t i = 0; i < expected.size() - 1; i += 2)
  {
    expected[i]     = static_cast<int>(i);
    expected[i + 1] = static_cast<int>(i);
  }

  // When initial value = 1 for the given input the first two
  // elements of the result are equal to 1.
  expected[0] = 1;
  expected[1] = 1;

  REQUIRE(d_out == expected);
  REQUIRE(d_block_aggregate[0] == 126);
}
