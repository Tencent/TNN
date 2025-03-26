/******************************************************************************
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

/******************************************************************************
 * Simple demonstration of cub::BlockReduce with dynamic shared memory
 *
 * To compile using the command line:
 *   nvcc -arch=sm_XX example_block_reduce_dyn_smem.cu -I../.. -lcudart -O3 -std=c++14
 *
 ******************************************************************************/

// Ensure printing of CUDA runtime errors to console (define before including cub.h)
#define CUB_STDERR

#include <cub/block/block_load.cuh>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_store.cuh>

#include <algorithm>
#include <iostream>

#include "../../test/test_util.h"
#include <stdio.h>

// Some implementation details rely on c++14
#if _CCCL_STD_VER >= 2014

using namespace cub;

//---------------------------------------------------------------------
// Globals, constants and aliases
//---------------------------------------------------------------------

/// Verbose output
bool g_verbose = false;

/// Default grid size
int g_grid_size = 1;

//---------------------------------------------------------------------
// Kernels
//---------------------------------------------------------------------

/**
 * Simple kernel for performing a block-wide reduction.
 */
template <int BLOCK_THREADS>
__global__ void BlockReduceKernel(int* d_in, // Tile of input
                                  int* d_out // Tile aggregate
)
{
  // Specialize BlockReduce type for our thread block
  using BlockReduceT = cub::BlockReduce<int, BLOCK_THREADS>;
  using TempStorageT = typename BlockReduceT::TempStorage;

  union ShmemLayout
  {
    TempStorageT reduce;
    int aggregate;
  };

  // shared memory byte-array
  extern __shared__ __align__(alignof(ShmemLayout)) char smem[];

  // cast to lvalue reference of expected type
  auto& temp_storage = reinterpret_cast<TempStorageT&>(smem);

  int data = d_in[threadIdx.x];

  // Compute sum
  int aggregate = BlockReduceT(temp_storage).Sum(data);

  // block-wide sync barrier necessary to re-use shared mem safely
  __syncthreads();
  int* smem_integers = reinterpret_cast<int*>(smem);
  if (threadIdx.x == 0)
  {
    smem_integers[0] = aggregate;
  }

  // sync to make new shared value available to all threads
  __syncthreads();
  aggregate = smem_integers[0];

  // all threads write the aggregate to output
  d_out[threadIdx.x] = aggregate;
}

//---------------------------------------------------------------------
// Host utilities
//---------------------------------------------------------------------

/**
 * Initialize reduction problem (and solution).
 * Returns the aggregate
 */
int Initialize(int* h_in, int num_items)
{
  int inclusive = 0;

  for (int i = 0; i < num_items; ++i)
  {
    h_in[i] = i % 17;
    inclusive += h_in[i];
  }

  return inclusive;
}

/**
 * Test thread block reduction
 */
template <int BLOCK_THREADS>
void Test()
{
  // Allocate host arrays
  int* h_in = new int[BLOCK_THREADS];

  // Initialize problem and reference output on host
  int h_aggregate = Initialize(h_in, BLOCK_THREADS);

  // Initialize device arrays
  int* d_in  = nullptr;
  int* d_out = nullptr;
  cudaMalloc((void**) &d_in, sizeof(int) * BLOCK_THREADS);
  cudaMalloc((void**) &d_out, sizeof(int) * BLOCK_THREADS);

  // Display input problem data
  if (g_verbose)
  {
    printf("Input data: ");
    for (int i = 0; i < BLOCK_THREADS; i++)
    {
      printf("%d, ", h_in[i]);
    }
    printf("\n\n");
  }

  // Copy problem to device
  cudaMemcpy(d_in, h_in, sizeof(int) * BLOCK_THREADS, cudaMemcpyHostToDevice);

  // determine necessary storage size:
  auto block_reduce_temp_bytes = sizeof(typename cub::BlockReduce<int, BLOCK_THREADS>::TempStorage);
  // finally, we need to make sure that we can hold at least one integer
  // needed in the kernel to exchange data after reduction
  auto smem_size = (std::max)(1 * sizeof(int), block_reduce_temp_bytes);

  // use default stream
  cudaStream_t stream = nullptr;

  // Run reduction kernel
  BlockReduceKernel<BLOCK_THREADS><<<g_grid_size, BLOCK_THREADS, smem_size, stream>>>(d_in, d_out);

  // Check total aggregate
  printf("\tAggregate: ");
  int compare = 0;
  for (int i = 0; i < BLOCK_THREADS; i++)
  {
    compare = compare || CompareDeviceResults(&h_aggregate, d_out + i, 1, g_verbose, g_verbose);
  }
  printf("%s\n", compare ? "FAIL" : "PASS");
  AssertEquals(0, compare);

  // Check for kernel errors and STDIO from the kernel, if any
  CubDebugExit(cudaPeekAtLastError());
  CubDebugExit(cudaDeviceSynchronize());

  // Cleanup
  if (h_in)
  {
    delete[] h_in;
  }
  if (d_in)
  {
    cudaFree(d_in);
  }
  if (d_out)
  {
    cudaFree(d_out);
  }
}

/**
 * Main
 */
int main(int argc, char** argv)
{
  // Initialize command line
  CommandLineArgs args(argc, argv);
  g_verbose = args.CheckCmdLineFlag("v");
  args.GetCmdLineArgument("grid-size", g_grid_size);

  // Print usage
  if (args.CheckCmdLineFlag("help"))
  {
    printf("%s "
           "[--device=<device-id>] "
           "[--grid-size=<grid size>] "
           "[--v] "
           "\n",
           argv[0]);
    exit(0);
  }

  // Initialize device
  CubDebugExit(args.DeviceInit());

  // Run tests
  Test<1024>();
  Test<512>();
  Test<256>();
  Test<128>();
  Test<64>();
  Test<32>();
  Test<16>();

  return 0;
}

#else // < C++14

int main() {}

#endif
