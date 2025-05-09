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

// Ensure printing of CUDA runtime errors to console
#include "cub/util_type.cuh"
#define CUB_STDERR

#include <cub/block/block_load.cuh>
#include <cub/block/block_radix_rank.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/radix_rank_sort_operations.cuh>
#include <cub/util_allocator.cuh>
#include <cub/util_vsmem.cuh>

#include <algorithm>
#include <iostream>
#include <memory>

#include "test_util.h"
#include <stdio.h>

bool g_verbose = false;
cub::CachingDeviceAllocator g_allocator(true);

template <cub::RadixRankAlgorithm RankAlgorithm,
          int BlockThreads,
          int ItemsPerThread,
          int RadixBits,
          cub::BlockScanAlgorithm ScanAlgorithm,
          int Descending,
          typename Key>
__launch_bounds__(BlockThreads, 1) __global__ void kernel(Key* d_keys, int* d_ranks)
{
  using block_radix_rank =
    cub::detail::block_radix_rank_t<RankAlgorithm, BlockThreads, RadixBits, Descending, ScanAlgorithm>;

  using storage_t = typename block_radix_rank::TempStorage;

  // Allocate temp storage in shared memory
  __shared__ storage_t temp_storage;

  // Items per thread
  Key keys[ItemsPerThread];
  int ranks[ItemsPerThread];

  constexpr bool uses_warp_striped_arrangement =
    RankAlgorithm == cub::RadixRankAlgorithm::RADIX_RANK_MATCH
    || RankAlgorithm == cub::RadixRankAlgorithm::RADIX_RANK_MATCH_EARLY_COUNTS_ANY
    || RankAlgorithm == cub::RadixRankAlgorithm::RADIX_RANK_MATCH_EARLY_COUNTS_ATOMIC_OR;

  if (uses_warp_striped_arrangement)
  {
    cub::LoadDirectWarpStriped(threadIdx.x, d_keys, keys);
  }
  else
  {
    cub::LoadDirectBlocked(threadIdx.x, d_keys, keys);
  }

  cub::BFEDigitExtractor<Key> extractor(0, RadixBits);
  block_radix_rank(temp_storage).RankKeys(keys, ranks, extractor);

  if (uses_warp_striped_arrangement)
  {
    cub::StoreDirectWarpStriped(threadIdx.x, d_ranks, ranks);
  }
  else
  {
    cub::StoreDirectBlocked(threadIdx.x, d_ranks, ranks);
  }
}

//---------------------------------------------------------------------
// Host testing subroutines
//---------------------------------------------------------------------

/**
 * Simple key-value pairing
 */
template <typename Key>
struct pair_t
{
  Key key;
  int value;

  bool operator<(const pair_t& b) const
  {
    return (key < b.key);
  }
};

template <bool DESCENDING, typename Key>
void Initialize(GenMode gen_mode, Key* h_keys, int* h_reference_ranks, int num_items, int num_bits)
{
  std::unique_ptr<pair_t<Key>[]> h_pairs_storage(new pair_t<Key>[num_items]);
  pair_t<Key>* h_pairs = h_pairs_storage.get();

  for (int i = 0; i < num_items; ++i)
  {
    InitValue(gen_mode, h_keys[i], i);

    // Mask off unwanted portions
    std::uint64_t base = 0;
    memcpy(&base, &h_keys[i], sizeof(Key));
    base &= (1ull << num_bits) - 1;
    memcpy(&h_keys[i], &base, sizeof(Key));

    h_pairs[i].key   = h_keys[i];
    h_pairs[i].value = i;
  }

  if (DESCENDING)
  {
    std::reverse(h_pairs, h_pairs + num_items);
  }

  std::stable_sort(h_pairs, h_pairs + num_items);

  if (DESCENDING)
  {
    std::reverse(h_pairs, h_pairs + num_items);
  }

  for (int i = 0; i < num_items; ++i)
  {
    h_reference_ranks[h_pairs[i].value] = i;
  }
}

template <cub::RadixRankAlgorithm RankAlgorithm,
          int BlockThreads,
          int ItemsPerThread,
          int RadixBits,
          cub::BlockScanAlgorithm ScanAlgorithm,
          int Descending,
          typename Key>
void TestDriver(GenMode gen_mode)
{
  constexpr int tile_size = BlockThreads * ItemsPerThread;

  // Allocate host arrays
  std::unique_ptr<Key[]> h_keys(new Key[tile_size]);
  std::unique_ptr<int[]> h_ranks(new int[tile_size]);
  std::unique_ptr<int[]> h_reference_ranks(new int[tile_size]);

  // Allocate device arrays
  Key* d_keys  = nullptr;
  int* d_ranks = nullptr;

  CubDebugExit(g_allocator.DeviceAllocate((void**) &d_keys, sizeof(Key) * tile_size));
  CubDebugExit(g_allocator.DeviceAllocate((void**) &d_ranks, sizeof(int) * tile_size));

  // Initialize problem and solution on host
  Initialize<Descending>(gen_mode, h_keys.get(), h_reference_ranks.get(), tile_size, RadixBits);

  // Copy problem to device
  CubDebugExit(cudaMemcpy(d_keys, h_keys.get(), sizeof(Key) * tile_size, cudaMemcpyHostToDevice));

  // Run kernel
  kernel<RankAlgorithm, BlockThreads, ItemsPerThread, RadixBits, ScanAlgorithm, Descending, Key>
    <<<1, BlockThreads>>>(d_keys, d_ranks);

  // Flush kernel output / errors
  CubDebugExit(cudaPeekAtLastError());
  CubDebugExit(cudaDeviceSynchronize());

  // Check keys results
  const bool compare = CompareDeviceResults(h_reference_ranks.get(), d_ranks, tile_size, g_verbose, g_verbose);
  AssertEquals(0, compare);

  if (d_keys)
  {
    CubDebugExit(g_allocator.DeviceFree(d_keys));
  }

  if (d_ranks)
  {
    CubDebugExit(g_allocator.DeviceFree(d_ranks));
  }
}

template <cub::RadixRankAlgorithm RankAlgorithm,
          int BlockThreads,
          int ItemsPerThread,
          int RadixBits,
          cub::BlockScanAlgorithm ScanAlgorithm,
          int Descending,
          typename Key>
void TestValid(cub::Int2Type<true> /*fits_smem_capacity*/)
{
  TestDriver<RankAlgorithm, BlockThreads, ItemsPerThread, RadixBits, ScanAlgorithm, Descending, Key>(UNIFORM);

  TestDriver<RankAlgorithm, BlockThreads, ItemsPerThread, RadixBits, ScanAlgorithm, Descending, Key>(INTEGER_SEED);
}

template <cub::RadixRankAlgorithm RankAlgorithm,
          int BlockThreads,
          int ItemsPerThread,
          int RadixBits,
          cub::BlockScanAlgorithm ScanAlgorithm,
          int Descending,
          typename Key>
void TestValid(cub::Int2Type<false> fits_smem_capacity)
{}

template <cub::RadixRankAlgorithm RankAlgorithm,
          int BlockThreads,
          int ItemsPerThread,
          int RadixBits,
          cub::BlockScanAlgorithm ScanAlgorithm,
          bool Descending,
          typename Key>
void Test()
{
  // Check size of smem storage for the target arch to make sure it will fit
  using block_radix_rank =
    cub::detail::block_radix_rank_t<RankAlgorithm, BlockThreads, RadixBits, Descending, ScanAlgorithm>;
  using storage_t = typename block_radix_rank::TempStorage;

  cub::Int2Type<(sizeof(storage_t) <= cub::detail::max_smem_per_block)> fits_smem_capacity;

  TestValid<RankAlgorithm, BlockThreads, ItemsPerThread, RadixBits, ScanAlgorithm, Descending, Key>(fits_smem_capacity);
}

template <cub::RadixRankAlgorithm RankAlgorithm,
          int BlockThreads,
          int ItemsPerThread,
          int RadixBits,
          cub::BlockScanAlgorithm ScanAlgorithm,
          typename Key>
void Test()
{
  Test<RankAlgorithm, BlockThreads, ItemsPerThread, RadixBits, ScanAlgorithm, true, Key>();
  Test<RankAlgorithm, BlockThreads, ItemsPerThread, RadixBits, ScanAlgorithm, false, Key>();
}

template <cub::RadixRankAlgorithm RankAlgorithm,
          int BlockThreads,
          int ItemsPerThread,
          int RadixBits,
          cub::BlockScanAlgorithm ScanAlgorithm>
void Test()
{
  Test<RankAlgorithm, BlockThreads, ItemsPerThread, RadixBits, ScanAlgorithm, std::uint8_t>();
  Test<RankAlgorithm, BlockThreads, ItemsPerThread, RadixBits, ScanAlgorithm, std::uint16_t>();
}

template <cub::RadixRankAlgorithm RankAlgorithm, int BlockThreads, int ItemsPerThread, int RadixBits>
void Test()
{
  Test<RankAlgorithm, BlockThreads, ItemsPerThread, RadixBits, cub::BLOCK_SCAN_RAKING>();
  Test<RankAlgorithm, BlockThreads, ItemsPerThread, RadixBits, cub::BLOCK_SCAN_WARP_SCANS>();
}

template <cub::RadixRankAlgorithm RankAlgorithm, int BlockThreads, int ItemsPerThread>
void Test()
{
  Test<RankAlgorithm, BlockThreads, ItemsPerThread, 1>();
  Test<RankAlgorithm, BlockThreads, ItemsPerThread, 5>();
}

template <cub::RadixRankAlgorithm RankAlgorithm, int BlockThreads>
void Test()
{
  Test<RankAlgorithm, BlockThreads, 1>();
  Test<RankAlgorithm, BlockThreads, 4>();
}

template <int BlockThreads>
void Test(cub::Int2Type<true> /* multiple of hw warp */)
{
  Test<cub::RadixRankAlgorithm::RADIX_RANK_MATCH, BlockThreads>();

  // TODO(senior-zero):
  // - RADIX_RANK_MATCH_EARLY_COUNTS_ANY
  // - RADIX_RANK_MATCH_EARLY_COUNTS_ATOMIC_OR
}

template <int BlockThreads>
void Test(cub::Int2Type<false> /* multiple of hw warp */)
{}

template <int BlockThreads>
void Test()
{
  Test<cub::RadixRankAlgorithm::RADIX_RANK_BASIC, BlockThreads>();
  Test<cub::RadixRankAlgorithm::RADIX_RANK_MEMOIZE, BlockThreads>();

  Test<BlockThreads>(cub::Int2Type<(BlockThreads % 32) == 0>{});
}

int main(int argc, char** argv)
{
  // Initialize command line
  CommandLineArgs args(argc, argv);
  g_verbose = args.CheckCmdLineFlag("v");

  // Print usage
  if (args.CheckCmdLineFlag("help"))
  {
    printf("%s "
           "[--device=<device-id>] "
           "[--v] "
           "\n",
           argv[0]);
    exit(0);
  }

  // Initialize device
  CubDebugExit(args.DeviceInit());

  Test<16>();
  Test<32>();
  Test<128>();
  Test<130>();

  g_allocator.FreeAllCached();

  return 0;
}
