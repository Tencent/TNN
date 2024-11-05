/******************************************************************************
 * Copyright (c) 2011-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "insert_nested_NVTX_range_guard.h"
// above header needs to be included first

#undef NDEBUG
#include <cub/device/device_scan.cuh>

#include <cassert>

#include <c2h/catch2_test_helper.cuh>

template <class ScanTileStateT>
__global__ void init_kernel(ScanTileStateT tile_state, int blocks_in_grid)
{
  tile_state.InitializeStatus(blocks_in_grid);
}

template <class MessageT>
__global__ void decoupled_look_back_kernel(cub::ScanTileState<MessageT> tile_state, MessageT* tile_data)
{
  using scan_op_t         = cub::Sum;
  using scan_tile_state_t = cub::ScanTileState<MessageT>;
  using tile_prefix_op    = cub::TilePrefixCallbackOp<MessageT, scan_op_t, scan_tile_state_t>;
  using temp_storage_t    = typename tile_prefix_op::TempStorage;

  // Allocate temp storage in shared memory
  __shared__ temp_storage_t temp_storage;

  scan_op_t scan_op{};
  constexpr unsigned int threads_in_warp = 32;
  const unsigned int tid                 = threadIdx.x;

  // Construct prefix op
  tile_prefix_op prefix(tile_state, temp_storage, scan_op);
  const unsigned int tile_idx = prefix.GetTileIdx();

  // "Compute" tile aggregate
  MessageT tile_aggregate = tile_data[tile_idx];

  if (tile_idx == 0)
  {
    // There are no blocks to look back to, immediately set the inclusive state
    if (tid == 0)
    {
      tile_state.SetInclusive(tile_idx, tile_aggregate);
      tile_data[tile_idx] = tile_aggregate;
    }
  }
  else
  {
    // Only the first warp in the block can perform the look back
    const unsigned int warp_id = tid / threads_in_warp;

    if (warp_id == 0)
    {
      // Perform the decoupled look-back
      // Invocation of the prefix will block until the look-back is complete.
      MessageT exclusive_prefix = prefix(tile_aggregate);

      if (tid == 0)
      {
        MessageT inclusive_prefix = scan_op(exclusive_prefix, tile_aggregate);
        tile_data[tile_idx]       = inclusive_prefix;
      }
    }
    __syncthreads();

    assert(tile_data[tile_idx] == prefix.GetInclusivePrefix());
    assert(tile_aggregate == prefix.GetBlockAggregate());
  }
}

using message_types = c2h::type_list<std::uint8_t, std::uint16_t, std::uint32_t, std::uint64_t>;

template <class MessageT>
c2h::host_vector<MessageT> compute_reference(const c2h::device_vector<MessageT>& tile_aggregates)
{
  if (tile_aggregates.empty())
  {
    return {};
  }

  c2h::host_vector<MessageT> reference = tile_aggregates;
  MessageT* h_reference                = thrust::raw_pointer_cast(reference.data());

  MessageT aggregate = h_reference[0];
  for (std::size_t i = 1; i < reference.size(); i++)
  {
    aggregate += h_reference[i];
    h_reference[i] = aggregate;
  }

  return reference;
}

C2H_TEST("Decoupled look-back works with various message types", "[decoupled look-back][device]", message_types)
{
  using message_t         = typename c2h::get<0, TestType>;
  using scan_tile_state_t = cub::ScanTileState<message_t>;

  constexpr int max_tiles = 1024 * 1024;
  const int num_tiles     = GENERATE_COPY(take(10, random(1, max_tiles)));

  c2h::device_vector<message_t> tile_data(num_tiles);
  message_t* d_tile_data = thrust::raw_pointer_cast(tile_data.data());

  c2h::gen(C2H_SEED(2), tile_data);
  c2h::host_vector<message_t> reference = compute_reference(tile_data);

  // Query temporary storage requirements
  std::size_t temp_storage_bytes{};
  scan_tile_state_t::AllocationSize(num_tiles, temp_storage_bytes);

  // Allocate temporary storage
  c2h::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
  std::uint8_t* d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

  // Initialize temporary storage
  scan_tile_state_t tile_status;
  cudaError_t status = tile_status.Init(num_tiles, d_temp_storage, temp_storage_bytes);
  REQUIRE(status == cudaSuccess);

  constexpr unsigned int threads_in_init_block = 256;
  const unsigned int blocks_in_init_grid       = ::cuda::ceil_div(num_tiles, threads_in_init_block);
  init_kernel<<<blocks_in_init_grid, threads_in_init_block>>>(tile_status, num_tiles);
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  // Launch decoupled look-back
  constexpr unsigned int threads_in_block = 256;
  decoupled_look_back_kernel<<<num_tiles, threads_in_block>>>(tile_status, d_tile_data);
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  REQUIRE(reference == tile_data);
}
