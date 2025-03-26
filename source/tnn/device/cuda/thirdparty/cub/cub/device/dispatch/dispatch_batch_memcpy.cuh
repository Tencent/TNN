/******************************************************************************
 * Copyright (c) 2011-2022, NVIDIA CORPORATION. All rights reserved.
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

/**
 * \file
 * cub::DispatchBatchMemcpy provides device-wide, parallel operations for copying data from a number
 * of given source buffers to their corresponding destination buffer.
 */

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/agent/agent_batch_memcpy.cuh>
#include <cub/agent/single_pass_scan_operators.cuh>
#include <cub/detail/temporary_storage.cuh>
#include <cub/thread/thread_search.cuh>
#include <cub/util_debug.cuh>
#include <cub/util_device.cuh>
#include <cub/util_ptx.cuh>

#include <thrust/system/cuda/detail/core/triple_chevron_launch.h>

#include <cuda/std/type_traits>

#include <cstdint>

CUB_NAMESPACE_BEGIN

namespace detail
{

/**
 * Parameterizable tuning policy type for AgentBatchMemcpy
 */
template <uint32_t _BLOCK_THREADS, uint32_t _BYTES_PER_THREAD>
struct AgentBatchMemcpyLargeBuffersPolicy
{
  /// Threads per thread block
  static constexpr uint32_t BLOCK_THREADS = _BLOCK_THREADS;
  /// The number of bytes each thread copies
  static constexpr uint32_t BYTES_PER_THREAD = _BYTES_PER_THREAD;
};

/**
 * Initialization kernel for tile status initialization (multi-block)
 */
template <typename BufferOffsetScanTileStateT, typename BlockOffsetScanTileStateT, typename TileOffsetT>
CUB_DETAIL_KERNEL_ATTRIBUTES void InitTileStateKernel(
  BufferOffsetScanTileStateT buffer_offset_scan_tile_state,
  BlockOffsetScanTileStateT block_offset_scan_tile_state,
  TileOffsetT num_tiles)
{
  // Initialize tile status
  buffer_offset_scan_tile_state.InitializeStatus(num_tiles);
  block_offset_scan_tile_state.InitializeStatus(num_tiles);
}

/**
 * Kernel that copies buffers that need to be copied by at least one (and potentially many) thread
 * blocks.
 */
template <typename ChainedPolicyT,
          typename BufferOffsetT,
          typename InputBufferIt,
          typename OutputBufferIt,
          typename BufferSizeIteratorT,
          typename BufferTileOffsetItT,
          typename TileT,
          typename TileOffsetT,
          bool IsMemcpy>
__launch_bounds__(int(ChainedPolicyT::ActivePolicy::AgentLargeBufferPolicyT::BLOCK_THREADS))
  CUB_DETAIL_KERNEL_ATTRIBUTES void MultiBlockBatchMemcpyKernel(
    InputBufferIt input_buffer_it,
    OutputBufferIt output_buffer_it,
    BufferSizeIteratorT buffer_sizes,
    BufferTileOffsetItT buffer_tile_offsets,
    TileT buffer_offset_tile,
    TileOffsetT last_tile_offset)
{
  using StatusWord    = typename TileT::StatusWord;
  using ActivePolicyT = typename ChainedPolicyT::ActivePolicy::AgentLargeBufferPolicyT;
  using BufferSizeT   = cub::detail::value_t<BufferSizeIteratorT>;
  /// Internal load/store type. For byte-wise memcpy, a single-byte type
  using AliasT =
    typename ::cuda::std::conditional<IsMemcpy,
                                      std::iterator_traits<char*>,
                                      std::iterator_traits<cub::detail::value_t<InputBufferIt>>>::type::value_type;
  /// Types of the input and output buffers
  using InputBufferT  = cub::detail::value_t<InputBufferIt>;
  using OutputBufferT = cub::detail::value_t<OutputBufferIt>;

  constexpr uint32_t BLOCK_THREADS    = ActivePolicyT::BLOCK_THREADS;
  constexpr uint32_t ITEMS_PER_THREAD = ActivePolicyT::BYTES_PER_THREAD;
  constexpr BufferSizeT TILE_SIZE     = static_cast<BufferSizeT>(BLOCK_THREADS * ITEMS_PER_THREAD);

  BufferOffsetT num_blev_buffers = buffer_offset_tile.LoadValid(last_tile_offset);

  uint32_t tile_id = blockIdx.x;

  // No block-level buffers => we're done here
  if (num_blev_buffers == 0)
  {
    return;
  }

  // While there's still tiles of bytes from block-level buffers to copied
  do
  {
    __shared__ BufferOffsetT block_buffer_id;

    // Make sure thread 0 does not overwrite the buffer id before other threads have finished with
    // the prior iteration of the loop
    CTA_SYNC();

    // Binary search the buffer that this tile belongs to
    if (threadIdx.x == 0)
    {
      block_buffer_id = UpperBound(buffer_tile_offsets, num_blev_buffers, tile_id) - 1;
    }

    // Make sure thread 0 has written the buffer this thread block is assigned to
    CTA_SYNC();

    const BufferOffsetT buffer_id = block_buffer_id;

    // The relative offset of this tile within the buffer it's assigned to
    BufferSizeT tile_offset_within_buffer =
      static_cast<BufferSizeT>(tile_id - buffer_tile_offsets[buffer_id]) * TILE_SIZE;

    // If the tile has already reached beyond the work of the end of the last buffer
    if (buffer_id >= num_blev_buffers - 1 && tile_offset_within_buffer > buffer_sizes[buffer_id])
    {
      return;
    }

    // Tiny remainders are copied without vectorizing laods
    if (buffer_sizes[buffer_id] - tile_offset_within_buffer <= 32)
    {
      BufferSizeT thread_offset = tile_offset_within_buffer + threadIdx.x;
      for (int i = 0; i < ITEMS_PER_THREAD; i++)
      {
        if (thread_offset < buffer_sizes[buffer_id])
        {
          const auto value = read_item<IsMemcpy, AliasT, InputBufferT>(input_buffer_it[buffer_id], thread_offset);
          write_item<IsMemcpy, AliasT, OutputBufferT>(output_buffer_it[buffer_id], thread_offset, value);
        }
        thread_offset += BLOCK_THREADS;
      }
    }
    else
    {
      copy_items<IsMemcpy, BLOCK_THREADS, InputBufferT, OutputBufferT, BufferSizeT>(
        input_buffer_it[buffer_id],
        output_buffer_it[buffer_id],
        (cub::min)(buffer_sizes[buffer_id] - tile_offset_within_buffer, TILE_SIZE),
        tile_offset_within_buffer);
    }

    tile_id += gridDim.x;
  } while (true);
}

/**
 * @brief Kernel that copies data from a batch of given source buffers to their corresponding
 * destination buffer. If a buffer's size is to large to be copied by a single thread block, that
 * buffer is put into a queue of buffers that will get picked up later on, where multiple blocks
 * collaborate on each of these buffers. All other buffers get copied straight away.
 *
 * @param input_buffer_it [in] Iterator providing the pointers to the source memory buffers
 * @param output_buffer_it [in] Iterator providing the pointers to the destination memory buffers
 * @param buffer_sizes [in] Iterator providing the number of bytes to be copied for each pair of
 * buffers
 * @param num_buffers [in] The total number of buffer pairs
 * @param blev_buffer_srcs [out] The source pointers of buffers that require block-level
 * collaboration
 * @param blev_buffer_dsts [out] The destination pointers of buffers that require block-level
 * collaboration
 * @param blev_buffer_sizes [out] The sizes of buffers that require block-level collaboration
 * @param blev_buffer_scan_state [in,out] Tile states for the prefix sum over the count of buffers
 * requiring block-level collaboration (to "stream compact" (aka "select") BLEV-buffers)
 * @param blev_block_scan_state [in,out] Tile states for the prefix sum over the number of thread
 * blocks getting assigned to each buffer that requires block-level collaboration
 */
template <typename ChainedPolicyT,
          typename InputBufferIt,
          typename OutputBufferIt,
          typename BufferSizeIteratorT,
          typename BufferOffsetT,
          typename BlevBufferSrcsOutItT,
          typename BlevBufferDstsOutItT,
          typename BlevBufferSizesOutItT,
          typename BlevBufferTileOffsetsOutItT,
          typename BlockOffsetT,
          typename BLevBufferOffsetTileState,
          typename BLevBlockOffsetTileState,
          bool IsMemcpy>
__launch_bounds__(int(ChainedPolicyT::ActivePolicy::AgentSmallBufferPolicyT::BLOCK_THREADS))
  CUB_DETAIL_KERNEL_ATTRIBUTES void BatchMemcpyKernel(
    InputBufferIt input_buffer_it,
    OutputBufferIt output_buffer_it,
    BufferSizeIteratorT buffer_sizes,
    BufferOffsetT num_buffers,
    BlevBufferSrcsOutItT blev_buffer_srcs,
    BlevBufferDstsOutItT blev_buffer_dsts,
    BlevBufferSizesOutItT blev_buffer_sizes,
    BlevBufferTileOffsetsOutItT blev_buffer_tile_offsets,
    BLevBufferOffsetTileState blev_buffer_scan_state,
    BLevBlockOffsetTileState blev_block_scan_state)
{
  // Internal type used for storing a buffer's size
  using BufferSizeT = cub::detail::value_t<BufferSizeIteratorT>;

  // Alias the correct tuning policy for the current compilation pass' architecture
  using AgentBatchMemcpyPolicyT = typename ChainedPolicyT::ActivePolicy::AgentSmallBufferPolicyT;

  // Block-level specialization
  using AgentBatchMemcpyT = AgentBatchMemcpy<
    AgentBatchMemcpyPolicyT,
    InputBufferIt,
    OutputBufferIt,
    BufferSizeIteratorT,
    BufferOffsetT,
    BlevBufferSrcsOutItT,
    BlevBufferDstsOutItT,
    BlevBufferSizesOutItT,
    BlevBufferTileOffsetsOutItT,
    BlockOffsetT,
    BLevBufferOffsetTileState,
    BLevBlockOffsetTileState,
    IsMemcpy>;

  // Shared memory for AgentBatchMemcpy
  __shared__ typename AgentBatchMemcpyT::TempStorage temp_storage;

  // Process this block's tile of input&output buffer pairs
  AgentBatchMemcpyT(
    temp_storage,
    input_buffer_it,
    output_buffer_it,
    buffer_sizes,
    num_buffers,
    blev_buffer_srcs,
    blev_buffer_dsts,
    blev_buffer_sizes,
    blev_buffer_tile_offsets,
    blev_buffer_scan_state,
    blev_block_scan_state)
    .ConsumeTile(blockIdx.x);
}

template <class BufferOffsetT, class BlockOffsetT>
struct DeviceBatchMemcpyPolicy
{
  static constexpr uint32_t BLOCK_THREADS         = 128U;
  static constexpr uint32_t BUFFERS_PER_THREAD    = 4U;
  static constexpr uint32_t TLEV_BYTES_PER_THREAD = 8U;

  static constexpr uint32_t LARGE_BUFFER_BLOCK_THREADS    = 256U;
  static constexpr uint32_t LARGE_BUFFER_BYTES_PER_THREAD = 32U;

  static constexpr uint32_t WARP_LEVEL_THRESHOLD  = 128;
  static constexpr uint32_t BLOCK_LEVEL_THRESHOLD = 8 * 1024;

  using buff_delay_constructor_t  = detail::default_delay_constructor_t<BufferOffsetT>;
  using block_delay_constructor_t = detail::default_delay_constructor_t<BlockOffsetT>;

  /// SM35
  struct Policy350 : ChainedPolicy<350, Policy350, Policy350>
  {
    static constexpr bool PREFER_POW2_BITS = true;
    using AgentSmallBufferPolicyT          = AgentBatchMemcpyPolicy<
               BLOCK_THREADS,
               BUFFERS_PER_THREAD,
               TLEV_BYTES_PER_THREAD,
               PREFER_POW2_BITS,
               LARGE_BUFFER_BLOCK_THREADS * LARGE_BUFFER_BYTES_PER_THREAD,
               WARP_LEVEL_THRESHOLD,
               BLOCK_LEVEL_THRESHOLD,
               buff_delay_constructor_t,
               block_delay_constructor_t>;

    using AgentLargeBufferPolicyT =
      AgentBatchMemcpyLargeBuffersPolicy<LARGE_BUFFER_BLOCK_THREADS, LARGE_BUFFER_BYTES_PER_THREAD>;
  };

  /// SM70
  struct Policy700 : ChainedPolicy<700, Policy700, Policy350>
  {
    static constexpr bool PREFER_POW2_BITS = false;
    using AgentSmallBufferPolicyT          = AgentBatchMemcpyPolicy<
               BLOCK_THREADS,
               BUFFERS_PER_THREAD,
               TLEV_BYTES_PER_THREAD,
               PREFER_POW2_BITS,
               LARGE_BUFFER_BLOCK_THREADS * LARGE_BUFFER_BYTES_PER_THREAD,
               WARP_LEVEL_THRESHOLD,
               BLOCK_LEVEL_THRESHOLD,
               buff_delay_constructor_t,
               block_delay_constructor_t>;

    using AgentLargeBufferPolicyT =
      AgentBatchMemcpyLargeBuffersPolicy<LARGE_BUFFER_BLOCK_THREADS, LARGE_BUFFER_BYTES_PER_THREAD>;
  };

  using MaxPolicy = Policy700;
};

/**
 * @tparam InputBufferIt **[inferred]** Random-access input iterator type providing the pointers
 * to the source memory buffers
 * @tparam OutputBufferIt **[inferred]** Random-access input iterator type providing the pointers
 * to the destination memory buffers
 * @tparam BufferSizeIteratorT **[inferred]** Random-access input iterator type providing the
 * number of bytes to be copied for each pair of buffers
 * @tparam BufferOffsetT Integer type large enough to hold any offset in [0, num_buffers)
 * @tparam BlockOffsetT Integer type large enough to hold any offset in [0,
 * num_thread_blocks_launched)
 */
template <typename InputBufferIt,
          typename OutputBufferIt,
          typename BufferSizeIteratorT,
          typename BufferOffsetT,
          typename BlockOffsetT,
          typename SelectedPolicy = DeviceBatchMemcpyPolicy<BufferOffsetT, BlockOffsetT>,
          bool IsMemcpy           = true>
struct DispatchBatchMemcpy : SelectedPolicy
{
  //------------------------------------------------------------------------------
  // TYPE ALIASES
  //------------------------------------------------------------------------------
  // Tile state for the single-pass prefix scan to "stream compact" (aka "select") the buffers
  // requiring block-level collaboration
  using BufferPartitionScanTileStateT = typename cub::ScanTileState<BufferOffsetT>;

  // Tile state for the single-pass prefix scan to keep track of how many blocks are assigned to
  // each of the buffers requiring block-level collaboration
  using BufferTileOffsetScanStateT = typename cub::ScanTileState<BlockOffsetT>;

  // Internal type used to keep track of a buffer's size
  using BufferSizeT = cub::detail::value_t<BufferSizeIteratorT>;

  //------------------------------------------------------------------------------
  // Member Veriables
  //------------------------------------------------------------------------------
  void* d_temp_storage;
  size_t& temp_storage_bytes;
  InputBufferIt input_buffer_it;
  OutputBufferIt output_buffer_it;
  BufferSizeIteratorT buffer_sizes;
  BufferOffsetT num_buffers;
  cudaStream_t stream;

  //------------------------------------------------------------------------------
  // Constructor
  //------------------------------------------------------------------------------
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE DispatchBatchMemcpy(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputBufferIt input_buffer_it,
    OutputBufferIt output_buffer_it,
    BufferSizeIteratorT buffer_sizes,
    BufferOffsetT num_buffers,
    cudaStream_t stream)
      : d_temp_storage(d_temp_storage)
      , temp_storage_bytes(temp_storage_bytes)
      , input_buffer_it(input_buffer_it)
      , output_buffer_it(output_buffer_it)
      , buffer_sizes(buffer_sizes)
      , num_buffers(num_buffers)
      , stream(stream)
  {}

  //------------------------------------------------------------------------------
  // Chained policy invocation
  //------------------------------------------------------------------------------
  /**
   * @brief Tuning policy invocation. This member function template is getting instantiated for all
   * tuning policies in the tuning policy chain. It is, however, *invoked* for the correct tuning
   * policy only.
   */
  template <typename ActivePolicyT>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t Invoke()
  {
    using MaxPolicyT = typename DispatchBatchMemcpy::MaxPolicy;

    // Single-pass prefix scan tile states for the prefix-sum over the number of block-level buffers
    using BLevBufferOffsetTileState = cub::ScanTileState<BufferOffsetT>;

    // Single-pass prefix scan tile states for the prefix sum over the number of thread blocks
    // assigned to each of the block-level buffers
    using BLevBlockOffsetTileState = cub::ScanTileState<BlockOffsetT>;

    cudaError error = cudaSuccess;

    enum : uint32_t
    {
      // Memory for the source pointers of the buffers that require block-level collaboration
      MEM_BLEV_BUFFER_SRCS = 0,
      // Memory for the destination pointers of the buffers that require block-level collaboration
      MEM_BLEV_BUFFER_DSTS,
      // Memory for the block-level buffers' sizes
      MEM_BLEV_BUFFER_SIZES,
      // Memory to keep track of the assignment of thread blocks to block-level buffers
      MEM_BLEV_BUFFER_TBLOCK,
      // Memory for the tile states of the prefix sum over the number of buffers that require
      // block-level collaboration
      MEM_BLEV_BUFFER_SCAN_STATE,
      // Memory for the scan tile states of the prefix sum over the number of thread block's
      // assigned up to and including a certain block-level buffer
      MEM_BLEV_BLOCK_SCAN_STATE,
      // Total number of distinct memory allocations in the temporary storage memory BLOB
      MEM_NUM_ALLOCATIONS
    };

    // Number of threads per block for initializing the grid states
    constexpr BlockOffsetT INIT_KERNEL_THREADS = 128U;

    // The number of buffers that get processed per thread block
    constexpr uint32_t TILE_SIZE = ActivePolicyT::AgentSmallBufferPolicyT::BLOCK_THREADS
                                 * ActivePolicyT::AgentSmallBufferPolicyT::BUFFERS_PER_THREAD;

    // The number of thread blocks (or tiles) required to process all of the given buffers
    BlockOffsetT num_tiles = ::cuda::ceil_div(num_buffers, TILE_SIZE);

    using BlevBufferSrcsOutT          = ::cuda::std::_If<IsMemcpy, const void*, cub::detail::value_t<InputBufferIt>>;
    using BlevBufferDstOutT           = ::cuda::std::_If<IsMemcpy, void*, cub::detail::value_t<OutputBufferIt>>;
    using BlevBufferSrcsOutItT        = BlevBufferSrcsOutT*;
    using BlevBufferDstsOutItT        = BlevBufferDstOutT*;
    using BlevBufferSizesOutItT       = BufferSizeT*;
    using BlevBufferTileOffsetsOutItT = BlockOffsetT*;

    temporary_storage::layout<MEM_NUM_ALLOCATIONS> temporary_storage_layout;

    auto blev_buffer_srcs_slot       = temporary_storage_layout.get_slot(MEM_BLEV_BUFFER_SRCS);
    auto blev_buffer_dsts_slot       = temporary_storage_layout.get_slot(MEM_BLEV_BUFFER_DSTS);
    auto blev_buffer_sizes_slot      = temporary_storage_layout.get_slot(MEM_BLEV_BUFFER_SIZES);
    auto blev_buffer_block_slot      = temporary_storage_layout.get_slot(MEM_BLEV_BUFFER_TBLOCK);
    auto blev_buffer_scan_slot       = temporary_storage_layout.get_slot(MEM_BLEV_BUFFER_SCAN_STATE);
    auto blev_buffer_block_scan_slot = temporary_storage_layout.get_slot(MEM_BLEV_BLOCK_SCAN_STATE);

    auto blev_buffer_srcs_alloc  = blev_buffer_srcs_slot->template create_alias<BlevBufferSrcsOutT>();
    auto blev_buffer_dsts_alloc  = blev_buffer_dsts_slot->template create_alias<BlevBufferDstOutT>();
    auto blev_buffer_sizes_alloc = blev_buffer_sizes_slot->template create_alias<BufferSizeT>();
    auto blev_buffer_block_alloc = blev_buffer_block_slot->template create_alias<BlockOffsetT>();
    auto blev_buffer_scan_alloc  = blev_buffer_scan_slot->template create_alias<uint8_t>();
    auto blev_block_scan_alloc   = blev_buffer_block_scan_slot->template create_alias<uint8_t>();

    std::size_t buffer_offset_scan_storage = 0;
    std::size_t blev_block_scan_storage    = 0;
    error =
      CubDebug(BLevBufferOffsetTileState::AllocationSize(static_cast<int32_t>(num_tiles), buffer_offset_scan_storage));
    if (error)
    {
      return error;
    }

    error =
      CubDebug(BLevBlockOffsetTileState::AllocationSize(static_cast<int32_t>(num_tiles), blev_block_scan_storage));
    if (error)
    {
      return error;
    }

    blev_buffer_srcs_alloc.grow(num_buffers);
    blev_buffer_dsts_alloc.grow(num_buffers);
    blev_buffer_sizes_alloc.grow(num_buffers);
    blev_buffer_block_alloc.grow(num_buffers);
    blev_buffer_scan_alloc.grow(buffer_offset_scan_storage);
    blev_block_scan_alloc.grow(blev_block_scan_storage);

    // Just return if no temporary storage is provided
    if (d_temp_storage == nullptr)
    {
      temp_storage_bytes = temporary_storage_layout.get_size();
      return error;
    }

    // Return if empty problem
    if (num_buffers == 0)
    {
      return error;
    }

    // Alias memory buffers into the storage blob
    error = CubDebug(temporary_storage_layout.map_to_buffer(d_temp_storage, temp_storage_bytes));
    if (cudaSuccess != error)
    {
      return error;
    }

    // Alias into temporary storage allocation
    BlevBufferSrcsOutItT d_blev_src_buffers          = blev_buffer_srcs_alloc.get();
    BlevBufferDstsOutItT d_blev_dst_buffers          = blev_buffer_dsts_alloc.get();
    BlevBufferSizesOutItT d_blev_buffer_sizes        = blev_buffer_sizes_alloc.get();
    BlevBufferTileOffsetsOutItT d_blev_block_offsets = blev_buffer_block_alloc.get();

    // Kernels' grid sizes
    BlockOffsetT init_grid_size         = ::cuda::ceil_div(num_tiles, INIT_KERNEL_THREADS);
    BlockOffsetT batch_memcpy_grid_size = num_tiles;

    // Kernels
    auto init_scan_states_kernel =
      InitTileStateKernel<BLevBufferOffsetTileState, BLevBlockOffsetTileState, BlockOffsetT>;
    auto batch_memcpy_non_blev_kernel = BatchMemcpyKernel<
      MaxPolicyT,
      InputBufferIt,
      OutputBufferIt,
      BufferSizeIteratorT,
      BufferOffsetT,
      BlevBufferSrcsOutItT,
      BlevBufferDstsOutItT,
      BlevBufferSizesOutItT,
      BlevBufferTileOffsetsOutItT,
      BlockOffsetT,
      BLevBufferOffsetTileState,
      BLevBlockOffsetTileState,
      IsMemcpy>;

    auto multi_block_memcpy_kernel = MultiBlockBatchMemcpyKernel<
      MaxPolicyT,
      BufferOffsetT,
      BlevBufferSrcsOutItT,
      BlevBufferDstsOutItT,
      BlevBufferSizesOutItT,
      BlevBufferTileOffsetsOutItT,
      BLevBufferOffsetTileState,
      BlockOffsetT,
      IsMemcpy>;

    constexpr uint32_t BLEV_BLOCK_THREADS = ActivePolicyT::AgentLargeBufferPolicyT::BLOCK_THREADS;

    // Get device ordinal
    int device_ordinal;
    error = CubDebug(cudaGetDevice(&device_ordinal));
    if (cudaSuccess != error)
    {
      return error;
    }

    // Get SM count
    int sm_count;
    error = CubDebug(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_ordinal));
    if (cudaSuccess != error)
    {
      return error;
    }

    // Get SM occupancy for the batch memcpy block-level buffers kernel
    int batch_memcpy_blev_occupancy;
    error = CubDebug(MaxSmOccupancy(batch_memcpy_blev_occupancy, multi_block_memcpy_kernel, BLEV_BLOCK_THREADS));
    if (cudaSuccess != error)
    {
      return error;
    }

    int batch_memcpy_blev_grid_size =
      static_cast<int>(sm_count * batch_memcpy_blev_occupancy * CUB_SUBSCRIPTION_FACTOR(0));

    // Construct the tile status for the buffer prefix sum
    BLevBufferOffsetTileState buffer_scan_tile_state;
    error = CubDebug(buffer_scan_tile_state.Init(
      static_cast<int32_t>(num_tiles), blev_buffer_scan_alloc.get(), buffer_offset_scan_storage));
    if (cudaSuccess != error)
    {
      return error;
    }

    // Construct the tile status for thread blocks-to-buffer-assignment prefix sum
    BLevBlockOffsetTileState block_scan_tile_state;
    error = CubDebug(block_scan_tile_state.Init(
      static_cast<int32_t>(num_tiles), blev_block_scan_alloc.get(), blev_block_scan_storage));
    if (cudaSuccess != error)
    {
      return error;
    }

#ifdef CUB_DETAIL_DEBUG_ENABLE_LOG
    _CubLog("Invoking "
            "InitTileStateKernel<<<%d, %d, 0, %lld>>>()\n",
            static_cast<int>(init_grid_size),
            INIT_KERNEL_THREADS,
            (long long) stream);
#endif

    // Invoke init_kernel to initialize buffer prefix sum-tile descriptors
    error = THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron(init_grid_size, INIT_KERNEL_THREADS, 0, stream)
              .doit(init_scan_states_kernel, buffer_scan_tile_state, block_scan_tile_state, num_tiles);

    // Check for failure to launch
    error = CubDebug(error);
    if (cudaSuccess != error)
    {
      return error;
    }

    // Sync the stream if specified to flush runtime errors
    error = CubDebug(detail::DebugSyncStream(stream));

    // Check for failure to launch
    if (cudaSuccess != error)
    {
      return error;
    }

#ifdef CUB_DETAIL_DEBUG_ENABLE_LOG
    _CubLog("Invoking "
            "BatchMemcpyKernel<<<%d, %d, 0, %lld>>>()\n",
            static_cast<int>(batch_memcpy_grid_size),
            ActivePolicyT::AgentSmallBufferPolicyT::BLOCK_THREADS,
            (long long) stream);
#endif

    // Invoke kernel to copy small buffers and put the larger ones into a queue that will get picked
    // up by next kernel
    error =
      THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron(
        batch_memcpy_grid_size, ActivePolicyT::AgentSmallBufferPolicyT::BLOCK_THREADS, 0, stream)
        .doit(batch_memcpy_non_blev_kernel,
              input_buffer_it,
              output_buffer_it,
              buffer_sizes,
              num_buffers,
              d_blev_src_buffers,
              d_blev_dst_buffers,
              d_blev_buffer_sizes,
              d_blev_block_offsets,
              buffer_scan_tile_state,
              block_scan_tile_state);

    // Check for failure to launch
    error = CubDebug(error);
    if (cudaSuccess != error)
    {
      return error;
    }

    // Sync the stream if specified to flush runtime errors
    error = CubDebug(detail::DebugSyncStream(stream));
    if (cudaSuccess != error)
    {
      return error;
    }

#ifdef CUB_DETAIL_DEBUG_ENABLE_LOG
    _CubLog("Invoking "
            "MultiBlockBatchMemcpyKernel<<<%d, %d, 0, %lld>>>()\n",
            static_cast<int>(batch_memcpy_blev_grid_size),
            BLEV_BLOCK_THREADS,
            (long long) stream);
#endif

    error =
      THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron(batch_memcpy_blev_grid_size, BLEV_BLOCK_THREADS, 0, stream)
        .doit(multi_block_memcpy_kernel,
              d_blev_src_buffers,
              d_blev_dst_buffers,
              d_blev_buffer_sizes,
              d_blev_block_offsets,
              buffer_scan_tile_state,
              batch_memcpy_grid_size - 1);

    // Check for failure to launch
    error = CubDebug(error);
    if (cudaSuccess != error)
    {
      return error;
    }

    // Sync the stream if specified to flush runtime errors
    error = CubDebug(detail::DebugSyncStream(stream));

    return error;
  }

  //------------------------------------------------------------------------------
  // Dispatch entrypoints
  //------------------------------------------------------------------------------
  /**
   * Internal dispatch routine
   */
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t Dispatch(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputBufferIt input_buffer_it,
    OutputBufferIt output_buffer_it,
    BufferSizeIteratorT buffer_sizes,
    BufferOffsetT num_buffers,
    cudaStream_t stream)
  {
    using MaxPolicyT = typename DispatchBatchMemcpy::MaxPolicy;

    cudaError_t error = cudaSuccess;

    // Get PTX version
    int ptx_version = 0;
    error           = CubDebug(PtxVersion(ptx_version));
    if (cudaSuccess != error)
    {
      return error;
    }

    // Create dispatch functor
    DispatchBatchMemcpy dispatch(
      d_temp_storage, temp_storage_bytes, input_buffer_it, output_buffer_it, buffer_sizes, num_buffers, stream);

    // Dispatch to chained policy
    error = CubDebug(MaxPolicyT::Invoke(ptx_version, dispatch));
    if (cudaSuccess != error)
    {
      return error;
    }
    return error;
  }
};

} // namespace detail

CUB_NAMESPACE_END
