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

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/agent/agent_three_way_partition.cuh>
#include <cub/device/dispatch/dispatch_scan.cuh>
#include <cub/device/dispatch/tuning/tuning_three_way_partition.cuh>
#include <cub/thread/thread_operators.cuh>
#include <cub/util_deprecated.cuh>
#include <cub/util_device.cuh>
#include <cub/util_math.cuh>

#include <thrust/system/cuda/detail/core/triple_chevron_launch.h>

#include <cstdio>
#include <iterator>

#include <nv/target>

CUB_NAMESPACE_BEGIN

/******************************************************************************
 * Kernel entry points
 *****************************************************************************/

template <typename ChainedPolicyT,
          typename InputIteratorT,
          typename FirstOutputIteratorT,
          typename SecondOutputIteratorT,
          typename UnselectedOutputIteratorT,
          typename NumSelectedIteratorT,
          typename ScanTileStateT,
          typename SelectFirstPartOp,
          typename SelectSecondPartOp,
          typename OffsetT>
__launch_bounds__(int(ChainedPolicyT::ActivePolicy::ThreeWayPartitionPolicy::BLOCK_THREADS))
  CUB_DETAIL_KERNEL_ATTRIBUTES void DeviceThreeWayPartitionKernel(
    InputIteratorT d_in,
    FirstOutputIteratorT d_first_part_out,
    SecondOutputIteratorT d_second_part_out,
    UnselectedOutputIteratorT d_unselected_out,
    NumSelectedIteratorT d_num_selected_out,
    ScanTileStateT tile_status,
    SelectFirstPartOp select_first_part_op,
    SelectSecondPartOp select_second_part_op,
    OffsetT num_items,
    int num_tiles)
{
  using AgentThreeWayPartitionPolicyT = typename ChainedPolicyT::ActivePolicy::ThreeWayPartitionPolicy;

  // Thread block type for selecting data from input tiles
  using AgentThreeWayPartitionT = AgentThreeWayPartition<
    AgentThreeWayPartitionPolicyT,
    InputIteratorT,
    FirstOutputIteratorT,
    SecondOutputIteratorT,
    UnselectedOutputIteratorT,
    SelectFirstPartOp,
    SelectSecondPartOp,
    OffsetT>;

  // Shared memory for AgentThreeWayPartition
  __shared__ typename AgentThreeWayPartitionT::TempStorage temp_storage;

  // Process tiles
  AgentThreeWayPartitionT(
    temp_storage,
    d_in,
    d_first_part_out,
    d_second_part_out,
    d_unselected_out,
    select_first_part_op,
    select_second_part_op,
    num_items)
    .ConsumeRange(num_tiles, tile_status, d_num_selected_out);
}

/**
 * @brief Initialization kernel for tile status initialization (multi-block)
 *
 * @tparam ScanTileStateT
 *   Tile status interface type
 *
 * @tparam NumSelectedIteratorT
 *   Output iterator type for recording the number of items selected
 *
 * @param[in] tile_state_1
 *   Tile status interface
 *
 * @param[in] tile_state_2
 *   Tile status interface
 *
 * @param[in] num_tiles
 *   Number of tiles
 *
 * @param[out] d_num_selected_out
 *   Pointer to the total number of items selected
 *   (i.e., length of @p d_selected_out)
 */
template <typename ScanTileStateT, typename NumSelectedIteratorT>
CUB_DETAIL_KERNEL_ATTRIBUTES void
DeviceThreeWayPartitionInitKernel(ScanTileStateT tile_state, int num_tiles, NumSelectedIteratorT d_num_selected_out)
{
  // Initialize tile status
  tile_state.InitializeStatus(num_tiles);

  // Initialize d_num_selected_out
  if (blockIdx.x == 0)
  {
    if (threadIdx.x < 2)
    {
      d_num_selected_out[threadIdx.x] = 0;
    }
  }
}

/******************************************************************************
 * Dispatch
 ******************************************************************************/

template <typename InputIteratorT,
          typename FirstOutputIteratorT,
          typename SecondOutputIteratorT,
          typename UnselectedOutputIteratorT,
          typename NumSelectedIteratorT,
          typename SelectFirstPartOp,
          typename SelectSecondPartOp,
          typename OffsetT,
          typename SelectedPolicy =
            detail::device_three_way_partition_policy_hub<cub::detail::value_t<InputIteratorT>, OffsetT>>
struct DispatchThreeWayPartitionIf
{
  /*****************************************************************************
   * Types and constants
   ****************************************************************************/

  using AccumPackHelperT = detail::three_way_partition::accumulator_pack_t<OffsetT>;
  using AccumPackT       = typename AccumPackHelperT::pack_t;
  using ScanTileStateT   = cub::ScanTileState<AccumPackT>;

  static constexpr int INIT_KERNEL_THREADS = 256;

  void* d_temp_storage;
  std::size_t& temp_storage_bytes;
  InputIteratorT d_in;
  FirstOutputIteratorT d_first_part_out;
  SecondOutputIteratorT d_second_part_out;
  UnselectedOutputIteratorT d_unselected_out;
  NumSelectedIteratorT d_num_selected_out;
  SelectFirstPartOp select_first_part_op;
  SelectSecondPartOp select_second_part_op;
  OffsetT num_items;
  cudaStream_t stream;

  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE DispatchThreeWayPartitionIf(
    void* d_temp_storage,
    std::size_t& temp_storage_bytes,
    InputIteratorT d_in,
    FirstOutputIteratorT d_first_part_out,
    SecondOutputIteratorT d_second_part_out,
    UnselectedOutputIteratorT d_unselected_out,
    NumSelectedIteratorT d_num_selected_out,
    SelectFirstPartOp select_first_part_op,
    SelectSecondPartOp select_second_part_op,
    OffsetT num_items,
    cudaStream_t stream)
      : d_temp_storage(d_temp_storage)
      , temp_storage_bytes(temp_storage_bytes)
      , d_in(d_in)
      , d_first_part_out(d_first_part_out)
      , d_second_part_out(d_second_part_out)
      , d_unselected_out(d_unselected_out)
      , d_num_selected_out(d_num_selected_out)
      , select_first_part_op(select_first_part_op)
      , select_second_part_op(select_second_part_op)
      , num_items(num_items)
      , stream(stream)
  {}

  /*****************************************************************************
   * Dispatch entrypoints
   ****************************************************************************/

  template <typename ActivePolicyT, typename ScanInitKernelPtrT, typename SelectIfKernelPtrT>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t
  Invoke(ScanInitKernelPtrT three_way_partition_init_kernel, SelectIfKernelPtrT three_way_partition_kernel)
  {
    cudaError error = cudaSuccess;

    constexpr int block_threads    = ActivePolicyT::ThreeWayPartitionPolicy::BLOCK_THREADS;
    constexpr int items_per_thread = ActivePolicyT::ThreeWayPartitionPolicy::ITEMS_PER_THREAD;

    do
    {
      // Get device ordinal
      int device_ordinal;
      error = CubDebug(cudaGetDevice(&device_ordinal));
      if (cudaSuccess != error)
      {
        break;
      }

      // Number of input tiles
      int tile_size = block_threads * items_per_thread;
      int num_tiles = static_cast<int>(::cuda::ceil_div(num_items, tile_size));

      // Specify temporary storage allocation requirements
      size_t allocation_sizes[1]; // bytes needed for tile status descriptors

      error = CubDebug(ScanTileStateT::AllocationSize(num_tiles, allocation_sizes[0]));
      if (cudaSuccess != error)
      {
        break;
      }

      // Compute allocation pointers into the single storage blob (or compute
      // the necessary size of the blob)
      void* allocations[1] = {};

      error = CubDebug(cub::AliasTemporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes));
      if (cudaSuccess != error)
      {
        break;
      }

      if (d_temp_storage == nullptr)
      {
        // Return if the caller is simply requesting the size of the storage
        // allocation
        break;
      }

      // Return if empty problem
      if (num_items == 0)
      {
        break;
      }

      // Construct the tile status interface
      ScanTileStateT tile_status;

      error = CubDebug(tile_status.Init(num_tiles, allocations[0], allocation_sizes[0]));
      if (cudaSuccess != error)
      {
        break;
      }

      // Log three_way_partition_init_kernel configuration
      int init_grid_size = CUB_MAX(1, ::cuda::ceil_div(num_tiles, INIT_KERNEL_THREADS));

#ifdef CUB_DETAIL_DEBUG_ENABLE_LOG
      _CubLog("Invoking three_way_partition_init_kernel<<<%d, %d, 0, %lld>>>()\n",
              init_grid_size,
              INIT_KERNEL_THREADS,
              reinterpret_cast<long long>(stream));
#endif // CUB_DETAIL_DEBUG_ENABLE_LOG

      // Invoke three_way_partition_init_kernel to initialize tile descriptors
      THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron(init_grid_size, INIT_KERNEL_THREADS, 0, stream)
        .doit(three_way_partition_init_kernel, tile_status, num_tiles, d_num_selected_out);

      // Check for failure to launch
      error = CubDebug(cudaPeekAtLastError());
      if (cudaSuccess != error)
      {
        break;
      }

      // Sync the stream if specified to flush runtime errors
      error = CubDebug(detail::DebugSyncStream(stream));
      if (cudaSuccess != error)
      {
        break;
      }

      // Get max x-dimension of grid
      int max_dim_x;
      error = CubDebug(cudaDeviceGetAttribute(&max_dim_x, cudaDevAttrMaxGridDimX, device_ordinal));
      if (cudaSuccess != error)
      {
        break;
      }

      // Get grid size for scanning tiles
      dim3 scan_grid_size;
      scan_grid_size.z = 1;
      scan_grid_size.y = ::cuda::ceil_div(num_tiles, max_dim_x);
      scan_grid_size.x = CUB_MIN(num_tiles, max_dim_x);

// Log select_if_kernel configuration
#ifdef CUB_DETAIL_DEBUG_ENABLE_LOG
      {
        // Get SM occupancy for select_if_kernel
        int range_select_sm_occupancy;
        error = CubDebug(MaxSmOccupancy(range_select_sm_occupancy, // out
                                        three_way_partition_kernel,
                                        block_threads));
        if (cudaSuccess != error)
        {
          break;
        }

        _CubLog("Invoking three_way_partition_kernel<<<{%d,%d,%d}, %d, 0, %lld>>>(), %d "
                "items per thread, %d SM occupancy\n",
                scan_grid_size.x,
                scan_grid_size.y,
                scan_grid_size.z,
                block_threads,
                reinterpret_cast<long long>(stream),
                items_per_thread,
                range_select_sm_occupancy);
      }
#endif // CUB_DETAIL_DEBUG_ENABLE_LOG

      // Invoke select_if_kernel
      THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron(scan_grid_size, block_threads, 0, stream)
        .doit(three_way_partition_kernel,
              d_in,
              d_first_part_out,
              d_second_part_out,
              d_unselected_out,
              d_num_selected_out,
              tile_status,
              select_first_part_op,
              select_second_part_op,
              num_items,
              num_tiles);

      // Check for failure to launch
      error = CubDebug(cudaPeekAtLastError());
      if (cudaSuccess != error)
      {
        break;
      }

      // Sync the stream if specified to flush runtime errors
      error = CubDebug(detail::DebugSyncStream(stream));
      if (cudaSuccess != error)
      {
        break;
      }
    } while (0);

    return error;
  }

  template <typename ActivePolicyT>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t Invoke()
  {
    using MaxPolicyT = typename SelectedPolicy::MaxPolicy;
    return Invoke<ActivePolicyT>(
      DeviceThreeWayPartitionInitKernel<ScanTileStateT, NumSelectedIteratorT>,
      DeviceThreeWayPartitionKernel<
        MaxPolicyT,
        InputIteratorT,
        FirstOutputIteratorT,
        SecondOutputIteratorT,
        UnselectedOutputIteratorT,
        NumSelectedIteratorT,
        ScanTileStateT,
        SelectFirstPartOp,
        SelectSecondPartOp,
        OffsetT>);
  }

  /**
   * Internal dispatch routine
   */
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t Dispatch(
    void* d_temp_storage,
    std::size_t& temp_storage_bytes,
    InputIteratorT d_in,
    FirstOutputIteratorT d_first_part_out,
    SecondOutputIteratorT d_second_part_out,
    UnselectedOutputIteratorT d_unselected_out,
    NumSelectedIteratorT d_num_selected_out,
    SelectFirstPartOp select_first_part_op,
    SelectSecondPartOp select_second_part_op,
    OffsetT num_items,
    cudaStream_t stream)
  {
    using MaxPolicyT = typename SelectedPolicy::MaxPolicy;

    cudaError error = cudaSuccess;

    do
    {
      // Get PTX version
      int ptx_version = 0;
      error           = CubDebug(cub::PtxVersion(ptx_version));
      if (cudaSuccess != error)
      {
        break;
      }

      DispatchThreeWayPartitionIf dispatch(
        d_temp_storage,
        temp_storage_bytes,
        d_in,
        d_first_part_out,
        d_second_part_out,
        d_unselected_out,
        d_num_selected_out,
        select_first_part_op,
        select_second_part_op,
        num_items,
        stream);

      // Dispatch
      error = CubDebug(MaxPolicyT::Invoke(ptx_version, dispatch));
      if (cudaSuccess != error)
      {
        break;
      }
    } while (0);

    return error;
  }

#ifndef DOXYGEN_SHOULD_SKIP_THIS // Do not document
  CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t Dispatch(
    void* d_temp_storage,
    std::size_t& temp_storage_bytes,
    InputIteratorT d_in,
    FirstOutputIteratorT d_first_part_out,
    SecondOutputIteratorT d_second_part_out,
    UnselectedOutputIteratorT d_unselected_out,
    NumSelectedIteratorT d_num_selected_out,
    SelectFirstPartOp select_first_part_op,
    SelectSecondPartOp select_second_part_op,
    OffsetT num_items,
    cudaStream_t stream,
    bool debug_synchronous)
  {
    CUB_DETAIL_RUNTIME_DEBUG_SYNC_USAGE_LOG

    return Dispatch(
      d_temp_storage,
      temp_storage_bytes,
      d_in,
      d_first_part_out,
      d_second_part_out,
      d_unselected_out,
      d_num_selected_out,
      select_first_part_op,
      select_second_part_op,
      num_items,
      stream);
  }
#endif // DOXYGEN_SHOULD_SKIP_THIS
};

CUB_NAMESPACE_END
