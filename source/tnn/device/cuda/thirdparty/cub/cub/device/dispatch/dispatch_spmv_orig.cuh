
/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
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
 * @file
 * cub::DeviceSpmv provides device-wide parallel operations for performing sparse-matrix * vector
 * multiplication (SpMV).
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

#include <cub/agent/agent_segment_fixup.cuh>
#include <cub/agent/agent_spmv_orig.cuh>
#include <cub/agent/single_pass_scan_operators.cuh>
#include <cub/grid/grid_queue.cuh>
#include <cub/thread/thread_search.cuh>
#include <cub/util_debug.cuh>
#include <cub/util_deprecated.cuh>
#include <cub/util_device.cuh>
#include <cub/util_math.cuh>
#include <cub/util_type.cuh>

#include <thrust/system/cuda/detail/core/triple_chevron_launch.h>

#include <cstdio>
#include <iterator>

#include <nv/target>

CUB_NAMESPACE_BEGIN

/******************************************************************************
 * SpMV kernel entry points
 *****************************************************************************/

/**
 * @brief Spmv search kernel. Identifies merge path starting coordinates for each tile.
 *
 * @tparam AgentSpmvPolicyT
 *   Parameterized SpmvPolicy tuning policy type
 *
 * @tparam ValueT
 *   Matrix and vector value type
 *
 * @tparam OffsetT
 *   Signed integer type for sequence offsets
 *
 * @param[in] spmv_params
 *   SpMV input parameter bundle
 */
template <typename AgentSpmvPolicyT, typename ValueT, typename OffsetT>
CUB_DETAIL_KERNEL_ATTRIBUTES void DeviceSpmv1ColKernel(SpmvParams<ValueT, OffsetT> spmv_params)
{
  using VectorValueIteratorT =
    CacheModifiedInputIterator<AgentSpmvPolicyT::VECTOR_VALUES_LOAD_MODIFIER, ValueT, OffsetT>;

  VectorValueIteratorT wrapped_vector_x(spmv_params.d_vector_x);

  int row_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (row_idx < spmv_params.num_rows)
  {
    OffsetT end_nonzero_idx = spmv_params.d_row_end_offsets[row_idx];
    OffsetT nonzero_idx     = spmv_params.d_row_end_offsets[row_idx - 1];

    ValueT value = 0.0;
    if (end_nonzero_idx != nonzero_idx)
    {
      value = spmv_params.d_values[nonzero_idx] * wrapped_vector_x[spmv_params.d_column_indices[nonzero_idx]];
    }

    spmv_params.d_vector_y[row_idx] = value;
  }
}

/**
 * @brief Spmv search kernel. Identifies merge path starting coordinates for each tile.
 *
 * @tparam SpmvPolicyT
 *   Parameterized SpmvPolicy tuning policy type
 *
 * @tparam OffsetT
 *   Signed integer type for sequence offsets
 *
 * @tparam CoordinateT
 *   Merge path coordinate type
 *
 * @tparam SpmvParamsT
 *   SpmvParams type
 *
 * @param[in] num_merge_tiles
 *   Number of SpMV merge tiles (spmv grid size)
 *
 * @param[out] d_tile_coordinates
 *   Pointer to the temporary array of tile starting coordinates
 *
 * @param[in] spmv_params
 *   SpMV input parameter bundle
 */
template <typename SpmvPolicyT, typename OffsetT, typename CoordinateT, typename SpmvParamsT>
CUB_DETAIL_KERNEL_ATTRIBUTES void
DeviceSpmvSearchKernel(int num_merge_tiles, CoordinateT* d_tile_coordinates, SpmvParamsT spmv_params)
{
  /// Constants
  enum
  {
    BLOCK_THREADS    = SpmvPolicyT::BLOCK_THREADS,
    ITEMS_PER_THREAD = SpmvPolicyT::ITEMS_PER_THREAD,
    TILE_ITEMS       = BLOCK_THREADS * ITEMS_PER_THREAD,
  };

  using RowOffsetsSearchIteratorT =
    CacheModifiedInputIterator<SpmvPolicyT::ROW_OFFSETS_SEARCH_LOAD_MODIFIER, OffsetT, OffsetT>;

  // Find the starting coordinate for all tiles (plus the end coordinate of the last one)
  int tile_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (tile_idx < num_merge_tiles + 1)
  {
    OffsetT diagonal = (tile_idx * TILE_ITEMS);
    CoordinateT tile_coordinate;
    CountingInputIterator<OffsetT> nonzero_indices(0);

    // Search the merge path
    MergePathSearch(
      diagonal,
      RowOffsetsSearchIteratorT(spmv_params.d_row_end_offsets),
      nonzero_indices,
      spmv_params.num_rows,
      spmv_params.num_nonzeros,
      tile_coordinate);

    // Output starting offset
    d_tile_coordinates[tile_idx] = tile_coordinate;
  }
}

/**
 * @brief Spmv agent entry point
 *
 * @tparam SpmvPolicyT
 *   Parameterized SpmvPolicy tuning policy type
 *
 * @tparam ScanTileStateT
 *   Tile status interface type
 *
 * @tparam ValueT
 *   Matrix and vector value type
 *
 * @tparam OffsetT
 *   Signed integer type for sequence offsets
 *
 * @tparam CoordinateT
 *   Merge path coordinate type
 *
 * @tparam HAS_ALPHA
 *   Whether the input parameter Alpha is 1
 *
 * @tparam HAS_BETA
 *   Whether the input parameter Beta is 0
 *
 * @param[in] spmv_params
 *   SpMV input parameter bundle
 *
 * @param[in] d_tile_coordinates
 *   Pointer to the temporary array of tile starting coordinates
 *
 * @param[out] d_tile_carry_pairs
 *   Pointer to the temporary array carry-out dot product row-ids, one per block
 *
 * @param[in] num_tiles
 *   Number of merge tiles
 *
 * @param[in] tile_state
 *   Tile status interface for fixup reduce-by-key kernel
 *
 * @param[in] num_segment_fixup_tiles
 *   Number of reduce-by-key tiles (fixup grid size)
 */
template <typename SpmvPolicyT,
          typename ScanTileStateT,
          typename ValueT,
          typename OffsetT,
          typename CoordinateT,
          bool HAS_ALPHA,
          bool HAS_BETA>
__launch_bounds__(int(SpmvPolicyT::BLOCK_THREADS)) CUB_DETAIL_KERNEL_ATTRIBUTES void DeviceSpmvKernel(
  SpmvParams<ValueT, OffsetT> spmv_params,
  CoordinateT* d_tile_coordinates,
  KeyValuePair<OffsetT, ValueT>* d_tile_carry_pairs,
  int num_tiles,
  ScanTileStateT tile_state,
  int num_segment_fixup_tiles)
{
  // Spmv agent type specialization
  using AgentSpmvT = AgentSpmv<SpmvPolicyT, ValueT, OffsetT, HAS_ALPHA, HAS_BETA>;

  // Shared memory for AgentSpmv
  __shared__ typename AgentSpmvT::TempStorage temp_storage;

  AgentSpmvT(temp_storage, spmv_params).ConsumeTile(d_tile_coordinates, d_tile_carry_pairs, num_tiles);

  // Initialize fixup tile status
  tile_state.InitializeStatus(num_segment_fixup_tiles);
}

/**
 * @tparam ValueT
 *   Matrix and vector value type
 *
 * @tparam OffsetT
 *   Signed integer type for sequence offsets
 *
 * @tparam HAS_BETA
 *   Whether the input parameter Beta is 0
 */
template <typename ValueT, typename OffsetT, bool HAS_BETA>
CUB_DETAIL_KERNEL_ATTRIBUTES void DeviceSpmvEmptyMatrixKernel(SpmvParams<ValueT, OffsetT> spmv_params)
{
  const int row = static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x);

  if (row < spmv_params.num_rows)
  {
    ValueT result = 0.0;

    _CCCL_IF_CONSTEXPR (HAS_BETA)
    {
      result += spmv_params.beta * spmv_params.d_vector_y[row];
    }

    spmv_params.d_vector_y[row] = result;
  }
}

/**
 * @brief Multi-block reduce-by-key sweep kernel entry point
 *
 * @tparam AgentSegmentFixupPolicyT
 *   Parameterized AgentSegmentFixupPolicy tuning policy type
 *
 * @tparam PairsInputIteratorT
 *   Random-access input iterator type for keys
 *
 * @tparam AggregatesOutputIteratorT
 *   Random-access output iterator type for values
 *
 * @tparam OffsetT
 *   Signed integer type for global offsets
 *
 * @tparam ScanTileStateT
 *   Tile status interface type
 *
 * @param[in] d_pairs_in
 *   Pointer to the array carry-out dot product row-ids, one per spmv block
 *
 * @param[in,out] d_aggregates_out
 *   Output value aggregates
 *
 * @param[in] num_items
 *   Total number of items to select from
 *
 * @param[in] num_tiles
 *   Total number of tiles for the entire problem
 *
 * @param[in] tile_state
 *   Tile status interface
 */
template <typename AgentSegmentFixupPolicyT,
          typename PairsInputIteratorT,
          typename AggregatesOutputIteratorT,
          typename OffsetT,
          typename ScanTileStateT>
__launch_bounds__(int(AgentSegmentFixupPolicyT::BLOCK_THREADS))
  CUB_DETAIL_KERNEL_ATTRIBUTES void DeviceSegmentFixupKernel(
    PairsInputIteratorT d_pairs_in,
    AggregatesOutputIteratorT d_aggregates_out,
    OffsetT num_items,
    int num_tiles,
    ScanTileStateT tile_state)
{
  // Thread block type for reducing tiles of value segments
  using AgentSegmentFixupT =
    AgentSegmentFixup<AgentSegmentFixupPolicyT,
                      PairsInputIteratorT,
                      AggregatesOutputIteratorT,
                      cub::Equality,
                      cub::Sum,
                      OffsetT>;

  // Shared memory for AgentSegmentFixup
  __shared__ typename AgentSegmentFixupT::TempStorage temp_storage;

  // Process tiles
  AgentSegmentFixupT(temp_storage, d_pairs_in, d_aggregates_out, cub::Equality(), cub::Sum())
    .ConsumeRange(num_items, num_tiles, tile_state);
}

/******************************************************************************
 * Dispatch
 ******************************************************************************/

/**
 * @brief Utility class for dispatching the appropriately-tuned kernels for DeviceSpmv
 *
 * @tparam ValueT
 *   Matrix and vector value type
 *
 * @tparam OffsetT
 *   Signed integer type for global offsets
 */
template <typename ValueT, typename OffsetT>
struct DispatchSpmv
{
  //---------------------------------------------------------------------
  // Constants and Types
  //---------------------------------------------------------------------

  enum
  {
    INIT_KERNEL_THREADS         = 128,
    EMPTY_MATRIX_KERNEL_THREADS = 128
  };

  // SpmvParams bundle type
  using SpmvParamsT = SpmvParams<ValueT, OffsetT>;

  // 2D merge path coordinate type
  using CoordinateT = typename CubVector<OffsetT, 2>::Type;

  // Tile status descriptor interface type
  using ScanTileStateT = ReduceByKeyScanTileState<ValueT, OffsetT>;

  // Tuple type for scanning (pairs accumulated segment-value with segment-index)
  using KeyValuePairT = KeyValuePair<OffsetT, ValueT>;

  //---------------------------------------------------------------------
  // Tuning policies
  //---------------------------------------------------------------------

  /// SM35
  struct Policy350
  {
    using SpmvPolicyT =
      AgentSpmvPolicy<(sizeof(ValueT) > 4) ? 96 : 128,
                      (sizeof(ValueT) > 4) ? 4 : 7,
                      LOAD_LDG,
                      LOAD_CA,
                      LOAD_LDG,
                      LOAD_LDG,
                      LOAD_LDG,
                      (sizeof(ValueT) > 4) ? true : false,
                      BLOCK_SCAN_WARP_SCANS>;

    using SegmentFixupPolicyT = AgentSegmentFixupPolicy<128, 3, BLOCK_LOAD_VECTORIZE, LOAD_LDG, BLOCK_SCAN_WARP_SCANS>;
  };

  /// SM37
  struct Policy370
  {
    using SpmvPolicyT =
      AgentSpmvPolicy<(sizeof(ValueT) > 4) ? 128 : 128,
                      (sizeof(ValueT) > 4) ? 9 : 14,
                      LOAD_LDG,
                      LOAD_CA,
                      LOAD_LDG,
                      LOAD_LDG,
                      LOAD_LDG,
                      false,
                      BLOCK_SCAN_WARP_SCANS>;

    using SegmentFixupPolicyT = AgentSegmentFixupPolicy<128, 3, BLOCK_LOAD_VECTORIZE, LOAD_LDG, BLOCK_SCAN_WARP_SCANS>;
  };

  /// SM50
  struct Policy500
  {
    using SpmvPolicyT =
      AgentSpmvPolicy<(sizeof(ValueT) > 4) ? 64 : 128,
                      (sizeof(ValueT) > 4) ? 6 : 7,
                      LOAD_LDG,
                      LOAD_DEFAULT,
                      (sizeof(ValueT) > 4) ? LOAD_LDG : LOAD_DEFAULT,
                      (sizeof(ValueT) > 4) ? LOAD_LDG : LOAD_DEFAULT,
                      LOAD_LDG,
                      (sizeof(ValueT) > 4) ? true : false,
                      (sizeof(ValueT) > 4) ? BLOCK_SCAN_WARP_SCANS : BLOCK_SCAN_RAKING_MEMOIZE>;

    using SegmentFixupPolicyT =
      AgentSegmentFixupPolicy<128, 3, BLOCK_LOAD_VECTORIZE, LOAD_LDG, BLOCK_SCAN_RAKING_MEMOIZE>;
  };

  /// SM60
  struct Policy600
  {
    using SpmvPolicyT =
      AgentSpmvPolicy<(sizeof(ValueT) > 4) ? 64 : 128,
                      (sizeof(ValueT) > 4) ? 5 : 7,
                      LOAD_DEFAULT,
                      LOAD_DEFAULT,
                      LOAD_DEFAULT,
                      LOAD_DEFAULT,
                      LOAD_DEFAULT,
                      false,
                      BLOCK_SCAN_WARP_SCANS>;

    using SegmentFixupPolicyT = AgentSegmentFixupPolicy<128, 3, BLOCK_LOAD_DIRECT, LOAD_LDG, BLOCK_SCAN_WARP_SCANS>;
  };

  //---------------------------------------------------------------------
  // Tuning policies of current PTX compiler pass
  //---------------------------------------------------------------------

#if (CUB_PTX_ARCH >= 600)
  using PtxPolicy = Policy600;

#elif (CUB_PTX_ARCH >= 500)
  using PtxPolicy = Policy500;

#elif (CUB_PTX_ARCH >= 370)
  using PtxPolicy = Policy370;

#else
  using PtxPolicy = Policy350;

#endif

  // "Opaque" policies (whose parameterizations aren't reflected in the type signature)
  struct PtxSpmvPolicyT : PtxPolicy::SpmvPolicyT
  {};
  struct PtxSegmentFixupPolicy : PtxPolicy::SegmentFixupPolicyT
  {};

  //---------------------------------------------------------------------
  // Utilities
  //---------------------------------------------------------------------

  /**
   * Initialize kernel dispatch configurations with the policies corresponding to the PTX assembly we will use
   */
  template <typename KernelConfig>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static void
  InitConfigs(int ptx_version, KernelConfig& spmv_config, KernelConfig& segment_fixup_config)
  {
    NV_IF_TARGET(
      NV_IS_DEVICE,
      ( // We're on the device, so initialize the kernel dispatch
        // configurations with the current PTX policy
        spmv_config.template Init<PtxSpmvPolicyT>(); segment_fixup_config.template Init<PtxSegmentFixupPolicy>();),
      (
        // We're on the host, so lookup and initialize the kernel dispatch
        // configurations with the policies that match the device's PTX
        // version
        if (ptx_version >= 600) {
          spmv_config.template Init<typename Policy600::SpmvPolicyT>();
          segment_fixup_config.template Init<typename Policy600::SegmentFixupPolicyT>();
        } else if (ptx_version >= 500) {
          spmv_config.template Init<typename Policy500::SpmvPolicyT>();
          segment_fixup_config.template Init<typename Policy500::SegmentFixupPolicyT>();
        } else if (ptx_version >= 370) {
          spmv_config.template Init<typename Policy370::SpmvPolicyT>();
          segment_fixup_config.template Init<typename Policy370::SegmentFixupPolicyT>();
        } else {
          spmv_config.template Init<typename Policy350::SpmvPolicyT>();
          segment_fixup_config.template Init<typename Policy350::SegmentFixupPolicyT>();
        }));
  }

  /**
   * Kernel kernel dispatch configuration.
   */
  struct KernelConfig
  {
    int block_threads;
    int items_per_thread;
    int tile_items;

    template <typename PolicyT>
    CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE void Init()
    {
      block_threads    = PolicyT::BLOCK_THREADS;
      items_per_thread = PolicyT::ITEMS_PER_THREAD;
      tile_items       = block_threads * items_per_thread;
    }
  };

  //---------------------------------------------------------------------
  // Dispatch entrypoints
  //---------------------------------------------------------------------

  /**
   * Internal dispatch routine for computing a device-wide reduction using the
   * specified kernel functions.
   *
   * If the input is larger than a single tile, this method uses two-passes of
   * kernel invocations.
   *
   * @tparam Spmv1ColKernelT
   *   Function type of cub::DeviceSpmv1ColKernel
   *
   * @tparam SpmvSearchKernelT
   *   Function type of cub::AgentSpmvSearchKernel
   *
   * @tparam SpmvKernelT
   *   Function type of cub::AgentSpmvKernel
   *
   * @tparam SegmentFixupKernelT
   *   Function type of cub::DeviceSegmentFixupKernelT
   *
   * @tparam SpmvEmptyMatrixKernelT
   *   Function type of cub::DeviceSpmvEmptyMatrixKernel
   *
   * @param[in] d_temp_storage
   *   Device-accessible allocation of temporary storage.
   *   When nullptr, the required allocation size is written to
   *   `temp_storage_bytes` and no work is done.
   *
   * @param[in,out] temp_storage_bytes
   *   Reference to size in bytes of \p d_temp_storage allocation
   *
   * @paramSpMV spmv_params
   *   input parameter bundle
   *
   * @param[in] stream
   *   CUDA stream to launch kernels within. Default is stream<sub>0</sub>.
   *
   * @param[in] spmv_1col_kernel
   *   Kernel function pointer to parameterization of DeviceSpmv1ColKernel
   *
   * @param[in] spmv_search_kernel
   *   Kernel function pointer to parameterization of AgentSpmvSearchKernel
   *
   * @param[in] spmv_kernel
   *   Kernel function pointer to parameterization of AgentSpmvKernel
   *
   * @param[in] segment_fixup_kernel
   *   Kernel function pointer to parameterization of cub::DeviceSegmentFixupKernel
   *
   * @param[in] spmv_empty_matrix_kernel
   *   Kernel function pointer to parameterization of cub::DeviceSpmvEmptyMatrixKernel
   *
   * @param[in] spmv_config
   *   Dispatch parameters that match the policy that @p spmv_kernel was compiled for
   *
   * @param[in] segment_fixup_config
   *   Dispatch parameters that match the policy that @p segment_fixup_kernel was compiled for
   */
  template <typename Spmv1ColKernelT,
            typename SpmvSearchKernelT,
            typename SpmvKernelT,
            typename SegmentFixupKernelT,
            typename SpmvEmptyMatrixKernelT>
  CUB_RUNTIME_FUNCTION _CCCL_VISIBILITY_HIDDEN _CCCL_FORCEINLINE static cudaError_t Dispatch(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    SpmvParamsT& spmv_params,
    cudaStream_t stream,
    Spmv1ColKernelT spmv_1col_kernel,
    SpmvSearchKernelT spmv_search_kernel,
    SpmvKernelT spmv_kernel,
    SegmentFixupKernelT segment_fixup_kernel,
    SpmvEmptyMatrixKernelT spmv_empty_matrix_kernel,
    KernelConfig spmv_config,
    KernelConfig segment_fixup_config)
  {
    cudaError error = cudaSuccess;
    do
    {
      if (spmv_params.num_rows < 0 || spmv_params.num_cols < 0)
      {
        return cudaErrorInvalidValue;
      }

      if (spmv_params.num_rows == 0 || spmv_params.num_cols == 0)
      { // Empty problem, no-op.
        if (d_temp_storage == nullptr)
        {
          temp_storage_bytes = 1;
        }

        break;
      }

      if (spmv_params.num_nonzeros == 0)
      {
        if (d_temp_storage == nullptr)
        {
          // Return if the caller is simply requesting the size of the storage allocation
          temp_storage_bytes = 1;
          break;
        }

        constexpr int threads_in_block = EMPTY_MATRIX_KERNEL_THREADS;
        const int blocks_in_grid       = ::cuda::ceil_div(spmv_params.num_rows, threads_in_block);

#ifdef CUB_DETAIL_DEBUG_ENABLE_LOG
        _CubLog("Invoking spmv_empty_matrix_kernel<<<%d, %d, 0, %lld>>>()\n",
                blocks_in_grid,
                threads_in_block,
                (long long) stream);
#endif // CUB_DETAIL_DEBUG_ENABLE_LOG
        error = THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron(blocks_in_grid, threads_in_block, 0, stream)
                  .doit(spmv_empty_matrix_kernel, spmv_params);

        if (CubDebug(error))
        {
          break;
        }

        // Sync the stream if specified to flush runtime errors
        error = detail::DebugSyncStream(stream);
        if (CubDebug(error))
        {
          break;
        }

        break;
      }

      if (spmv_params.num_cols == 1)
      {
        if (d_temp_storage == nullptr)
        {
          // Return if the caller is simply requesting the size of the storage allocation
          temp_storage_bytes = 1;
          break;
        }

        // Get search/init grid dims
        int degen_col_kernel_block_size = INIT_KERNEL_THREADS;
        int degen_col_kernel_grid_size  = ::cuda::ceil_div(spmv_params.num_rows, degen_col_kernel_block_size);

#ifdef CUB_DETAIL_DEBUG_ENABLE_LOG
        _CubLog("Invoking spmv_1col_kernel<<<%d, %d, 0, %lld>>>()\n",
                degen_col_kernel_grid_size,
                degen_col_kernel_block_size,
                (long long) stream);
#endif // CUB_DETAIL_DEBUG_ENABLE_LOG

        // Invoke spmv_search_kernel
        THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron(
          degen_col_kernel_grid_size, degen_col_kernel_block_size, 0, stream)
          .doit(spmv_1col_kernel, spmv_params);

        // Check for failure to launch
        if (CubDebug(error = cudaPeekAtLastError()))
        {
          break;
        }

        // Sync the stream if specified to flush runtime errors
        error = detail::DebugSyncStream(stream);
        if (CubDebug(error))
        {
          break;
        }

        break;
      }

      // Get device ordinal
      int device_ordinal;
      if (CubDebug(error = cudaGetDevice(&device_ordinal)))
      {
        break;
      }

      // Get SM count
      int sm_count;
      if (CubDebug(error = cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_ordinal)))
      {
        break;
      }

      // Get max x-dimension of grid
      int max_dim_x;
      if (CubDebug(error = cudaDeviceGetAttribute(&max_dim_x, cudaDevAttrMaxGridDimX, device_ordinal)))
      {
        break;
      }

      // Total number of spmv work items
      int num_merge_items = spmv_params.num_rows + spmv_params.num_nonzeros;

      // Tile sizes of kernels
      int merge_tile_size         = spmv_config.block_threads * spmv_config.items_per_thread;
      int segment_fixup_tile_size = segment_fixup_config.block_threads * segment_fixup_config.items_per_thread;

      // Number of tiles for kernels
      int num_merge_tiles         = ::cuda::ceil_div(num_merge_items, merge_tile_size);
      int num_segment_fixup_tiles = ::cuda::ceil_div(num_merge_tiles, segment_fixup_tile_size);

      // Get SM occupancy for kernels
      int spmv_sm_occupancy;
      if (CubDebug(error = MaxSmOccupancy(spmv_sm_occupancy, spmv_kernel, spmv_config.block_threads)))
      {
        break;
      }

      int segment_fixup_sm_occupancy;
      if (CubDebug(error = MaxSmOccupancy(
                     segment_fixup_sm_occupancy, segment_fixup_kernel, segment_fixup_config.block_threads)))
      {
        break;
      }

      // Get grid dimensions
      dim3 spmv_grid_size(CUB_MIN(num_merge_tiles, max_dim_x), ::cuda::ceil_div(num_merge_tiles, max_dim_x), 1);

      dim3 segment_fixup_grid_size(
        CUB_MIN(num_segment_fixup_tiles, max_dim_x), ::cuda::ceil_div(num_segment_fixup_tiles, max_dim_x), 1);

      // Get the temporary storage allocation requirements
      size_t allocation_sizes[3];
      if (CubDebug(error = ScanTileStateT::AllocationSize(num_segment_fixup_tiles, allocation_sizes[0])))
      {
        break; // bytes needed for reduce-by-key tile status descriptors
      }
      allocation_sizes[1] = num_merge_tiles * sizeof(KeyValuePairT); // bytes needed for block carry-out pairs
      allocation_sizes[2] = (num_merge_tiles + 1) * sizeof(CoordinateT); // bytes needed for tile starting coordinates

      // Alias the temporary allocations from the single storage blob (or compute the necessary size of the blob)
      void* allocations[3] = {};
      if (CubDebug(error = AliasTemporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes)))
      {
        break;
      }
      if (d_temp_storage == nullptr)
      {
        // Return if the caller is simply requesting the size of the storage allocation
        break;
      }

      // Construct the tile status interface
      ScanTileStateT tile_state;
      if (CubDebug(error = tile_state.Init(num_segment_fixup_tiles, allocations[0], allocation_sizes[0])))
      {
        break;
      }

      // Alias the other allocations
      KeyValuePairT* d_tile_carry_pairs = (KeyValuePairT*) allocations[1]; // Agent carry-out pairs
      CoordinateT* d_tile_coordinates   = (CoordinateT*) allocations[2]; // Agent starting coordinates

      // Get search/init grid dims
      int search_block_size = INIT_KERNEL_THREADS;
      int search_grid_size  = ::cuda::ceil_div(num_merge_tiles + 1, search_block_size);

      if (search_grid_size < sm_count)
      //            if (num_merge_tiles < spmv_sm_occupancy * sm_count)
      {
        // Not enough spmv tiles to saturate the device: have spmv blocks search their own staring coords
        d_tile_coordinates = nullptr;
      }
      else
      {
// Use separate search kernel if we have enough spmv tiles to saturate the device

// Log spmv_search_kernel configuration
#ifdef CUB_DETAIL_DEBUG_ENABLE_LOG
        _CubLog("Invoking spmv_search_kernel<<<%d, %d, 0, %lld>>>()\n",
                search_grid_size,
                search_block_size,
                (long long) stream);
#endif // CUB_DETAIL_DEBUG_ENABLE_LOG

        // Invoke spmv_search_kernel
        THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron(search_grid_size, search_block_size, 0, stream)
          .doit(spmv_search_kernel, num_merge_tiles, d_tile_coordinates, spmv_params);

        // Check for failure to launch
        if (CubDebug(error = cudaPeekAtLastError()))
        {
          break;
        }

        // Sync the stream if specified to flush runtime errors
        error = detail::DebugSyncStream(stream);
        if (CubDebug(error))
        {
          break;
        }
      }

// Log spmv_kernel configuration
#ifdef CUB_DETAIL_DEBUG_ENABLE_LOG
      _CubLog("Invoking spmv_kernel<<<{%d,%d,%d}, %d, 0, %lld>>>(), %d items per thread, %d SM occupancy\n",
              spmv_grid_size.x,
              spmv_grid_size.y,
              spmv_grid_size.z,
              spmv_config.block_threads,
              (long long) stream,
              spmv_config.items_per_thread,
              spmv_sm_occupancy);
#endif // CUB_DETAIL_DEBUG_ENABLE_LOG

      // Invoke spmv_kernel
      THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron(spmv_grid_size, spmv_config.block_threads, 0, stream)
        .doit(spmv_kernel,
              spmv_params,
              d_tile_coordinates,
              d_tile_carry_pairs,
              num_merge_tiles,
              tile_state,
              num_segment_fixup_tiles);

      // Check for failure to launch
      if (CubDebug(error = cudaPeekAtLastError()))
      {
        break;
      }

      // Sync the stream if specified to flush runtime errors
      error = detail::DebugSyncStream(stream);
      if (CubDebug(error))
      {
        break;
      }

      // Run reduce-by-key fixup if necessary
      if (num_merge_tiles > 1)
      {
// Log segment_fixup_kernel configuration
#ifdef CUB_DETAIL_DEBUG_ENABLE_LOG
        _CubLog("Invoking segment_fixup_kernel<<<{%d,%d,%d}, %d, 0, %lld>>>(), %d items per thread, %d SM occupancy\n",
                segment_fixup_grid_size.x,
                segment_fixup_grid_size.y,
                segment_fixup_grid_size.z,
                segment_fixup_config.block_threads,
                (long long) stream,
                segment_fixup_config.items_per_thread,
                segment_fixup_sm_occupancy);
#endif // CUB_DETAIL_DEBUG_ENABLE_LOG

        // Invoke segment_fixup_kernel
        THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron(
          segment_fixup_grid_size, segment_fixup_config.block_threads, 0, stream)
          .doit(segment_fixup_kernel,
                d_tile_carry_pairs,
                spmv_params.d_vector_y,
                num_merge_tiles,
                num_segment_fixup_tiles,
                tile_state);

        // Check for failure to launch
        if (CubDebug(error = cudaPeekAtLastError()))
        {
          break;
        }

        // Sync the stream if specified to flush runtime errors
        error = detail::DebugSyncStream(stream);
        if (CubDebug(error))
        {
          break;
        }
      }
    } while (0);

    return error;
  }

#ifndef DOXYGEN_SHOULD_SKIP_THIS // Do not document
  template <typename Spmv1ColKernelT,
            typename SpmvSearchKernelT,
            typename SpmvKernelT,
            typename SegmentFixupKernelT,
            typename SpmvEmptyMatrixKernelT>
  CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED CUB_RUNTIME_FUNCTION _CCCL_VISIBILITY_HIDDEN
  _CCCL_FORCEINLINE static cudaError_t
  Dispatch(void* d_temp_storage,
           size_t& temp_storage_bytes,
           SpmvParamsT& spmv_params,
           cudaStream_t stream,
           bool debug_synchronous,
           Spmv1ColKernelT spmv_1col_kernel,
           SpmvSearchKernelT spmv_search_kernel,
           SpmvKernelT spmv_kernel,
           SegmentFixupKernelT segment_fixup_kernel,
           SpmvEmptyMatrixKernelT spmv_empty_matrix_kernel,
           KernelConfig spmv_config,
           KernelConfig segment_fixup_config)
  {
    CUB_DETAIL_RUNTIME_DEBUG_SYNC_USAGE_LOG

    return Dispatch<Spmv1ColKernelT, SpmvSearchKernelT, SpmvKernelT, SegmentFixupKernelT, SpmvEmptyMatrixKernelT>(
      d_temp_storage,
      temp_storage_bytes,
      spmv_params,
      stream,
      spmv_1col_kernel,
      spmv_search_kernel,
      spmv_kernel,
      segment_fixup_kernel,
      spmv_empty_matrix_kernel,
      spmv_config,
      segment_fixup_config);
  }
#endif // DOXYGEN_SHOULD_SKIP_THIS

  /**
   * @brief Internal dispatch routine for computing a device-wide reduction
   *
   * @param[in] d_temp_storage
   *   Device-accessible allocation of temporary storage.
   *   When nullptr, the required allocation size is written to
   *   `temp_storage_bytes` and no work is done.
   *
   * @param[in,out] temp_storage_bytes
   *   Reference to size in bytes of `d_temp_storage` allocation
   *
   * @param SpMV spmv_params
   *   input parameter bundle
   *
   * @param[in] stream
   *   **[optional]** CUDA stream to launch kernels within. Default is stream<sub>0</sub>.
   */
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t
  Dispatch(void* d_temp_storage, size_t& temp_storage_bytes, SpmvParamsT& spmv_params, cudaStream_t stream = 0)
  {
    cudaError error = cudaSuccess;
    do
    {
      // Get PTX version
      int ptx_version = 0;
      if (CubDebug(error = PtxVersion(ptx_version)))
      {
        break;
      }

      // Get kernel kernel dispatch configurations
      KernelConfig spmv_config, segment_fixup_config;
      InitConfigs(ptx_version, spmv_config, segment_fixup_config);

      constexpr bool has_alpha = false;
      constexpr bool has_beta  = false;

      if (CubDebug(
            error = Dispatch(
              d_temp_storage,
              temp_storage_bytes,
              spmv_params,
              stream,
              DeviceSpmv1ColKernel<PtxSpmvPolicyT, ValueT, OffsetT>,
              DeviceSpmvSearchKernel<PtxSpmvPolicyT, OffsetT, CoordinateT, SpmvParamsT>,
              DeviceSpmvKernel<PtxSpmvPolicyT, ScanTileStateT, ValueT, OffsetT, CoordinateT, has_alpha, has_beta>,
              DeviceSegmentFixupKernel<PtxSegmentFixupPolicy, KeyValuePairT*, ValueT*, OffsetT, ScanTileStateT>,
              DeviceSpmvEmptyMatrixKernel<ValueT, OffsetT, has_beta>,
              spmv_config,
              segment_fixup_config)))
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
    size_t& temp_storage_bytes,
    SpmvParamsT& spmv_params,
    cudaStream_t stream,
    bool debug_synchronous)
  {
    CUB_DETAIL_RUNTIME_DEBUG_SYNC_USAGE_LOG

    return Dispatch(d_temp_storage, temp_storage_bytes, spmv_params, stream);
  }
#endif // DOXYGEN_SHOULD_SKIP_THIS
};

CUB_NAMESPACE_END
