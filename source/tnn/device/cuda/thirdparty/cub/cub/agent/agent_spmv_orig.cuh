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
 * \file
 * cub::AgentSpmv implements a stateful abstraction of CUDA thread blocks for participating in device-wide SpMV.
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

#include <cub/block/block_exchange.cuh>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/iterator/cache_modified_input_iterator.cuh>
#include <cub/iterator/counting_input_iterator.cuh>
#include <cub/thread/thread_operators.cuh>
#include <cub/thread/thread_search.cuh>
#include <cub/util_type.cuh>

#include <cuda/std/type_traits>

#include <iterator>

CUB_NAMESPACE_BEGIN

/******************************************************************************
 * Tuning policy
 ******************************************************************************/

/**
 * @param Parameterizable tuning policy type for AgentSpmv
 *
 * @tparam _BLOCK_THREADS
 *   Threads per thread block
 *
 * @tparam _ITEMS_PER_THREAD
 *   Items per thread (per tile of input)
 *
 * @tparam _ROW_OFFSETS_SEARCH_LOAD_MODIFIER
 *   Cache load modifier for reading CSR row-offsets during search
 *
 * @tparam _ROW_OFFSETS_LOAD_MODIFIER
 *   Cache load modifier for reading CSR row-offsets
 *
 * @tparam _COLUMN_INDICES_LOAD_MODIFIER
 *   Cache load modifier for reading CSR column-indices
 *
 * @tparam _VALUES_LOAD_MODIFIER
 *   Cache load modifier for reading CSR values
 *
 * @tparam _VECTOR_VALUES_LOAD_MODIFIER
 *   Cache load modifier for reading vector values
 *
 * @tparam _DIRECT_LOAD_NONZEROS
 *   Whether to load nonzeros directly from global during sequential merging (vs. pre-staged through
 * shared memory)
 *
 * @tparam _SCAN_ALGORITHM
 *   The BlockScan algorithm to use
 */
template <int _BLOCK_THREADS,
          int _ITEMS_PER_THREAD,
          CacheLoadModifier _ROW_OFFSETS_SEARCH_LOAD_MODIFIER,
          CacheLoadModifier _ROW_OFFSETS_LOAD_MODIFIER,
          CacheLoadModifier _COLUMN_INDICES_LOAD_MODIFIER,
          CacheLoadModifier _VALUES_LOAD_MODIFIER,
          CacheLoadModifier _VECTOR_VALUES_LOAD_MODIFIER,
          bool _DIRECT_LOAD_NONZEROS,
          BlockScanAlgorithm _SCAN_ALGORITHM>
struct AgentSpmvPolicy
{
  enum
  {
    /// Threads per thread block
    BLOCK_THREADS = _BLOCK_THREADS,

    /// Items per thread (per tile of input)
    ITEMS_PER_THREAD = _ITEMS_PER_THREAD,

    /// Whether to load nonzeros directly from global during sequential merging (pre-staged through
    /// shared memory)
    DIRECT_LOAD_NONZEROS = _DIRECT_LOAD_NONZEROS,
  };

  /// Cache load modifier for reading CSR row-offsets
  static constexpr CacheLoadModifier ROW_OFFSETS_SEARCH_LOAD_MODIFIER = _ROW_OFFSETS_SEARCH_LOAD_MODIFIER;

  /// Cache load modifier for reading CSR row-offsets
  static constexpr CacheLoadModifier ROW_OFFSETS_LOAD_MODIFIER = _ROW_OFFSETS_LOAD_MODIFIER;

  /// Cache load modifier for reading CSR column-indices
  static constexpr CacheLoadModifier COLUMN_INDICES_LOAD_MODIFIER = _COLUMN_INDICES_LOAD_MODIFIER;

  /// Cache load modifier for reading CSR values
  static constexpr CacheLoadModifier VALUES_LOAD_MODIFIER = _VALUES_LOAD_MODIFIER;

  /// Cache load modifier for reading vector values
  static constexpr CacheLoadModifier VECTOR_VALUES_LOAD_MODIFIER = _VECTOR_VALUES_LOAD_MODIFIER;

  /// The BlockScan algorithm to use
  static constexpr BlockScanAlgorithm SCAN_ALGORITHM = _SCAN_ALGORITHM;
};

/******************************************************************************
 * Thread block abstractions
 ******************************************************************************/

/**
 * @tparam ValueT
 *   Matrix and vector value type
 *
 * @tparam OffsetT
 *   Signed integer type for sequence offsets
 */
template <typename ValueT, typename OffsetT>
struct SpmvParams
{
  /// Pointer to the array of \p num_nonzeros values of the corresponding nonzero elements of matrix
  /// <b>A</b>.
  const ValueT* d_values;

  /// Pointer to the array of \p m offsets demarcating the end of every row in \p d_column_indices
  /// and \p d_values
  const OffsetT* d_row_end_offsets;

  /// Pointer to the array of \p num_nonzeros column-indices of the corresponding nonzero elements
  /// of matrix <b>A</b>.  (Indices are zero-valued.)
  const OffsetT* d_column_indices;

  /// Pointer to the array of \p num_cols values corresponding to the dense input vector <em>x</em>
  const ValueT* d_vector_x;

  /// Pointer to the array of \p num_rows values corresponding to the dense output vector <em>y</em>
  ValueT* d_vector_y;

  /// Number of rows of matrix <b>A</b>.
  int num_rows;

  /// Number of columns of matrix <b>A</b>.
  int num_cols;

  /// Number of nonzero elements of matrix <b>A</b>.
  int num_nonzeros;

  /// Alpha multiplicand
  ValueT alpha;

  /// Beta addend-multiplicand
  ValueT beta;
};

/**
 * @brief AgentSpmv implements a stateful abstraction of CUDA thread blocks for participating in device-wide SpMV.
 *
 * @tparam AgentSpmvPolicyT
 *   Parameterized AgentSpmvPolicy tuning policy type
 *
 * @tparam ValueT
 *   Matrix and vector value type
 *
 * @tparam OffsetT
 *   Signed integer type for sequence offsets
 *
 * @tparam HAS_ALPHA
 *   Whether the input parameter \p alpha is 1
 *
 * @tparam HAS_BETA
 *   Whether the input parameter \p beta is 0
 *
 * @tparam LEGACY_PTX_ARCH
 *   PTX compute capability (unused)
 */
template <typename AgentSpmvPolicyT,
          typename ValueT,
          typename OffsetT,
          bool HAS_ALPHA,
          bool HAS_BETA,
          int LEGACY_PTX_ARCH = 0>
struct AgentSpmv
{
  //---------------------------------------------------------------------
  // Types and constants
  //---------------------------------------------------------------------

  /// Constants
  enum
  {
    BLOCK_THREADS    = AgentSpmvPolicyT::BLOCK_THREADS,
    ITEMS_PER_THREAD = AgentSpmvPolicyT::ITEMS_PER_THREAD,
    TILE_ITEMS       = BLOCK_THREADS * ITEMS_PER_THREAD,
  };

  /// 2D merge path coordinate type
  using CoordinateT = typename CubVector<OffsetT, 2>::Type;

  /// Input iterator wrapper types (for applying cache modifiers)

  using RowOffsetsSearchIteratorT =
    CacheModifiedInputIterator<AgentSpmvPolicyT::ROW_OFFSETS_SEARCH_LOAD_MODIFIER, OffsetT, OffsetT>;

  using RowOffsetsIteratorT = CacheModifiedInputIterator<AgentSpmvPolicyT::ROW_OFFSETS_LOAD_MODIFIER, OffsetT, OffsetT>;

  using ColumnIndicesIteratorT =
    CacheModifiedInputIterator<AgentSpmvPolicyT::COLUMN_INDICES_LOAD_MODIFIER, OffsetT, OffsetT>;

  using ValueIteratorT = CacheModifiedInputIterator<AgentSpmvPolicyT::VALUES_LOAD_MODIFIER, ValueT, OffsetT>;

  using VectorValueIteratorT =
    CacheModifiedInputIterator<AgentSpmvPolicyT::VECTOR_VALUES_LOAD_MODIFIER, ValueT, OffsetT>;

  // Tuple type for scanning (pairs accumulated segment-value with segment-index)
  using KeyValuePairT = KeyValuePair<OffsetT, ValueT>;

  // Reduce-value-by-segment scan operator
  using ReduceBySegmentOpT = ReduceByKeyOp<cub::Sum>;

  // BlockReduce specialization
  using BlockReduceT = BlockReduce<ValueT, BLOCK_THREADS, BLOCK_REDUCE_WARP_REDUCTIONS>;

  // BlockScan specialization
  using BlockScanT = BlockScan<KeyValuePairT, BLOCK_THREADS, AgentSpmvPolicyT::SCAN_ALGORITHM>;

  // BlockScan specialization
  using BlockPrefixSumT = BlockScan<ValueT, BLOCK_THREADS, AgentSpmvPolicyT::SCAN_ALGORITHM>;

  // BlockExchange specialization
  using BlockExchangeT = BlockExchange<ValueT, BLOCK_THREADS, ITEMS_PER_THREAD>;

  /// Merge item type (either a non-zero value or a row-end offset)
  union MergeItem
  {
    // Value type to pair with index type OffsetT
    // (NullType if loading values directly during merge)
    using MergeValueT = ::cuda::std::_If<AgentSpmvPolicyT::DIRECT_LOAD_NONZEROS, NullType, ValueT>;

    OffsetT row_end_offset;
    MergeValueT nonzero;
  };

  /// Shared memory type required by this thread block
  struct _TempStorage
  {
    CoordinateT tile_coords[2];

    union Aliasable
    {
      // Smem needed for tile of merge items
      MergeItem merge_items[ITEMS_PER_THREAD + TILE_ITEMS + 1];

      // Smem needed for block exchange
      typename BlockExchangeT::TempStorage exchange;

      // Smem needed for block-wide reduction
      typename BlockReduceT::TempStorage reduce;

      // Smem needed for tile scanning
      typename BlockScanT::TempStorage scan;

      // Smem needed for tile prefix sum
      typename BlockPrefixSumT::TempStorage prefix_sum;

    } aliasable;
  };

  /// Temporary storage type (unionable)
  struct TempStorage : Uninitialized<_TempStorage>
  {};

  //---------------------------------------------------------------------
  // Per-thread fields
  //---------------------------------------------------------------------

  /// Reference to temp_storage
  _TempStorage& temp_storage;

  SpmvParams<ValueT, OffsetT>& spmv_params;

  /// Wrapped pointer to the array of \p num_nonzeros values of the corresponding nonzero elements
  /// of matrix <b>A</b>.
  ValueIteratorT wd_values;

  /// Wrapped Pointer to the array of \p m offsets demarcating the end of every row in \p
  /// d_column_indices and \p d_values
  RowOffsetsIteratorT wd_row_end_offsets;

  /// Wrapped Pointer to the array of \p num_nonzeros column-indices of the corresponding nonzero
  /// elements of matrix <b>A</b>.  (Indices are zero-valued.)
  ColumnIndicesIteratorT wd_column_indices;

  /// Wrapped Pointer to the array of \p num_cols values corresponding to the dense input vector
  /// <em>x</em>
  VectorValueIteratorT wd_vector_x;

  /// Wrapped Pointer to the array of \p num_cols values corresponding to the dense input vector
  /// <em>x</em>
  VectorValueIteratorT wd_vector_y;

  //---------------------------------------------------------------------
  // Interface
  //---------------------------------------------------------------------

  /**
   * @param temp_storage
   *   Reference to temp_storage
   *
   * @param spmv_params
   *   SpMV input parameter bundle
   */
  _CCCL_DEVICE _CCCL_FORCEINLINE AgentSpmv(TempStorage& temp_storage, SpmvParams<ValueT, OffsetT>& spmv_params)
      : temp_storage(temp_storage.Alias())
      , spmv_params(spmv_params)
      , wd_values(spmv_params.d_values)
      , wd_row_end_offsets(spmv_params.d_row_end_offsets)
      , wd_column_indices(spmv_params.d_column_indices)
      , wd_vector_x(spmv_params.d_vector_x)
      , wd_vector_y(spmv_params.d_vector_y)
  {}

  /**
   * @brief Consume a merge tile, specialized for direct-load of nonzeros
   *
   * @param is_direct_load
   *   Marker type indicating whether to load nonzeros directly during path-discovery or beforehand in batch
   */
  _CCCL_DEVICE _CCCL_FORCEINLINE KeyValuePairT
  ConsumeTile(int tile_idx, CoordinateT tile_start_coord, CoordinateT tile_end_coord, Int2Type<true> is_direct_load)
  {
    int tile_num_rows               = tile_end_coord.x - tile_start_coord.x;
    int tile_num_nonzeros           = tile_end_coord.y - tile_start_coord.y;
    OffsetT* s_tile_row_end_offsets = &temp_storage.aliasable.merge_items[0].row_end_offset;

    // Gather the row end-offsets for the merge tile into shared memory
    for (int item = threadIdx.x; item < tile_num_rows + ITEMS_PER_THREAD; item += BLOCK_THREADS)
    {
      const OffsetT offset =
        (cub::min)(static_cast<OffsetT>(tile_start_coord.x + item), static_cast<OffsetT>(spmv_params.num_rows - 1));
      s_tile_row_end_offsets[item] = wd_row_end_offsets[offset];
    }

    CTA_SYNC();

    // Search for the thread's starting coordinate within the merge tile
    CountingInputIterator<OffsetT> tile_nonzero_indices(tile_start_coord.y);
    CoordinateT thread_start_coord;

    MergePathSearch(
      OffsetT(threadIdx.x * ITEMS_PER_THREAD), // Diagonal
      s_tile_row_end_offsets, // List A
      tile_nonzero_indices, // List B
      tile_num_rows,
      tile_num_nonzeros,
      thread_start_coord);

    CTA_SYNC(); // Perf-sync

    // Compute the thread's merge path segment
    CoordinateT thread_current_coord = thread_start_coord;
    KeyValuePairT scan_segment[ITEMS_PER_THREAD];

    ValueT running_total = 0.0;

#pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
    {
      OffsetT nonzero_idx = CUB_MIN(tile_nonzero_indices[thread_current_coord.y], spmv_params.num_nonzeros - 1);
      OffsetT column_idx  = wd_column_indices[nonzero_idx];
      ValueT value        = wd_values[nonzero_idx];

      ValueT vector_value = wd_vector_x[column_idx];

      ValueT nonzero = value * vector_value;

      OffsetT row_end_offset = s_tile_row_end_offsets[thread_current_coord.x];

      if (tile_nonzero_indices[thread_current_coord.y] < row_end_offset)
      {
        // Move down (accumulate)
        running_total += nonzero;
        scan_segment[ITEM].value = running_total;
        scan_segment[ITEM].key   = tile_num_rows;
        ++thread_current_coord.y;
      }
      else
      {
        // Move right (reset)
        scan_segment[ITEM].value = running_total;
        scan_segment[ITEM].key   = thread_current_coord.x;
        running_total            = 0.0;
        ++thread_current_coord.x;
      }
    }

    CTA_SYNC();

    // Block-wide reduce-value-by-segment
    KeyValuePairT tile_carry;
    ReduceBySegmentOpT scan_op;
    KeyValuePairT scan_item;

    scan_item.value = running_total;
    scan_item.key   = thread_current_coord.x;

    BlockScanT(temp_storage.aliasable.scan).ExclusiveScan(scan_item, scan_item, scan_op, tile_carry);

    if (tile_num_rows > 0)
    {
      if (threadIdx.x == 0)
      {
        scan_item.key = -1;
      }

// Direct scatter
#pragma unroll
      for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
      {
        if (scan_segment[ITEM].key < tile_num_rows)
        {
          if (scan_item.key == scan_segment[ITEM].key)
          {
            scan_segment[ITEM].value = scan_item.value + scan_segment[ITEM].value;
          }

          if (HAS_ALPHA)
          {
            scan_segment[ITEM].value *= spmv_params.alpha;
          }

          if (HAS_BETA)
          {
            // Update the output vector element
            ValueT addend = spmv_params.beta * wd_vector_y[tile_start_coord.x + scan_segment[ITEM].key];
            scan_segment[ITEM].value += addend;
          }

          // Set the output vector element
          spmv_params.d_vector_y[tile_start_coord.x + scan_segment[ITEM].key] = scan_segment[ITEM].value;
        }
      }
    }

    // Return the tile's running carry-out
    return tile_carry;
  }

  /**
   * @brief Consume a merge tile, specialized for indirect load of nonzeros
   *
   * @param is_direct_load
   *   Marker type indicating whether to load nonzeros directly during path-discovery or beforehand in batch
   */
  _CCCL_DEVICE _CCCL_FORCEINLINE KeyValuePairT
  ConsumeTile(int tile_idx, CoordinateT tile_start_coord, CoordinateT tile_end_coord, Int2Type<false> is_direct_load)
  {
    int tile_num_rows     = tile_end_coord.x - tile_start_coord.x;
    int tile_num_nonzeros = tile_end_coord.y - tile_start_coord.y;

#if (CUB_PTX_ARCH >= 520)

    OffsetT* s_tile_row_end_offsets = &temp_storage.aliasable.merge_items[0].row_end_offset;
    ValueT* s_tile_nonzeros         = &temp_storage.aliasable.merge_items[tile_num_rows + ITEMS_PER_THREAD].nonzero;

// Gather the nonzeros for the merge tile into shared memory
#  pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
    {
      int nonzero_idx = threadIdx.x + (ITEM * BLOCK_THREADS);

      ValueIteratorT a          = wd_values + tile_start_coord.y + nonzero_idx;
      ColumnIndicesIteratorT ci = wd_column_indices + tile_start_coord.y + nonzero_idx;
      ValueT* s                 = s_tile_nonzeros + nonzero_idx;

      if (nonzero_idx < tile_num_nonzeros)
      {
        OffsetT column_idx = *ci;
        ValueT value       = *a;

        ValueT vector_value = wd_vector_x[column_idx];

        ValueT nonzero = value * vector_value;

        *s = nonzero;
      }
    }

#else

    OffsetT* s_tile_row_end_offsets = &temp_storage.aliasable.merge_items[0].row_end_offset;
    ValueT* s_tile_nonzeros         = &temp_storage.aliasable.merge_items[tile_num_rows + ITEMS_PER_THREAD].nonzero;

    // Gather the nonzeros for the merge tile into shared memory
    if (tile_num_nonzeros > 0)
    {
#  pragma unroll
      for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
      {
        int nonzero_idx = threadIdx.x + (ITEM * BLOCK_THREADS);
        nonzero_idx     = CUB_MIN(nonzero_idx, tile_num_nonzeros - 1);

        OffsetT column_idx = wd_column_indices[tile_start_coord.y + nonzero_idx];
        ValueT value       = wd_values[tile_start_coord.y + nonzero_idx];

        ValueT vector_value = wd_vector_x[column_idx];

        ValueT nonzero = value * vector_value;

        s_tile_nonzeros[nonzero_idx] = nonzero;
      }
    }

#endif

// Gather the row end-offsets for the merge tile into shared memory
#pragma unroll 1
    for (int item = threadIdx.x; item < tile_num_rows + ITEMS_PER_THREAD; item += BLOCK_THREADS)
    {
      const OffsetT offset =
        (cub::min)(static_cast<OffsetT>(tile_start_coord.x + item), static_cast<OffsetT>(spmv_params.num_rows - 1));
      s_tile_row_end_offsets[item] = wd_row_end_offsets[offset];
    }

    CTA_SYNC();

    // Search for the thread's starting coordinate within the merge tile
    CountingInputIterator<OffsetT> tile_nonzero_indices(tile_start_coord.y);
    CoordinateT thread_start_coord;

    MergePathSearch(
      OffsetT(threadIdx.x * ITEMS_PER_THREAD), // Diagonal
      s_tile_row_end_offsets, // List A
      tile_nonzero_indices, // List B
      tile_num_rows,
      tile_num_nonzeros,
      thread_start_coord);

    CTA_SYNC(); // Perf-sync

    // Compute the thread's merge path segment
    CoordinateT thread_current_coord = thread_start_coord;
    KeyValuePairT scan_segment[ITEMS_PER_THREAD];
    ValueT running_total = 0.0;

    OffsetT row_end_offset = s_tile_row_end_offsets[thread_current_coord.x];
    ValueT nonzero         = s_tile_nonzeros[thread_current_coord.y];

#pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
    {
      if (tile_nonzero_indices[thread_current_coord.y] < row_end_offset)
      {
        // Move down (accumulate)
        scan_segment[ITEM].value = nonzero;
        running_total += nonzero;
        ++thread_current_coord.y;
        nonzero = s_tile_nonzeros[thread_current_coord.y];
      }
      else
      {
        // Move right (reset)
        scan_segment[ITEM].value = 0.0;
        running_total            = 0.0;
        ++thread_current_coord.x;
        row_end_offset = s_tile_row_end_offsets[thread_current_coord.x];
      }

      scan_segment[ITEM].key = thread_current_coord.x;
    }

    CTA_SYNC();

    // Block-wide reduce-value-by-segment
    KeyValuePairT tile_carry;
    ReduceBySegmentOpT scan_op;
    KeyValuePairT scan_item;

    scan_item.value = running_total;
    scan_item.key   = thread_current_coord.x;

    BlockScanT(temp_storage.aliasable.scan).ExclusiveScan(scan_item, scan_item, scan_op, tile_carry);

    if (threadIdx.x == 0)
    {
      scan_item.key   = thread_start_coord.x;
      scan_item.value = 0.0;
    }

    if (tile_num_rows > 0)
    {
      CTA_SYNC();

      // Scan downsweep and scatter
      ValueT* s_partials = &temp_storage.aliasable.merge_items[0].nonzero;

      if (scan_item.key != scan_segment[0].key)
      {
        s_partials[scan_item.key] = scan_item.value;
      }
      else
      {
        scan_segment[0].value += scan_item.value;
      }

#pragma unroll
      for (int ITEM = 1; ITEM < ITEMS_PER_THREAD; ++ITEM)
      {
        if (scan_segment[ITEM - 1].key != scan_segment[ITEM].key)
        {
          s_partials[scan_segment[ITEM - 1].key] = scan_segment[ITEM - 1].value;
        }
        else
        {
          scan_segment[ITEM].value += scan_segment[ITEM - 1].value;
        }
      }

      CTA_SYNC();

#pragma unroll 1
      for (int item = threadIdx.x; item < tile_num_rows; item += BLOCK_THREADS)
      {
        spmv_params.d_vector_y[tile_start_coord.x + item] = s_partials[item];
      }
    }

    // Return the tile's running carry-out
    return tile_carry;
  }

  /**
   * @brief Consume input tile
   *
   * @param[in] d_tile_coordinates
   *   Pointer to the temporary array of tile starting coordinates
   *
   * @param[out] d_tile_carry_pairs
   *   Pointer to the temporary array carry-out dot product row-ids, one per block
   *
   * @param[in] num_merge_tiles
   *   Number of merge tiles
   */
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  ConsumeTile(CoordinateT* d_tile_coordinates, KeyValuePairT* d_tile_carry_pairs, int num_merge_tiles)
  {
    int tile_idx = (blockIdx.x * gridDim.y) + blockIdx.y; // Current tile index

    if (tile_idx >= num_merge_tiles)
    {
      return;
    }

    // Read our starting coordinates
    if (threadIdx.x < 2)
    {
      if (d_tile_coordinates == nullptr)
      {
        // Search our starting coordinates
        OffsetT diagonal = (tile_idx + threadIdx.x) * TILE_ITEMS;
        CoordinateT tile_coord;
        CountingInputIterator<OffsetT> nonzero_indices(0);

        // Search the merge path
        MergePathSearch(
          diagonal,
          RowOffsetsSearchIteratorT(spmv_params.d_row_end_offsets),
          nonzero_indices,
          spmv_params.num_rows,
          spmv_params.num_nonzeros,
          tile_coord);

        temp_storage.tile_coords[threadIdx.x] = tile_coord;
      }
      else
      {
        temp_storage.tile_coords[threadIdx.x] = d_tile_coordinates[tile_idx + threadIdx.x];
      }
    }

    CTA_SYNC();

    CoordinateT tile_start_coord = temp_storage.tile_coords[0];
    CoordinateT tile_end_coord   = temp_storage.tile_coords[1];

    // Consume multi-segment tile
    KeyValuePairT tile_carry =
      ConsumeTile(tile_idx, tile_start_coord, tile_end_coord, Int2Type<AgentSpmvPolicyT::DIRECT_LOAD_NONZEROS>());

    // Output the tile's carry-out
    if (threadIdx.x == 0)
    {
      if (HAS_ALPHA)
      {
        tile_carry.value *= spmv_params.alpha;
      }

      tile_carry.key += tile_start_coord.x;
      if (tile_carry.key >= spmv_params.num_rows)
      {
        // FIXME: This works around an invalid memory access in the
        // fixup kernel. The underlying issue needs to be debugged and
        // properly fixed, but this hack prevents writes to
        // out-of-bounds addresses. It doesn't appear to have an effect
        // on the validity of the results, since this only affects the
        // carry-over from last tile in the input.
        tile_carry.key   = spmv_params.num_rows - 1;
        tile_carry.value = ValueT{};
      };

      d_tile_carry_pairs[tile_idx] = tile_carry;
    }
  }
};

CUB_NAMESPACE_END
