
/******************************************************************************
 * Copyright (c) NVIDIA CORPORATION.  All rights reserved.
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
 * cub::DeviceSelect::UniqueByKey provides device-wide, parallel operations for selecting unique
 * items by key from sequences of data items residing within device-accessible memory.
 */

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/agent/agent_unique_by_key.cuh>
#include <cub/device/dispatch/dispatch_scan.cuh>
#include <cub/device/dispatch/tuning/tuning_unique_by_key.cuh>
#include <cub/util_deprecated.cuh>
#include <cub/util_device.cuh>
#include <cub/util_math.cuh>
#include <cub/util_vsmem.cuh>

#include <iterator>

CUB_NAMESPACE_BEGIN

/******************************************************************************
 * Kernel entry points
 *****************************************************************************/

/**
 * @brief Unique by key kernel entry point (multi-block)
 *
 * @tparam KeyInputIteratorT
 *   Random-access input iterator type for keys
 *
 * @tparam ValueInputIteratorT
 *   Random-access input iterator type for values
 *
 * @tparam KeyOutputIteratorT
 *   Random-access output iterator type for keys
 *
 * @tparam ValueOutputIteratorT
 *   Random-access output iterator type for values
 *
 * @tparam NumSelectedIteratorT
 *   Output iterator type for recording the number of items selected
 *
 * @tparam ScanTileStateT
 *   Tile status interface type
 *
 * @tparam EqualityOpT
 *   Equality operator type
 *
 * @tparam OffsetT
 *   Signed integer type for global offsets
 *
 * @param[in] d_keys_in
 *   Pointer to the input sequence of keys
 *
 * @param[in] d_values_in
 *   Pointer to the input sequence of values
 *
 * @param[out] d_keys_out
 *   Pointer to the output sequence of selected data items
 *
 * @param[out] d_values_out
 *   Pointer to the output sequence of selected data items
 *
 * @param[out] d_num_selected_out
 *   Pointer to the total number of items selected
 *   (i.e., length of @p d_keys_out or @p d_values_out)
 *
 * @param[in] tile_state
 *   Tile status interface
 *
 * @param[in] equality_op
 *   Equality operator
 *
 * @param[in] num_items
 *   Total number of input items
 *   (i.e., length of @p d_keys_in or @p d_values_in)
 *
 * @param[in] num_tiles
 *   Total number of tiles for the entire problem
 *
 * @param[in] vsmem
 *   Memory to support virtual shared memory
 */
template <typename ChainedPolicyT,
          typename KeyInputIteratorT,
          typename ValueInputIteratorT,
          typename KeyOutputIteratorT,
          typename ValueOutputIteratorT,
          typename NumSelectedIteratorT,
          typename ScanTileStateT,
          typename EqualityOpT,
          typename OffsetT>
__launch_bounds__(int(
  cub::detail::vsmem_helper_default_fallback_policy_t<
    typename ChainedPolicyT::ActivePolicy::UniqueByKeyPolicyT,
    AgentUniqueByKey,
    KeyInputIteratorT,
    ValueInputIteratorT,
    KeyOutputIteratorT,
    ValueOutputIteratorT,
    EqualityOpT,
    OffsetT>::agent_policy_t::BLOCK_THREADS))
  CUB_DETAIL_KERNEL_ATTRIBUTES void DeviceUniqueByKeySweepKernel(
    KeyInputIteratorT d_keys_in,
    ValueInputIteratorT d_values_in,
    KeyOutputIteratorT d_keys_out,
    ValueOutputIteratorT d_values_out,
    NumSelectedIteratorT d_num_selected_out,
    ScanTileStateT tile_state,
    EqualityOpT equality_op,
    OffsetT num_items,
    int num_tiles,
    cub::detail::vsmem_t vsmem)
{
  using VsmemHelperT = cub::detail::vsmem_helper_default_fallback_policy_t<
    typename ChainedPolicyT::ActivePolicy::UniqueByKeyPolicyT,
    AgentUniqueByKey,
    KeyInputIteratorT,
    ValueInputIteratorT,
    KeyOutputIteratorT,
    ValueOutputIteratorT,
    EqualityOpT,
    OffsetT>;

  using AgentUniqueByKeyPolicyT = typename VsmemHelperT::agent_policy_t;

  // Thread block type for selecting data from input tiles
  using AgentUniqueByKeyT = typename VsmemHelperT::agent_t;

  // Static shared memory allocation
  __shared__ typename VsmemHelperT::static_temp_storage_t static_temp_storage;

  // Get temporary storage
  typename AgentUniqueByKeyT::TempStorage& temp_storage =
    VsmemHelperT::get_temp_storage(static_temp_storage, vsmem, (blockIdx.x * gridDim.y) + blockIdx.y);

  // Process tiles
  AgentUniqueByKeyT(temp_storage, d_keys_in, d_values_in, d_keys_out, d_values_out, equality_op, num_items)
    .ConsumeRange(num_tiles, tile_state, d_num_selected_out);

  // If applicable, hints to discard modified cache lines for vsmem
  VsmemHelperT::discard_temp_storage(temp_storage);
}

/******************************************************************************
 * Dispatch
 ******************************************************************************/

/**
 * @brief Utility class for dispatching the appropriately-tuned kernels for DeviceSelect
 *
 * @tparam KeyInputIteratorT
 *   Random-access input iterator type for keys
 *
 * @tparam ValueInputIteratorT
 *   Random-access input iterator type for values
 *
 * @tparam KeyOutputIteratorT
 *   Random-access output iterator type for keys
 *
 * @tparam ValueOutputIteratorT
 *   Random-access output iterator type for values
 *
 * @tparam NumSelectedIteratorT
 *   Output iterator type for recording the number of items selected
 *
 * @tparam EqualityOpT
 *   Equality operator type
 *
 * @tparam OffsetT
 *   Signed integer type for global offsets
 */
template <typename KeyInputIteratorT,
          typename ValueInputIteratorT,
          typename KeyOutputIteratorT,
          typename ValueOutputIteratorT,
          typename NumSelectedIteratorT,
          typename EqualityOpT,
          typename OffsetT,
          typename SelectedPolicy = DeviceUniqueByKeyPolicy<KeyInputIteratorT, ValueInputIteratorT>>
struct DispatchUniqueByKey : SelectedPolicy
{
  /******************************************************************************
   * Types and constants
   ******************************************************************************/

  enum
  {
    INIT_KERNEL_THREADS = 128,
  };

  // The input key and value type
  using KeyT   = typename std::iterator_traits<KeyInputIteratorT>::value_type;
  using ValueT = typename std::iterator_traits<ValueInputIteratorT>::value_type;

  // Tile status descriptor interface type
  using ScanTileStateT = ScanTileState<OffsetT>;

  /// Device-accessible allocation of temporary storage.  When nullptr, the required allocation size
  /// is written to `temp_storage_bytes` and no work is done.
  void* d_temp_storage;

  /// Reference to size in bytes of `d_temp_storage` allocation
  size_t& temp_storage_bytes;

  /// Pointer to the input sequence of keys
  KeyInputIteratorT d_keys_in;

  /// Pointer to the input sequence of values
  ValueInputIteratorT d_values_in;

  /// Pointer to the output sequence of selected data items
  KeyOutputIteratorT d_keys_out;

  /// Pointer to the output sequence of selected data items
  ValueOutputIteratorT d_values_out;

  /// Pointer to the total number of items selected
  /// (i.e., length of @p d_keys_out or @p d_values_out)
  NumSelectedIteratorT d_num_selected_out;

  /// Equality operator
  EqualityOpT equality_op;

  /// Total number of input items (i.e., length of @p d_keys_in or @p d_values_in)
  OffsetT num_items;

  /// **[optional]** CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
  cudaStream_t stream;

  /**
   * @param[in] d_temp_storage
   *   Device-accessible allocation of temporary storage.
   *   When nullptr, the required allocation size is written to
   *   `temp_storage_bytes` and no work is done.
   *
   * @tparam temp_storage_bytes
   *   [in,out] Reference to size in bytes of `d_temp_storage` allocation
   *
   * @param[in] d_keys_in
   *   Pointer to the input sequence of keys
   *
   * @param[in] d_values_in
   *   Pointer to the input sequence of values
   *
   * @param[out] d_keys_out
   *   Pointer to the output sequence of selected data items
   *
   * @param[out] d_values_out
   *   Pointer to the output sequence of selected data items
   *
   * @param[out] d_num_selected_out
   *   Pointer to the total number of items selected
   *   (i.e., length of @p d_keys_out or @p d_values_out)
   *
   * @param[in] equality_op
   *   Equality operator
   *
   * @param[in] num_items
   *   Total number of input items (i.e., length of @p d_keys_in or @p d_values_in)
   *
   * @param[in] stream
   *   **[optional]** CUDA stream to launch kernels within.
   *   Default is stream<sub>0</sub>.
   */
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE DispatchUniqueByKey(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    KeyInputIteratorT d_keys_in,
    ValueInputIteratorT d_values_in,
    KeyOutputIteratorT d_keys_out,
    ValueOutputIteratorT d_values_out,
    NumSelectedIteratorT d_num_selected_out,
    EqualityOpT equality_op,
    OffsetT num_items,
    cudaStream_t stream)
      : d_temp_storage(d_temp_storage)
      , temp_storage_bytes(temp_storage_bytes)
      , d_keys_in(d_keys_in)
      , d_values_in(d_values_in)
      , d_keys_out(d_keys_out)
      , d_values_out(d_values_out)
      , d_num_selected_out(d_num_selected_out)
      , equality_op(equality_op)
      , num_items(num_items)
      , stream(stream)
  {}

#ifndef DOXYGEN_SHOULD_SKIP_THIS // Do not document
  CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE DispatchUniqueByKey(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    KeyInputIteratorT d_keys_in,
    ValueInputIteratorT d_values_in,
    KeyOutputIteratorT d_keys_out,
    ValueOutputIteratorT d_values_out,
    NumSelectedIteratorT d_num_selected_out,
    EqualityOpT equality_op,
    OffsetT num_items,
    cudaStream_t stream,
    bool debug_synchronous)
      : d_temp_storage(d_temp_storage)
      , temp_storage_bytes(temp_storage_bytes)
      , d_keys_in(d_keys_in)
      , d_values_in(d_values_in)
      , d_keys_out(d_keys_out)
      , d_values_out(d_values_out)
      , d_num_selected_out(d_num_selected_out)
      , equality_op(equality_op)
      , num_items(num_items)
      , stream(stream)
  {
    CUB_DETAIL_RUNTIME_DEBUG_SYNC_USAGE_LOG
  }
#endif // DOXYGEN_SHOULD_SKIP_THIS

  /******************************************************************************
   * Dispatch entrypoints
   ******************************************************************************/

  template <typename ActivePolicyT, typename InitKernel, typename ScanKernel>
  CUB_RUNTIME_FUNCTION _CCCL_HOST _CCCL_FORCEINLINE cudaError_t Invoke(InitKernel init_kernel, ScanKernel scan_kernel)
  {
    using Policy = typename ActivePolicyT::UniqueByKeyPolicyT;

    using VsmemHelperT = cub::detail::vsmem_helper_default_fallback_policy_t<
      Policy,
      AgentUniqueByKey,
      KeyInputIteratorT,
      ValueInputIteratorT,
      KeyOutputIteratorT,
      ValueOutputIteratorT,
      EqualityOpT,
      OffsetT>;

    cudaError error = cudaSuccess;
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
      constexpr auto block_threads    = VsmemHelperT::agent_policy_t::BLOCK_THREADS;
      constexpr auto items_per_thread = VsmemHelperT::agent_policy_t::ITEMS_PER_THREAD;
      int tile_size                   = block_threads * items_per_thread;
      int num_tiles                   = static_cast<int>(::cuda::ceil_div(num_items, tile_size));
      const auto vsmem_size           = num_tiles * VsmemHelperT::vsmem_per_block;

      // Specify temporary storage allocation requirements
      size_t allocation_sizes[2] = {0, vsmem_size};

      // Bytes needed for tile status descriptors
      error = CubDebug(ScanTileStateT::AllocationSize(num_tiles, allocation_sizes[0]));
      if (cudaSuccess != error)
      {
        break;
      }

      // Compute allocation pointers into the single storage blob (or compute the necessary size of the blob)
      void* allocations[2] = {nullptr, nullptr};

      error = CubDebug(AliasTemporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes));
      if (cudaSuccess != error)
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
      error = CubDebug(tile_state.Init(num_tiles, allocations[0], allocation_sizes[0]));
      if (cudaSuccess != error)
      {
        break;
      }

      // Log init_kernel configuration
      num_tiles          = CUB_MAX(1, num_tiles);
      int init_grid_size = ::cuda::ceil_div(num_tiles, INIT_KERNEL_THREADS);

#ifdef CUB_DETAIL_DEBUG_ENABLE_LOG
      _CubLog("Invoking init_kernel<<<%d, %d, 0, %lld>>>()\n", init_grid_size, INIT_KERNEL_THREADS, (long long) stream);
#endif // CUB_DETAIL_DEBUG_ENABLE_LOG

      // Invoke init_kernel to initialize tile descriptors
      THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron(init_grid_size, INIT_KERNEL_THREADS, 0, stream)
        .doit(init_kernel, tile_state, num_tiles, d_num_selected_out);

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

      // Return if empty problem
      if (num_items == 0)
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
        // Get SM occupancy for unique_by_key_kernel
        int scan_sm_occupancy;
        error = CubDebug(MaxSmOccupancy(scan_sm_occupancy, // out
                                        scan_kernel,
                                        block_threads));
        if (cudaSuccess != error)
        {
          break;
        }

        _CubLog("Invoking unique_by_key_kernel<<<{%d,%d,%d}, %d, 0, "
                "%lld>>>(), %d items per thread, %d SM occupancy\n",
                scan_grid_size.x,
                scan_grid_size.y,
                scan_grid_size.z,
                block_threads,
                (long long) stream,
                items_per_thread,
                scan_sm_occupancy);
      }
#endif // CUB_DETAIL_DEBUG_ENABLE_LOG

      // Invoke select_if_kernel
      error =
        THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron(scan_grid_size, block_threads, 0, stream)
          .doit(scan_kernel,
                d_keys_in,
                d_values_in,
                d_keys_out,
                d_values_out,
                d_num_selected_out,
                tile_state,
                equality_op,
                num_items,
                num_tiles,
                cub::detail::vsmem_t{allocations[1]});

      // Check for failure to launch
      error = CubDebug(error);
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
  CUB_RUNTIME_FUNCTION _CCCL_HOST _CCCL_FORCEINLINE cudaError_t Invoke()
  {
    using MaxPolicyT = typename DispatchUniqueByKey::MaxPolicy;

    // Ensure kernels are instantiated.
    return Invoke<ActivePolicyT>(
      DeviceCompactInitKernel<ScanTileStateT, NumSelectedIteratorT>,
      DeviceUniqueByKeySweepKernel<
        MaxPolicyT,
        KeyInputIteratorT,
        ValueInputIteratorT,
        KeyOutputIteratorT,
        ValueOutputIteratorT,
        NumSelectedIteratorT,
        ScanTileStateT,
        EqualityOpT,
        OffsetT>);
  }

  /**
   * @brief Internal dispatch routine
   *
   * @param[in] d_temp_storage
   *   Device-accessible allocation of temporary storage.
   *   When nullptr, the required allocation size is written to
   *   `temp_storage_bytes` and no work is done.
   *
   * @param[in,out] &temp_storage_bytes
   *   Reference to size in bytes of `d_temp_storage` allocation
   *
   * @param[in] d_keys_in
   *   Pointer to the input sequence of keys
   *
   * @param[in] d_values_in
   *   Pointer to the input sequence of values
   *
   * @param[out] d_keys_out
   *   Pointer to the output sequence of selected data items
   *
   * @param[out] d_values_out
   *   Pointer to the output sequence of selected data items
   *
   * @param[out] d_num_selected_out
   *   Pointer to the total number of items selected
   *   (i.e., length of @p d_keys_out or @p d_values_out)
   *
   * @param[in] equality_op
   *   Equality operator
   *
   * @param[in] num_items
   *   Total number of input items (i.e., the length of @p d_in)
   *
   * @param[in] stream
   *   **[optional]** CUDA stream to launch kernels within.
   *   Default is stream<sub>0</sub>.
   */
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t Dispatch(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    KeyInputIteratorT d_keys_in,
    ValueInputIteratorT d_values_in,
    KeyOutputIteratorT d_keys_out,
    ValueOutputIteratorT d_values_out,
    NumSelectedIteratorT d_num_selected_out,
    EqualityOpT equality_op,
    OffsetT num_items,
    cudaStream_t stream)
  {
    using MaxPolicyT = typename DispatchUniqueByKey::MaxPolicy;

    cudaError_t error;
    do
    {
      // Get PTX version
      int ptx_version = 0;
      error           = CubDebug(PtxVersion(ptx_version));
      if (cudaSuccess != error)
      {
        break;
      }

      // Create dispatch functor
      DispatchUniqueByKey dispatch(
        d_temp_storage,
        temp_storage_bytes,
        d_keys_in,
        d_values_in,
        d_keys_out,
        d_values_out,
        d_num_selected_out,
        equality_op,
        num_items,
        stream);

      // Dispatch to chained policy
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
    size_t& temp_storage_bytes,
    KeyInputIteratorT d_keys_in,
    ValueInputIteratorT d_values_in,
    KeyOutputIteratorT d_keys_out,
    ValueOutputIteratorT d_values_out,
    NumSelectedIteratorT d_num_selected_out,
    EqualityOpT equality_op,
    OffsetT num_items,
    cudaStream_t stream,
    bool debug_synchronous)
  {
    CUB_DETAIL_RUNTIME_DEBUG_SYNC_USAGE_LOG

    return Dispatch(
      d_temp_storage,
      temp_storage_bytes,
      d_keys_in,
      d_values_in,
      d_keys_out,
      d_values_out,
      d_num_selected_out,
      equality_op,
      num_items,
      stream);
  }
#endif // DOXYGEN_SHOULD_SKIP_THIS
};

CUB_NAMESPACE_END
