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

#include <cub/agent/agent_merge_sort.cuh>
#include <cub/util_deprecated.cuh>
#include <cub/util_device.cuh>
#include <cub/util_math.cuh>
#include <cub/util_namespace.cuh>
#include <cub/util_vsmem.cuh>

#include <thrust/detail/integer_math.h>
#include <thrust/system/cuda/detail/core/triple_chevron_launch.h>

#include <cuda/std/type_traits>

CUB_NAMESPACE_BEGIN

namespace detail
{

/**
 * @brief Helper class template that provides two agent template instantiations: one instantiated with the default
 * policy and one with the fallback policy. This helps to avoid having to enlist all the agent's template parameters
 * twice: once for the default agent and once for the fallback agent
 */
template <typename DefaultPolicyT, typename FallbackPolicyT, template <typename...> class AgentT, typename... AgentParamsT>
struct dual_policy_agent_helper_t
{
  using default_agent_t  = AgentT<DefaultPolicyT, AgentParamsT...>;
  using fallback_agent_t = AgentT<FallbackPolicyT, AgentParamsT...>;

  static constexpr auto default_size  = sizeof(typename default_agent_t::TempStorage);
  static constexpr auto fallback_size = sizeof(typename fallback_agent_t::TempStorage);
};

/**
 * @brief Helper class template for merge sort-specific virtual shared memory handling. The merge sort algorithm in its
 * current implementation relies on the fact that both the sorting as well as the merging kernels use the same tile
 * size. This circumstance needs to be respected when determining whether the fallback policy for large user types is
 * applicable: we must either use the fallback for both or for none of the two agents.
 */
template <typename DefaultPolicyT,
          typename KeyInputIteratorT,
          typename ValueInputIteratorT,
          typename KeyIteratorT,
          typename ValueIteratorT,
          typename OffsetT,
          typename CompareOpT,
          typename KeyT,
          typename ValueT>
class merge_sort_vsmem_helper_t
{
private:
  // Default fallback policy with a smaller tile size
  using fallback_policy_t = cub::detail::policy_wrapper_t<DefaultPolicyT, 64, 1>;

  // Helper for the `AgentBlockSort` template with one member type alias for the agent template instantiated with the
  // default policy and one instantiated with the fallback policy
  using block_sort_helper_t = dual_policy_agent_helper_t<
    DefaultPolicyT,
    fallback_policy_t,
    AgentBlockSort,
    KeyInputIteratorT,
    ValueInputIteratorT,
    KeyIteratorT,
    ValueIteratorT,
    OffsetT,
    CompareOpT,
    KeyT,
    ValueT>;
  using default_block_sort_agent_t  = typename block_sort_helper_t::default_agent_t;
  using fallback_block_sort_agent_t = typename block_sort_helper_t::fallback_agent_t;

  // Helper for the `AgentMerge` template with one member type alias for the agent template instantiated with the
  // default policy and one instantiated with the fallback policy
  using merge_helper_t = dual_policy_agent_helper_t<
    DefaultPolicyT,
    fallback_policy_t,
    AgentMerge,
    KeyIteratorT,
    ValueIteratorT,
    OffsetT,
    CompareOpT,
    KeyT,
    ValueT>;
  using default_merge_agent_t  = typename merge_helper_t::default_agent_t;
  using fallback_merge_agent_t = typename merge_helper_t::fallback_agent_t;

  // Use fallback if either (a) the default block sort or (b) the block merge agent exceed the maximum shared memory
  // available per block and both (1) the fallback block sort and (2) the fallback merge agent would not exceed the
  // available shared memory
  static constexpr auto max_default_size = (cub::max)(block_sort_helper_t::default_size, merge_helper_t::default_size);
  static constexpr auto max_fallback_size =
    (cub::max)(block_sort_helper_t::fallback_size, merge_helper_t::fallback_size);
  static constexpr bool uses_fallback_policy =
    (max_default_size > max_smem_per_block) && (max_fallback_size <= max_smem_per_block);

public:
  using policy_t = ::cuda::std::_If<uses_fallback_policy, fallback_policy_t, DefaultPolicyT>;
  using block_sort_agent_t =
    ::cuda::std::_If<uses_fallback_policy, fallback_block_sort_agent_t, default_block_sort_agent_t>;
  using merge_agent_t = ::cuda::std::_If<uses_fallback_policy, fallback_merge_agent_t, default_merge_agent_t>;
};
} // namespace detail

template <typename ChainedPolicyT,
          typename KeyInputIteratorT,
          typename ValueInputIteratorT,
          typename KeyIteratorT,
          typename ValueIteratorT,
          typename OffsetT,
          typename CompareOpT,
          typename KeyT,
          typename ValueT>
__launch_bounds__(
  cub::detail::merge_sort_vsmem_helper_t<
    typename ChainedPolicyT::ActivePolicy::MergeSortPolicy,
    KeyInputIteratorT,
    ValueInputIteratorT,
    KeyIteratorT,
    ValueIteratorT,
    OffsetT,
    CompareOpT,
    KeyT,
    ValueT>::policy_t::BLOCK_THREADS)
  CUB_DETAIL_KERNEL_ATTRIBUTES void DeviceMergeSortBlockSortKernel(
    bool ping,
    KeyInputIteratorT keys_in,
    ValueInputIteratorT items_in,
    KeyIteratorT keys_out,
    ValueIteratorT items_out,
    OffsetT keys_count,
    KeyT* tmp_keys_out,
    ValueT* tmp_items_out,
    CompareOpT compare_op,
    cub::detail::vsmem_t vsmem)
{
  using MergeSortHelperT = cub::detail::merge_sort_vsmem_helper_t<
    typename ChainedPolicyT::ActivePolicy::MergeSortPolicy,
    KeyInputIteratorT,
    ValueInputIteratorT,
    KeyIteratorT,
    ValueIteratorT,
    OffsetT,
    CompareOpT,
    KeyT,
    ValueT>;

  using ActivePolicyT = typename MergeSortHelperT::policy_t;

  using AgentBlockSortT = typename MergeSortHelperT::block_sort_agent_t;

  using VSmemHelperT = cub::detail::vsmem_helper_impl<AgentBlockSortT>;

  // Static shared memory allocation
  __shared__ typename VSmemHelperT::static_temp_storage_t static_temp_storage;

  // Get temporary storage
  typename AgentBlockSortT::TempStorage& temp_storage = VSmemHelperT::get_temp_storage(static_temp_storage, vsmem);

  AgentBlockSortT agent(
    ping,
    temp_storage,
    THRUST_NS_QUALIFIER::cuda_cub::core::make_load_iterator(ActivePolicyT(), keys_in),
    THRUST_NS_QUALIFIER::cuda_cub::core::make_load_iterator(ActivePolicyT(), items_in),
    keys_count,
    keys_out,
    items_out,
    tmp_keys_out,
    tmp_items_out,
    compare_op);

  agent.Process();

  // If applicable, hints to discard modified cache lines for vsmem
  VSmemHelperT::discard_temp_storage(temp_storage);
}

template <typename KeyIteratorT, typename OffsetT, typename CompareOpT, typename KeyT>
CUB_DETAIL_KERNEL_ATTRIBUTES void DeviceMergeSortPartitionKernel(
  bool ping,
  KeyIteratorT keys_ping,
  KeyT* keys_pong,
  OffsetT keys_count,
  OffsetT num_partitions,
  OffsetT* merge_partitions,
  CompareOpT compare_op,
  OffsetT target_merged_tiles_number,
  int items_per_tile)
{
  OffsetT partition_idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (partition_idx < num_partitions)
  {
    AgentPartition<KeyIteratorT, OffsetT, CompareOpT, KeyT> agent(
      ping,
      keys_ping,
      keys_pong,
      keys_count,
      partition_idx,
      merge_partitions,
      compare_op,
      target_merged_tiles_number,
      items_per_tile,
      num_partitions);

    agent.Process();
  }
}

template <typename ChainedPolicyT,
          typename KeyInputIteratorT,
          typename ValueInputIteratorT,
          typename KeyIteratorT,
          typename ValueIteratorT,
          typename OffsetT,
          typename CompareOpT,
          typename KeyT,
          typename ValueT>
__launch_bounds__(
  cub::detail::merge_sort_vsmem_helper_t<
    typename ChainedPolicyT::ActivePolicy::MergeSortPolicy,
    KeyInputIteratorT,
    ValueInputIteratorT,
    KeyIteratorT,
    ValueIteratorT,
    OffsetT,
    CompareOpT,
    KeyT,
    ValueT>::policy_t::BLOCK_THREADS)
  CUB_DETAIL_KERNEL_ATTRIBUTES void DeviceMergeSortMergeKernel(
    bool ping,
    KeyIteratorT keys_ping,
    ValueIteratorT items_ping,
    OffsetT keys_count,
    KeyT* keys_pong,
    ValueT* items_pong,
    CompareOpT compare_op,
    OffsetT* merge_partitions,
    OffsetT target_merged_tiles_number,
    cub::detail::vsmem_t vsmem)
{
  using MergeSortHelperT = cub::detail::merge_sort_vsmem_helper_t<
    typename ChainedPolicyT::ActivePolicy::MergeSortPolicy,
    KeyInputIteratorT,
    ValueInputIteratorT,
    KeyIteratorT,
    ValueIteratorT,
    OffsetT,
    CompareOpT,
    KeyT,
    ValueT>;

  using ActivePolicyT = typename MergeSortHelperT::policy_t;

  using AgentMergeT = typename MergeSortHelperT::merge_agent_t;

  using VSmemHelperT = cub::detail::vsmem_helper_impl<AgentMergeT>;

  // Static shared memory allocation
  __shared__ typename VSmemHelperT::static_temp_storage_t static_temp_storage;

  // Get temporary storage
  typename AgentMergeT::TempStorage& temp_storage = VSmemHelperT::get_temp_storage(static_temp_storage, vsmem);

  AgentMergeT agent(
    ping,
    temp_storage,
    THRUST_NS_QUALIFIER::cuda_cub::core::make_load_iterator(ActivePolicyT(), keys_ping),
    THRUST_NS_QUALIFIER::cuda_cub::core::make_load_iterator(ActivePolicyT(), items_ping),
    THRUST_NS_QUALIFIER::cuda_cub::core::make_load_iterator(ActivePolicyT(), keys_pong),
    THRUST_NS_QUALIFIER::cuda_cub::core::make_load_iterator(ActivePolicyT(), items_pong),
    keys_count,
    keys_pong,
    items_pong,
    keys_ping,
    items_ping,
    compare_op,
    merge_partitions,
    target_merged_tiles_number);

  agent.Process();

  // If applicable, hints to discard modified cache lines for vsmem
  VSmemHelperT::discard_temp_storage(temp_storage);
}

/*******************************************************************************
 * Policy
 ******************************************************************************/

template <typename KeyIteratorT>
struct DeviceMergeSortPolicy
{
  using KeyT = cub::detail::value_t<KeyIteratorT>;

  //----------------------------------------------------------------------------
  // Architecture-specific tuning policies
  //----------------------------------------------------------------------------

  struct Policy350 : ChainedPolicy<350, Policy350, Policy350>
  {
    using MergeSortPolicy =
      AgentMergeSortPolicy<256,
                           Nominal4BItemsToItems<KeyT>(11),
                           cub::BLOCK_LOAD_WARP_TRANSPOSE,
                           cub::LOAD_LDG,
                           cub::BLOCK_STORE_WARP_TRANSPOSE>;
  };

// NVBug 3384810
#if defined(_NVHPC_CUDA)
  using Policy520 = Policy350;
#else
  struct Policy520 : ChainedPolicy<520, Policy520, Policy350>
  {
    using MergeSortPolicy =
      AgentMergeSortPolicy<512,
                           Nominal4BItemsToItems<KeyT>(15),
                           cub::BLOCK_LOAD_WARP_TRANSPOSE,
                           cub::LOAD_LDG,
                           cub::BLOCK_STORE_WARP_TRANSPOSE>;
  };
#endif

  struct Policy600 : ChainedPolicy<600, Policy600, Policy520>
  {
    using MergeSortPolicy =
      AgentMergeSortPolicy<256,
                           Nominal4BItemsToItems<KeyT>(17),
                           cub::BLOCK_LOAD_WARP_TRANSPOSE,
                           cub::LOAD_DEFAULT,
                           cub::BLOCK_STORE_WARP_TRANSPOSE>;
  };

  /// MaxPolicy
  using MaxPolicy = Policy600;
};

template <typename KeyInputIteratorT,
          typename ValueInputIteratorT,
          typename KeyIteratorT,
          typename ValueIteratorT,
          typename OffsetT,
          typename CompareOpT,
          typename SelectedPolicy = DeviceMergeSortPolicy<KeyIteratorT>>
struct DispatchMergeSort : SelectedPolicy
{
  using KeyT   = cub::detail::value_t<KeyIteratorT>;
  using ValueT = cub::detail::value_t<ValueIteratorT>;

  /// Whether or not there are values to be trucked along with keys
  static constexpr bool KEYS_ONLY = std::is_same<ValueT, NullType>::value;

  // Problem state

  /// Device-accessible allocation of temporary storage. When nullptr, the required
  /// allocation size is written to \p temp_storage_bytes and no work is done.
  void* d_temp_storage;

  /// Reference to size in bytes of \p d_temp_storage allocation
  std::size_t& temp_storage_bytes;

  /// Pointer to the input sequence of unsorted input keys
  KeyInputIteratorT d_input_keys;

  /// Pointer to the input sequence of unsorted input values
  ValueInputIteratorT d_input_items;

  /// Pointer to the output sequence of sorted input keys
  KeyIteratorT d_output_keys;

  /// Pointer to the output sequence of sorted input values
  ValueIteratorT d_output_items;

  /// Number of items to sort
  OffsetT num_items;

  /// Comparison function object which returns true if the first argument is
  /// ordered before the second
  CompareOpT compare_op;

  /// CUDA stream to launch kernels within. Default is stream<sub>0</sub>.
  cudaStream_t stream;

  int ptx_version;

  // Constructor
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE DispatchMergeSort(
    void* d_temp_storage,
    std::size_t& temp_storage_bytes,
    KeyInputIteratorT d_input_keys,
    ValueInputIteratorT d_input_items,
    KeyIteratorT d_output_keys,
    ValueIteratorT d_output_items,
    OffsetT num_items,
    CompareOpT compare_op,
    cudaStream_t stream,
    int ptx_version)
      : d_temp_storage(d_temp_storage)
      , temp_storage_bytes(temp_storage_bytes)
      , d_input_keys(d_input_keys)
      , d_input_items(d_input_items)
      , d_output_keys(d_output_keys)
      , d_output_items(d_output_items)
      , num_items(num_items)
      , compare_op(compare_op)
      , stream(stream)
      , ptx_version(ptx_version)
  {}

  CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE DispatchMergeSort(
    void* d_temp_storage,
    std::size_t& temp_storage_bytes,
    KeyInputIteratorT d_input_keys,
    ValueInputIteratorT d_input_items,
    KeyIteratorT d_output_keys,
    ValueIteratorT d_output_items,
    OffsetT num_items,
    CompareOpT compare_op,
    cudaStream_t stream,
    bool debug_synchronous,
    int ptx_version)
      : d_temp_storage(d_temp_storage)
      , temp_storage_bytes(temp_storage_bytes)
      , d_input_keys(d_input_keys)
      , d_input_items(d_input_items)
      , d_output_keys(d_output_keys)
      , d_output_items(d_output_items)
      , num_items(num_items)
      , compare_op(compare_op)
      , stream(stream)
      , ptx_version(ptx_version)
  {
    CUB_DETAIL_RUNTIME_DEBUG_SYNC_USAGE_LOG
  }

  // Invocation
  template <typename ActivePolicyT>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t Invoke()
  {
    using MergePolicyT = typename ActivePolicyT::MergeSortPolicy;

    using merge_sort_helper_t = cub::detail::merge_sort_vsmem_helper_t<
      MergePolicyT,
      KeyInputIteratorT,
      ValueInputIteratorT,
      KeyIteratorT,
      ValueIteratorT,
      OffsetT,
      CompareOpT,
      KeyT,
      ValueT>;

    using BlockSortVSmemHelperT  = cub::detail::vsmem_helper_impl<typename merge_sort_helper_t::block_sort_agent_t>;
    using MergeAgentVSmemHelperT = cub::detail::vsmem_helper_impl<typename merge_sort_helper_t::merge_agent_t>;

    using MaxPolicyT = typename DispatchMergeSort::MaxPolicy;

    cudaError error = cudaSuccess;

    if (num_items == 0)
    {
      if (d_temp_storage == nullptr)
      {
        temp_storage_bytes = 0;
      }
      return error;
    }

    do
    {
      constexpr auto tile_size = merge_sort_helper_t::policy_t::ITEMS_PER_TILE;
      const auto num_tiles     = ::cuda::ceil_div(num_items, tile_size);

      const auto merge_partitions_size         = static_cast<std::size_t>(1 + num_tiles) * sizeof(OffsetT);
      const auto temporary_keys_storage_size   = static_cast<std::size_t>(num_items * sizeof(KeyT));
      const auto temporary_values_storage_size = static_cast<std::size_t>(num_items * sizeof(ValueT)) * !KEYS_ONLY;

      /**
       * Merge sort supports large types, which can lead to excessive shared memory size requirements. In these cases,
       * merge sort allocates virtual shared memory that resides in global memory.
       */
      const std::size_t block_sort_smem_size       = num_tiles * BlockSortVSmemHelperT::vsmem_per_block;
      const std::size_t merge_smem_size            = num_tiles * MergeAgentVSmemHelperT::vsmem_per_block;
      const std::size_t virtual_shared_memory_size = (cub::max)(block_sort_smem_size, merge_smem_size);

      void* allocations[4]            = {nullptr, nullptr, nullptr, nullptr};
      std::size_t allocation_sizes[4] = {
        merge_partitions_size, temporary_keys_storage_size, temporary_values_storage_size, virtual_shared_memory_size};

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

      const int num_passes = static_cast<int>(THRUST_NS_QUALIFIER::detail::log2_ri(num_tiles));

      /*
       * The algorithm consists of stages. At each stage, there are input and output arrays. There are two pairs of
       * arrays allocated (keys and items). One pair is from function arguments and another from temporary storage. Ping
       * is a helper variable that controls which of these two pairs of arrays is an input and which is an output for a
       * current stage. If the ping is true - the current stage stores its result in the temporary storage. The
       * temporary storage acts as input data otherwise.
       *
       * Block sort is executed before the main loop. It stores its result in  the pair of arrays that will be an input
       * of the next stage. The initial value of the ping variable is selected so that the result of the final stage is
       * stored in the input arrays.
       */
      bool ping = num_passes % 2 == 0;

      auto merge_partitions = static_cast<OffsetT*>(allocations[0]);
      auto keys_buffer      = static_cast<KeyT*>(allocations[1]);
      auto items_buffer     = static_cast<ValueT*>(allocations[2]);

      // Invoke DeviceMergeSortBlockSortKernel
      THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron(
        static_cast<int>(num_tiles), merge_sort_helper_t::policy_t::BLOCK_THREADS, 0, stream)
        .doit(
          DeviceMergeSortBlockSortKernel<
            MaxPolicyT,
            KeyInputIteratorT,
            ValueInputIteratorT,
            KeyIteratorT,
            ValueIteratorT,
            OffsetT,
            CompareOpT,
            KeyT,
            ValueT>,
          ping,
          d_input_keys,
          d_input_items,
          d_output_keys,
          d_output_items,
          num_items,
          keys_buffer,
          items_buffer,
          compare_op,
          cub::detail::vsmem_t{allocations[3]});

      error = CubDebug(detail::DebugSyncStream(stream));
      if (cudaSuccess != error)
      {
        break;
      }

      // Check for failure to launch
      error = CubDebug(cudaPeekAtLastError());
      if (cudaSuccess != error)
      {
        break;
      }

      const OffsetT num_partitions              = num_tiles + 1;
      constexpr int threads_per_partition_block = 256;
      const int partition_grid_size = static_cast<int>(::cuda::ceil_div(num_partitions, threads_per_partition_block));

      error = CubDebug(detail::DebugSyncStream(stream));
      if (cudaSuccess != error)
      {
        break;
      }

      // Check for failure to launch
      error = CubDebug(cudaPeekAtLastError());
      if (cudaSuccess != error)
      {
        break;
      }

      for (int pass = 0; pass < num_passes; ++pass, ping = !ping)
      {
        const OffsetT target_merged_tiles_number = OffsetT(2) << pass;

        // Partition
        THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron(
          partition_grid_size, threads_per_partition_block, 0, stream)
          .doit(DeviceMergeSortPartitionKernel<KeyIteratorT, OffsetT, CompareOpT, KeyT>,
                ping,
                d_output_keys,
                keys_buffer,
                num_items,
                num_partitions,
                merge_partitions,
                compare_op,
                target_merged_tiles_number,
                tile_size);

        error = CubDebug(detail::DebugSyncStream(stream));
        if (cudaSuccess != error)
        {
          break;
        }

        // Check for failure to launch
        error = CubDebug(cudaPeekAtLastError());
        if (cudaSuccess != error)
        {
          break;
        }

        // Merge
        THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron(
          static_cast<int>(num_tiles), static_cast<int>(merge_sort_helper_t::policy_t::BLOCK_THREADS), 0, stream)
          .doit(
            DeviceMergeSortMergeKernel<MaxPolicyT,
                                       KeyInputIteratorT,
                                       ValueInputIteratorT,
                                       KeyIteratorT,
                                       ValueIteratorT,
                                       OffsetT,
                                       CompareOpT,
                                       KeyT,
                                       ValueT>,
            ping,
            d_output_keys,
            d_output_items,
            num_items,
            keys_buffer,
            items_buffer,
            compare_op,
            merge_partitions,
            target_merged_tiles_number,
            cub::detail::vsmem_t{allocations[3]});

        error = CubDebug(detail::DebugSyncStream(stream));
        if (cudaSuccess != error)
        {
          break;
        }

        // Check for failure to launch
        error = CubDebug(cudaPeekAtLastError());
        if (cudaSuccess != error)
        {
          break;
        }
      }
    } while (0);

    return error;
  }

  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t Dispatch(
    void* d_temp_storage,
    std::size_t& temp_storage_bytes,
    KeyInputIteratorT d_input_keys,
    ValueInputIteratorT d_input_items,
    KeyIteratorT d_output_keys,
    ValueIteratorT d_output_items,
    OffsetT num_items,
    CompareOpT compare_op,
    cudaStream_t stream)
  {
    using MaxPolicyT = typename DispatchMergeSort::MaxPolicy;

    cudaError error = cudaSuccess;
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
      DispatchMergeSort dispatch(
        d_temp_storage,
        temp_storage_bytes,
        d_input_keys,
        d_input_items,
        d_output_keys,
        d_output_items,
        num_items,
        compare_op,
        stream,
        ptx_version);

      // Dispatch to chained policy
      error = CubDebug(MaxPolicyT::Invoke(ptx_version, dispatch));
      if (cudaSuccess != error)
      {
        break;
      }
    } while (0);

    return error;
  }

  CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t Dispatch(
    void* d_temp_storage,
    std::size_t& temp_storage_bytes,
    KeyInputIteratorT d_input_keys,
    ValueInputIteratorT d_input_items,
    KeyIteratorT d_output_keys,
    ValueIteratorT d_output_items,
    OffsetT num_items,
    CompareOpT compare_op,
    cudaStream_t stream,
    bool debug_synchronous)
  {
    CUB_DETAIL_RUNTIME_DEBUG_SYNC_USAGE_LOG

    return Dispatch(
      d_temp_storage,
      temp_storage_bytes,
      d_input_keys,
      d_input_items,
      d_output_keys,
      d_output_items,
      num_items,
      compare_op,
      stream);
  }
};

CUB_NAMESPACE_END
