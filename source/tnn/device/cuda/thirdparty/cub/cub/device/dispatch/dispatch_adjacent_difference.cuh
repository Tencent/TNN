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

#include <cub/agent/agent_adjacent_difference.cuh>
#include <cub/detail/type_traits.cuh>
#include <cub/util_debug.cuh>
#include <cub/util_deprecated.cuh>
#include <cub/util_device.cuh>
#include <cub/util_math.cuh>
#include <cub/util_namespace.cuh>

#include <thrust/system/cuda/detail/core/triple_chevron_launch.h>

#include <iterator>

CUB_NAMESPACE_BEGIN

template <typename AgentDifferenceInitT, typename InputIteratorT, typename InputT, typename OffsetT>
CUB_DETAIL_KERNEL_ATTRIBUTES void
DeviceAdjacentDifferenceInitKernel(InputIteratorT first, InputT* result, OffsetT num_tiles, int items_per_tile)
{
  const int tile_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  AgentDifferenceInitT::Process(tile_idx, first, result, num_tiles, items_per_tile);
}

template <typename ChainedPolicyT,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename DifferenceOpT,
          typename OffsetT,
          typename InputT,
          bool MayAlias,
          bool ReadLeft>
CUB_DETAIL_KERNEL_ATTRIBUTES void DeviceAdjacentDifferenceDifferenceKernel(
  InputIteratorT input,
  InputT* first_tile_previous,
  OutputIteratorT result,
  DifferenceOpT difference_op,
  OffsetT num_items)
{
  using ActivePolicyT = typename ChainedPolicyT::ActivePolicy::AdjacentDifferencePolicy;

  // It is OK to introspect the return type or parameter types of the
  // `operator()` function of `__device__` extended lambda within device code.
  using OutputT = detail::invoke_result_t<DifferenceOpT, InputT, InputT>;

  using Agent =
    AgentDifference<ActivePolicyT,
                    InputIteratorT,
                    OutputIteratorT,
                    DifferenceOpT,
                    OffsetT,
                    InputT,
                    OutputT,
                    MayAlias,
                    ReadLeft>;

  __shared__ typename Agent::TempStorage storage;

  Agent agent(storage, input, first_tile_previous, result, difference_op, num_items);

  int tile_idx      = static_cast<int>(blockIdx.x);
  OffsetT tile_base = static_cast<OffsetT>(tile_idx) * ActivePolicyT::ITEMS_PER_TILE;

  agent.Process(tile_idx, tile_base);
}

template <typename InputIteratorT, bool MayAlias = true>
struct DeviceAdjacentDifferencePolicy
{
  using ValueT = typename std::iterator_traits<InputIteratorT>::value_type;

  //------------------------------------------------------------------------------
  // Architecture-specific tuning policies
  //------------------------------------------------------------------------------

  struct Policy300 : ChainedPolicy<300, Policy300, Policy300>
  {
    using AdjacentDifferencePolicy =
      AgentAdjacentDifferencePolicy<128,
                                    Nominal8BItemsToItems<ValueT>(7),
                                    BLOCK_LOAD_WARP_TRANSPOSE,
                                    LOAD_DEFAULT,
                                    BLOCK_STORE_WARP_TRANSPOSE>;
  };

  struct Policy350 : ChainedPolicy<350, Policy350, Policy300>
  {
    using AdjacentDifferencePolicy =
      AgentAdjacentDifferencePolicy<128,
                                    Nominal8BItemsToItems<ValueT>(7),
                                    BLOCK_LOAD_WARP_TRANSPOSE,
                                    MayAlias ? LOAD_CA : LOAD_LDG,
                                    BLOCK_STORE_WARP_TRANSPOSE>;
  };

  using MaxPolicy = Policy350;
};

template <typename InputIteratorT,
          typename OutputIteratorT,
          typename DifferenceOpT,
          typename OffsetT,
          bool MayAlias,
          bool ReadLeft,
          typename SelectedPolicy = DeviceAdjacentDifferencePolicy<InputIteratorT, MayAlias>>
struct DispatchAdjacentDifference : public SelectedPolicy
{
  using InputT = typename std::iterator_traits<InputIteratorT>::value_type;

  void* d_temp_storage;
  std::size_t& temp_storage_bytes;
  InputIteratorT d_input;
  OutputIteratorT d_output;
  OffsetT num_items;
  DifferenceOpT difference_op;
  cudaStream_t stream;

  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE DispatchAdjacentDifference(
    void* d_temp_storage,
    std::size_t& temp_storage_bytes,
    InputIteratorT d_input,
    OutputIteratorT d_output,
    OffsetT num_items,
    DifferenceOpT difference_op,
    cudaStream_t stream)
      : d_temp_storage(d_temp_storage)
      , temp_storage_bytes(temp_storage_bytes)
      , d_input(d_input)
      , d_output(d_output)
      , num_items(num_items)
      , difference_op(difference_op)
      , stream(stream)
  {}

#ifndef DOXYGEN_SHOULD_SKIP_THIS // Do not document
  CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED
  CUB_DEPRECATED CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE DispatchAdjacentDifference(
    void* d_temp_storage,
    std::size_t& temp_storage_bytes,
    InputIteratorT d_input,
    OutputIteratorT d_output,
    OffsetT num_items,
    DifferenceOpT difference_op,
    cudaStream_t stream,
    bool debug_synchronous)
      : d_temp_storage(d_temp_storage)
      , temp_storage_bytes(temp_storage_bytes)
      , d_input(d_input)
      , d_output(d_output)
      , num_items(num_items)
      , difference_op(difference_op)
      , stream(stream)
  {
    CUB_DETAIL_RUNTIME_DEBUG_SYNC_USAGE_LOG
  }
#endif // DOXYGEN_SHOULD_SKIP_THIS

  /// Invocation
  template <typename ActivePolicyT>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t Invoke()
  {
    using AdjacentDifferencePolicyT = typename ActivePolicyT::AdjacentDifferencePolicy;

    using MaxPolicyT = typename DispatchAdjacentDifference::MaxPolicy;

    cudaError error = cudaSuccess;

    do
    {
      constexpr int tile_size = AdjacentDifferencePolicyT::ITEMS_PER_TILE;
      const int num_tiles     = static_cast<int>(::cuda::ceil_div(num_items, tile_size));

      std::size_t first_tile_previous_size = MayAlias * num_tiles * sizeof(InputT);

      void* allocations[1]            = {nullptr};
      std::size_t allocation_sizes[1] = {MayAlias * first_tile_previous_size};

      error = CubDebug(AliasTemporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes));

      if (cudaSuccess != error)
      {
        break;
      }

      if (d_temp_storage == nullptr)
      {
        // Return if the caller is simply requesting the size of the storage
        // allocation

        if (temp_storage_bytes == 0)
        {
          temp_storage_bytes = 1;
        }

        break;
      }

      if (num_items == OffsetT{})
      {
        break;
      }

      auto first_tile_previous = reinterpret_cast<InputT*>(allocations[0]);

      if (MayAlias)
      {
        using AgentDifferenceInitT = AgentDifferenceInit<InputIteratorT, InputT, OffsetT, ReadLeft>;

        constexpr int init_block_size = AgentDifferenceInitT::BLOCK_THREADS;
        const int init_grid_size      = ::cuda::ceil_div(num_tiles, init_block_size);

#ifdef CUB_DETAIL_DEBUG_ENABLE_LOG
        _CubLog("Invoking DeviceAdjacentDifferenceInitKernel"
                "<<<%d, %d, 0, %lld>>>()\n",
                init_grid_size,
                init_block_size,
                reinterpret_cast<long long>(stream));
#endif // CUB_DETAIL_DEBUG_ENABLE_LOG

        THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron(init_grid_size, init_block_size, 0, stream)
          .doit(DeviceAdjacentDifferenceInitKernel<AgentDifferenceInitT, InputIteratorT, InputT, OffsetT>,
                d_input,
                first_tile_previous,
                num_tiles,
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
      }

#ifdef CUB_DETAIL_DEBUG_ENABLE_LOG
      _CubLog("Invoking DeviceAdjacentDifferenceDifferenceKernel"
              "<<<%d, %d, 0, %lld>>>()\n",
              num_tiles,
              AdjacentDifferencePolicyT::BLOCK_THREADS,
              reinterpret_cast<long long>(stream));
#endif // CUB_DETAIL_DEBUG_ENABLE_LOG

      THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron(
        num_tiles, AdjacentDifferencePolicyT::BLOCK_THREADS, 0, stream)
        .doit(DeviceAdjacentDifferenceDifferenceKernel<
                MaxPolicyT,
                InputIteratorT,
                OutputIteratorT,
                DifferenceOpT,
                OffsetT,
                InputT,
                MayAlias,
                ReadLeft>,
              d_input,
              first_tile_previous,
              d_output,
              difference_op,
              num_items);

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
    } while (0);

    return error;
  }

  CUB_RUNTIME_FUNCTION static cudaError_t Dispatch(
    void* d_temp_storage,
    std::size_t& temp_storage_bytes,
    InputIteratorT d_input,
    OutputIteratorT d_output,
    OffsetT num_items,
    DifferenceOpT difference_op,
    cudaStream_t stream)
  {
    using MaxPolicyT = typename DispatchAdjacentDifference::MaxPolicy;

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
      DispatchAdjacentDifference dispatch(
        d_temp_storage, temp_storage_bytes, d_input, d_output, num_items, difference_op, stream);

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
  CUB_RUNTIME_FUNCTION static cudaError_t Dispatch(
    void* d_temp_storage,
    std::size_t& temp_storage_bytes,
    InputIteratorT d_input,
    OutputIteratorT d_output,
    OffsetT num_items,
    DifferenceOpT difference_op,
    cudaStream_t stream,
    bool debug_synchronous)
  {
    CUB_DETAIL_RUNTIME_DEBUG_SYNC_USAGE_LOG

    return Dispatch(d_temp_storage, temp_storage_bytes, d_input, d_output, num_items, difference_op, stream);
  }
#endif // DOXYGEN_SHOULD_SKIP_THIS
};

CUB_NAMESPACE_END
