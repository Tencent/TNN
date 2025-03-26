/******************************************************************************
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

#include <cub/config.cuh>

#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/util_device.cuh>
#include <cub/util_type.cuh>
#include <cub/util_vsmem.cuh>

#include "catch2/catch.hpp"
#include "catch2_test_launch_helper.h"
#include <c2h/catch2_test_helper.cuh>

//----------------------------------------------------------------------------
// Helper section
//----------------------------------------------------------------------------
template <int Size>
struct large_custom_t
{
  uint8_t data[Size];
};

struct kernel_test_info_t
{
  // Whether the kernel is using virtual shared memory
  bool uses_vsmem_ptr;
  // Whether the kernel is using the fallback agent
  bool uses_fallback_agent;
  // Whether the kernel is using the fallback policy
  bool uses_fallback_policy;
};

struct launch_config_test_info_t
{
  // The tile size that's assumed during launch configuration
  std::size_t config_assumes_tile_size;
  // The block size that's assumed during launch configuration
  std::size_t config_assumes_block_threads;
  // The total amount of virtual shared memory that has to be allocated
  std::size_t config_vsmem_per_block;
};

// CUB_NAMESPACE_BEGIN

//----------------------------------------------------------------------------
// Tuning policy definition
//----------------------------------------------------------------------------
template <int BlockThreads, int ItemsPerThread>
struct agent_dummy_algorithm_policy_t
{
  static constexpr int ITEMS_PER_THREAD = ItemsPerThread;
  static constexpr int BLOCK_THREADS    = BlockThreads;
};

//----------------------------------------------------------------------------
// Agent template definition
//----------------------------------------------------------------------------
template <typename ActivePolicyT, typename InputIteratorT, typename OutputIteratorT, typename OffsetT>
struct agent_dummy_algorithm_t
{
  static constexpr auto block_threads    = ActivePolicyT::BLOCK_THREADS;
  static constexpr auto items_per_thread = ActivePolicyT::ITEMS_PER_THREAD;
  static constexpr auto tile_size        = block_threads * items_per_thread;

  using item_t = cub::detail::value_t<InputIteratorT>;

  using block_load_t = cub::BlockLoad<item_t, block_threads, items_per_thread, cub::BLOCK_LOAD_TRANSPOSE>;

  using block_store_t = cub::BlockStore<item_t, block_threads, items_per_thread, cub::BLOCK_STORE_TRANSPOSE>;

  // We are intentionally not aliasing the TempStorage here to double the required shared memory of the test and be able
  // to use a smaller `large_custom_t`, as we experienced slow compilation times for large a `large_custom_t`.
  struct _temp_storage_t
  {
    typename block_load_t::TempStorage load;
    typename block_store_t::TempStorage store;
  };

  struct TempStorage : cub::Uninitialized<_temp_storage_t>
  {};

  _temp_storage_t& temp_storage; ///< Reference to temp_storage
  InputIteratorT d_in; ///< Input data
  OutputIteratorT d_out; ///< Output data

  __device__ __forceinline__
  agent_dummy_algorithm_t(TempStorage& temp_storage, InputIteratorT d_in, OutputIteratorT d_out)
      : temp_storage(temp_storage.Alias())
      , d_in(d_in)
      , d_out(d_out)
  {}

  __device__ __forceinline__ void consume_tile(int tile_idx, OffsetT total_num_items)
  {
    const OffsetT tile_offset   = static_cast<OffsetT>(tile_idx) * static_cast<OffsetT>(tile_size);
    const OffsetT num_remaining = total_num_items - tile_offset;

    // Load items into a blocked arrangement
    item_t items[items_per_thread]{};
    block_load_t(temp_storage.load).Load(d_in + tile_offset, items, num_remaining);

    // Store items from blocked arrangement
    block_store_t(temp_storage.store).Store(d_out + tile_offset, items, num_remaining);
  }
};

//----------------------------------------------------------------------------
// Kernel template definition
//----------------------------------------------------------------------------
template <typename ChainedPolicyT, typename InputIteratorT, typename OutputIteratorT, typename OffsetT>
void __global__ __launch_bounds__(
  cub::detail::vsmem_helper_fallback_policy_t<
    typename ChainedPolicyT::ActivePolicy::DummyAlgorithmPolicy,
    typename ChainedPolicyT::ActivePolicy::FallbackDummyAlgorithmPolicy,
    agent_dummy_algorithm_t,
    InputIteratorT,
    OutputIteratorT,
    OffsetT>::agent_policy_t::BLOCK_THREADS)
  dummy_algorithm_kernel(
    InputIteratorT d_in,
    OutputIteratorT d_out,
    OffsetT num_items,
    kernel_test_info_t* kernel_test_info,
    cub::detail::vsmem_t vsmem)
{
  using active_policy_t   = typename ChainedPolicyT::ActivePolicy;
  using default_policy_t  = typename active_policy_t::DummyAlgorithmPolicy;
  using fallback_policy_t = typename active_policy_t::FallbackDummyAlgorithmPolicy;
  using fallback_agent_t  = agent_dummy_algorithm_t<fallback_policy_t, InputIteratorT, OutputIteratorT, OffsetT>;

  using vsmem_helper_t = cub::detail::vsmem_helper_fallback_policy_t<
    default_policy_t,
    fallback_policy_t,
    agent_dummy_algorithm_t,
    InputIteratorT,
    OutputIteratorT,
    OffsetT>;

  using agent_t = typename vsmem_helper_t::agent_t;

  // Static shared memory allocation
  __shared__ typename vsmem_helper_t::static_temp_storage_t static_temp_storage;

  // Get temporary storage
  typename agent_t::TempStorage& temp_storage = vsmem_helper_t::get_temp_storage(static_temp_storage, vsmem);

  // Populate test meta data
  kernel_test_info->uses_vsmem_ptr =
    (reinterpret_cast<char*>(&temp_storage)
     == (static_cast<char*>(vsmem.gmem_ptr) + (blockIdx.x * vsmem_helper_t::vsmem_per_block)));
  kernel_test_info->uses_fallback_agent =
    ::cuda::std::is_same<typename vsmem_helper_t::agent_t, fallback_agent_t>::value;
  kernel_test_info->uses_fallback_policy =
    ::cuda::std::is_same<typename vsmem_helper_t::agent_policy_t, fallback_policy_t>::value;

  // Instantiate the algorithm's agent
  agent_t agent(temp_storage, d_in, d_out);

  // Process this thread block's tile
  agent.consume_tile(blockIdx.x, num_items);

  // If applicable, hints to discard modified cache lines for vsmem
  vsmem_helper_t::discard_temp_storage(temp_storage);
}

//----------------------------------------------------------------------------
// Tuning policy chain
//----------------------------------------------------------------------------
template <typename InputIteratorT>
struct device_dummy_algorithm_policy_t
{
  using item_t = cub::detail::value_t<InputIteratorT>;

  static constexpr int FALLBACK_BLOCK_THREADS = 64;

  struct policy_350 : cub::ChainedPolicy<350, policy_350, policy_350>
  {
    using DummyAlgorithmPolicy = agent_dummy_algorithm_policy_t<256, cub::Nominal4BItemsToItems<item_t>(17)>;

    // The fallback policy that's used if there's insufficient shared memory for the default policy,
    // yet still sufficient memory for the fallback policy
    using FallbackDummyAlgorithmPolicy = cub::detail::policy_wrapper_t<DummyAlgorithmPolicy, FALLBACK_BLOCK_THREADS>;
  };

  /// MaxPolicy
  using max_policy_t = policy_350;
};

//----------------------------------------------------------------------------
// Dispatch layer
//----------------------------------------------------------------------------
template <typename InputIteratorT,
          typename OutputIteratorT,
          typename OffsetT,
          typename SelectedPolicy = device_dummy_algorithm_policy_t<InputIteratorT>>
struct dispatch_dummy_algorithm_t : SelectedPolicy
{
  using item_t = cub::detail::value_t<InputIteratorT>;

  /// Device-accessible allocation of temporary storage. When nullptr, the required
  /// allocation size is written to \p temp_storage_bytes and no work is done.
  void* d_temp_storage;

  /// Reference to size in bytes of \p d_temp_storage allocation
  std::size_t& temp_storage_bytes;

  InputIteratorT d_in;
  OutputIteratorT d_out;
  OffsetT num_items;
  kernel_test_info_t* kernel_test_info;
  launch_config_test_info_t* launch_config_info;
  cudaStream_t stream;
  int ptx_version;

  CUB_RUNTIME_FUNCTION __forceinline__ dispatch_dummy_algorithm_t(
    void* d_temp_storage,
    std::size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OutputIteratorT d_out,
    OffsetT num_items,
    kernel_test_info_t* kernel_test_info,
    launch_config_test_info_t* launch_config_info,
    cudaStream_t stream,
    int ptx_version)
      : d_temp_storage(d_temp_storage)
      , temp_storage_bytes(temp_storage_bytes)
      , d_in(d_in)
      , d_out(d_out)
      , num_items(num_items)
      , kernel_test_info(kernel_test_info)
      , launch_config_info(launch_config_info)
      , stream(stream)
      , ptx_version(ptx_version)
  {}

  /**
   * @brief During compilation, CUB's dispatch mechanism (more specifically, `cub::ChainedPolicy`)
   * instantiates the `Invoke` function template for *all* tunings policies that are defined in the
   * algorithm's chain of tuning policies. At runtime, when an algorithm is invoked, of all the
   * instantiated `Invoke`-function templates, `cub::ChainedPolicy` makes sure to *call* only the
   * first tuning policy whose PTX version is less-than-or-equal to the GPU's SM version to which
   * the algorithm is dispatched.
   * Since the `Invoke`function template is instantiated for *all* tunings policies in the chain, we
   * want to avoid making any of the algorithm's kernel template parameters depend on the
   * `ActivePolicyT` template argument of the `Invoke` function template, as that would result in
   * multiple kernel template instances.
   */
  template <typename ActivePolicyT>
  CUB_RUNTIME_FUNCTION __forceinline__ cudaError_t Invoke()
  {
    using max_policy_t = typename dispatch_dummy_algorithm_t::max_policy_t;

    using vsmem_helper_t = cub::detail::vsmem_helper_fallback_policy_t<
      typename ActivePolicyT::DummyAlgorithmPolicy,
      typename ActivePolicyT::FallbackDummyAlgorithmPolicy,
      agent_dummy_algorithm_t,
      InputIteratorT,
      OutputIteratorT,
      OffsetT>;

    // Empty problem size
    if (num_items == 0)
    {
      if (d_temp_storage)
      {
        return cudaSuccess;
      }
      else
      {
        temp_storage_bytes = 0;
        return cudaSuccess;
      }
    }

    // Compute launch configurations
    constexpr auto block_threads    = vsmem_helper_t::agent_policy_t::BLOCK_THREADS;
    constexpr auto items_per_thread = vsmem_helper_t::agent_policy_t::ITEMS_PER_THREAD;
    constexpr auto tile_size        = block_threads * items_per_thread;
    const auto num_tiles            = ::cuda::ceil_div(num_items, tile_size);
    const auto total_vsmem          = num_tiles * vsmem_helper_t::vsmem_per_block;

    // Get device ordinal
    cudaError error = cudaSuccess;

    // Compute temporary storage requirements
    void* allocations[1]            = {nullptr};
    std::size_t allocation_sizes[1] = {total_vsmem};
    error = cub::AliasTemporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes);
    if (cudaSuccess != error)
    {
      return error;
    }

    // Return if the caller is simply requesting the size of the storage allocation
    if (d_temp_storage == nullptr)
    {
      return error;
    }
    launch_config_info->config_assumes_tile_size     = static_cast<std::size_t>(tile_size);
    launch_config_info->config_assumes_block_threads = static_cast<std::size_t>(block_threads);
    launch_config_info->config_vsmem_per_block       = vsmem_helper_t::vsmem_per_block;

    THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron(num_tiles, block_threads, 0, stream)
      .doit(dummy_algorithm_kernel<max_policy_t, InputIteratorT, OutputIteratorT, OffsetT>,
            d_in,
            d_out,
            num_items,
            kernel_test_info,
            cub::detail::vsmem_t{allocations[0]});
    return cudaPeekAtLastError();
  }

  /**
   * @brief Static member function as the entry point for algorithm dispatch
   */
  CUB_RUNTIME_FUNCTION static cudaError_t dispatch(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OutputIteratorT d_out,
    OffsetT num_items,
    kernel_test_info_t* kernel_test_info,
    launch_config_test_info_t* launch_config_info,
    cudaStream_t stream = 0)
  {
    using max_policy_t = typename dispatch_dummy_algorithm_t::max_policy_t;

    // Get PTX version
    int ptx_version = 0;
    cudaError error = cub::PtxVersion(ptx_version);
    if (cudaSuccess != error)
    {
      return error;
    }

    // Create dispatch functor
    dispatch_dummy_algorithm_t dispatch(
      d_temp_storage,
      temp_storage_bytes,
      d_in,
      d_out,
      num_items,
      kernel_test_info,
      launch_config_info,
      stream,
      ptx_version);

    // Dispatch to chained policy
    error = max_policy_t::Invoke(ptx_version, dispatch);
    if (cudaSuccess != error)
    {
      return error;
    }

    return error;
  }
};

//----------------------------------------------------------------------------
// Device-scope interface layer
//----------------------------------------------------------------------------
template <typename InputIteratorT, typename OutputIteratorT, typename OffsetT>
CUB_RUNTIME_FUNCTION static cudaError_t device_dummy_algorithm(
  void* d_temp_storage,
  size_t& temp_storage_bytes,
  InputIteratorT d_in,
  OutputIteratorT d_out,
  OffsetT num_items,
  kernel_test_info_t* kernel_test_info,
  launch_config_test_info_t* launch_config_info,
  cudaStream_t stream = 0)
{
  using dispatch_dummy_algorithm_t = dispatch_dummy_algorithm_t<InputIteratorT, OutputIteratorT, OffsetT>;
  return dispatch_dummy_algorithm_t::dispatch(
    d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, kernel_test_info, launch_config_info, stream);
}

// CUB_NAMESPACE_END

DECLARE_LAUNCH_WRAPPER(device_dummy_algorithm, dummy_algorithm);

// %PARAM% TEST_LAUNCH lid 0:1:2

using type_list = c2h::type_list<large_custom_t<1>, large_custom_t<80>, large_custom_t<128>, large_custom_t<512>>;

C2H_TEST("Virtual shared memory works within algorithms", "[util][vsmem]", type_list)
{
  using item_t   = typename c2h::get<0, TestType>;
  using offset_t = int32_t;

  constexpr offset_t target_size = 10000000;
  const offset_t num_items       = target_size / sizeof(item_t);

  // Prepare input and output buffers for a simple copy algorithm test
  c2h::device_vector<uint8_t> in(num_items * sizeof(item_t));
  c2h::device_vector<uint8_t> out(num_items * sizeof(item_t));
  auto const in_ptr  = reinterpret_cast<item_t*>(thrust::raw_pointer_cast(in.data()));
  auto const out_ptr = reinterpret_cast<item_t*>(thrust::raw_pointer_cast(out.data()));

  // Generate some random noise input data
  c2h::gen(C2H_SEED(1), in);

  // Query default and fallback policies and agents so we can confirm vsmem
  using default_policy_t  = typename device_dummy_algorithm_policy_t<item_t*>::policy_350::DummyAlgorithmPolicy;
  using default_agent_t   = agent_dummy_algorithm_t<default_policy_t, item_t*, item_t*, offset_t>;
  using fallback_policy_t = typename device_dummy_algorithm_policy_t<item_t*>::policy_350::FallbackDummyAlgorithmPolicy;
  using fallback_agent_t  = agent_dummy_algorithm_t<fallback_policy_t, item_t*, item_t*, offset_t>;

  // Get the information as it is expected from the vsmem helper to work as epxected
  std::size_t default_smem_size  = sizeof(typename default_agent_t::TempStorage);
  std::size_t fallback_smem_size = sizeof(typename fallback_agent_t::TempStorage);
  bool expected_to_use_fallback =
    default_smem_size > cub::detail::max_smem_per_block && fallback_smem_size <= cub::detail::max_smem_per_block;
  std::size_t expected_smem_per_block = expected_to_use_fallback ? fallback_smem_size : default_smem_size;
  bool expected_needs_vsmem           = expected_smem_per_block > cub::detail::max_smem_per_block;
  std::size_t expected_block_threads =
    expected_to_use_fallback ? fallback_policy_t::BLOCK_THREADS : default_policy_t::BLOCK_THREADS;
  std::size_t expected_items_per_thread =
    expected_to_use_fallback ? fallback_policy_t::ITEMS_PER_THREAD : default_policy_t::ITEMS_PER_THREAD;
  std::size_t expected_tile_size       = expected_block_threads * expected_items_per_thread;
  std::size_t expected_vsmem_per_block = (expected_needs_vsmem ? expected_smem_per_block : 0ULL);

  // Setup vsmem test
  launch_config_test_info_t* launch_config_info = nullptr;
  cudaMallocHost(&launch_config_info, sizeof(launch_config_test_info_t));
  c2h::device_vector<kernel_test_info_t> device_kernel_test_info(1);
  dummy_algorithm(
    in_ptr, out_ptr, num_items, thrust::raw_pointer_cast(device_kernel_test_info.data()), launch_config_info);

  // Make sure the algorithm worked correctly
  REQUIRE(in == out);

  // Make sure the kernel information retrieved from the vsmem helper is correct
  c2h::host_vector<kernel_test_info_t> kernel_test_info = device_kernel_test_info;
  REQUIRE(kernel_test_info[0].uses_vsmem_ptr == expected_needs_vsmem);
  REQUIRE(kernel_test_info[0].uses_fallback_agent == expected_to_use_fallback);
  REQUIRE(kernel_test_info[0].uses_fallback_policy == expected_to_use_fallback);

  // Make sure the launch configuration information retrieved from the vsmem helper is correct
  REQUIRE(launch_config_info->config_assumes_tile_size == expected_tile_size);
  REQUIRE(launch_config_info->config_assumes_block_threads == expected_block_threads);
  if (expected_vsmem_per_block == 0)
  {
    REQUIRE(launch_config_info->config_vsmem_per_block == 0);
  }
  else
  {
    // The virtual shared memory helper pads vsmem to a multiple of a line size, hence the range check
    REQUIRE(launch_config_info->config_vsmem_per_block >= expected_vsmem_per_block);
  }

  cudaFreeHost(launch_config_info);
}
