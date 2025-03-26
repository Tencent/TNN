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

#include <cub/iterator/cache_modified_output_iterator.cuh>
#include <cub/warp/warp_store.cuh>

#include <c2h/catch2_test_helper.cuh>
#include <c2h/fill_striped.cuh>

template <cub::WarpStoreAlgorithm StoreAlgorithm,
          int LOGICAL_WARP_THREADS,
          int ITEMS_PER_THREAD,
          int TOTAL_WARPS,
          typename T,
          typename OutputIteratorT,
          typename ActionT>
__global__ void warp_store_kernel(OutputIteratorT output_iterator, ActionT action)
{
  using warp_store_t = cub::WarpStore<T, ITEMS_PER_THREAD, StoreAlgorithm, LOGICAL_WARP_THREADS>;
  using storage_t    = typename warp_store_t::TempStorage;

  constexpr int tile_size = ITEMS_PER_THREAD * LOGICAL_WARP_THREADS;
  __shared__ storage_t storage[TOTAL_WARPS];

  const int tid = cub::RowMajorTid(blockDim.x, blockDim.y, blockDim.z);
  T reg[ITEMS_PER_THREAD];

  for (int item = 0; item < ITEMS_PER_THREAD; item++)
  {
    reg[item] = static_cast<T>(tid * ITEMS_PER_THREAD + item);
  }

  const int warp_id = tid / LOGICAL_WARP_THREADS;
  warp_store_t store(storage[warp_id]);

  action(store, output_iterator + (warp_id * tile_size), reg);
}

template <cub::WarpStoreAlgorithm StoreAlgorithm,
          int LOGICAL_WARP_THREADS,
          int ITEMS_PER_THREAD,
          int TOTAL_WARPS,
          typename T,
          typename OutputIteratorT,
          typename ActionT>
void warp_store(OutputIteratorT output_iterator, ActionT action)
{
  warp_store_kernel<StoreAlgorithm, LOGICAL_WARP_THREADS, ITEMS_PER_THREAD, TOTAL_WARPS, T, OutputIteratorT, ActionT>
    <<<1, TOTAL_WARPS * LOGICAL_WARP_THREADS>>>(output_iterator, action);
}

struct guarded_store_t
{
  int valid_items;
  template <cub::WarpStoreAlgorithm StoreAlgorithm,
            int LOGICAL_WARP_THREADS,
            int ITEMS_PER_THREAD,
            typename T,
            typename OutputIteratorT>
  __device__ void operator()(cub::WarpStore<T, ITEMS_PER_THREAD, StoreAlgorithm, LOGICAL_WARP_THREADS> store,
                             OutputIteratorT output,
                             T (&reg)[ITEMS_PER_THREAD])
  {
    store.Store(output, reg, valid_items);
  }
};

struct unguarded_store_t
{
  template <cub::WarpStoreAlgorithm StoreAlgorithm,
            int LOGICAL_WARP_THREADS,
            int ITEMS_PER_THREAD,
            typename T,
            typename OutputIteratorT>
  __device__ void operator()(cub::WarpStore<T, ITEMS_PER_THREAD, StoreAlgorithm, LOGICAL_WARP_THREADS> store,
                             OutputIteratorT output,
                             T (&reg)[ITEMS_PER_THREAD])
  {
    store.Store(output, reg);
  }
};

template <cub::WarpStoreAlgorithm StoreAlgorithm,
          int LOGICAL_WARP_THREADS,
          int ITEMS_PER_THREAD,
          int TOTAL_WARPS,
          typename T>
c2h::device_vector<T> compute_reference(int valid_items)
{
  constexpr int tile_size        = LOGICAL_WARP_THREADS * ITEMS_PER_THREAD;
  constexpr int total_item_count = TOTAL_WARPS * tile_size;
  c2h::device_vector<T> d_input(total_item_count);

  _CCCL_IF_CONSTEXPR (StoreAlgorithm == cub::WarpStoreAlgorithm::WARP_STORE_STRIPED)
  {
    c2h::host_vector<T> input(total_item_count);
    fill_striped<ITEMS_PER_THREAD, LOGICAL_WARP_THREADS, ITEMS_PER_THREAD * TOTAL_WARPS>(input.begin());
    d_input = input;
  }
  else
  {
    c2h::gen(c2h::modulo_t{d_input.size()}, d_input);
  }
  if (valid_items != total_item_count)
  {
    for (int warp_id = 0; warp_id < TOTAL_WARPS; warp_id++)
    {
      thrust::fill(c2h::device_policy,
                   d_input.begin() + warp_id * tile_size + valid_items,
                   d_input.begin() + (warp_id + 1) * tile_size,
                   T{});
    }
  }
  return d_input;
}

// %PARAM% LWT lwt 4:16:32
// %PARAM% ALGO_TYPE alg 0:1:2:3

using types                = c2h::type_list<std::uint8_t, std::uint16_t, std::int32_t, std::int64_t>;
using items_per_thread     = c2h::enum_type_list<int, 1, 4, 7>;
using logical_warp_threads = c2h::enum_type_list<int, LWT>;
using algorithms =
  c2h::enum_type_list<cub::WarpStoreAlgorithm,
                      cub::WarpStoreAlgorithm::WARP_STORE_DIRECT,
                      cub::WarpStoreAlgorithm::WARP_STORE_STRIPED,
                      cub::WarpStoreAlgorithm::WARP_STORE_TRANSPOSE,
                      cub::WarpStoreAlgorithm::WARP_STORE_VECTORIZE>;
using algorithm = c2h::enum_type_list<cub::WarpStoreAlgorithm, c2h::get<ALGO_TYPE, algorithms>::value>;

using cache_store_modifier =
  c2h::enum_type_list<cub::CacheStoreModifier,
                      cub::CacheStoreModifier::STORE_DEFAULT,
                      cub::CacheStoreModifier::STORE_WB,
                      cub::CacheStoreModifier::STORE_CG,
                      cub::CacheStoreModifier::STORE_CS,
                      cub::CacheStoreModifier::STORE_WT,
                      cub::CacheStoreModifier::STORE_VOLATILE>;

constexpr int guarded_store_tests_count = 30;

template <int logical_warp_threads>
struct total_warps_t
{
private:
  static constexpr int max_warps      = 2;
  static constexpr bool is_arch_warp  = (logical_warp_threads == CUB_WARP_THREADS(0));
  static constexpr bool is_pow_of_two = ((logical_warp_threads & (logical_warp_threads - 1)) == 0);
  static constexpr int total_warps    = (is_arch_warp || is_pow_of_two) ? max_warps : 1;

public:
  static constexpr int value()
  {
    return total_warps;
  }
};

template <class TestType>
struct params_t
{
  using type = typename c2h::get<0, TestType>;

  static constexpr int logical_warp_threads          = c2h::get<1, TestType>::value;
  static constexpr int items_per_thread              = c2h::get<2, TestType>::value;
  static constexpr cub::WarpStoreAlgorithm algorithm = c2h::get<3, TestType>::value;
  static constexpr int total_warps                   = total_warps_t<logical_warp_threads>::value();
  static constexpr int tile_size                     = logical_warp_threads * items_per_thread;
  static constexpr int total_item_count              = total_warps * tile_size;
};

C2H_TEST("Warp store guarded range works with pointer",
         "[store][warp]",
         types,
         logical_warp_threads,
         items_per_thread,
         algorithm)
{
  using params = params_t<TestType>;
  using type   = typename params::type;

  c2h::device_vector<type> d_out(params::total_item_count, type{});
  const int valid_items = GENERATE_COPY(take(guarded_store_tests_count, random(0, params::tile_size - 1)));
  auto out              = thrust::raw_pointer_cast(d_out.data());
  warp_store<params::algorithm, params::logical_warp_threads, params::items_per_thread, params::total_warps, type>(
    out, guarded_store_t{valid_items});
  auto d_expected_output =
    compute_reference<params::algorithm, params::logical_warp_threads, params::items_per_thread, params::total_warps, type>(
      valid_items);
  REQUIRE(d_expected_output == d_out);
}

C2H_TEST("Warp store guarded range works with cache modified iterator",
         "[store][warp]",
         types,
         logical_warp_threads,
         items_per_thread,
         algorithm,
         cache_store_modifier)
{
  using params                                     = params_t<TestType>;
  using type                                       = typename params::type;
  constexpr cub::CacheStoreModifier store_modifier = c2h::get<4, TestType>::value;

  c2h::device_vector<type> d_out(params::total_item_count, type{});
  const int valid_items = GENERATE_COPY(take(guarded_store_tests_count, random(0, params::tile_size - 1)));
  auto out = cub::CacheModifiedOutputIterator<store_modifier, type>(thrust::raw_pointer_cast(d_out.data()));
  warp_store<params::algorithm, params::logical_warp_threads, params::items_per_thread, params::total_warps, type>(
    out, guarded_store_t{valid_items});
  auto d_expected_output =
    compute_reference<params::algorithm, params::logical_warp_threads, params::items_per_thread, params::total_warps, type>(
      valid_items);
  REQUIRE(d_expected_output == d_out);
}

C2H_TEST("Warp store unguarded range works with pointer",
         "[store][warp]",
         types,
         logical_warp_threads,
         items_per_thread,
         algorithm)
{
  using params = params_t<TestType>;
  using type   = typename params::type;

  c2h::device_vector<type> d_out(params::total_item_count, type{});
  constexpr int valid_items = params::tile_size;
  auto out                  = thrust::raw_pointer_cast(d_out.data());
  warp_store<params::algorithm, params::logical_warp_threads, params::items_per_thread, params::total_warps, type>(
    out, unguarded_store_t{});
  auto d_expected_output =
    compute_reference<params::algorithm, params::logical_warp_threads, params::items_per_thread, params::total_warps, type>(
      valid_items);
  REQUIRE(d_expected_output == d_out);
}

C2H_TEST("Warp store unguarded range works with cache modified iterator",
         "[store][warp]",
         types,
         logical_warp_threads,
         items_per_thread,
         algorithm,
         cache_store_modifier)
{
  using params                                     = params_t<TestType>;
  using type                                       = typename params::type;
  constexpr cub::CacheStoreModifier store_modifier = c2h::get<4, TestType>::value;

  c2h::device_vector<type> d_out(params::total_item_count, type{});
  constexpr int valid_items = params::tile_size;
  auto out = cub::CacheModifiedOutputIterator<store_modifier, type>(thrust::raw_pointer_cast(d_out.data()));
  warp_store<params::algorithm, params::logical_warp_threads, params::items_per_thread, params::total_warps, type>(
    out, unguarded_store_t{});
  auto d_expected_output =
    compute_reference<params::algorithm, params::logical_warp_threads, params::items_per_thread, params::total_warps, type>(
      valid_items);
  REQUIRE(d_expected_output == d_out);
}
