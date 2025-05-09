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

#include <cub/iterator/cache_modified_input_iterator.cuh>
#include <cub/warp/warp_load.cuh>

#include <thrust/sequence.h>

#include <c2h/catch2_test_helper.cuh>
#include <c2h/fill_striped.cuh>

template <cub::WarpLoadAlgorithm LoadAlgorithm,
          int LOGICAL_WARP_THREADS,
          int ITEMS_PER_THREAD,
          int TOTAL_WARPS,
          typename T,
          typename InputIteratorT,
          typename ActionT>
__global__ void warp_load_kernel(InputIteratorT input_iterator, ActionT action, int* error_counter)
{
  using warp_load_t = cub::WarpLoad<T, ITEMS_PER_THREAD, LoadAlgorithm, LOGICAL_WARP_THREADS>;
  using storage_t   = typename warp_load_t::TempStorage;

  constexpr int tile_size = ITEMS_PER_THREAD * LOGICAL_WARP_THREADS;

  __shared__ storage_t storage[TOTAL_WARPS];

  const int linear_tid = threadIdx.x;

  const int warp_id = linear_tid / LOGICAL_WARP_THREADS;
  warp_load_t load(storage[warp_id]);

  // Test WarpLoad specialization
  T reg[ITEMS_PER_THREAD];
  action.load(load, input_iterator + (warp_id * tile_size), reg);

  // Verify data was loaded as expected
  action.verify(reg, error_counter);
}

template <cub::WarpLoadAlgorithm LoadAlgorithm,
          int LOGICAL_WARP_THREADS,
          int ITEMS_PER_THREAD,
          int TOTAL_WARPS,
          typename T,
          typename InputIteratorT,
          typename ActionT>
void warp_load(InputIteratorT input_iterator, ActionT action, int* error_counter)
{
  warp_load_kernel<LoadAlgorithm, LOGICAL_WARP_THREADS, ITEMS_PER_THREAD, TOTAL_WARPS, T, InputIteratorT, ActionT>
    <<<1, TOTAL_WARPS * LOGICAL_WARP_THREADS>>>(input_iterator, action, error_counter);
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
}

/**
 * @brief WarpLoad test specialisation for guarded loads
 */
template <cub::WarpLoadAlgorithm LoadAlgorithm, int LOGICAL_WARP_THREADS, typename T>
struct guarded_load_t
{
  int valid_items;
  T oob_default;

  template <int ITEMS_PER_THREAD, typename InputIteratorT>
  __device__ void load(cub::WarpLoad<T, ITEMS_PER_THREAD, LoadAlgorithm, LOGICAL_WARP_THREADS> load,
                       InputIteratorT input,
                       T (&reg)[ITEMS_PER_THREAD])
  {
    load.Load(input, reg, valid_items, oob_default);
  }

  template <int ITEMS_PER_THREAD>
  __device__ void verify(T (&reg)[ITEMS_PER_THREAD], int* error_counter)
  {
    const auto linear_tid = cub::RowMajorTid(blockDim.x, blockDim.y, blockDim.z);
    const auto lane_id    = linear_tid % LOGICAL_WARP_THREADS;
    for (int item = 0; item < ITEMS_PER_THREAD; item++)
    {
      const auto expected_value = static_cast<T>(linear_tid * ITEMS_PER_THREAD + item);

      const bool is_oob = LoadAlgorithm == cub::WarpLoadAlgorithm::WARP_LOAD_STRIPED
                          ? item * LOGICAL_WARP_THREADS + lane_id >= valid_items
                          : lane_id * ITEMS_PER_THREAD + item >= valid_items;

      if (is_oob)
      {
        if (reg[item] != oob_default)
        {
          atomicAdd(error_counter, 1);
        }
      }
      else if (reg[item] != expected_value)
      {
        atomicAdd(error_counter, 1);
      }
    }
  }
};

/**
 * @brief WarpLoad test specialisation for unguarded loads
 */
struct unguarded_load_t
{
  template <cub::WarpLoadAlgorithm LoadAlgorithm,
            int LOGICAL_WARP_THREADS,
            int ITEMS_PER_THREAD,
            typename T,
            typename InputIteratorT>
  __device__ void load(cub::WarpLoad<T, ITEMS_PER_THREAD, LoadAlgorithm, LOGICAL_WARP_THREADS> load,
                       InputIteratorT input,
                       T (&reg)[ITEMS_PER_THREAD])
  {
    load.Load(input, reg);
  }

  template <typename T, int ITEMS_PER_THREAD>
  __device__ void verify(T (&reg)[ITEMS_PER_THREAD], int* error_counter)
  {
    for (int item = 0; item < ITEMS_PER_THREAD; item++)
    {
      const auto expected_value = static_cast<T>(threadIdx.x * ITEMS_PER_THREAD + item);

      if (reg[item] != expected_value)
      {
        atomicAdd(error_counter, 1);
      }
    }
  }
};

template <cub::WarpLoadAlgorithm LoadAlgorithm, int LOGICAL_WARP_THREADS, int ITEMS_PER_THREAD, int TOTAL_WARPS, typename T>
c2h::device_vector<T> generate_input()
{
  constexpr int tile_size = LOGICAL_WARP_THREADS * ITEMS_PER_THREAD;
  constexpr int num_items = TOTAL_WARPS * tile_size;

  c2h::device_vector<T> d_input(num_items);

  _CCCL_IF_CONSTEXPR (LoadAlgorithm == cub::WarpLoadAlgorithm::WARP_LOAD_STRIPED)
  {
    c2h::host_vector<T> h_input(num_items);

    // In this case we need different stripe pattern, so the
    // items/threads parameters are swapped

    constexpr int FAKE_BLOCK_SIZE = ITEMS_PER_THREAD * TOTAL_WARPS;

    fill_striped<ITEMS_PER_THREAD, LOGICAL_WARP_THREADS, FAKE_BLOCK_SIZE>(h_input.begin());
    d_input = h_input;
  }
  else
  {
    c2h::gen(c2h::modulo_t{num_items}, d_input);
  }

  return d_input;
}

// %PARAM% LWT lwt 4:16:32
// %PARAM% ALGO_TYPE alg 0:1:2:3

using types                = c2h::type_list<std::uint8_t, std::uint16_t, std::int32_t, std::int64_t>;
using items_per_thread     = c2h::enum_type_list<int, 1, 4, 7>;
using logical_warp_threads = c2h::enum_type_list<int, LWT>;
using algorithms =
  c2h::enum_type_list<cub::WarpLoadAlgorithm,
                      cub::WarpLoadAlgorithm::WARP_LOAD_DIRECT,
                      cub::WarpLoadAlgorithm::WARP_LOAD_STRIPED,
                      cub::WarpLoadAlgorithm::WARP_LOAD_TRANSPOSE,
                      cub::WarpLoadAlgorithm::WARP_LOAD_VECTORIZE>;
using algorithm = c2h::enum_type_list<cub::WarpLoadAlgorithm, c2h::get<ALGO_TYPE, algorithms>::value>;

using cache_load_modifier =
  c2h::enum_type_list<cub::CacheLoadModifier,
                      cub::CacheLoadModifier::LOAD_DEFAULT,
                      cub::CacheLoadModifier::LOAD_CA,
                      cub::CacheLoadModifier::LOAD_CG,
                      cub::CacheLoadModifier::LOAD_CS,
                      cub::CacheLoadModifier::LOAD_CV,
                      cub::CacheLoadModifier::LOAD_LDG,
                      cub::CacheLoadModifier::LOAD_VOLATILE>;

constexpr int guarded_load_tests_count = 30;

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

  static constexpr int logical_warp_threads         = c2h::get<1, TestType>::value;
  static constexpr int items_per_thread             = c2h::get<2, TestType>::value;
  static constexpr cub::WarpLoadAlgorithm algorithm = c2h::get<3, TestType>::value;
  static constexpr int total_warps                  = total_warps_t<logical_warp_threads>::value();
  static constexpr int tile_size                    = logical_warp_threads * items_per_thread;
  static constexpr int total_item_count             = total_warps * tile_size;
};

C2H_TEST(
  "Warp load guarded range works with pointer", "[load][warp]", types, logical_warp_threads, items_per_thread, algorithm)
{
  using params     = params_t<TestType>;
  using type       = typename params::type;
  using delegate_t = guarded_load_t<params::algorithm, params::logical_warp_threads, type>;

  const int valid_items  = GENERATE_COPY(take(guarded_load_tests_count, random(0, params::tile_size - 1)));
  const auto oob_default = static_cast<type>(valid_items);

  auto d_in =
    generate_input<params::algorithm, params::logical_warp_threads, params::items_per_thread, params::total_warps, type>();
  c2h::device_vector<int> d_error_counter(1, 0);

  warp_load<params::algorithm, params::logical_warp_threads, params::items_per_thread, params::total_warps, type>(
    thrust::raw_pointer_cast(d_in.data()),
    delegate_t{valid_items, oob_default},
    thrust::raw_pointer_cast(d_error_counter.data()));

  const int num_errors               = d_error_counter[0];
  constexpr int expected_error_count = 0;
  REQUIRE(num_errors == expected_error_count);
}

C2H_TEST("Warp load guarded range works with cache modified iterator",
         "[load][warp]",
         types,
         logical_warp_threads,
         items_per_thread,
         algorithm,
         cache_load_modifier)
{
  using params     = params_t<TestType>;
  using type       = typename params::type;
  using delegate_t = guarded_load_t<params::algorithm, params::logical_warp_threads, type>;
  constexpr cub::CacheLoadModifier load_modifier = c2h::get<4, TestType>::value;

  const int valid_items  = GENERATE_COPY(take(guarded_load_tests_count, random(0, params::tile_size - 1)));
  const auto oob_default = static_cast<type>(valid_items);

  auto d_in =
    generate_input<params::algorithm, params::logical_warp_threads, params::items_per_thread, params::total_warps, type>();
  auto in_it = cub::CacheModifiedInputIterator<load_modifier, type>(thrust::raw_pointer_cast(d_in.data()));
  c2h::device_vector<int> d_error_counter(1, 0);

  warp_load<params::algorithm, params::logical_warp_threads, params::items_per_thread, params::total_warps, type>(
    in_it, delegate_t{valid_items, oob_default}, thrust::raw_pointer_cast(d_error_counter.data()));

  const auto num_errors              = d_error_counter[0];
  constexpr int expected_error_count = 0;
  REQUIRE(num_errors == expected_error_count);
}

C2H_TEST("Warp load unguarded range works with pointer",
         "[load][warp]",
         types,
         logical_warp_threads,
         items_per_thread,
         algorithm)
{
  using params     = params_t<TestType>;
  using type       = typename params::type;
  using delegate_t = unguarded_load_t;

  auto d_in =
    generate_input<params::algorithm, params::logical_warp_threads, params::items_per_thread, params::total_warps, type>();
  c2h::device_vector<int> d_error_counter(1, 0);

  warp_load<params::algorithm, params::logical_warp_threads, params::items_per_thread, params::total_warps, type>(
    thrust::raw_pointer_cast(d_in.data()), delegate_t{}, thrust::raw_pointer_cast(d_error_counter.data()));

  const auto num_errors              = d_error_counter[0];
  constexpr int expected_error_count = 0;
  REQUIRE(num_errors == expected_error_count);
}

C2H_TEST("Warp load unguarded range works with cache modified iterator",
         "[load][warp]",
         types,
         logical_warp_threads,
         items_per_thread,
         algorithm,
         cache_load_modifier)
{
  using params                                   = params_t<TestType>;
  using type                                     = typename params::type;
  using delegate_t                               = unguarded_load_t;
  constexpr cub::CacheLoadModifier load_modifier = c2h::get<4, TestType>::value;

  auto d_in =
    generate_input<params::algorithm, params::logical_warp_threads, params::items_per_thread, params::total_warps, type>();
  auto in_it = cub::CacheModifiedInputIterator<load_modifier, type>(thrust::raw_pointer_cast(d_in.data()));
  c2h::device_vector<int> d_error_counter(1, 0);

  warp_load<params::algorithm, params::logical_warp_threads, params::items_per_thread, params::total_warps, type>(
    in_it, delegate_t{}, thrust::raw_pointer_cast(d_error_counter.data()));

  const auto num_errors              = d_error_counter[0];
  constexpr int expected_error_count = 0;
  REQUIRE(num_errors == expected_error_count);
}
