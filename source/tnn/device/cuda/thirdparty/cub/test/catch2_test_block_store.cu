/******************************************************************************
 * Copyright (c) 2011-2022, NVIDIA CORPORATION.  All rights reserved.
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

#include <cub/block/block_store.cuh>
#include <cub/iterator/cache_modified_output_iterator.cuh>
#include <cub/iterator/discard_output_iterator.cuh>
#include <cub/util_allocator.cuh>
#include <cub/util_arch.cuh>

#include <c2h/catch2_test_helper.cuh>

template <int ItemsPerThread, int ThreadsInBlock, cub::BlockStoreAlgorithm /* StoreAlgorithm */>
struct output_idx
{
  static __device__ int get(int item)
  {
    return static_cast<int>(threadIdx.x) * ItemsPerThread + item;
  }
};

template <int ItemsPerThread, int ThreadsInBlock>
struct output_idx<ItemsPerThread, ThreadsInBlock, cub::BlockStoreAlgorithm::BLOCK_STORE_STRIPED>
{
  static __device__ int get(int item)
  {
    return static_cast<int>(threadIdx.x) + ThreadsInBlock * item;
  }
};

template <typename InputIteratorT,
          typename OutputIteratorT,
          int ItemsPerThread,
          int ThreadsInBlock,
          cub::BlockStoreAlgorithm StoreAlgorithm>
__global__ void kernel(std::integral_constant<bool, true>, InputIteratorT input, OutputIteratorT output, int num_items)
{
  using input_t       = cub::detail::value_t<InputIteratorT>;
  using block_store_t = cub::BlockStore<input_t, ThreadsInBlock, ItemsPerThread, StoreAlgorithm>;
  using storage_t     = typename block_store_t::TempStorage;

  __shared__ storage_t storage;
  block_store_t block_store(storage);

  input_t data[ItemsPerThread];

  for (int i = 0; i < ItemsPerThread; i++)
  {
    const int idx = output_idx<ItemsPerThread, ThreadsInBlock, StoreAlgorithm>::get(i);

    if (idx < num_items)
    {
      data[i] = input[idx];
    }
  }

  if (ItemsPerThread * ThreadsInBlock == num_items)
  {
    block_store.Store(output, data);
  }
  else
  {
    block_store.Store(output, data, num_items);
  }
}

template <typename InputIteratorT,
          typename OutputIteratorT,
          int ItemsPerThread,
          int ThreadsInBlock,
          cub::BlockStoreAlgorithm /* StoreAlgorithm */>
__global__ void kernel(std::integral_constant<bool, false>, InputIteratorT input, OutputIteratorT output, int num_items)
{
  for (int i = 0; i < ItemsPerThread; i++)
  {
    const int idx = output_idx<ItemsPerThread, ThreadsInBlock, cub::BlockStoreAlgorithm::BLOCK_STORE_DIRECT>::get(i);

    if (idx < num_items)
    {
      output[idx] = input[idx];
    }
  }
}

template <int ItemsPerThread,
          int ThreadsInBlock,
          cub::BlockStoreAlgorithm StoreAlgorithm,
          typename InputIteratorT,
          typename OutputIteratorT>
void block_store(InputIteratorT input, OutputIteratorT output, int num_items)
{
  using input_t                       = cub::detail::value_t<InputIteratorT>;
  using block_store_t                 = cub::BlockStore<input_t, ThreadsInBlock, ItemsPerThread, StoreAlgorithm>;
  using storage_t                     = typename block_store_t::TempStorage;
  constexpr bool sufficient_resources = sizeof(storage_t) <= cub::detail::max_smem_per_block;

  kernel<InputIteratorT, OutputIteratorT, ItemsPerThread, ThreadsInBlock, StoreAlgorithm>
    <<<1, ThreadsInBlock>>>(std::integral_constant<bool, sufficient_resources>{}, input, output, num_items);

  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
}

// %PARAM% IPT it 1:11

using types     = c2h::type_list<std::uint8_t, std::int32_t, std::int64_t>;
using vec_types = c2h::type_list<long2, double2>;

using even_threads_in_block = c2h::enum_type_list<int, 32, 128>;
using odd_threads_in_block  = c2h::enum_type_list<int, 15, 65>;
using a_block_size          = c2h::enum_type_list<int, 256>;

using items_per_thread = c2h::enum_type_list<int, IPT>;
using store_algorithm =
  c2h::enum_type_list<cub::BlockStoreAlgorithm,
                      cub::BlockStoreAlgorithm::BLOCK_STORE_DIRECT,
                      cub::BlockStoreAlgorithm::BLOCK_STORE_STRIPED,
                      cub::BlockStoreAlgorithm::BLOCK_STORE_VECTORIZE,
                      cub::BlockStoreAlgorithm::BLOCK_STORE_TRANSPOSE,
                      cub::BlockStoreAlgorithm::BLOCK_STORE_WARP_TRANSPOSE,
                      cub::BlockStoreAlgorithm::BLOCK_STORE_WARP_TRANSPOSE_TIMESLICED>;

using odd_store_algorithm =
  c2h::enum_type_list<cub::BlockStoreAlgorithm,
                      cub::BlockStoreAlgorithm::BLOCK_STORE_DIRECT,
                      cub::BlockStoreAlgorithm::BLOCK_STORE_STRIPED,
                      cub::BlockStoreAlgorithm::BLOCK_STORE_VECTORIZE,
                      cub::BlockStoreAlgorithm::BLOCK_STORE_TRANSPOSE>;

template <class TestType>
struct params_t
{
  using type = typename c2h::get<0, TestType>;

  static constexpr int items_per_thread                     = c2h::get<1, TestType>::value;
  static constexpr int threads_in_block                     = c2h::get<2, TestType>::value;
  static constexpr int tile_size                            = items_per_thread * threads_in_block;
  static constexpr cub::BlockStoreAlgorithm store_algorithm = c2h::get<3, TestType>::value;
};

C2H_TEST("Block store works with even block sizes",
         "[store][block]",
         types,
         items_per_thread,
         even_threads_in_block,
         store_algorithm)
{
  using params = params_t<TestType>;
  using type   = typename params::type;

  c2h::device_vector<type> d_input(GENERATE_COPY(take(10, random(0, params::tile_size))));
  c2h::gen(C2H_SEED(10), d_input);

  c2h::device_vector<type> d_output(d_input.size());

  block_store<params::items_per_thread, params::threads_in_block, params::store_algorithm>(
    thrust::raw_pointer_cast(d_input.data()),
    thrust::raw_pointer_cast(d_output.data()),
    static_cast<int>(d_input.size()));

  REQUIRE(d_input == d_output);
}

C2H_TEST("Block store works with even odd sizes",
         "[store][block]",
         types,
         items_per_thread,
         odd_threads_in_block,
         odd_store_algorithm)
{
  using params = params_t<TestType>;
  using type   = typename params::type;

  c2h::device_vector<type> d_input(GENERATE_COPY(take(10, random(0, params::tile_size))));
  c2h::gen(C2H_SEED(10), d_input);

  c2h::device_vector<type> d_output(d_input.size());

  block_store<params::items_per_thread, params::threads_in_block, params::store_algorithm>(
    thrust::raw_pointer_cast(d_input.data()),
    thrust::raw_pointer_cast(d_output.data()),
    static_cast<int>(d_input.size()));

  REQUIRE(d_input == d_output);
}

C2H_TEST("Block store works with even vector types",
         "[store][block]",
         vec_types,
         items_per_thread,
         a_block_size,
         store_algorithm)
{
  using params = params_t<TestType>;
  using type   = typename params::type;

  c2h::device_vector<type> d_input(GENERATE_COPY(take(10, random(0, params::tile_size))));
  c2h::gen(C2H_SEED(10), d_input);

  c2h::device_vector<type> d_output(d_input.size());

  block_store<params::items_per_thread, params::threads_in_block, params::store_algorithm>(
    thrust::raw_pointer_cast(d_input.data()),
    thrust::raw_pointer_cast(d_output.data()),
    static_cast<int>(d_input.size()));

  REQUIRE(d_input == d_output);
}

C2H_TEST("Block store works with custom types", "[store][block]", items_per_thread, store_algorithm)
{
  using type                                                = c2h::custom_type_t<c2h::equal_comparable_t>;
  constexpr int items_per_thread                            = c2h::get<0, TestType>::value;
  constexpr int threads_in_block                            = 64;
  constexpr int tile_size                                   = items_per_thread * threads_in_block;
  static constexpr cub::BlockStoreAlgorithm store_algorithm = c2h::get<1, TestType>::value;

  c2h::device_vector<type> d_input(GENERATE_COPY(take(10, random(0, tile_size))));
  c2h::gen(C2H_SEED(10), d_input);

  c2h::device_vector<type> d_output(d_input.size());

  block_store<items_per_thread, threads_in_block, store_algorithm>(
    thrust::raw_pointer_cast(d_input.data()),
    thrust::raw_pointer_cast(d_output.data()),
    static_cast<int>(d_input.size()));

  REQUIRE(d_input == d_output);
}

C2H_TEST("Block store works with caching iterators", "[store][block]", items_per_thread, store_algorithm)
{
  using type                                                = int;
  constexpr int items_per_thread                            = c2h::get<0, TestType>::value;
  constexpr int threads_in_block                            = 64;
  constexpr int tile_size                                   = items_per_thread * threads_in_block;
  static constexpr cub::BlockStoreAlgorithm store_algorithm = c2h::get<1, TestType>::value;

  c2h::device_vector<type> d_input(GENERATE_COPY(take(10, random(0, tile_size))));
  c2h::gen(C2H_SEED(10), d_input);

  c2h::device_vector<type> d_output(d_input.size());
  cub::CacheModifiedOutputIterator<cub::CacheStoreModifier::STORE_DEFAULT, type> out(
    thrust::raw_pointer_cast(d_output.data()));

  block_store<items_per_thread, threads_in_block, store_algorithm>(
    thrust::raw_pointer_cast(d_input.data()), out, static_cast<int>(d_input.size()));

  REQUIRE(d_input == d_output);
}
