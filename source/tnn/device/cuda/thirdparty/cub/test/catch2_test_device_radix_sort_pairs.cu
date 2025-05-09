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

#include "insert_nested_NVTX_range_guard.h"
// above header needs to be included first

#include <cub/device/device_radix_sort.cuh>
#include <cub/util_type.cuh>

#include <thrust/memory.h>

#include <cuda/std/type_traits>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <new> // bad_alloc

#include "catch2_large_array_sort_helper.cuh"
#include "catch2_radix_sort_helper.cuh"
#include "catch2_test_launch_helper.h"
#include <c2h/catch2_test_helper.cuh>

// %PARAM% TEST_LAUNCH lid 0:1:2

DECLARE_LAUNCH_WRAPPER(cub::DeviceRadixSort::SortPairs, sort_pairs);
DECLARE_LAUNCH_WRAPPER(cub::DeviceRadixSort::SortPairsDescending, sort_pairs_descending);

using custom_value_t = c2h::custom_type_t<c2h::equal_comparable_t>;
using value_types    = c2h::type_list<cuda::std::uint8_t, cuda::std::uint64_t, custom_value_t>;

// cub::detail::ChooseOffsetsT only selected 32/64 bit unsigned types:
using num_items_types = c2h::type_list<cuda::std::uint32_t, cuda::std::uint64_t>;

C2H_TEST("DeviceRadixSort::SortPairs: Basic testing", "[pairs][radix][sort][device]", value_types, num_items_types)
{
  using key_t       = cuda::std::uint32_t;
  using value_t     = c2h::get<0, TestType>;
  using num_items_t = c2h::get<1, TestType>;

  constexpr num_items_t min_num_items = 1 << 5;
  constexpr num_items_t max_num_items = 1 << 20;
  const num_items_t num_items =
    GENERATE_COPY(num_items_t{0}, num_items_t{1}, take(5, random(min_num_items, max_num_items)));

  c2h::device_vector<key_t> in_keys(num_items);
  c2h::device_vector<key_t> out_keys(num_items);

  c2h::device_vector<value_t> in_values(num_items);
  c2h::device_vector<value_t> out_values(num_items);

  const int num_key_seeds   = 1;
  const int num_value_seeds = 1;
  c2h::gen(C2H_SEED(num_key_seeds), in_keys);
  c2h::gen(C2H_SEED(num_value_seeds), in_values);

  const bool is_descending = GENERATE(false, true);

  if (is_descending)
  {
    sort_pairs_descending(
      thrust::raw_pointer_cast(in_keys.data()),
      thrust::raw_pointer_cast(out_keys.data()),
      thrust::raw_pointer_cast(in_values.data()),
      thrust::raw_pointer_cast(out_values.data()),
      num_items,
      begin_bit<key_t>(),
      end_bit<key_t>());
  }
  else
  {
    sort_pairs(thrust::raw_pointer_cast(in_keys.data()),
               thrust::raw_pointer_cast(out_keys.data()),
               thrust::raw_pointer_cast(in_values.data()),
               thrust::raw_pointer_cast(out_values.data()),
               num_items,
               begin_bit<key_t>(),
               end_bit<key_t>());
  }

  auto refs        = radix_sort_reference(in_keys, in_values, is_descending);
  auto& ref_keys   = refs.first;
  auto& ref_values = refs.second;

  REQUIRE(ref_keys == out_keys);
  REQUIRE(ref_values == out_values);
}

C2H_TEST("DeviceRadixSort::SortPairs: DoubleBuffer API", "[pairs][radix][sort][device]", value_types)
{
  using key_t   = cuda::std::uint32_t;
  using value_t = c2h::get<0, TestType>;

  constexpr std::size_t max_num_items = 1 << 18;
  const std::size_t num_items         = GENERATE_COPY(take(1, random(max_num_items / 2, max_num_items)));

  c2h::device_vector<key_t> in_keys(num_items);
  c2h::device_vector<key_t> out_keys(num_items);

  c2h::device_vector<value_t> in_values(num_items);
  c2h::device_vector<value_t> out_values(num_items);

  const int num_key_seeds   = 1;
  const int num_value_seeds = 1;
  c2h::gen(C2H_SEED(num_key_seeds), in_keys);
  c2h::gen(C2H_SEED(num_value_seeds), in_values);

  const bool is_descending = GENERATE(false, true);

  cub::DoubleBuffer<key_t> key_buffer(
    thrust::raw_pointer_cast(in_keys.data()), thrust::raw_pointer_cast(out_keys.data()));
  cub::DoubleBuffer<value_t> value_buffer(
    thrust::raw_pointer_cast(in_values.data()), thrust::raw_pointer_cast(out_values.data()));

  double_buffer_sort_t action(is_descending);
  action.initialize();
  launch(action, key_buffer, value_buffer, num_items, begin_bit<key_t>(), end_bit<key_t>());

  key_buffer.selector   = action.selector();
  value_buffer.selector = action.selector();
  action.finalize();

  auto refs        = radix_sort_reference(in_keys, in_values, is_descending);
  auto& ref_keys   = refs.first;
  auto& ref_values = refs.second;

  auto& keys   = key_buffer.selector == 0 ? in_keys : out_keys;
  auto& values = value_buffer.selector == 0 ? in_values : out_values;

  REQUIRE(ref_keys == keys);
  REQUIRE(ref_values == values);
}

template <typename key_t, typename value_t, typename num_items_t>
void do_large_offset_test(std::size_t num_items)
{
  const bool is_descending = GENERATE(false, true);

  CAPTURE(num_items, is_descending);

  try
  {
    large_array_sort_helper<key_t, value_t> arrays;
    arrays.initialize_for_stable_pair_sort(C2H_SEED(1), num_items, is_descending);

    TIME(c2h::cpu_timer timer);

    double_buffer_sort_t action(is_descending);
    action.initialize();
    const num_items_t typed_num_items = static_cast<num_items_t>(num_items);
    launch(action, arrays.keys_buffer, arrays.values_buffer, typed_num_items, begin_bit<key_t>(), end_bit<key_t>());

    TIME(timer.print_elapsed_seconds_and_reset("Device sort"));

    arrays.keys_buffer.selector   = action.selector();
    arrays.values_buffer.selector = action.selector();
    action.finalize();

    auto& keys   = arrays.keys_buffer.selector == 0 ? arrays.keys_in : arrays.keys_out;
    auto& values = arrays.values_buffer.selector == 0 ? arrays.values_in : arrays.values_out;

    arrays.verify_stable_pair_sort(num_items, is_descending, keys, values);
  }
  catch (std::bad_alloc& e)
  {
    (void) e;
#ifdef DEBUG_CHECKED_ALLOC_FAILURE
    const std::size_t num_bytes = num_items * (sizeof(key_t) + sizeof(value_t));
    std::cerr
      << "Skipping radix sort test with " << num_items << " elements (" << num_bytes << " bytes): " << e.what() << "\n";
#endif // DEBUG_CHECKED_ALLOC_FAILURE
  }
}

C2H_TEST("DeviceRadixSort::SortPairs: 32-bit overflow check", "[large][pairs][radix][sort][device]")
{
  using key_t       = std::uint8_t;
  using value_t     = std::uint8_t;
  using num_items_t = std::uint32_t;

  // Test problem size at the maximum offset value to ensure that internal calculations do not overflow.
  const std::size_t num_items = std::numeric_limits<num_items_t>::max();

  do_large_offset_test<key_t, value_t, num_items_t>(num_items);
}

C2H_TEST("DeviceRadixSort::SortPairs: Large Offsets", "[large][pairs][radix][sort][device]")
{
  using key_t       = std::uint8_t;
  using value_t     = std::uint8_t;
  using num_items_t = std::uint64_t;

  constexpr std::size_t min_num_items = std::size_t{1} << 32;
  constexpr std::size_t max_num_items = min_num_items + (std::size_t{1} << 30);
  const std::size_t num_items         = GENERATE_COPY(take(1, random(min_num_items, max_num_items)));

  do_large_offset_test<key_t, value_t, num_items_t>(num_items);
}
