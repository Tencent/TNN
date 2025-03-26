/******************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

#include <cub/device/device_merge_sort.cuh>

#include <thrust/copy.h>
#include <thrust/equal.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/reverse_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reverse.h>
#include <thrust/sequence.h>

#include <algorithm>

#include "catch2_test_device_merge_sort_common.cuh"
#include "catch2_test_launch_helper.h"
#include <c2h/catch2_test_helper.cuh>

// %PARAM% TEST_LAUNCH lid 0:1:2

DECLARE_LAUNCH_WRAPPER(cub::DeviceMergeSort::SortPairs, sort_pairs);
DECLARE_LAUNCH_WRAPPER(cub::DeviceMergeSort::SortPairsCopy, sort_pairs_copy);
DECLARE_LAUNCH_WRAPPER(cub::DeviceMergeSort::StableSortPairs, stable_sort_pairs);

DECLARE_LAUNCH_WRAPPER(cub::DeviceMergeSort::SortKeys, sort_keys);
DECLARE_LAUNCH_WRAPPER(cub::DeviceMergeSort::SortKeysCopy, sort_keys_copy);
DECLARE_LAUNCH_WRAPPER(cub::DeviceMergeSort::StableSortKeys, stable_sort_keys);
DECLARE_LAUNCH_WRAPPER(cub::DeviceMergeSort::StableSortKeysCopy, stable_sort_keys_copy);

C2H_TEST("DeviceMergeSort::SortKeysCopy works with iterators", "[merge][sort][device]")
{
  using key_t    = std::uint32_t;
  using offset_t = std::int32_t;

  // Prepare input
  const offset_t num_items = GENERATE_COPY(take(2, random(1, 1000000)), values({500, 1000000}));
  auto keys_counting_it    = thrust::make_counting_iterator(key_t{});
  auto keys_in_it          = thrust::make_reverse_iterator(keys_counting_it + num_items);

  // Perform sort
  c2h::device_vector<key_t> keys_out(num_items, static_cast<key_t>(42));
  sort_keys_copy(keys_in_it, keys_out.begin(), num_items, custom_less_op_t{});

  // Verify results
  auto keys_expected_it = keys_counting_it;
  bool keys_equal       = thrust::equal(c2h::device_policy, keys_out.cbegin(), keys_out.cend(), keys_expected_it);
  REQUIRE(keys_equal == true);
}

C2H_TEST("DeviceMergeSort::StableSortKeysCopy works with iterators and is stable", "[merge][sort][device]")
{
  using key_t    = std::uint32_t;
  using offset_t = std::int32_t;

  // Prepare input (ensure we have multiple sort keys that compare equal to check stability)
  const offset_t num_items = GENERATE_COPY(take(2, random(1, 1000000)), values({500, 1000000}));
  auto sort_key_it = thrust::make_transform_iterator(thrust::make_counting_iterator(key_t{}), mod_op_t<key_t>{128});
  auto key_idx_it  = thrust::make_counting_iterator(offset_t{});
  auto keys_in_it  = thrust::make_zip_iterator(sort_key_it, key_idx_it);

  // Perform sort
  c2h::device_vector<thrust::tuple<key_t, offset_t>> keys_out(
    num_items, thrust::tuple<key_t, offset_t>{static_cast<key_t>(42), static_cast<offset_t>(42)});
  stable_sort_keys_copy(keys_in_it, keys_out.begin(), num_items, compare_first_lt_op_t{});

  // Verify results
  c2h::host_vector<thrust::tuple<key_t, offset_t>> keys_expected(num_items);
  thrust::copy(keys_in_it, keys_in_it + num_items, keys_expected.begin());
  std::stable_sort(keys_expected.begin(), keys_expected.end(), compare_first_lt_op_t{});

  REQUIRE(keys_expected == keys_out);
}

C2H_TEST("DeviceMergeSort::SortKeys works with iterators", "[merge][sort][device]")
{
  using key_t    = std::uint32_t;
  using offset_t = std::int32_t;

  // Prepare input
  const offset_t num_items = GENERATE_COPY(take(2, random(1, 1000000)), values({500, 1000000}));
  c2h::device_vector<key_t> keys_in_out(num_items);
  thrust::sequence(c2h::device_policy, keys_in_out.begin(), keys_in_out.end());
  auto keys_in_it = thrust::make_reverse_iterator(keys_in_out.end());

  // Perform sort
  sort_keys(keys_in_it, num_items, custom_less_op_t{});

  // Verify results
  auto keys_counting_it = thrust::make_counting_iterator(key_t{});
  auto keys_expected_it = thrust::make_reverse_iterator(keys_counting_it + num_items);
  bool keys_equal       = thrust::equal(keys_in_out.cbegin(), keys_in_out.cend(), keys_expected_it);
  REQUIRE(keys_equal == true);
}

C2H_TEST("DeviceMergeSort::StableSortKeys works with iterators", "[merge][sort][device]")
{
  using key_t    = std::uint32_t;
  using offset_t = std::int32_t;

  // Prepare input
  const offset_t num_items = GENERATE_COPY(take(2, random(1, 1000000)), values({500, 1000000}));
  c2h::device_vector<key_t> keys_in_out(num_items);
  thrust::sequence(c2h::device_policy, keys_in_out.begin(), keys_in_out.end());
  auto keys_in_it = thrust::make_reverse_iterator(keys_in_out.end());

  // Perform sort
  stable_sort_keys(keys_in_it, num_items, custom_less_op_t{});

  // Verify results
  auto keys_counting_it = thrust::make_counting_iterator(key_t{});
  auto keys_expected_it = thrust::make_reverse_iterator(keys_counting_it + num_items);
  bool keys_equal       = thrust::equal(c2h::device_policy, keys_in_out.cbegin(), keys_in_out.cend(), keys_expected_it);
  REQUIRE(keys_equal == true);
}

C2H_TEST("DeviceMergeSort::SortPairsCopy works with iterators", "[merge][sort][device]")
{
  using key_t    = std::uint32_t;
  using data_t   = std::uint64_t;
  using offset_t = std::int32_t;

  // Prepare input
  const offset_t num_items = GENERATE_COPY(take(2, random(1, 1000000)), values({500, 1000000}));
  auto key_counting_it     = thrust::make_counting_iterator(key_t{});
  auto keys_in             = thrust::make_reverse_iterator(key_counting_it + num_items);
  auto values_in           = thrust::make_counting_iterator(data_t{}) + num_items;

  // Perform sort
  c2h::device_vector<key_t> keys_out(num_items, static_cast<key_t>(42));
  c2h::device_vector<data_t> values_out(num_items, static_cast<data_t>(42));
  sort_pairs_copy(keys_in, values_in, keys_out.begin(), values_out.begin(), num_items, custom_less_op_t{});

  // Verify results
  auto keys_expected_it   = key_counting_it;
  auto values_expected_it = thrust::make_reverse_iterator(values_in + num_items);
  bool keys_equal         = thrust::equal(c2h::device_policy, keys_out.cbegin(), keys_out.cend(), keys_expected_it);
  bool values_equal = thrust::equal(c2h::device_policy, values_out.cbegin(), values_out.cend(), values_expected_it);
  REQUIRE(keys_equal == true);
  REQUIRE(values_equal == true);
}

C2H_TEST("DeviceMergeSort::SortPairs works with iterators", "[merge][sort][device]")
{
  using key_t    = std::uint32_t;
  using data_t   = std::uint64_t;
  using offset_t = std::int32_t;

  // Prepare input
  const offset_t num_items = GENERATE_COPY(take(2, random(1, 1000000)), values({500, 1000000}));
  c2h::device_vector<key_t> keys_in_out(num_items);
  c2h::device_vector<data_t> values_in_out(num_items);
  thrust::sequence(c2h::device_policy, keys_in_out.begin(), keys_in_out.end());
  thrust::sequence(c2h::device_policy, values_in_out.begin(), values_in_out.end());
  thrust::reverse(c2h::device_policy, values_in_out.begin(), values_in_out.end());
  auto keys_in_it = thrust::make_reverse_iterator(keys_in_out.end());

  // Perform sort
  sort_pairs(keys_in_it, values_in_out.begin(), num_items, custom_less_op_t{});

  // Verify results
  auto keys_counting_it   = thrust::make_counting_iterator(key_t{});
  auto keys_expected_it   = thrust::make_reverse_iterator(keys_counting_it + num_items);
  auto values_expected_it = thrust::make_counting_iterator(data_t{});
  bool keys_equal = thrust::equal(c2h::device_policy, keys_in_out.cbegin(), keys_in_out.cend(), keys_expected_it);
  bool values_equal =
    thrust::equal(c2h::device_policy, values_in_out.cbegin(), values_in_out.cend(), values_expected_it);
  REQUIRE(keys_equal == true);
  REQUIRE(values_equal == true);
}

C2H_TEST("DeviceMergeSort::StableSortPairs works with iterators", "[merge][sort][device]")
{
  using key_t    = std::uint32_t;
  using data_t   = std::uint64_t;
  using offset_t = std::int32_t;

  // Prepare input
  const offset_t num_items = GENERATE_COPY(take(2, random(1, 1000000)), values({500, 1000000}));
  c2h::device_vector<key_t> keys_in_out(num_items);
  c2h::device_vector<data_t> values_in_out(num_items);
  thrust::sequence(c2h::device_policy, keys_in_out.begin(), keys_in_out.end());
  thrust::sequence(c2h::device_policy, values_in_out.begin(), values_in_out.end());
  thrust::reverse(c2h::device_policy, values_in_out.begin(), values_in_out.end());
  auto keys_in_it = thrust::make_reverse_iterator(keys_in_out.end());

  // Perform sort
  stable_sort_pairs(keys_in_it, values_in_out.begin(), num_items, custom_less_op_t{});

  // Verify results
  auto keys_counting_it   = thrust::make_counting_iterator(key_t{});
  auto keys_expected_it   = thrust::make_reverse_iterator(keys_counting_it + num_items);
  auto values_expected_it = thrust::make_counting_iterator(data_t{});
  bool keys_equal = thrust::equal(c2h::device_policy, keys_in_out.cbegin(), keys_in_out.cend(), keys_expected_it);
  bool values_equal =
    thrust::equal(c2h::device_policy, values_in_out.cbegin(), values_in_out.cend(), values_expected_it);
  REQUIRE(keys_equal == true);
  REQUIRE(values_equal == true);
}
