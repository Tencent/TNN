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

#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/iterator/zip_iterator.h>

#include <algorithm>

#include "catch2_test_device_merge_sort_common.cuh"
#include "catch2_test_launch_helper.h"
#include <c2h/catch2_test_helper.cuh>

// %PARAM% TEST_LAUNCH lid 0:1:2

DECLARE_LAUNCH_WRAPPER(cub::DeviceMergeSort::StableSortKeys, stable_sort_keys);
DECLARE_LAUNCH_WRAPPER(cub::DeviceMergeSort::StableSortPairs, stable_sort_pairs);

using key_types =
  c2h::type_list<c2h::custom_type_t<c2h::equal_comparable_t, c2h::less_comparable_t, c2h::huge_data<512>::type>,
                 c2h::custom_type_t<c2h::equal_comparable_t, c2h::less_comparable_t, c2h::huge_data<1024>::type>>;

C2H_TEST("DeviceMergeSort::StableSortKeys works for large types", "[merge][sort][device]", key_types)
{
  using key_t    = typename c2h::get<0, TestType>;
  using offset_t = std::int32_t;

  // Prepare input
  const offset_t num_items = GENERATE_COPY(take(2, random(1, 10000)));
  c2h::device_vector<key_t> keys_in_out(num_items);
  c2h::gen(C2H_SEED(2), keys_in_out);

  // Prepare host data for verification
  c2h::host_vector<key_t> keys_expected(keys_in_out);
  std::stable_sort(keys_expected.begin(), keys_expected.end(), custom_less_op_t{});

  // Perform sort
  stable_sort_keys(thrust::raw_pointer_cast(keys_in_out.data()), num_items, custom_less_op_t{});

  // Verify results
  REQUIRE(keys_expected == keys_in_out);
}

C2H_TEST("DeviceMergeSort::StableSortPairs works for large types", "[merge][sort][device]", key_types)
{
  using key_t    = typename c2h::get<0, TestType>;
  using data_t   = std::uint32_t;
  using offset_t = std::int32_t;

  // Prepare input
  const offset_t num_items = GENERATE_COPY(take(2, random(1, 10000)));
  c2h::device_vector<key_t> keys_in_out(num_items);
  c2h::device_vector<data_t> values_in_out(num_items);
  c2h::gen(C2H_SEED(2), keys_in_out);
  c2h::gen(C2H_SEED(1), values_in_out);

  // Prepare host data for verification
  c2h::host_vector<key_t> keys_expected(keys_in_out);
  c2h::host_vector<data_t> values_expected(values_in_out);
  auto zipped_expected_it = thrust::make_zip_iterator(keys_expected.begin(), values_expected.begin());
  std::stable_sort(zipped_expected_it, zipped_expected_it + num_items, compare_first_lt_op_t{});

  // Perform sort
  stable_sort_pairs(thrust::raw_pointer_cast(keys_in_out.data()),
                    thrust::raw_pointer_cast(values_in_out.data()),
                    num_items,
                    custom_less_op_t{});

  // Verify results
  REQUIRE(keys_expected == keys_in_out);
  REQUIRE(values_expected == values_in_out);
}
