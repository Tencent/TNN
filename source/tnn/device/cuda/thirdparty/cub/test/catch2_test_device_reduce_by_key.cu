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

#include <cub/device/device_reduce.cuh>

#include "catch2_test_device_reduce.cuh"
#include "catch2_test_launch_helper.h"
#include <c2h/catch2_test_helper.cuh>
#include <c2h/custom_type.cuh>
#include <c2h/extended_types.cuh>

DECLARE_LAUNCH_WRAPPER(cub::DeviceReduce::ReduceByKey, device_reduce_by_key);

// %PARAM% TEST_LAUNCH lid 0:1:2
// %PARAM% TEST_TYPES types 0:1:2:3

// List of types to test
using custom_t =
  c2h::custom_type_t<c2h::accumulateable_t,
                     c2h::equal_comparable_t,
                     c2h::lexicographical_less_comparable_t,
                     c2h::lexicographical_greater_comparable_t>;

#if TEST_TYPES == 0
using full_type_list = c2h::type_list<type_triple<std::uint8_t>, type_triple<std::int8_t, std::int32_t, custom_t>>;
#elif TEST_TYPES == 1
using full_type_list = c2h::type_list<type_triple<std::int32_t>, type_triple<std::int64_t>>;
#elif TEST_TYPES == 2
using full_type_list = c2h::type_list<type_triple<uchar3, uchar3, custom_t>, type_triple<ulonglong4>>;
#elif TEST_TYPES == 3
// clang-format off
using full_type_list = c2h::type_list<
type_triple<custom_t>
#if TEST_HALF_T
, type_triple<half_t> // testing half
#endif
#if TEST_BF_T
, type_triple<bfloat16_t> // testing bf16
#endif
>;
// clang-format on
#endif

C2H_TEST("Device reduce-by-key works", "[by_key][reduce][device]", full_type_list)
{
  using params   = params_t<TestType>;
  using value_t  = typename params::item_t;
  using output_t = typename params::output_t;
  using key_t    = typename params::type_pair_t::key_t;
  using offset_t = uint32_t;

  constexpr offset_t min_items = 1;
  constexpr offset_t max_items = 1000000;

  // Number of items
  const offset_t num_items = GENERATE_COPY(
    take(2, random(min_items, max_items)),
    values({
      min_items,
      max_items,
    }));
  INFO("Test num_items: " << num_items);

  // Range of segment sizes to generate (a segment is a series of consecutive equal keys)
  const std::tuple<offset_t, offset_t> seg_size_range =
    GENERATE_COPY(table<offset_t, offset_t>({{1, 1}, {1, num_items}, {num_items, num_items}}));
  INFO("Test seg_size_range: [" << std::get<0>(seg_size_range) << ", " << std::get<1>(seg_size_range) << "]");

  // Generate input segments
  c2h::device_vector<offset_t> segment_offsets = c2h::gen_uniform_offsets<offset_t>(
    C2H_SEED(1), num_items, std::get<0>(seg_size_range), std::get<1>(seg_size_range));

  // Get array of keys from segment offsets
  const offset_t num_segments = static_cast<offset_t>(segment_offsets.size() - 1);
  c2h::device_vector<key_t> segment_keys(num_items);
  c2h::init_key_segments(segment_offsets, segment_keys);
  auto d_keys_it = thrust::raw_pointer_cast(segment_keys.data());

  // Generate input data
  c2h::device_vector<value_t> in_values(num_items);
  c2h::gen(C2H_SEED(2), in_values);
  auto d_values_it = thrust::raw_pointer_cast(in_values.data());

  SECTION("sum")
  {
    using op_t = cub::Sum;

    // Binary reduction operator
    auto reduction_op = unwrap_op(reference_extended_fp(d_values_it), op_t{});

    // Prepare verification data
    using accum_t = ::cuda::std::__accumulator_t<op_t, value_t, output_t>;
    c2h::host_vector<output_t> expected_result(num_segments);
    compute_segmented_problem_reference(in_values, segment_offsets, reduction_op, accum_t{}, expected_result.begin());
    c2h::host_vector<key_t> expected_keys = compute_unique_keys_reference(segment_keys);

    // Run test
    c2h::device_vector<offset_t> num_unique_keys(1);
    c2h::device_vector<key_t> out_unique_keys(num_segments);
    c2h::device_vector<output_t> out_result(num_segments);
    auto d_out_it      = thrust::raw_pointer_cast(out_result.data());
    auto d_keys_out_it = thrust::raw_pointer_cast(out_unique_keys.data());
    device_reduce_by_key(
      d_keys_it,
      d_keys_out_it,
      unwrap_it(d_values_it),
      unwrap_it(d_out_it),
      thrust::raw_pointer_cast(num_unique_keys.data()),
      reduction_op,
      num_items);

    // Verify result
    REQUIRE(num_segments == num_unique_keys[0]);
    REQUIRE(expected_result == out_result);
    REQUIRE(expected_keys == out_unique_keys);
  }

  SECTION("min")
  {
    using op_t = cub::Min;

    // Prepare verification data
    c2h::host_vector<output_t> expected_result(num_segments);
    compute_segmented_problem_reference(
      in_values, segment_offsets, op_t{}, cub::NumericTraits<value_t>::Max(), expected_result.begin());
    c2h::host_vector<key_t> expected_keys = compute_unique_keys_reference(segment_keys);

    // Run test
    c2h::device_vector<offset_t> num_unique_keys(1);
    c2h::device_vector<key_t> out_unique_keys(num_segments);
    c2h::device_vector<output_t> out_result(num_segments);
    auto d_result_out_it = thrust::raw_pointer_cast(out_result.data());
    auto d_keys_out_it   = thrust::raw_pointer_cast(out_unique_keys.data());
    device_reduce_by_key(
      d_keys_it,
      d_keys_out_it,
      unwrap_it(d_values_it),
      unwrap_it(d_result_out_it),
      thrust::raw_pointer_cast(num_unique_keys.data()),
      op_t{},
      num_items);

    // Verify result
    REQUIRE(num_segments == num_unique_keys[0]);
    REQUIRE(expected_result == out_result);
    REQUIRE(expected_keys == out_unique_keys);
  }
}
