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

#include <cub/device/device_scan.cuh>

#include <cstdint>
#include <type_traits>

#include "catch2_test_device_reduce.cuh"
#include "catch2_test_device_scan.cuh"
#include "catch2_test_launch_helper.h"
#include <c2h/catch2_test_helper.cuh>
#include <c2h/custom_type.cuh>
#include <c2h/extended_types.cuh>

DECLARE_LAUNCH_WRAPPER(cub::DeviceScan::ExclusiveSumByKey, device_exclusive_sum_by_key);
DECLARE_LAUNCH_WRAPPER(cub::DeviceScan::ExclusiveScanByKey, device_exclusive_scan_by_key);
DECLARE_LAUNCH_WRAPPER(cub::DeviceScan::InclusiveSumByKey, device_inclusive_sum_by_key);
DECLARE_LAUNCH_WRAPPER(cub::DeviceScan::InclusiveScanByKey, device_inclusive_scan_by_key);

// %PARAM% TEST_LAUNCH lid 0:1:2
// %PARAM% TEST_TYPES types 0:1:2:3

// List of types to test
using custom_t =
  c2h::custom_type_t<c2h::accumulateable_t,
                     c2h::equal_comparable_t,
                     c2h::lexicographical_less_comparable_t,
                     c2h::lexicographical_greater_comparable_t>;

// type_quad's parameters and defaults:
// type_quad<value_in_t, value_out_t=value_in_t, key_t=int32_t, equality_op_t=cub::Equality>
#if TEST_TYPES == 0
using full_type_list = c2h::type_list<type_quad<std::uint8_t, std::int32_t, float>,
                                      type_quad<std::int8_t, std::int8_t, std::int32_t, Mod2Equality>>;
#elif TEST_TYPES == 1
using full_type_list = c2h::type_list<type_quad<std::int32_t>, type_quad<std::uint64_t>>;
#elif TEST_TYPES == 2
using full_type_list =
  c2h::type_list<type_quad<uchar3, uchar3, custom_t>, type_quad<ulonglong4, ulonglong4, std::uint8_t, Mod2Equality>>;
#elif TEST_TYPES == 3
// clang-format off
using full_type_list = c2h::type_list<
type_quad<custom_t, custom_t, custom_t>
#if TEST_HALF_T
, type_quad<half_t> // testing half
#endif
#if TEST_BF_T
, type_quad<bfloat16_t> // testing bf16
#endif
>;
// clang-format on
#endif

C2H_TEST("Device scan works with all device interfaces", "[by_key][scan][device]", full_type_list)
{
  using params   = params_t<TestType>;
  using key_t    = typename params::type_pair_t::key_t;
  using value_t  = typename params::item_t;
  using output_t = typename params::output_t;
  using offset_t = std::uint32_t;
  using eq_op_t  = typename params::type_pair_t::eq_op_t;

  constexpr offset_t min_items = 1;
  constexpr offset_t max_items = 1000000;

  // Generate the input sizes to test for
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
  c2h::device_vector<key_t> segment_keys(num_items);
  c2h::init_key_segments(segment_offsets, segment_keys);
  auto d_keys_it = thrust::raw_pointer_cast(segment_keys.data());

  // Generate input data
  c2h::device_vector<value_t> in_values(num_items);
  c2h::gen(C2H_SEED(2), in_values, std::numeric_limits<value_t>::min());
  auto d_values_it = thrust::raw_pointer_cast(in_values.data());

// Skip DeviceScan::InclusiveSum and DeviceScan::ExclusiveSum tests for extended floating-point
// types because of unbounded epsilon due to pseudo associativity of the addition operation over
// floating point numbers
#if TEST_TYPES != 3
  SECTION("inclusive sum")
  {
    using op_t = cub::Sum;

    // Prepare verification data
    c2h::host_vector<output_t> expected_result(num_items);
    compute_inclusive_scan_by_key_reference(in_values, segment_keys, expected_result.begin(), op_t{}, eq_op_t{});

    // Run test
    c2h::device_vector<output_t> out_values(num_items);
    auto d_values_out_it = thrust::raw_pointer_cast(out_values.data());
    device_inclusive_sum_by_key(d_keys_it, d_values_it, d_values_out_it, num_items, eq_op_t{});

    // Verify result
    REQUIRE(expected_result == out_values);

    // Run test in-place
    _CCCL_IF_CONSTEXPR (std::is_same<value_t, output_t>::value)
    {
      // Copy input values to memory allocated for output values, to ensure in_values are
      // unchanged for a (potentially) subsequent test that uses in_values as input
      out_values            = in_values;
      auto values_in_out_it = thrust::raw_pointer_cast(out_values.data());
      device_inclusive_sum_by_key(d_keys_it, values_in_out_it, values_in_out_it, num_items, eq_op_t{});

      // Verify result
      REQUIRE(expected_result == out_values);
    }
  }

  SECTION("exclusive sum")
  {
    using op_t = cub::Sum;

    // Prepare verification data
    c2h::host_vector<output_t> expected_result(num_items);
    compute_exclusive_scan_by_key_reference(
      in_values, segment_keys, expected_result.begin(), op_t{}, eq_op_t{}, output_t{});

    // Run test
    c2h::device_vector<output_t> out_values(num_items);
    auto d_values_out_it = thrust::raw_pointer_cast(out_values.data());
    device_exclusive_sum_by_key(d_keys_it, d_values_it, d_values_out_it, num_items, eq_op_t{});

    // Verify result
    REQUIRE(expected_result == out_values);

    // Run test in-place
    _CCCL_IF_CONSTEXPR (std::is_same<value_t, output_t>::value)
    {
      // Copy input values to memory allocated for output values, to ensure in_values are
      // unchanged for a (potentially) subsequent test that uses in_values as input
      out_values            = in_values;
      auto values_in_out_it = thrust::raw_pointer_cast(out_values.data());
      device_exclusive_sum_by_key(d_keys_it, values_in_out_it, values_in_out_it, num_items, eq_op_t{});

      // Verify result
      REQUIRE(expected_result == out_values);
    }
  }
#endif

  SECTION("inclusive scan")
  {
    using op_t = cub::Min;

    // Prepare verification data
    c2h::host_vector<output_t> expected_result(num_items);
    compute_inclusive_scan_by_key_reference(in_values, segment_keys, expected_result.begin(), op_t{}, eq_op_t{});

    // Run test
    c2h::device_vector<output_t> out_values(num_items);
    auto d_values_out_it = thrust::raw_pointer_cast(out_values.data());
    device_inclusive_scan_by_key(
      d_keys_it, unwrap_it(d_values_it), unwrap_it(d_values_out_it), op_t{}, num_items, eq_op_t{});

    // Verify result
    REQUIRE(expected_result == out_values);

    // Run test in-place
    _CCCL_IF_CONSTEXPR (std::is_same<value_t, output_t>::value)
    {
      // Copy input values to memory allocated for output values, to ensure in_values are
      // unchanged for a (potentially) subsequent test that uses in_values as input
      out_values            = in_values;
      auto values_in_out_it = thrust::raw_pointer_cast(out_values.data());
      device_inclusive_scan_by_key(
        d_keys_it, unwrap_it(values_in_out_it), unwrap_it(values_in_out_it), op_t{}, num_items, eq_op_t{});

      // Verify result
      REQUIRE(expected_result == out_values);
    }
  }

  SECTION("exclusive scan")
  {
    using op_t = cub::Sum;

    // Scan operator
    auto scan_op = unwrap_op(reference_extended_fp(d_values_it), op_t{});

    // Prepare verification data
    c2h::host_vector<output_t> expected_result(num_items);
    compute_exclusive_scan_by_key_reference(
      in_values, segment_keys, expected_result.begin(), scan_op, eq_op_t{}, output_t{});

    // Run test
    c2h::device_vector<output_t> out_values(num_items);
    auto d_values_out_it = thrust::raw_pointer_cast(out_values.data());
    using init_t         = cub::detail::value_t<decltype(unwrap_it(d_values_out_it))>;
    device_exclusive_scan_by_key(
      d_keys_it, unwrap_it(d_values_it), unwrap_it(d_values_out_it), scan_op, init_t{}, num_items, eq_op_t{});

    // Verify result
    REQUIRE(expected_result == out_values);

    // Run test in-place
    _CCCL_IF_CONSTEXPR (std::is_same<value_t, output_t>::value)
    {
      // Copy input values to memory allocated for output values, to ensure in_values are
      // unchanged for a (potentially) subsequent test that uses in_values as input
      out_values            = in_values;
      auto values_in_out_it = thrust::raw_pointer_cast(out_values.data());
      device_exclusive_scan_by_key(
        d_keys_it, unwrap_it(values_in_out_it), unwrap_it(values_in_out_it), scan_op, init_t{}, num_items, eq_op_t{});

      // Verify result
      REQUIRE(expected_result == out_values);
    }
  }
}

#if TEST_TYPES == 0
using key_alias_type_list = c2h::type_list<std::uint8_t>;
#elif TEST_TYPES == 1
using key_alias_type_list = c2h::type_list<std::int32_t>;
#elif TEST_TYPES == 2
using key_alias_type_list = c2h::type_list<float>;
#elif TEST_TYPES == 3
using key_alias_type_list = c2h::type_list<custom_t>;
#endif

C2H_TEST("Device scan works when memory for keys and results alias one another",
         "[by_key][scan][device]",
         key_alias_type_list)
{
  using key_t    = typename c2h::get<0, TestType>;
  using value_t  = key_t;
  using output_t = key_t;
  using offset_t = std::uint32_t;

  constexpr offset_t min_items = 1;
  constexpr offset_t max_items = 1000000;

  // Generate the input sizes to test for
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
  c2h::device_vector<key_t> segment_keys(num_items);
  c2h::init_key_segments(segment_offsets, segment_keys);
  auto d_keys_it = thrust::raw_pointer_cast(segment_keys.data());

  // Generate input data
  c2h::device_vector<value_t> in_values(num_items);
  c2h::gen(C2H_SEED(2), in_values, std::numeric_limits<value_t>::min());
  auto d_values_it = thrust::raw_pointer_cast(in_values.data());

  SECTION("inclusive sum")
  {
    using op_t = cub::Sum;

    // Prepare verification data
    c2h::host_vector<output_t> expected_result(num_items);
    compute_inclusive_scan_by_key_reference(in_values, segment_keys, expected_result.begin(), op_t{}, cub::Equality{});

    // Run test
    auto d_values_out_it = d_keys_it;
    device_inclusive_sum_by_key(d_keys_it, d_values_it, d_values_out_it, num_items, cub::Equality{});

    // Verify result
    REQUIRE(expected_result == segment_keys);
  }

  SECTION("exclusive sum")
  {
    using op_t = cub::Sum;

    // Prepare verification data
    c2h::host_vector<output_t> expected_result(num_items);
    compute_exclusive_scan_by_key_reference(
      in_values, segment_keys, expected_result.begin(), op_t{}, cub::Equality{}, output_t{});

    // Run test
    auto d_values_out_it = d_keys_it;
    device_exclusive_sum_by_key(d_keys_it, d_values_it, d_values_out_it, num_items, cub::Equality{});

    // Verify result
    REQUIRE(expected_result == segment_keys);
  }

  SECTION("inclusive scan")
  {
    using op_t = cub::Min;

    // Prepare verification data
    c2h::host_vector<output_t> expected_result(num_items);
    compute_inclusive_scan_by_key_reference(in_values, segment_keys, expected_result.begin(), op_t{}, cub::Equality{});

    // Run test
    auto d_values_out_it = d_keys_it;
    device_inclusive_scan_by_key(d_keys_it, d_values_it, d_values_out_it, op_t{}, num_items, cub::Equality{});

    // Verify result
    REQUIRE(expected_result == segment_keys);
  }

  SECTION("exclusive scan")
  {
    using op_t = cub::Sum;

    // Scan operator
    auto scan_op = op_t{};

    // Prepare verification data
    c2h::host_vector<output_t> expected_result(num_items);
    compute_exclusive_scan_by_key_reference(
      in_values, segment_keys, expected_result.begin(), scan_op, cub::Equality{}, output_t{});

    // Run test
    auto d_values_out_it = d_keys_it;
    using init_t         = value_t;
    device_exclusive_scan_by_key(d_keys_it, d_values_it, d_values_out_it, scan_op, init_t{}, num_items, cub::Equality{});

    // Verify result
    REQUIRE(expected_result == segment_keys);
  }
}
