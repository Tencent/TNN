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

// %PARAM% TEST_LAUNCH lid 0:1
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
using full_type_list = c2h::type_list<type_quad<custom_t, custom_t, custom_t>>;
#endif

/**
 * @brief Input data generation mode
 */
enum class gen_data_t : int
{
  /// Uniform random data generation
  GEN_TYPE_RANDOM,
  /// Constant value as input data
  GEN_TYPE_CONST
};

C2H_TEST("Device scan works with fancy iterators", "[by_key][scan][device]", full_type_list)
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
  auto d_keys_it = segment_keys.begin();
  c2h::host_vector<key_t> h_segment_keys(segment_keys);

  // Prepare input data
  value_t default_constant{};
  init_default_constant(default_constant);
  auto values_in_it = thrust::make_constant_iterator(default_constant);

  SECTION("inclusive sum")
  {
    using op_t = cub::Sum;

    // Prepare verification data
    c2h::host_vector<output_t> expected_result(num_items);
    compute_inclusive_scan_by_key_reference(
      values_in_it, h_segment_keys.cbegin(), expected_result.begin(), op_t{}, eq_op_t{}, num_items);

    // Run test
    c2h::device_vector<output_t> out_values(num_items);
    device_inclusive_sum_by_key(d_keys_it, values_in_it, out_values.begin(), num_items, eq_op_t{});

    // Verify result
    REQUIRE(expected_result == out_values);
  }

  SECTION("exclusive sum")
  {
    using op_t = cub::Sum;

    // Prepare verification data
    c2h::host_vector<output_t> expected_result(num_items);
    compute_exclusive_scan_by_key_reference(
      values_in_it, h_segment_keys.cbegin(), expected_result.begin(), op_t{}, eq_op_t{}, output_t{}, num_items);

    // Run test
    c2h::device_vector<output_t> out_values(num_items);
    device_exclusive_sum_by_key(d_keys_it, values_in_it, out_values.begin(), num_items, eq_op_t{});

    // Verify result
    REQUIRE(expected_result == out_values);
  }

  SECTION("inclusive scan")
  {
    using op_t = cub::Min;

    // Prepare verification data
    c2h::host_vector<output_t> expected_result(num_items);
    compute_inclusive_scan_by_key_reference(
      values_in_it, h_segment_keys.cbegin(), expected_result.begin(), op_t{}, eq_op_t{}, num_items);

    // Run test
    c2h::device_vector<output_t> out_values(num_items);
    device_inclusive_scan_by_key(d_keys_it, values_in_it, out_values.begin(), op_t{}, num_items, eq_op_t{});

    // Verify result
    REQUIRE(expected_result == out_values);
  }

  SECTION("exclusive scan")
  {
    using op_t = cub::Sum;

    // Scan operator
    auto scan_op = op_t{};

    // Prepare verification data
    c2h::host_vector<output_t> expected_result(num_items);
    compute_exclusive_scan_by_key_reference(
      values_in_it, h_segment_keys.cbegin(), expected_result.begin(), scan_op, eq_op_t{}, output_t{}, num_items);

    // Run test
    c2h::device_vector<output_t> out_values(num_items);
    using init_t = value_t;
    device_exclusive_scan_by_key(d_keys_it, values_in_it, out_values.begin(), scan_op, init_t{}, num_items, eq_op_t{});

    // Verify result
    REQUIRE(expected_result == out_values);
  }
}
