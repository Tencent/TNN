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

#include <thrust/iterator/constant_iterator.h>

#include <cstdint>

#include "catch2_test_device_reduce.cuh"
#include "catch2_test_launch_helper.h"
#include <c2h/catch2_test_helper.cuh>
#include <c2h/custom_type.cuh>
#include <c2h/extended_types.cuh>

DECLARE_LAUNCH_WRAPPER(cub::DeviceReduce::Reduce, device_reduce);
DECLARE_LAUNCH_WRAPPER(cub::DeviceReduce::Sum, device_sum);
DECLARE_LAUNCH_WRAPPER(cub::DeviceReduce::Min, device_min);
DECLARE_LAUNCH_WRAPPER(cub::DeviceReduce::ArgMin, device_arg_min);
DECLARE_LAUNCH_WRAPPER(cub::DeviceReduce::Max, device_max);
DECLARE_LAUNCH_WRAPPER(cub::DeviceReduce::ArgMax, device_arg_max);

// %PARAM% TEST_LAUNCH lid 0:1:2
// %PARAM% TEST_TYPES types 0:1:2:3:4

// List of types to test
using custom_t =
  c2h::custom_type_t<c2h::accumulateable_t,
                     c2h::equal_comparable_t,
                     c2h::lexicographical_less_comparable_t,
                     c2h::lexicographical_greater_comparable_t>;

#if TEST_TYPES == 0
using full_type_list = c2h::type_list<type_pair<std::uint8_t>, type_pair<std::int8_t, std::int32_t>>;
#elif TEST_TYPES == 1
using full_type_list = c2h::type_list<type_pair<std::int32_t>, type_pair<std::int64_t>>;
#elif TEST_TYPES == 2
using full_type_list = c2h::type_list<type_pair<uchar3>, type_pair<ulonglong4>>;
#elif TEST_TYPES == 3
// clang-format off
using full_type_list = c2h::type_list<
type_pair<custom_t>
#if TEST_HALF_T
, type_pair<half_t> // testing half
#endif
#if TEST_BF_T
, type_pair<bfloat16_t> // testing bf16

>;
#endif
// clang-format on
#elif TEST_TYPES == 4
// DPX SIMD instructions
using full_type_list = c2h::type_list<type_pair<std::uint16_t>, type_pair<std::int16_t>>;
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

C2H_TEST("Device reduce works with all device interfaces", "[reduce][device]", full_type_list)
{
  using params   = params_t<TestType>;
  using item_t   = typename params::item_t;
  using output_t = typename params::output_t;
  using offset_t = int32_t;

  constexpr int max_items    = 5000000;
  constexpr int min_items    = 1;
  constexpr int num_segments = 1;

  // Generate the input sizes to test for
  const int num_items = GENERATE_COPY(
    take(3, random(min_items, max_items)),
    values({
      min_items,
      max_items,
    }));

  // Input data generation to test
  const gen_data_t data_gen_mode = GENERATE_COPY(gen_data_t::GEN_TYPE_RANDOM, gen_data_t::GEN_TYPE_CONST);

  // Generate input data
  c2h::device_vector<item_t> in_items(num_items);
  if (data_gen_mode == gen_data_t::GEN_TYPE_RANDOM)
  {
    c2h::gen(C2H_SEED(2), in_items);
  }
  else
  {
    item_t default_constant{};
    init_default_constant(default_constant);
    thrust::fill(c2h::device_policy, in_items.begin(), in_items.end(), default_constant);
  }
  auto d_in_it = thrust::raw_pointer_cast(in_items.data());

#if TEST_TYPES != 4
  SECTION("reduce")
  {
    using op_t = cub::Sum;

    // Binary reduction operator
    auto reduction_op = unwrap_op(reference_extended_fp(d_in_it), op_t{});

    // Prepare verification data
    using accum_t = ::cuda::std::__accumulator_t<op_t, item_t, output_t>;
    output_t expected_result =
      static_cast<output_t>(compute_single_problem_reference(in_items, reduction_op, accum_t{}));

    // Run test
    c2h::device_vector<output_t> out_result(num_segments);
    auto d_out_it = thrust::raw_pointer_cast(out_result.data());
    using init_t  = cub::detail::value_t<decltype(unwrap_it(d_out_it))>;
    device_reduce(unwrap_it(d_in_it), unwrap_it(d_out_it), num_items, reduction_op, init_t{});

    // Verify result
    REQUIRE(expected_result == out_result[0]);
  }
#endif // TEST_TYPES != 4

// Skip DeviceReduce::Sum tests for extended floating-point types because of unbounded epsilon due
// to pseudo associativity of the addition operation over floating point numbers
#if TEST_TYPES != 3
  SECTION("sum")
  {
    using op_t    = cub::Sum;
    using accum_t = ::cuda::std::__accumulator_t<op_t, item_t, output_t>;

    // Prepare verification data
    output_t expected_result = static_cast<output_t>(compute_single_problem_reference(in_items, op_t{}, accum_t{}));

    // Run test
    c2h::device_vector<output_t> out_result(num_segments);
    auto d_out_it = unwrap_it(thrust::raw_pointer_cast(out_result.data()));
    device_sum(d_in_it, d_out_it, num_items);

    // Verify result
    REQUIRE(expected_result == out_result[0]);
  }
#endif

  SECTION("min")
  {
    // Prepare verification data
    c2h::host_vector<item_t> host_items(in_items);
    auto expected_result = *std::min_element(host_items.cbegin(), host_items.cend());

    // Run test
    c2h::device_vector<output_t> out_result(num_segments);
    auto d_out_it = thrust::raw_pointer_cast(out_result.data());
    device_min(unwrap_it(d_in_it), unwrap_it(d_out_it), num_items);

    // Verify result
    REQUIRE(expected_result == out_result[0]);
  }

  SECTION("max")
  {
    // Prepare verification data
    c2h::host_vector<item_t> host_items(in_items);
    auto expected_result = *std::max_element(host_items.cbegin(), host_items.cend());

    // Run test
    c2h::device_vector<output_t> out_result(num_segments);
    auto d_out_it = thrust::raw_pointer_cast(out_result.data());
    device_max(unwrap_it(d_in_it), unwrap_it(d_out_it), num_items);

    // Verify result
    REQUIRE(expected_result == out_result[0]);
  }

#if TEST_TYPES != 4
  SECTION("argmax")
  {
    // Prepare verification data
    c2h::host_vector<item_t> host_items(in_items);
    auto expected_result = std::max_element(host_items.cbegin(), host_items.cend());

    // Run test

    using result_t = cub::KeyValuePair<int, unwrap_value_t<output_t>>;
    c2h::device_vector<result_t> out_result(num_segments);
    device_arg_max(unwrap_it(d_in_it), thrust::raw_pointer_cast(out_result.data()), num_items);

    // Verify result
    result_t gpu_result = out_result[0];
    output_t gpu_value  = static_cast<output_t>(gpu_result.value); // Explicitly rewrap the gpu value
    REQUIRE(expected_result[0] == gpu_value);
    REQUIRE((expected_result - host_items.cbegin()) == gpu_result.key);
  }

  SECTION("argmin")
  {
    // Prepare verification data
    c2h::host_vector<item_t> host_items(in_items);
    auto expected_result = std::min_element(host_items.cbegin(), host_items.cend());

    // Run test
    using result_t = cub::KeyValuePair<int, unwrap_value_t<output_t>>;
    c2h::device_vector<result_t> out_result(num_segments);
    device_arg_min(unwrap_it(d_in_it), thrust::raw_pointer_cast(out_result.data()), num_items);

    // Verify result
    result_t gpu_result = out_result[0];
    output_t gpu_value  = static_cast<output_t>(gpu_result.value); // Explicitly rewrap the gpu value
    REQUIRE(expected_result[0] == gpu_value);
    REQUIRE((expected_result - host_items.cbegin()) == gpu_result.key);
  }
#endif
}
