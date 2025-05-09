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
#include <thrust/iterator/discard_iterator.h>

#include <cstdint>

#include "catch2_test_device_reduce.cuh"
#include "catch2_test_launch_helper.h"
#include <c2h/catch2_test_helper.cuh>
#include <c2h/custom_type.cuh>

DECLARE_LAUNCH_WRAPPER(cub::DeviceReduce::Reduce, device_reduce);
DECLARE_LAUNCH_WRAPPER(cub::DeviceReduce::Sum, device_sum);

// %PARAM% TEST_LAUNCH lid 0:1:2

// List of types to test
using custom_t           = c2h::custom_type_t<c2h::accumulateable_t, c2h::equal_comparable_t>;
using iterator_type_list = c2h::type_list<type_pair<custom_t>, type_pair<std::int64_t>>;

/**
 * @brief Helper function to test large problem sizes, including problems requiring 64-bit offset
 * types.
 */
template <typename T, typename offset_t>
void test_big_indices_helper(offset_t num_items)
{
  thrust::constant_iterator<T> const_iter(T{1});
  c2h::device_vector<std::size_t> out(1);
  std::size_t* d_out = thrust::raw_pointer_cast(out.data());
  device_sum(const_iter, d_out, num_items);
  std::size_t result = out[0];

  REQUIRE(result == num_items);
}

C2H_TEST("Device sum works for big indices", "[reduce][device]")
{
  test_big_indices_helper<std::size_t, std::uint32_t>(1ull << 30);
  test_big_indices_helper<std::size_t, std::uint32_t>(1ull << 31);
  test_big_indices_helper<std::size_t, std::uint32_t>((1ull << 32) - 1);
  test_big_indices_helper<std::size_t, std::uint64_t>(1ull << 33);
}

C2H_TEST("Device reduce works with fancy input iterators", "[reduce][device]", iterator_type_list)
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

  // Prepare input data
  item_t default_constant{};
  init_default_constant(default_constant);
  auto in_it = thrust::make_constant_iterator(default_constant);

  using op_t   = cub::Sum;
  using init_t = output_t;

  // Binary reduction operator
  auto reduction_op = op_t{};

  // Prepare verification data
  using accum_t            = ::cuda::std::__accumulator_t<op_t, item_t, init_t>;
  output_t expected_result = compute_single_problem_reference(in_it, in_it + num_items, reduction_op, accum_t{});

  // Run test
  c2h::device_vector<output_t> out_result(num_segments);
  auto d_out_it = thrust::raw_pointer_cast(out_result.data());
  device_reduce(in_it, d_out_it, num_items, reduction_op, init_t{});

  // Verify result
  REQUIRE(expected_result == out_result[0]);
}

C2H_TEST("Device reduce compiles with discard output iterator", "[reduce][device]", iterator_type_list)
{
  using params   = params_t<TestType>;
  using item_t   = typename params::item_t;
  using output_t = typename params::output_t;

  constexpr int max_items = 5000000;
  constexpr int min_items = 1;

  // Generate the input sizes to test for
  const int num_items = GENERATE_COPY(values({
    min_items,
    max_items,
  }));

  // Prepare input data
  item_t default_constant{};
  init_default_constant(default_constant);
  auto in_it = thrust::make_constant_iterator(default_constant);

  using op_t   = cub::Sum;
  using init_t = output_t;

  // Binary reduction operator
  auto reduction_op = op_t{};

  // Run test
  device_reduce(in_it, thrust::make_discard_iterator(), num_items, reduction_op, init_t{});
}
