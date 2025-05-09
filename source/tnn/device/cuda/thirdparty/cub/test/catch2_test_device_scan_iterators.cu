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

#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>

#include <cstdint>

#include "catch2_test_device_reduce.cuh"
#include "catch2_test_device_scan.cuh"
#include "catch2_test_launch_helper.h"
#include <c2h/catch2_test_helper.cuh>
#include <c2h/custom_type.cuh>
#include <c2h/extended_types.cuh>

DECLARE_LAUNCH_WRAPPER(cub::DeviceScan::ExclusiveSum, device_exclusive_sum);
DECLARE_LAUNCH_WRAPPER(cub::DeviceScan::ExclusiveScan, device_exclusive_scan);
DECLARE_LAUNCH_WRAPPER(cub::DeviceScan::InclusiveSum, device_inclusive_sum);
DECLARE_LAUNCH_WRAPPER(cub::DeviceScan::InclusiveScan, device_inclusive_scan);

// %PARAM% TEST_LAUNCH lid 0:1

// List of types to test
using custom_t =
  c2h::custom_type_t<c2h::accumulateable_t,
                     c2h::equal_comparable_t,
                     c2h::lexicographical_less_comparable_t,
                     c2h::lexicographical_greater_comparable_t>;

using iterator_type_list = c2h::type_list<type_pair<std::int8_t>, type_pair<custom_t>, type_pair<uchar3>>;

C2H_TEST("Device scan works with iterators", "[scan][device]", iterator_type_list)
{
  using params   = params_t<TestType>;
  using input_t  = typename params::item_t;
  using output_t = typename params::output_t;
  using offset_t = int32_t;

  constexpr offset_t min_items = 1;
  constexpr offset_t max_items = 1000000;

  // Generate the input sizes to test for
  const offset_t num_items = GENERATE_COPY(
    take(3, random(min_items, max_items)),
    values({
      min_items,
      max_items,
    }));

  // Prepare input iterator
  input_t default_constant{};
  init_default_constant(default_constant);
  auto in_it = thrust::make_constant_iterator(default_constant);

  SECTION("inclusive sum")
  {
    using op_t    = cub::Sum;
    using accum_t = ::cuda::std::__accumulator_t<op_t, input_t, input_t>;

    // Prepare verification data
    c2h::host_vector<output_t> expected_result(num_items);
    compute_inclusive_scan_reference(in_it, in_it + num_items, expected_result.begin(), op_t{}, accum_t{});

    // Run test
    c2h::device_vector<output_t> out_result(num_items);
    auto d_out_it = thrust::raw_pointer_cast(out_result.data());
    device_inclusive_sum(in_it, d_out_it, num_items);

    // Verify result
    REQUIRE(expected_result == out_result);
  }

  SECTION("exclusive sum")
  {
    using op_t    = cub::Sum;
    using accum_t = ::cuda::std::__accumulator_t<op_t, input_t, input_t>;

    // Prepare verification data
    c2h::host_vector<output_t> expected_result(num_items);
    compute_exclusive_scan_reference(in_it, in_it + num_items, expected_result.begin(), accum_t{}, op_t{});

    // Run test
    c2h::device_vector<output_t> out_result(num_items);
    auto d_out_it = thrust::raw_pointer_cast(out_result.data());
    device_exclusive_sum(in_it, d_out_it, num_items);

    // Verify result
    REQUIRE(expected_result == out_result);
  }

  SECTION("inclusive scan")
  {
    using op_t    = cub::Min;
    using accum_t = ::cuda::std::__accumulator_t<op_t, input_t, input_t>;

    // Prepare verification data
    c2h::host_vector<output_t> expected_result(num_items);
    compute_inclusive_scan_reference(
      in_it, in_it + num_items, expected_result.begin(), op_t{}, cub::NumericTraits<accum_t>::Max());

    // Run test
    c2h::device_vector<output_t> out_result(num_items);
    auto d_out_it = thrust::raw_pointer_cast(out_result.data());
    device_inclusive_scan(in_it, d_out_it, op_t{}, num_items);

    // Verify result
    REQUIRE(expected_result == out_result);
  }

  SECTION("exclusive scan")
  {
    using op_t    = cub::Sum;
    using accum_t = ::cuda::std::__accumulator_t<op_t, input_t, input_t>;

    // Prepare verification data
    c2h::host_vector<output_t> expected_result(num_items);
    compute_exclusive_scan_reference(in_it, in_it + num_items, expected_result.begin(), accum_t{}, op_t{});

    // Run test
    c2h::device_vector<output_t> out_result(num_items);
    auto d_out_it = thrust::raw_pointer_cast(out_result.data());
    device_exclusive_scan(in_it, d_out_it, op_t{}, input_t{}, num_items);

    // Verify result
    REQUIRE(expected_result == out_result);
  }

  SECTION("exclusive scan with future-init value")
  {
    using op_t    = cub::Sum;
    using accum_t = ::cuda::std::__accumulator_t<op_t, input_t, input_t>;

    // Prepare verification data
    accum_t init_value{};
    init_default_constant(init_value);
    c2h::host_vector<output_t> expected_result(num_items);
    compute_exclusive_scan_reference(in_it, in_it + num_items, expected_result.begin(), init_value, op_t{});

    // Run test
    c2h::device_vector<output_t> out_result(num_items);
    auto d_out_it = thrust::raw_pointer_cast(out_result.data());
    using init_t  = cub::detail::value_t<decltype(unwrap_it(d_out_it))>;
    c2h::device_vector<init_t> d_initial_value(1);
    d_initial_value[0]     = static_cast<init_t>(init_value);
    auto future_init_value = cub::FutureValue<init_t>(thrust::raw_pointer_cast(d_initial_value.data()));
    device_exclusive_scan(in_it, d_out_it, op_t{}, future_init_value, num_items);

    // Verify result
    REQUIRE(expected_result == out_result);
  }
}

class custom_input_t
{
  char m_val{};

public:
  __host__ __device__ explicit custom_input_t(char val)
      : m_val(val)
  {}

  __host__ __device__ int get() const
  {
    return static_cast<int>(m_val);
  }
};

class custom_accumulator_t
{
  int m_val{0};
  int m_magic_value{42};

  __host__ __device__ custom_accumulator_t(int val)
      : m_val(val)
  {}

public:
  __host__ __device__ custom_accumulator_t() {}

  __host__ __device__ custom_accumulator_t(const custom_accumulator_t& in)
      : m_val(in.is_valid() * in.get())
      , m_magic_value(in.is_valid() * 42)
  {}

  __host__ __device__ custom_accumulator_t(const custom_input_t& in)
      : m_val(in.get())
      , m_magic_value(42)
  {}

  __host__ __device__ void operator=(const custom_input_t& in)
  {
    if (this->is_valid())
    {
      m_val = in.get();
    }
  }

  __host__ __device__ void operator=(const custom_accumulator_t& in)
  {
    if (this->is_valid() && in.is_valid())
    {
      m_val = in.get();
    }
  }

  __host__ __device__ custom_accumulator_t operator+(const custom_input_t& in) const
  {
    const int multiplier = this->is_valid();
    return {(m_val + in.get()) * multiplier};
  }

  __host__ __device__ custom_accumulator_t operator+(const custom_accumulator_t& in) const
  {
    const int multiplier = this->is_valid() && in.is_valid();
    return {(m_val + in.get()) * multiplier};
  }

  __host__ __device__ int get() const
  {
    return m_val;
  }

  __host__ __device__ bool is_valid() const
  {
    return m_magic_value == 42;
  }
};

class custom_output_t
{
  int* m_d_ok_count{};
  int m_expected{};

public:
  __host__ __device__ custom_output_t(int* d_ok_count, int expected)
      : m_d_ok_count(d_ok_count)
      , m_expected(expected)
  {}

  __device__ void operator=(const custom_accumulator_t& accum) const
  {
    const int ok = accum.is_valid() && (accum.get() == m_expected);
    atomicAdd(m_d_ok_count, ok);
  }
};

struct index_to_custom_output_op
{
  int* d_ok_count;

  __host__ __device__ __forceinline__ custom_output_t operator()(int index)
  {
    return custom_output_t{d_ok_count, index};
  }
};

C2H_TEST("Device scan works complex accumulator types", "[scan][device]")
{
  constexpr int num_items = 2 * 1024 * 1024;

  custom_accumulator_t init{};

  c2h::device_vector<custom_input_t> d_input(static_cast<size_t>(num_items), custom_input_t{1});
  c2h::device_vector<custom_output_t> d_output{static_cast<size_t>(num_items), custom_output_t{nullptr, 0}};
  c2h::device_vector<int> d_ok_count(1);

  auto index_it = thrust::make_counting_iterator(0);
  thrust::transform(
    c2h::device_policy,
    index_it,
    index_it + num_items,
    d_output.begin(),
    index_to_custom_output_op{thrust::raw_pointer_cast(d_ok_count.data())});

  auto d_in_it  = thrust::raw_pointer_cast(d_input.data());
  auto d_out_it = thrust::raw_pointer_cast(d_output.data());
  device_exclusive_scan(d_in_it, d_out_it, cub::Sum{}, init, num_items);

  REQUIRE(d_ok_count[0] == num_items);
}
