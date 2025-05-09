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

#include <cstdint>

#include "catch2_test_device_reduce.cuh"
#include "catch2_test_launch_helper.h"
#include <c2h/catch2_test_helper.cuh>
#include <c2h/custom_type.cuh>
#include <c2h/extended_types.cuh>

DECLARE_LAUNCH_WRAPPER(cub::DeviceReduce::TransformReduce, device_transform_reduce);

// %PARAM% TEST_LAUNCH lid 0:1:2

using types = c2h::type_list<std::uint32_t, std::uint64_t>;

template <class T>
struct square_t
{
  __host__ __device__ T operator()(const T& x) const
  {
    return x * x;
  }
};

C2H_TEST("Device transform reduce works with pointers", "[reduce][device]", types)
{
  using item_t         = c2h::get<0, TestType>;
  using init_t         = item_t;
  using offset_t       = std::int32_t;
  using reduction_op_t = cub::Sum;
  using transform_op_t = square_t<item_t>;

  constexpr int max_items = 5000000;
  constexpr int min_items = 1;

  const int num_items = GENERATE_COPY(take(3, random(min_items, max_items)));

  item_t init{42};
  c2h::device_vector<item_t> out(1);
  c2h::device_vector<item_t> in(num_items + 1);
  c2h::gen(C2H_SEED(2), in);

  item_t* d_in  = thrust::raw_pointer_cast(in.data());
  item_t* d_out = thrust::raw_pointer_cast(out.data());

  const c2h::host_vector<item_t> h_in = in;
  c2h::host_vector<item_t> h_transformed_in(h_in.size() - 1);

  SECTION("when aligned")
  {
    device_transform_reduce(d_in, d_out, num_items, reduction_op_t{}, transform_op_t{}, init);

    std::transform(h_in.begin(), h_in.end() - 1, h_transformed_in.begin(), transform_op_t{});
    const item_t expected = std::accumulate(h_transformed_in.begin(), h_transformed_in.end(), init);

    INFO("num_items: " << num_items);
    REQUIRE(expected == out[0]);
  }

  SECTION("when unaligned")
  {
    device_transform_reduce(d_in + 1, d_out, num_items, reduction_op_t{}, transform_op_t{}, init);

    std::transform(h_in.begin() + 1, h_in.end(), h_transformed_in.begin(), transform_op_t{});
    const item_t expected = std::accumulate(h_transformed_in.begin(), h_transformed_in.end(), init);

    INFO("num_items: " << num_items);
    REQUIRE(expected == out[0]);
  }
}

C2H_TEST("Device transform reduce works with iterators", "[reduce][device]", types)
{
  using item_t         = c2h::get<0, TestType>;
  using init_t         = item_t;
  using offset_t       = std::int32_t;
  using reduction_op_t = cub::Sum;
  using transform_op_t = square_t<item_t>;

  constexpr int max_items = 5000000;
  constexpr int min_items = 1;

  const int num_items = GENERATE_COPY(take(3, random(min_items, max_items)));

  const item_t magic_val{2};
  c2h::device_vector<item_t> in(num_items, magic_val);
  c2h::device_vector<item_t> out(1);

  device_transform_reduce(in.begin(), out.begin(), num_items, reduction_op_t{}, transform_op_t{}, init_t{});

  const item_t expected = num_items * magic_val * magic_val;
  const item_t actual   = out[0];

  INFO("num_items: " << num_items);
  REQUIRE(expected == actual);
}

struct input_t
{
  std::uint32_t a;
  std::uint32_t b;
};

struct transformed_input_t
{
  std::uint64_t a;
  std::uint64_t b;
};

struct init_t
{
  char a;
  char b;
};

struct accum_t
{
  std::uint64_t a;
  std::uint64_t b;

  __host__ __device__ accum_t()
      : a{42}
      , b{42}
  {}

  __host__ __device__ accum_t(const transformed_input_t& other)
      : a{other.a}
      , b{other.b}
  {}

  __host__ __device__ accum_t(const init_t& other)
      : a{static_cast<std::uint64_t>(other.a)}
      , b{static_cast<std::uint64_t>(other.b)}
  {}

  __host__ __device__ accum_t& operator=(const transformed_input_t& other)
  {
    a = other.a;
    b = other.b;
    return *this;
  }
};

struct output_t
{
  std::uint64_t a;
  std::uint64_t b;

  __host__ __device__ output_t()
      : a{42}
      , b{42}
  {}

  __host__ __device__ output_t(const accum_t& other)
      : a{other.a}
      , b{other.b}
  {}

  __host__ __device__ output_t(const init_t& other)
      : a{static_cast<std::uint64_t>(other.a)}
      , b{static_cast<std::uint64_t>(other.b)}
  {}
};

struct transform_op_t
{
  __host__ __device__ transformed_input_t operator()(const input_t& x) const
  {
    return {static_cast<std::uint64_t>(x.a * x.a), static_cast<std::uint64_t>(x.b * x.b)};
  }
};

struct reduction_op_t
{
  __host__ __device__ accum_t operator()(accum_t x, accum_t y) const
  {
    accum_t result{};
    result.a = x.a + y.a;
    result.b = x.b + y.b;
    return result;
  }
};

C2H_TEST("Device transform reduce doesn't let input type into reduction op", "[reduce][device]")
{
  constexpr int max_items = 5000000;
  constexpr int min_items = 1;

  const int num_items = GENERATE_COPY(take(3, random(min_items, max_items)));

  const init_t init{3, 3};
  const input_t magic_val{2, 2};

  c2h::device_vector<input_t> in(num_items, magic_val);
  c2h::device_vector<output_t> out(1);

  input_t* d_in   = thrust::raw_pointer_cast(in.data());
  output_t* d_out = thrust::raw_pointer_cast(out.data());

  device_transform_reduce(d_in, d_out, num_items, reduction_op_t{}, transform_op_t{}, init);

  const std::uint64_t expected = num_items * magic_val.a * magic_val.a + init.a;
  const output_t actual        = out[0];

  INFO("num_items: " << num_items);
  REQUIRE(expected == actual.a);
  REQUIRE(expected == actual.b);
}
