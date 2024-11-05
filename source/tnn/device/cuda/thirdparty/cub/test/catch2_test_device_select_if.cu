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

#include <cub/device/device_select.cuh>
#include <cub/device/dispatch/dispatch_select_if.cuh>

#include <thrust/distance.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/logical.h>
#include <thrust/partition.h>
#include <thrust/reverse.h>

#include <cuda/std/limits>

#include <algorithm>

#include "catch2_test_device_select_common.cuh"
#include "catch2_test_launch_helper.h"
#include <c2h/catch2_test_helper.cuh>

DECLARE_LAUNCH_WRAPPER(cub::DeviceSelect::If, select_if);

// %PARAM% TEST_LAUNCH lid 0:1:2

struct equal_to_default_t
{
  template <typename T>
  __host__ __device__ bool operator()(const T& a) const
  {
    return a == T{};
  }
};

struct always_false_t
{
  template <typename T>
  __device__ bool operator()(const T&) const
  {
    return false;
  }
};

struct always_true_t
{
  template <typename T>
  __device__ bool operator()(const T&) const
  {
    return true;
  }
};

using all_types =
  c2h::type_list<std::uint8_t,
                 std::uint16_t,
                 std::uint32_t,
                 std::uint64_t,
                 ulonglong2,
                 ulonglong4,
                 int,
                 long2,
                 c2h::custom_type_t<c2h::less_comparable_t, c2h::equal_comparable_t>>;

using types = c2h::
  type_list<std::uint8_t, std::uint32_t, ulonglong4, c2h::custom_type_t<c2h::less_comparable_t, c2h::equal_comparable_t>>;

C2H_TEST("DeviceSelect::If can run with empty input", "[device][select_if]", types)
{
  using type = typename c2h::get<0, TestType>;

  constexpr int num_items = 0;
  c2h::device_vector<type> in(num_items);
  c2h::device_vector<type> out(num_items);

  // Needs to be device accessible
  c2h::device_vector<int> num_selected_out(1, 42);
  int* d_num_selected_out = thrust::raw_pointer_cast(num_selected_out.data());

  select_if(in.begin(), out.begin(), d_num_selected_out, num_items, always_true_t{});

  REQUIRE(num_selected_out[0] == 0);
}

C2H_TEST("DeviceSelect::If handles all matched", "[device][select_if]", types)
{
  using type = typename c2h::get<0, TestType>;

  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)));
  c2h::device_vector<type> in(num_items);
  c2h::device_vector<type> out(num_items);
  c2h::gen(C2H_SEED(2), in);

  // Needs to be device accessible
  c2h::device_vector<int> num_selected_out(1, 0);
  int* d_first_num_selected_out = thrust::raw_pointer_cast(num_selected_out.data());

  select_if(in.begin(), out.begin(), d_first_num_selected_out, num_items, always_true_t{});

  REQUIRE(num_selected_out[0] == num_items);
  REQUIRE(out == in);
}

C2H_TEST("DeviceSelect::If handles no matched", "[device][select_if]", types)
{
  using type = typename c2h::get<0, TestType>;

  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)));
  c2h::device_vector<type> in(num_items);
  c2h::device_vector<type> out(0);
  c2h::gen(C2H_SEED(2), in);

  // Needs to be device accessible
  c2h::device_vector<int> num_selected_out(1, 0);
  int* d_first_num_selected_out = thrust::raw_pointer_cast(num_selected_out.data());

  select_if(in.begin(), out.begin(), d_first_num_selected_out, num_items, always_false_t{});

  REQUIRE(num_selected_out[0] == 0);
}

C2H_TEST("DeviceSelect::If does not change input", "[device][select_if]", types)
{
  using type = typename c2h::get<0, TestType>;

  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)));
  c2h::device_vector<type> in(num_items);
  c2h::device_vector<type> out(num_items);
  c2h::gen(C2H_SEED(2), in);

  // just pick one of the input elements as boundary
  less_than_t<type> le{in[num_items / 2]};

  // Needs to be device accessible
  c2h::device_vector<int> num_selected_out(1, 0);
  int* d_first_num_selected_out = thrust::raw_pointer_cast(num_selected_out.data());

  // copy input first
  c2h::device_vector<type> reference = in;

  select_if(in.begin(), out.begin(), d_first_num_selected_out, num_items, le);

  REQUIRE(reference == in);
}

C2H_TEST("DeviceSelect::If is stable", "[device][select_if]")
{
  using type = c2h::custom_type_t<c2h::less_comparable_t, c2h::equal_comparable_t>;

  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)));
  c2h::device_vector<type> in(num_items);
  c2h::device_vector<type> out(num_items);
  c2h::gen(C2H_SEED(2), in);

  // just pick one of the input elements as boundary
  less_than_t<type> le{in[num_items / 2]};

  // Needs to be device accessible
  c2h::device_vector<int> num_selected_out(1, 0);
  int* d_first_num_selected_out = thrust::raw_pointer_cast(num_selected_out.data());

  select_if(in.begin(), out.begin(), d_first_num_selected_out, num_items, le);

  // Ensure that we create the same output as std
  c2h::host_vector<type> reference = in;
  std::stable_partition(reference.begin(), reference.end(), le);

  // Ensure that we did not overwrite other elements
  const auto boundary = out.begin() + num_selected_out[0];
  REQUIRE(thrust::all_of(c2h::device_policy, boundary, out.end(), equal_to_default_t{}));

  out.resize(num_selected_out[0]);
  reference.resize(num_selected_out[0]);
  REQUIRE(reference == out);
}

C2H_TEST("DeviceSelect::If works with iterators", "[device][select_if]", all_types)
{
  using type = typename c2h::get<0, TestType>;

  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)));
  c2h::device_vector<type> in(num_items);
  c2h::device_vector<type> out(num_items);
  c2h::gen(C2H_SEED(2), in);

  // just pick one of the input elements as boundary
  less_than_t<type> le{in[num_items / 2]};

  // Needs to be device accessible
  c2h::device_vector<int> num_selected_out(1, 0);
  int* d_first_num_selected_out = thrust::raw_pointer_cast(num_selected_out.data());

  select_if(in.begin(), out.begin(), d_first_num_selected_out, num_items, le);

  const auto boundary = out.begin() + num_selected_out[0];
  REQUIRE(thrust::all_of(c2h::device_policy, out.begin(), boundary, le));
  REQUIRE(thrust::all_of(c2h::device_policy, boundary, out.end(), equal_to_default_t{}));
}

C2H_TEST("DeviceSelect::If works with pointers", "[device][select_if]", types)
{
  using type = typename c2h::get<0, TestType>;

  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)));
  c2h::device_vector<type> in(num_items);
  c2h::device_vector<type> out(num_items);
  c2h::gen(C2H_SEED(2), in);

  // just pick one of the input elements as boundary
  less_than_t<type> le{in[num_items / 2]};

  // Needs to be device accessible
  c2h::device_vector<int> num_selected_out(1, 0);
  int* d_first_num_selected_out = thrust::raw_pointer_cast(num_selected_out.data());

  select_if(
    thrust::raw_pointer_cast(in.data()), thrust::raw_pointer_cast(out.data()), d_first_num_selected_out, num_items, le);

  const auto boundary = out.begin() + num_selected_out[0];
  REQUIRE(thrust::all_of(c2h::device_policy, out.begin(), boundary, le));
  REQUIRE(thrust::all_of(c2h::device_policy, boundary, out.end(), equal_to_default_t{}));
}

C2H_TEST("DeviceSelect::If works in place", "[device][select_if]", types)
{
  using type = typename c2h::get<0, TestType>;

  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)));
  c2h::device_vector<type> in(num_items);
  c2h::gen(C2H_SEED(2), in);

  // just pick one of the input elements as boundary
  less_than_t<type> le{in[num_items / 2]};

  // Needs to be device accessible
  c2h::device_vector<int> num_selected_out(1, 0);
  int* d_first_num_selected_out = thrust::raw_pointer_cast(num_selected_out.data());

  // Ensure that we create the same output as std
  c2h::host_vector<type> reference = in;
  std::stable_partition(reference.begin(), reference.end(), le);

  select_if(in.begin(), d_first_num_selected_out, num_items, le);

  in.resize(num_selected_out[0]);
  reference.resize(num_selected_out[0]);
  REQUIRE(reference == in);
}

template <class T>
struct convertible_from_T
{
  T val_;

  convertible_from_T() = default;
  __host__ __device__ convertible_from_T(const T& val) noexcept
      : val_(val)
  {}
  __host__ __device__ convertible_from_T& operator=(const T& val) noexcept
  {
    val_ = val;
  }
  // Converting back to T helps satisfy all the machinery that T supports
  __host__ __device__ operator T() const noexcept
  {
    return val_;
  }
};

C2H_TEST("DeviceSelect::If works with a different output type", "[device][select_if]")
{
  using type = c2h::custom_type_t<c2h::less_comparable_t, c2h::equal_comparable_t>;

  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)));
  c2h::device_vector<type> in(num_items);
  c2h::device_vector<convertible_from_T<type>> out(num_items);
  c2h::gen(C2H_SEED(2), in);

  // just pick one of the input elements as boundary
  less_than_t<type> le{in[num_items / 2]};

  // Needs to be device accessible
  c2h::device_vector<int> num_selected_out(1, 0);
  int* d_first_num_selected_out = thrust::raw_pointer_cast(num_selected_out.data());

  select_if(in.begin(), out.begin(), d_first_num_selected_out, num_items, le);

  const auto boundary = out.begin() + num_selected_out[0];
  REQUIRE(thrust::all_of(c2h::device_policy, out.begin(), boundary, le));
  REQUIRE(thrust::all_of(c2h::device_policy, boundary, out.end(), equal_to_default_t{}));
}

C2H_TEST("DeviceSelect::If works for very large number of items", "[device][select_if]")
try
{
  using type     = std::int64_t;
  using offset_t = std::int64_t;

  // The partition size (the maximum number of items processed by a single kernel invocation) is an important boundary
  constexpr auto max_partition_size = static_cast<offset_t>(::cuda::std::numeric_limits<std::int32_t>::max());

  offset_t num_items = GENERATE_COPY(
    values({
      offset_t{2} * max_partition_size + offset_t{20000000}, // 3 partitions
      offset_t{2} * max_partition_size, // 2 partitions
      max_partition_size + offset_t{1}, // 2 partitions
      max_partition_size, // 1 partitions
      max_partition_size - offset_t{1} // 1 partitions
    }),
    take(2, random(max_partition_size - offset_t{1000000}, max_partition_size + offset_t{1000000})));

  // Input
  auto in = thrust::make_counting_iterator(static_cast<type>(0));

  // Needs to be device accessible
  c2h::device_vector<offset_t> num_selected_out(1, 0);
  offset_t* d_first_num_selected_out = thrust::raw_pointer_cast(num_selected_out.data());

  // Run test
  constexpr offset_t match_every_nth = 1000000;
  offset_t expected_num_copied       = (num_items + match_every_nth - offset_t{1}) / match_every_nth;
  c2h::device_vector<type> out(expected_num_copied);
  select_if(
    in, out.begin(), d_first_num_selected_out, num_items, mod_n<offset_t>{static_cast<offset_t>(match_every_nth)});

  // Ensure that we created the correct output
  REQUIRE(num_selected_out[0] == expected_num_copied);
  auto expected_out_it =
    thrust::make_transform_iterator(in, multiply_n<offset_t>{static_cast<offset_t>(match_every_nth)});
  bool all_results_correct = thrust::equal(out.cbegin(), out.cend(), expected_out_it);
  REQUIRE(all_results_correct == true);
}
catch (std::bad_alloc&)
{
  // Exceeding memory is not a failure.
}

C2H_TEST("DeviceSelect::If works for very large number of output items", "[device][select_if]")
try
{
  using type     = std::uint8_t;
  using offset_t = std::int64_t;

  // The partition size (the maximum number of items processed by a single kernel invocation) is an important boundary
  constexpr auto max_partition_size = static_cast<offset_t>(::cuda::std::numeric_limits<std::int32_t>::max());

  offset_t num_items = GENERATE_COPY(
    values({
      offset_t{2} * max_partition_size + offset_t{20000000}, // 3 partitions
      offset_t{2} * max_partition_size, // 2 partitions
      max_partition_size + offset_t{1}, // 2 partitions
      max_partition_size, // 1 partitions
      max_partition_size - offset_t{1} // 1 partitions
    }),
    take(2, random(max_partition_size - offset_t{1000000}, max_partition_size + offset_t{1000000})));

  // Prepare input iterator: it[i] = (i%mod)+(i/div)
  static constexpr offset_t mod = 200;
  static constexpr offset_t div = 1000000000;
  auto in                       = thrust::make_transform_iterator(
    thrust::make_counting_iterator(offset_t{0}), modx_and_add_divy<offset_t, type>{mod, div});

  // Prepare output
  c2h::device_vector<type> out(num_items);

  // Needs to be device accessible
  c2h::device_vector<offset_t> num_selected_out(1, 0);
  offset_t* d_first_num_selected_out = thrust::raw_pointer_cast(num_selected_out.data());

  // Run test
  select_if(in, out.begin(), d_first_num_selected_out, num_items, always_true_t{});

  // Ensure that we created the correct output
  REQUIRE(num_selected_out[0] == num_items);
  bool all_results_correct = thrust::equal(out.cbegin(), out.cend(), in);
  REQUIRE(all_results_correct == true);
}
catch (std::bad_alloc&)
{
  // Exceeding memory is not a failure.
}
