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

#include <cub/device/device_partition.cuh>

#include <thrust/partition.h>
#include <thrust/random.h>
#include <thrust/reduce.h>
#include <thrust/shuffle.h>
#include <thrust/tabulate.h>

#include <cuda/std/utility>

#include "catch2_test_launch_helper.h"
#include "cub/util_type.cuh"
#include <c2h/catch2_test_helper.cuh>

DECLARE_LAUNCH_WRAPPER(cub::DevicePartition::If, partition);

// %PARAM% TEST_LAUNCH lid 0:1:2

using types = c2h::type_list<std::int32_t, std::int64_t>;

template <typename T>
struct less_than_t
{
  T compare;

  explicit __host__ less_than_t(T compare)
      : compare(compare)
  {}

  __device__ bool operator()(const T& a) const
  {
    return a < compare;
  }
};

template <typename T>
struct equal_to_t
{
  T compare;

  explicit __host__ equal_to_t(T compare)
      : compare(compare)
  {}

  __device__ bool operator()(const T& a) const
  {
    return a == compare;
  }
};

template <typename T>
struct greater_or_equal_t
{
  T compare;

  explicit __host__ greater_or_equal_t(T compare)
      : compare(compare)
  {}

  __device__ bool operator()(const T& a) const
  {
    return a >= compare;
  }
};

template <typename ValueT>
struct count_to_pair_t
{
  template <typename OffsetT>
  __device__ __host__ cuda::std::pair<ValueT, std::uint32_t> operator()(OffsetT id)
  {
    return cuda::std::make_pair(static_cast<ValueT>(id), id);
  }
};

C2H_TEST("Device three-way partition can handle empty problems", "[partition][device]", types)
{
  using type = typename c2h::get<0, TestType>;

  constexpr int num_items = 0;

  type* in{};
  type* d_first_part_out{};
  type* d_second_part_out{};
  type* d_unselected_out{};
  type* d_num_selected_out{};

  less_than_t<type> le(type{0});
  greater_or_equal_t<type> ge(type{1});

  partition(in, d_first_part_out, d_second_part_out, d_unselected_out, d_num_selected_out, num_items, le, ge);
}

template <typename T>
struct three_way_partition_result_t
{
  three_way_partition_result_t() = delete;
  three_way_partition_result_t(int num_items)
      : first_part(num_items)
      , second_part(num_items)
      , unselected(num_items)
  {}

  c2h::device_vector<T> first_part;
  c2h::device_vector<T> second_part;
  c2h::device_vector<T> unselected;

  int num_items_in_first_part{};
  int num_items_in_second_part{};
  int num_unselected_items{};

  bool operator==(const three_way_partition_result_t<T>& other) const
  {
    return std::tie(num_items_in_first_part,
                    num_items_in_second_part,
                    num_unselected_items,
                    first_part,
                    second_part,
                    unselected)
        == std::tie(other.num_items_in_first_part,
                    other.num_items_in_second_part,
                    other.num_unselected_items,
                    other.first_part,
                    other.second_part,
                    other.unselected);
  }
};

template <typename FirstPartSelectionOp, typename SecondPartSelectionOp, typename T>
three_way_partition_result_t<T>
cub_partition(FirstPartSelectionOp first_selector, SecondPartSelectionOp second_selector, c2h::device_vector<T>& in)
{
  const int num_items = static_cast<int>(in.size());
  three_way_partition_result_t<T> result(num_items);

  T* d_in              = thrust::raw_pointer_cast(in.data());
  T* d_first_part_out  = thrust::raw_pointer_cast(result.first_part.data());
  T* d_second_part_out = thrust::raw_pointer_cast(result.second_part.data());
  T* d_unselected_out  = thrust::raw_pointer_cast(result.unselected.data());

  c2h::device_vector<int> num_selected_out(2);
  int* d_num_selected_out = thrust::raw_pointer_cast(num_selected_out.data());

  partition(
    d_in,
    d_first_part_out,
    d_second_part_out,
    d_unselected_out,
    d_num_selected_out,
    num_items,
    first_selector,
    second_selector);

  c2h::host_vector<int> h_num_selected_out(num_selected_out);

  result.num_items_in_first_part  = h_num_selected_out[0];
  result.num_items_in_second_part = h_num_selected_out[1];

  result.num_unselected_items = num_items - h_num_selected_out[0] - h_num_selected_out[1];

  return result;
}

template <typename FirstPartSelectionOp, typename SecondPartSelectionOp, typename T>
three_way_partition_result_t<T>
thrust_partition(FirstPartSelectionOp first_selector, SecondPartSelectionOp second_selector, c2h::device_vector<T>& in)
{
  const int num_items = static_cast<int>(in.size());
  three_way_partition_result_t<T> result(num_items);

  c2h::device_vector<T> intermediate_result(num_items);

  auto intermediate_iterators = thrust::partition_copy(
    c2h::device_policy, in.begin(), in.end(), result.first_part.begin(), intermediate_result.begin(), first_selector);

  result.num_items_in_first_part =
    static_cast<int>(thrust::distance(result.first_part.begin(), intermediate_iterators.first));

  auto final_iterators = thrust::partition_copy(
    c2h::device_policy,
    intermediate_result.begin(),
    intermediate_result.begin() + (num_items - result.num_items_in_first_part),
    result.second_part.begin(),
    result.unselected.begin(),
    second_selector);

  result.num_items_in_second_part =
    static_cast<int>(thrust::distance(result.second_part.begin(), final_iterators.first));

  result.num_unselected_items = static_cast<int>(thrust::distance(result.unselected.begin(), final_iterators.second));

  return result;
}

C2H_TEST("Device three-way partition is stable", "[partition][device]", types)
{
  using type      = typename c2h::get<0, TestType>;
  using pair_type = cuda::std::pair<type, std::uint32_t>;

  const int num_items = GENERATE_COPY(take(10, random(1, 1000000)));
  c2h::device_vector<pair_type> in(num_items);

  thrust::tabulate(c2h::device_policy, in.begin(), in.end(), count_to_pair_t<type>{});

  pair_type first_unselected_val = cuda::std::make_pair(static_cast<type>(num_items / 3), std::uint32_t{});

  pair_type first_val_of_second_part = cuda::std::make_pair(static_cast<type>(2 * num_items / 3), std::uint32_t{});

  less_than_t<pair_type> le(first_unselected_val);
  greater_or_equal_t<pair_type> ge(first_val_of_second_part);

  auto cub_result    = cub_partition(le, ge, in);
  auto thrust_result = thrust_partition(le, ge, in);

  REQUIRE(cub_result == thrust_result);
}

C2H_TEST("Device three-way partition handles empty first part", "[partition][device]", types)
{
  using type = typename c2h::get<0, TestType>;

  const int num_items = GENERATE_COPY(take(10, random(1, 1000000)));
  c2h::device_vector<type> in(num_items);
  thrust::sequence(c2h::device_policy, in.begin(), in.end());

  type first_unselected_val     = type{0};
  type first_val_of_second_part = static_cast<type>(num_items / 2);

  less_than_t<type> le(first_unselected_val);
  greater_or_equal_t<type> ge(first_val_of_second_part);

  auto cub_result    = cub_partition(le, ge, in);
  auto thrust_result = thrust_partition(le, ge, in);

  REQUIRE(cub_result == thrust_result);
  REQUIRE(cub_result.num_items_in_first_part == 0);
}

C2H_TEST("Device three-way partition handles empty second part", "[partition][device]", types)
{
  using type = typename c2h::get<0, TestType>;

  const int num_items = GENERATE_COPY(take(10, random(1, 1000000)));
  c2h::device_vector<type> in(num_items);
  thrust::sequence(c2h::device_policy, in.begin(), in.end());

  type first_unselected_val     = static_cast<type>(num_items / 2);
  type first_val_of_second_part = type{0}; // empty set for unsigned types

  less_than_t<type> le(first_unselected_val);
  greater_or_equal_t<type> ge(first_val_of_second_part);

  auto cub_result    = cub_partition(ge, le, in);
  auto thrust_result = thrust_partition(ge, le, in);

  REQUIRE(cub_result == thrust_result);
  REQUIRE(cub_result.num_items_in_second_part == 0);
}

C2H_TEST("Device three-way partition handles empty unselected part", "[partition][device]", types)
{
  using type = typename c2h::get<0, TestType>;

  const int num_items = GENERATE_COPY(take(10, random(1, 1000000)));
  c2h::device_vector<type> in(num_items);
  thrust::sequence(c2h::device_policy, in.begin(), in.end());

  type first_unselected_val = static_cast<type>(num_items / 2);

  less_than_t<type> le(first_unselected_val);
  greater_or_equal_t<type> ge(first_unselected_val);

  auto cub_result    = cub_partition(le, ge, in);
  auto thrust_result = thrust_partition(le, ge, in);

  REQUIRE(cub_result == thrust_result);
  REQUIRE(cub_result.num_unselected_items == 0);
}

C2H_TEST("Device three-way partition handles only unselected items", "[partition][device]", types)
{
  using type = typename c2h::get<0, TestType>;

  const int num_items = GENERATE_COPY(take(10, random(1, 1000000)));
  c2h::device_vector<type> in(num_items);
  thrust::sequence(c2h::device_policy, in.begin(), in.end());

  type first_unselected_val = type{0};

  less_than_t<type> le(first_unselected_val);

  auto cub_result    = cub_partition(le, le, in);
  auto thrust_result = thrust_partition(le, le, in);

  REQUIRE(cub_result == thrust_result);
  REQUIRE(cub_result.num_unselected_items == num_items);
  REQUIRE(cub_result.num_items_in_first_part == 0);
  REQUIRE(cub_result.num_items_in_second_part == 0);
}

C2H_TEST("Device three-way partition handles reverse iterator", "[partition][device]", types)
{
  using type = typename c2h::get<0, TestType>;

  const int num_items               = GENERATE_COPY(take(10, random(1, 1000000)));
  const int num_items_in_first_part = num_items / 3;
  const int num_unselected_items    = 2 * num_items / 3;

  type first_part_val{0};
  type second_part_val{1};
  type unselected_part_val{2};

  c2h::device_vector<type> in(num_items, second_part_val);
  thrust::fill_n(c2h::device_policy, in.begin(), num_items_in_first_part, first_part_val);
  thrust::fill_n(c2h::device_policy, in.begin() + num_items_in_first_part, num_unselected_items, unselected_part_val);

  thrust::shuffle(c2h::device_policy, in.begin(), in.end(), thrust::default_random_engine{});

  c2h::device_vector<type> first_and_unselected_part(num_items);

  equal_to_t<type> first_selector{first_part_val};
  equal_to_t<type> second_selector{second_part_val};

  c2h::device_vector<int> num_selected_out(2);

  partition(
    in.cbegin(),
    first_and_unselected_part.begin(),
    thrust::make_discard_iterator(),
    first_and_unselected_part.rbegin(),
    num_selected_out.begin(),
    num_items,
    first_selector,
    second_selector);

  c2h::device_vector<int> h_num_selected_out = num_selected_out;

  REQUIRE(h_num_selected_out[0] == num_items_in_first_part);

  const auto actual_num_unselected_items = thrust::count(
    c2h::device_policy,
    first_and_unselected_part.rbegin(),
    first_and_unselected_part.rbegin() + num_unselected_items,
    unselected_part_val);

  REQUIRE(actual_num_unselected_items == num_unselected_items);

  const auto actual_num_items_in_first_part = thrust::count(
    c2h::device_policy,
    first_and_unselected_part.begin(),
    first_and_unselected_part.begin() + num_items_in_first_part,
    first_part_val);

  REQUIRE(actual_num_items_in_first_part == num_items_in_first_part);
}

C2H_TEST("Device three-way partition handles single output", "[partition][device]", types)
{
  using type = typename c2h::get<0, TestType>;

  const int num_items          = GENERATE_COPY(take(10, random(1, 1000000)));
  int num_items_in_first_part  = num_items / 3;
  int num_unselected_items     = 2 * num_items / 3;
  int num_items_in_second_part = num_items - num_items_in_first_part - num_unselected_items;

  type first_part_val{0};
  type second_part_val{1};
  type unselected_part_val{2};

  c2h::device_vector<type> in(num_items, second_part_val);
  thrust::fill_n(c2h::device_policy, in.begin(), num_items_in_first_part, first_part_val);
  thrust::fill_n(c2h::device_policy, in.begin() + num_items_in_first_part, num_unselected_items, unselected_part_val);

  thrust::shuffle(c2h::device_policy, in.begin(), in.end(), thrust::default_random_engine{});

  c2h::device_vector<type> output(num_items);

  equal_to_t<type> first_selector{first_part_val};
  equal_to_t<type> second_selector{second_part_val};

  c2h::device_vector<int> num_selected_out(2);

  partition(
    in.cbegin(),
    output.begin(),
    output.begin() + num_items_in_first_part,
    output.rbegin(),
    num_selected_out.begin(),
    num_items,
    first_selector,
    second_selector);

  c2h::device_vector<int> h_num_selected_out(num_selected_out);

  REQUIRE(h_num_selected_out[0] == num_items_in_first_part);
  REQUIRE(h_num_selected_out[1] == num_items_in_second_part);

  const auto actual_num_unselected_items =
    thrust::count(c2h::device_policy, output.rbegin(), output.rbegin() + num_unselected_items, unselected_part_val);
  REQUIRE(actual_num_unselected_items == num_unselected_items);

  const auto actual_num_items_in_first_part =
    thrust::count(c2h::device_policy, output.begin(), output.begin() + num_items_in_first_part, first_part_val);
  REQUIRE(actual_num_items_in_first_part == num_items_in_first_part);

  const auto actual_num_items_in_second_part = thrust::count(
    c2h::device_policy,
    output.begin() + num_items_in_first_part,
    output.begin() + num_items_in_first_part + num_items_in_second_part,
    second_part_val);
  REQUIRE(actual_num_items_in_second_part == num_items_in_second_part);
}
