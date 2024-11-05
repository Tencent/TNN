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

#include <cub/device/device_select.cuh>

#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/logical.h>

#include <algorithm>

#include "catch2_test_launch_helper.h"
#include <c2h/catch2_test_helper.cuh>

template <typename PredOpT>
struct predicate_op_wrapper_t
{
  PredOpT if_pred;
  template <typename FlagT, typename ItemT>
  __host__ __device__ bool operator()(thrust::tuple<FlagT, ItemT> tuple) const
  {
    const auto flag = thrust::get<0>(tuple);
    return static_cast<bool>(if_pred(flag));
  }
};

template <class T, class FlagT, class Pred>
static c2h::host_vector<T>
get_reference(c2h::device_vector<T> const& in, c2h::device_vector<FlagT> const& flags, Pred if_predicate)
{
  c2h::host_vector<T> reference   = in;
  c2h::host_vector<FlagT> h_flags = flags;
  // Zips flags and items
  auto zipped_in_it = thrust::make_zip_iterator(h_flags.cbegin(), reference.cbegin());

  // Discards the flags part and only keeps the items
  auto zipped_out_it = thrust::make_zip_iterator(thrust::make_discard_iterator(), reference.begin());

  auto end =
    std::copy_if(zipped_in_it, zipped_in_it + in.size(), zipped_out_it, predicate_op_wrapper_t<Pred>{if_predicate});
  reference.resize(thrust::distance(zipped_out_it, end));
  return reference;
}

DECLARE_LAUNCH_WRAPPER(cub::DeviceSelect::FlaggedIf, select_flagged_if);

// %PARAM% TEST_LAUNCH lid 0:1:2

using custom_t = c2h::custom_type_t<c2h::equal_comparable_t>;

template <typename T>
struct is_even_t
{
  __host__ __device__ bool operator()(T const& elem) const
  {
    return !(elem % 2);
  }
};

template <>
struct is_even_t<custom_t>
{
  __host__ __device__ bool operator()(custom_t elem) const
  {
    return !(elem.key % 2);
  }
};

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
  c2h::type_list<std::uint8_t, std::uint16_t, std::uint32_t, std::uint64_t, ulonglong2, ulonglong4, int, long2, custom_t>;

using types = c2h::type_list<std::uint8_t, std::uint32_t, ulonglong4, custom_t>;

using flag_types = c2h::type_list<std::uint8_t, std::uint64_t, custom_t>;

C2H_TEST("DeviceSelect::FlaggedIf can run with empty input", "[device][select_flagged_if]", types)
{
  using type = typename c2h::get<0, TestType>;

  constexpr int num_items = 0;
  c2h::device_vector<type> in(num_items);
  c2h::device_vector<type> out(num_items);
  c2h::device_vector<int> flags(num_items);

  // Needs to be device accessible
  c2h::device_vector<int> num_selected_out(1, 0);
  int* d_num_selected_out = thrust::raw_pointer_cast(num_selected_out.data());

  select_flagged_if(in.begin(), flags.begin(), out.begin(), d_num_selected_out, num_items, always_true_t{});

  REQUIRE(num_selected_out[0] == 0);
}

C2H_TEST("DeviceSelect::FlaggedIf handles all matched", "[device][select_flagged_if]", types)
{
  using type = typename c2h::get<0, TestType>;

  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)));
  c2h::device_vector<type> in(num_items);
  c2h::device_vector<type> out(num_items);
  c2h::device_vector<int> flags(num_items);
  c2h::gen(C2H_SEED(2), in);

  // Needs to be device accessible
  c2h::device_vector<int> num_selected_out(1, 0);
  int* d_first_num_selected_out = thrust::raw_pointer_cast(num_selected_out.data());

  select_flagged_if(in.begin(), flags.begin(), out.begin(), d_first_num_selected_out, num_items, always_true_t{});

  REQUIRE(num_selected_out[0] == num_items);
  REQUIRE(out == in);
}

C2H_TEST("DeviceSelect::FlaggedIf handles no matched", "[device][select_flagged_if]", types)
{
  using type = typename c2h::get<0, TestType>;

  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)));
  c2h::device_vector<type> in(num_items);
  c2h::device_vector<type> out(0);
  c2h::gen(C2H_SEED(2), in);

  c2h::device_vector<int> flags(num_items, 0);

  // Needs to be device accessible
  c2h::device_vector<int> num_selected_out(1, 0);
  int* d_first_num_selected_out = thrust::raw_pointer_cast(num_selected_out.data());

  select_flagged_if(in.begin(), flags.begin(), out.begin(), d_first_num_selected_out, num_items, always_false_t{});

  REQUIRE(num_selected_out[0] == 0);
}

C2H_TEST("DeviceSelect::FlaggedIf does not change input and is stable",
         "[device][select_flagged_if]",
         c2h::type_list<std::uint8_t, std::uint64_t>,
         flag_types)
{
  using input_type = typename c2h::get<0, TestType>;
  using flag_type  = typename c2h::get<1, TestType>;

  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)));
  c2h::device_vector<input_type> in(num_items);
  c2h::device_vector<input_type> out(num_items);
  c2h::gen(C2H_SEED(2), in);

  is_even_t<flag_type> is_even{};

  c2h::device_vector<flag_type> flags(num_items);
  c2h::gen(C2H_SEED(1), flags);
  const c2h::host_vector<input_type> reference_out = get_reference(in, flags, is_even);
  const int num_selected                           = static_cast<int>(reference_out.size());

  // Needs to be device accessible
  c2h::device_vector<int> num_selected_out(1, 0);
  int* d_num_selected_out = thrust::raw_pointer_cast(num_selected_out.data());

  // copy input first
  c2h::device_vector<input_type> reference_in = in;

  select_flagged_if(in.begin(), flags.begin(), out.begin(), d_num_selected_out, num_items, is_even);

  REQUIRE(num_selected == num_selected_out[0]);
  REQUIRE(reference_in == in);

  // Ensure that we did not overwrite other elements
  const auto boundary = out.begin() + num_selected_out[0];
  REQUIRE(thrust::all_of(c2h::device_policy, boundary, out.end(), equal_to_default_t{}));

  out.resize(num_selected_out[0]);
  REQUIRE(reference_out == out);
}

C2H_TEST("DeviceSelect::FlaggedIf works with iterators", "[device][select_if]", all_types, flag_types)
{
  using input_type = typename c2h::get<0, TestType>;
  using flag_type  = typename c2h::get<1, TestType>;

  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)));
  c2h::device_vector<input_type> in(num_items);
  c2h::device_vector<input_type> out(num_items);
  c2h::gen(C2H_SEED(2), in);

  is_even_t<flag_type> is_even{};

  c2h::device_vector<flag_type> flags(num_items);
  c2h::gen(C2H_SEED(1), flags);
  const c2h::host_vector<input_type> reference = get_reference(in, flags, is_even);
  const int num_selected                       = static_cast<int>(reference.size());

  // Needs to be device accessible
  c2h::device_vector<int> num_selected_out(1, 0);
  int* d_first_num_selected_out = thrust::raw_pointer_cast(num_selected_out.data());

  select_flagged_if(in.begin(), flags.begin(), out.begin(), d_first_num_selected_out, num_items, is_even);

  out.resize(num_selected_out[0]);
  REQUIRE(num_selected == num_selected_out[0]);
  REQUIRE(reference == out);
}

C2H_TEST("DeviceSelect::FlaggedIf works with pointers", "[device][select_flagged]", types, flag_types)
{
  using input_type = typename c2h::get<0, TestType>;
  using flag_type  = typename c2h::get<1, TestType>;

  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)));
  c2h::device_vector<input_type> in(num_items);
  c2h::device_vector<input_type> out(num_items);
  c2h::gen(C2H_SEED(2), in);

  is_even_t<flag_type> is_even{};

  c2h::device_vector<flag_type> flags(num_items);
  c2h::gen(C2H_SEED(1), flags);

  const c2h::host_vector<input_type> reference = get_reference(in, flags, is_even);
  const int num_selected                       = static_cast<int>(reference.size());

  // Needs to be device accessible
  c2h::device_vector<int> num_selected_out(1, 0);
  int* d_num_selected_out = thrust::raw_pointer_cast(num_selected_out.data());

  select_flagged_if(
    thrust::raw_pointer_cast(in.data()),
    thrust::raw_pointer_cast(flags.data()),
    thrust::raw_pointer_cast(out.data()),
    d_num_selected_out,
    num_items,
    is_even);

  out.resize(num_selected_out[0]);
  REQUIRE(num_selected == num_selected_out[0]);
  REQUIRE(reference == out);
}
