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

#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>

#include <algorithm>

#include "catch2_test_launch_helper.h"
#include <c2h/catch2_test_helper.cuh>

template <class T>
inline T to_bound(const unsigned long long bound)
{
  return static_cast<T>(bound);
}

template <>
inline ulonglong2 to_bound(const unsigned long long bound)
{
  return {bound, bound};
}

template <>
inline ulonglong4 to_bound(const unsigned long long bound)
{
  return {bound, bound, bound, bound};
}

template <>
inline long2 to_bound(const unsigned long long bound)
{
  return {static_cast<long>(bound), static_cast<long>(bound)};
}

template <>
inline c2h::custom_type_t<c2h::equal_comparable_t> to_bound(const unsigned long long bound)
{
  c2h::custom_type_t<c2h::equal_comparable_t> val;
  val.key = bound;
  val.val = bound;
  return val;
}

template <typename HugeDataTypeT>
struct index_to_huge_type_op_t
{
  template <typename ValueType>
  __device__ __host__ HugeDataTypeT operator()(const ValueType& val)
  {
    HugeDataTypeT return_val{};
    return_val.key = val;
    return_val.val = val;
    return return_val;
  }
};

template <typename ValueT>
struct index_to_value_t
{
  template <typename IndexT>
  __host__ __device__ __forceinline__ ValueT operator()(IndexT index)
  {
    if (static_cast<std::uint64_t>(index) == 4300000000ULL)
    {
      return static_cast<ValueT>(1);
    }
    else
    {
      return static_cast<ValueT>(0);
    }
  }
};

DECLARE_LAUNCH_WRAPPER(cub::DeviceSelect::UniqueByKey, select_unique_by_key);

// %PARAM% TEST_LAUNCH lid 0:1:2

struct equal_to_default_t
{
  template <typename T>
  __host__ __device__ bool operator()(const T& a) const
  {
    return a == T{};
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
                 c2h::custom_type_t<c2h::equal_comparable_t>>;

using huge_types = c2h::type_list<c2h::custom_type_t<c2h::equal_comparable_t, c2h::huge_data<128>::type>,
                                  c2h::custom_type_t<c2h::equal_comparable_t, c2h::huge_data<256>::type>>;

using types = c2h::type_list<std::uint8_t, std::uint32_t>;

C2H_TEST("DeviceSelect::UniqueByKey can run with empty input", "[device][select_unique_by_key]", types)
{
  using type = typename c2h::get<0, TestType>;

  constexpr int num_items = 0;
  c2h::device_vector<type> empty(num_items);

  // Needs to be device accessible
  c2h::device_vector<int> num_selected_out(1, 0);
  int* d_num_selected_out = thrust::raw_pointer_cast(num_selected_out.data());

  select_unique_by_key(empty.begin(), empty.begin(), empty.begin(), empty.begin(), d_num_selected_out, num_items);

  REQUIRE(num_selected_out[0] == 0);
}

C2H_TEST("DeviceSelect::UniqueByKey handles none equal", "[device][select_unique_by_key]", types)
{
  using type = typename c2h::get<0, TestType>;

  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)));
  c2h::device_vector<type> vals_in(num_items);
  c2h::device_vector<type> vals_out(num_items);

  // Ensure we copy the right value
  c2h::gen(C2H_SEED(2), vals_in);

  // Needs to be device accessible
  c2h::device_vector<int> num_selected_out(1, 0);
  int* d_first_num_selected_out = thrust::raw_pointer_cast(num_selected_out.data());

  select_unique_by_key(
    thrust::counting_iterator<type>(0),
    vals_in.begin(),
    thrust::discard_iterator<>(),
    vals_out.begin(),
    d_first_num_selected_out,
    num_items);

  REQUIRE(num_selected_out[0] == num_items);
  REQUIRE(vals_in == vals_out);
}

C2H_TEST("DeviceSelect::UniqueByKey handles all equal", "[device][select_unique_by_key]", types)
{
  using type     = typename c2h::get<0, TestType>;
  using val_type = c2h::custom_type_t<c2h::equal_comparable_t>;

  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)));
  c2h::device_vector<type> keys_in(num_items, static_cast<type>(1));
  c2h::device_vector<val_type> vals_in(num_items);
  c2h::device_vector<type> keys_out(1);
  c2h::device_vector<val_type> vals_out(1);

  // Ensure we copy the right value
  c2h::gen(C2H_SEED(2), vals_in);

  // Needs to be device accessible
  c2h::device_vector<int> num_selected_out(1, 0);
  int* d_first_num_selected_out = thrust::raw_pointer_cast(num_selected_out.data());

  select_unique_by_key(
    keys_in.begin(), vals_in.begin(), keys_out.begin(), vals_out.begin(), d_first_num_selected_out, num_items);

  // At least one item is selected
  REQUIRE(num_selected_out[0] == 1);
  REQUIRE(keys_in[0] == keys_out[0]);
  REQUIRE(vals_in[0] == vals_out[0]);
}

C2H_TEST("DeviceSelect::UniqueByKey does not change input", "[device][select_unique_by_key]", types)
{
  using type     = typename c2h::get<0, TestType>;
  using val_type = c2h::custom_type_t<c2h::equal_comparable_t>;

  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)));
  c2h::device_vector<type> keys_in(num_items);
  c2h::device_vector<val_type> vals_in(num_items);
  c2h::gen(C2H_SEED(2), keys_in, to_bound<type>(0), to_bound<type>(42));
  c2h::gen(C2H_SEED(1), vals_in);

  // Needs to be device accessible
  c2h::device_vector<int> num_selected_out(1, 0);
  int* d_first_num_selected_out = thrust::raw_pointer_cast(num_selected_out.data());

  c2h::device_vector<type> reference_keys     = keys_in;
  c2h::device_vector<val_type> reference_vals = vals_in;

  select_unique_by_key(
    keys_in.begin(),
    vals_in.begin(),
    thrust::discard_iterator<>(),
    thrust::discard_iterator<>(),
    d_first_num_selected_out,
    num_items);

  // At least one item is selected
  REQUIRE(reference_keys == keys_in);
  REQUIRE(reference_vals == vals_in);
}

template <typename EqualityOpT>
struct project_first
{
  EqualityOpT equality_op;
  template <typename Tuple>
  __host__ __device__ bool operator()(const Tuple& lhs, const Tuple& rhs) const
  {
    return equality_op(thrust::get<0>(lhs), thrust::get<0>(rhs));
  }
};

template <typename T>
struct custom_equality_op
{
  T div_val;
  __host__ __device__ __forceinline__ bool operator()(const T& lhs, const T& rhs) const
  {
    return (lhs / div_val) == (rhs / div_val);
  }
};

C2H_TEST("DeviceSelect::UniqueByKey works with iterators", "[device][select_unique_by_key]", all_types)
{
  using type     = typename c2h::get<0, TestType>;
  using val_type = c2h::custom_type_t<c2h::equal_comparable_t>;

  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)));
  c2h::device_vector<type> keys_in(num_items);
  c2h::device_vector<val_type> vals_in(num_items);
  c2h::device_vector<type> keys_out(num_items);
  c2h::device_vector<val_type> vals_out(num_items);
  c2h::gen(C2H_SEED(2), keys_in, to_bound<type>(0), to_bound<type>(42));
  c2h::gen(C2H_SEED(1), vals_in);

  // Needs to be device accessible
  c2h::device_vector<int> num_selected_out(1, 0);
  int* d_first_num_selected_out = thrust::raw_pointer_cast(num_selected_out.data());

  select_unique_by_key(
    keys_in.begin(), vals_in.begin(), keys_out.begin(), vals_out.begin(), d_first_num_selected_out, num_items);

  // Ensure that we create the same output as std
  c2h::host_vector<type> reference_keys     = keys_in;
  c2h::host_vector<val_type> reference_vals = vals_in;
  const auto zip_begin                      = thrust::make_zip_iterator(reference_keys.begin(), reference_vals.begin());
  const auto zip_end                        = thrust::make_zip_iterator(reference_keys.end(), reference_vals.end());
  const auto boundary = std::unique(zip_begin, zip_end, project_first<cub::Equality>{cub::Equality{}});
  REQUIRE((boundary - zip_begin) == num_selected_out[0]);

  keys_out.resize(num_selected_out[0]);
  vals_out.resize(num_selected_out[0]);
  reference_keys.resize(num_selected_out[0]);
  reference_vals.resize(num_selected_out[0]);
  REQUIRE(reference_keys == keys_out);
  REQUIRE(reference_vals == vals_out);
}

C2H_TEST("DeviceSelect::UniqueByKey works with pointers", "[device][select_unique_by_key]", types)
{
  using type     = typename c2h::get<0, TestType>;
  using val_type = c2h::custom_type_t<c2h::equal_comparable_t>;

  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)));
  c2h::device_vector<type> keys_in(num_items);
  c2h::device_vector<val_type> vals_in(num_items);
  c2h::device_vector<type> keys_out(num_items);
  c2h::device_vector<val_type> vals_out(num_items);
  c2h::gen(C2H_SEED(2), keys_in, to_bound<type>(0), to_bound<type>(42));
  c2h::gen(C2H_SEED(1), vals_in);

  // Needs to be device accessible
  c2h::device_vector<int> num_selected_out(1, 0);
  int* d_first_num_selected_out = thrust::raw_pointer_cast(num_selected_out.data());

  select_unique_by_key(
    thrust::raw_pointer_cast(keys_in.data()),
    thrust::raw_pointer_cast(vals_in.data()),
    thrust::raw_pointer_cast(keys_out.data()),
    thrust::raw_pointer_cast(vals_out.data()),
    d_first_num_selected_out,
    num_items);

  // Ensure that we create the same output as std
  c2h::host_vector<type> reference_keys     = keys_in;
  c2h::host_vector<val_type> reference_vals = vals_in;
  const auto zip_begin                      = thrust::make_zip_iterator(reference_keys.begin(), reference_vals.begin());
  const auto zip_end                        = thrust::make_zip_iterator(reference_keys.end(), reference_vals.end());
  const auto boundary = std::unique(zip_begin, zip_end, project_first<cub::Equality>{cub::Equality{}});
  REQUIRE((boundary - zip_begin) == num_selected_out[0]);

  keys_out.resize(num_selected_out[0]);
  vals_out.resize(num_selected_out[0]);
  reference_keys.resize(num_selected_out[0]);
  reference_vals.resize(num_selected_out[0]);
  REQUIRE(reference_keys == keys_out);
  REQUIRE(reference_vals == vals_out);
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

C2H_TEST("DeviceSelect::UniqueByKey works with a different output type", "[device][select_unique_by_key]", types)
{
  using type     = typename c2h::get<0, TestType>;
  using val_type = c2h::custom_type_t<c2h::equal_comparable_t>;

  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)));
  c2h::device_vector<type> keys_in(num_items);
  c2h::device_vector<val_type> vals_in(num_items);
  c2h::device_vector<type> keys_out(num_items);
  c2h::device_vector<convertible_from_T<val_type>> vals_out(num_items);
  c2h::gen(C2H_SEED(2), keys_in, to_bound<type>(0), to_bound<type>(42));
  c2h::gen(C2H_SEED(1), vals_in);

  // Needs to be device accessible
  c2h::device_vector<int> num_selected_out(1, 0);
  int* d_first_num_selected_out = thrust::raw_pointer_cast(num_selected_out.data());

  select_unique_by_key(
    keys_in.begin(), vals_in.begin(), keys_out.begin(), vals_out.begin(), d_first_num_selected_out, num_items);

  // Ensure that we create the same output as std
  c2h::host_vector<type> reference_keys     = keys_in;
  c2h::host_vector<val_type> reference_vals = vals_in;
  const auto zip_begin                      = thrust::make_zip_iterator(reference_keys.begin(), reference_vals.begin());
  const auto zip_end                        = thrust::make_zip_iterator(reference_keys.end(), reference_vals.end());
  const auto boundary = std::unique(zip_begin, zip_end, project_first<cub::Equality>{cub::Equality{}});
  REQUIRE((boundary - zip_begin) == num_selected_out[0]);

  keys_out.resize(num_selected_out[0]);
  vals_out.resize(num_selected_out[0]);
  reference_keys.resize(num_selected_out[0]);
  reference_vals.resize(num_selected_out[0]);
  REQUIRE(reference_keys == keys_out);
  REQUIRE(reference_vals == vals_out);
}

C2H_TEST("DeviceSelect::UniqueByKey works and uses vsmem for large types",
         "[device][select_unique_by_key][vsmem]",
         huge_types)
{
  using type     = std::uint32_t;
  using val_type = typename c2h::get<0, TestType>;

  const int num_items = GENERATE_COPY(take(2, random(1, 100000)));
  c2h::device_vector<type> keys_in(num_items);
  c2h::device_vector<type> keys_out(num_items);
  c2h::device_vector<val_type> vals_out(num_items);
  c2h::gen(C2H_SEED(2), keys_in, to_bound<type>(0), to_bound<type>(42));

  auto vals_it =
    thrust::make_transform_iterator(thrust::make_counting_iterator(0U), index_to_huge_type_op_t<val_type>{});

  // Needs to be device accessible
  c2h::device_vector<int> num_selected_out(1, 0);
  int* d_first_num_selected_out = thrust::raw_pointer_cast(num_selected_out.data());

  select_unique_by_key(
    thrust::raw_pointer_cast(keys_in.data()),
    vals_it,
    thrust::raw_pointer_cast(keys_out.data()),
    thrust::raw_pointer_cast(vals_out.data()),
    d_first_num_selected_out,
    num_items);

  // Ensure that we create the same output as std
  c2h::host_vector<type> reference_keys = keys_in;
  c2h::host_vector<val_type> reference_vals(num_items);
  thrust::copy(vals_it, vals_it + num_items, reference_vals.begin());

  const auto zip_begin = thrust::make_zip_iterator(reference_keys.begin(), reference_vals.begin());
  const auto zip_end   = thrust::make_zip_iterator(reference_keys.end(), reference_vals.end());
  const auto boundary  = std::unique(zip_begin, zip_end, project_first<cub::Equality>{cub::Equality{}});
  REQUIRE((boundary - zip_begin) == num_selected_out[0]);

  keys_out.resize(num_selected_out[0]);
  vals_out.resize(num_selected_out[0]);
  reference_keys.resize(num_selected_out[0]);
  reference_vals.resize(num_selected_out[0]);
  REQUIRE(reference_keys == keys_out);
  REQUIRE(reference_vals == vals_out);
}

C2H_TEST("DeviceSelect::UniqueByKey works for very large input that need 64-bit offset types",
         "[device][select_unique_by_key]")
{
  using type       = std::int32_t;
  using index_type = std::int64_t;

  const std::size_t num_items = 4400000000ULL;
  c2h::host_vector<type> reference_keys{static_cast<type>(0), static_cast<type>(1), static_cast<type>(0)};
  c2h::host_vector<index_type> reference_values{0, 4300000000ULL, 4300000001ULL};

  auto keys_in   = thrust::make_transform_iterator(thrust::make_counting_iterator(0ULL), index_to_value_t<type>{});
  auto values_in = thrust::make_counting_iterator(0ULL);
  c2h::device_vector<type> keys_out(reference_keys.size());
  c2h::device_vector<index_type> values_out(reference_values.size());

  // Needs to be device accessible
  c2h::device_vector<int> num_selected_out(1, 0);
  int* d_first_num_selected_out = thrust::raw_pointer_cast(num_selected_out.data());

  // Run test
  select_unique_by_key(keys_in, values_in, keys_out.begin(), values_out.begin(), d_first_num_selected_out, num_items);

  // Ensure that we created the correct output
  REQUIRE(reference_keys.size() == static_cast<std::size_t>(num_selected_out[0]));
  REQUIRE(reference_keys == keys_out);
  REQUIRE(reference_values == values_out);
}

C2H_TEST("DeviceSelect::UniqueByKey works for very large outputs that needs 64-bit offset types",
         "[device][select_unique_by_key]")
{
  using type       = std::int32_t;
  using index_type = std::int64_t;

  constexpr std::size_t num_items = 4400000000ULL;

  auto keys_in   = thrust::make_counting_iterator(0ULL);
  auto values_in = thrust::make_counting_iterator(0ULL);

  // Needs to be device accessible
  c2h::device_vector<index_type> num_selected_out(1, 0);
  index_type* d_first_num_selected_out = thrust::raw_pointer_cast(num_selected_out.data());

  // Run test
  select_unique_by_key(
    keys_in,
    values_in,
    thrust::make_discard_iterator(),
    thrust::make_discard_iterator(),
    d_first_num_selected_out,
    num_items);

  // Ensure that we created the correct output
  REQUIRE(num_items == static_cast<std::size_t>(num_selected_out[0]));
}

C2H_TEST("DeviceSelect::UniqueByKey works with a custom equality operator", "[device][select_unique_by_key]")
{
  using type        = std::int32_t;
  using custom_op_t = custom_equality_op<type>;
  using val_type    = std::uint64_t;
  using index_type  = std::int64_t;

  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)));
  auto keys_in        = thrust::make_counting_iterator(static_cast<type>(0));
  auto values_in      = thrust::make_counting_iterator(0ULL);
  c2h::device_vector<type> keys_out(num_items);
  c2h::device_vector<val_type> vals_out(num_items);

  // Needs to be device accessible
  c2h::device_vector<index_type> num_selected_out(1, 0);
  index_type* d_first_num_selected_out = thrust::raw_pointer_cast(num_selected_out.data());

  // Run test
  select_unique_by_key(
    keys_in,
    values_in,
    keys_out.begin(),
    vals_out.begin(),
    d_first_num_selected_out,
    num_items,
    custom_op_t{static_cast<type>(8)});

  // Ensure that we create the same output as std
  c2h::host_vector<type> reference_keys(num_items);
  c2h::host_vector<val_type> reference_vals(num_items);
  thrust::copy(keys_in, keys_in + num_items, reference_keys.begin());
  thrust::copy(values_in, values_in + num_items, reference_vals.begin());
  const auto zip_begin = thrust::make_zip_iterator(reference_keys.begin(), reference_vals.begin());
  const auto zip_end   = thrust::make_zip_iterator(reference_keys.end(), reference_vals.end());
  const auto boundary  = std::unique(zip_begin, zip_end, project_first<custom_op_t>{custom_op_t{static_cast<type>(8)}});
  REQUIRE((boundary - zip_begin) == static_cast<std::ptrdiff_t>(num_selected_out[0]));

  keys_out.resize(num_selected_out[0]);
  vals_out.resize(num_selected_out[0]);
  reference_keys.resize(num_selected_out[0]);
  reference_vals.resize(num_selected_out[0]);
  REQUIRE(reference_keys == keys_out);
  REQUIRE(reference_vals == vals_out);
}
