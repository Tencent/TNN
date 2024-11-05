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

#include <cub/device/device_radix_sort.cuh>

#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/device_vector.h> // for examples
#include <thrust/gather.h>
#include <thrust/reverse.h>
#include <thrust/sequence.h>

#include <algorithm>
#include <bitset>
#include <climits>
#include <limits>

#include "catch2_radix_sort_helper.cuh"
#include "catch2_test_launch_helper.h"
#include "cub/util_type.cuh"
#include <c2h/catch2_test_helper.cuh>

DECLARE_LAUNCH_WRAPPER(cub::DeviceRadixSort::SortKeys, sort_keys);
DECLARE_LAUNCH_WRAPPER(cub::DeviceRadixSort::SortPairs, sort_pairs);
DECLARE_LAUNCH_WRAPPER(cub::DeviceRadixSort::SortKeysDescending, sort_keys_descending);
DECLARE_LAUNCH_WRAPPER(cub::DeviceRadixSort::SortPairsDescending, sort_pairs_descending);

// %PARAM% TEST_LAUNCH lid 0:1

using key   = c2h::custom_type_t<c2h::equal_comparable_t,
                                 c2h::lexicographical_less_comparable_t,
                                 c2h::lexicographical_greater_comparable_t>;
using value = std::size_t;

struct key_decomposer_t
{
  template <template <typename> class... Ps>
  __host__ __device__ ::cuda::std::tuple<std::size_t&> operator()(c2h::custom_type_t<Ps...>& key) const
  {
    return {key.key};
  }
};

struct pair_decomposer_t
{
  template <template <typename> class... Ps>
  __host__ __device__ ::cuda::std::tuple<std::size_t&, std::size_t&> operator()(c2h::custom_type_t<Ps...>& key) const
  {
    return {key.key, key.val};
  }
};

constexpr std::size_t bits_per_size_t = sizeof(std::size_t) * CHAR_BIT;
constexpr std::size_t bits_per_pair_t = bits_per_size_t * 2;

template <template <typename> class... Ps>
std::bitset<bits_per_pair_t> to_bitset(c2h::custom_type_t<Ps...>& key, int begin_bit, int end_bit)
{
  std::bitset<bits_per_pair_t> bits(key.key);
  bits <<= bits_per_size_t;
  bits |= key.val;

  for (int bit = 0; bit < begin_bit; bit++)
  {
    bits.reset(bit);
  }

  for (int bit = end_bit; bit < static_cast<int>(bits_per_pair_t); bit++)
  {
    bits.reset(bit);
  }

  return bits;
}

template <template <typename> class... Ps>
void from_bitset(std::bitset<bits_per_pair_t> bits, c2h::custom_type_t<Ps...>& pair)
{
  pair.key = (bits >> bits_per_size_t).to_ullong();
  bits <<= bits_per_size_t;
  bits >>= bits_per_size_t;
  pair.val = bits.to_ullong();
}

static c2h::host_vector<key> get_striped_keys(c2h::host_vector<key> keys, int begin_bit, int end_bit)
{
  if ((begin_bit > 0) || (end_bit < static_cast<int>(bits_per_pair_t)))
  {
    for (std::size_t i = 0; i < keys.size(); i++)
    {
      from_bitset(to_bitset(keys[i], begin_bit, end_bit), keys[i]);
    }
  }

  return keys;
}

static c2h::host_vector<std::size_t>
get_permutation(const c2h::host_vector<key>& h_keys, bool is_descending, int begin_bit, int end_bit)
{
  c2h::host_vector<key> h_striped_keys = get_striped_keys(h_keys, begin_bit, end_bit);

  c2h::host_vector<std::size_t> h_permutation(h_keys.size());
  thrust::sequence(h_permutation.begin(), h_permutation.end());

  std::stable_sort(h_permutation.begin(), h_permutation.end(), [&](std::size_t a, std::size_t b) {
    if (is_descending)
    {
      return h_striped_keys[a] > h_striped_keys[b];
    }

    return h_striped_keys[a] < h_striped_keys[b];
  });

  return h_permutation;
}

static c2h::device_vector<key>
reference_sort_keys(const c2h::device_vector<key>& d_keys, bool is_descending, int begin_bit, int end_bit)
{
  c2h::host_vector<key> h_keys(d_keys);
  c2h::host_vector<std::size_t> h_permutation = get_permutation(h_keys, is_descending, begin_bit, end_bit);
  c2h::host_vector<key> result(d_keys.size());
  thrust::gather(h_permutation.cbegin(), h_permutation.cend(), h_keys.cbegin(), result.begin());
  return result;
}

static std::pair<c2h::device_vector<key>, c2h::device_vector<value>> reference_sort_pairs(
  const c2h::device_vector<key>& d_keys,
  const c2h::device_vector<value>& d_values,
  bool is_descending,
  int begin_bit,
  int end_bit)
{
  c2h::host_vector<key> h_keys(d_keys);
  c2h::host_vector<value> h_values(d_values);
  c2h::host_vector<std::size_t> h_permutation = get_permutation(h_keys, is_descending, begin_bit, end_bit);

  c2h::host_vector<key> result_keys(d_keys.size());
  c2h::host_vector<value> result_values(d_values.size());
  thrust::gather(h_permutation.cbegin(),
                 h_permutation.cend(),
                 thrust::make_zip_iterator(h_keys.cbegin(), h_values.cbegin()),
                 thrust::make_zip_iterator(result_keys.begin(), result_values.begin()));

  return std::make_pair(result_keys, result_values);
}

C2H_TEST("Device radix sort works with parts of custom i128_t", "[radix][sort][device]")
{
  constexpr int max_items = 1 << 18;
  const int num_items     = GENERATE_COPY(take(4, random(max_items / 2, max_items)));

  c2h::device_vector<key> in_keys(num_items);
  c2h::device_vector<key> out_keys(num_items);
  c2h::gen(C2H_SEED(10), in_keys);

  auto reference_keys = reference_sort_keys(in_keys, false, 64, 128);
  sort_keys(
    thrust::raw_pointer_cast(in_keys.data()), thrust::raw_pointer_cast(out_keys.data()), num_items, key_decomposer_t{});

  REQUIRE(reference_keys == out_keys);
}

C2H_TEST("Device radix descending sort works with custom i128_t", "[radix][sort][device]")
{
  constexpr int max_items = 1 << 18;
  const int num_items     = GENERATE_COPY(take(4, random(max_items / 2, max_items)));

  c2h::device_vector<key> in_keys(num_items);
  c2h::device_vector<key> out_keys(num_items);
  c2h::gen(C2H_SEED(10), in_keys);

  const bool is_descending = GENERATE(false, true);
  auto reference_keys      = reference_sort_keys(in_keys, is_descending, 0, 128);

  if (is_descending)
  {
    sort_keys_descending(
      thrust::raw_pointer_cast(in_keys.data()),
      thrust::raw_pointer_cast(out_keys.data()),
      num_items,
      pair_decomposer_t{});
  }
  else
  {
    sort_keys(thrust::raw_pointer_cast(in_keys.data()),
              thrust::raw_pointer_cast(out_keys.data()),
              num_items,
              pair_decomposer_t{});
  }

  REQUIRE(reference_keys == out_keys);
}

C2H_TEST("Device radix sort can sort pairs with custom i128_t keys", "[radix][sort][device]")
{
  constexpr int max_items = 1 << 18;
  const int num_items     = GENERATE_COPY(take(4, random(max_items / 2, max_items)));

  c2h::device_vector<key> in_keys(num_items);
  c2h::device_vector<key> out_keys(num_items);

  c2h::device_vector<value> in_values(num_items);
  c2h::device_vector<value> out_values(num_items);
  c2h::gen(C2H_SEED(10), in_keys);
  c2h::gen(C2H_SEED(1), in_values);

  const bool is_descending = GENERATE(false, true);
  auto reference           = reference_sort_pairs(in_keys, in_values, is_descending, 0, 128);

  if (is_descending)
  {
    sort_pairs_descending(
      thrust::raw_pointer_cast(in_keys.data()),
      thrust::raw_pointer_cast(out_keys.data()),
      thrust::raw_pointer_cast(in_values.data()),
      thrust::raw_pointer_cast(out_values.data()),
      num_items,
      pair_decomposer_t{});
  }
  else
  {
    sort_pairs(thrust::raw_pointer_cast(in_keys.data()),
               thrust::raw_pointer_cast(out_keys.data()),
               thrust::raw_pointer_cast(in_values.data()),
               thrust::raw_pointer_cast(out_values.data()),
               num_items,
               pair_decomposer_t{});
  }

  REQUIRE(reference.first == out_keys);
  REQUIRE(reference.second == out_values);
}

C2H_TEST("Device radix sort works with custom i128_t (db)", "[radix][sort][device]")
{
  constexpr int max_items = 1 << 18;
  const int num_items     = GENERATE_COPY(take(4, random(max_items / 2, max_items)));

  c2h::device_vector<key> keys_1(num_items);
  c2h::device_vector<key> keys_2(num_items);
  c2h::gen(C2H_SEED(2), keys_1);

  key* d_keys_1 = thrust::raw_pointer_cast(keys_1.data());
  key* d_keys_2 = thrust::raw_pointer_cast(keys_2.data());

  cub::DoubleBuffer<key> keys(d_keys_1, d_keys_2);

  const bool is_descending = GENERATE(false, true);
  auto reference_keys      = reference_sort_keys(keys_1, is_descending, 0, 128);

  double_buffer_sort_t action(is_descending);
  action.initialize();
  launch(action, keys, num_items, pair_decomposer_t{});

  keys.selector = action.selector();
  action.finalize();

  c2h::device_vector<key>& out_keys = keys.Current() == d_keys_1 ? keys_1 : keys_2;

  REQUIRE(reference_keys == out_keys);
}

C2H_TEST("Device radix sort works with custom i128_t keys (db)", "[radix][sort][device]")
{
  constexpr int max_items = 1 << 18;
  const int num_items     = GENERATE_COPY(take(4, random(max_items / 2, max_items)));

  c2h::device_vector<key> keys_1(num_items);
  c2h::device_vector<key> keys_2(num_items);
  c2h::gen(C2H_SEED(2), keys_1);

  c2h::device_vector<value> values_1(num_items);
  c2h::device_vector<value> values_2(num_items);
  c2h::gen(C2H_SEED(1), values_1);

  key* d_keys_1 = thrust::raw_pointer_cast(keys_1.data());
  key* d_keys_2 = thrust::raw_pointer_cast(keys_2.data());

  value* d_values_1 = thrust::raw_pointer_cast(values_1.data());
  value* d_values_2 = thrust::raw_pointer_cast(values_2.data());

  cub::DoubleBuffer<key> keys(d_keys_1, d_keys_2);
  cub::DoubleBuffer<value> values(d_values_1, d_values_2);

  const bool is_descending = GENERATE(false, true);

  auto reference_keys = reference_sort_pairs(keys_1, values_1, is_descending, 0, 128);

  double_buffer_sort_t action(is_descending);
  action.initialize();
  launch(action, keys, values, num_items, pair_decomposer_t{});

  keys.selector   = action.selector();
  values.selector = action.selector();
  action.finalize();

  c2h::device_vector<key>& out_keys     = keys.Current() == d_keys_1 ? keys_1 : keys_2;
  c2h::device_vector<value>& out_values = values.Current() == d_values_1 ? values_1 : values_2;

  REQUIRE(reference_keys.first == out_keys);
  REQUIRE(reference_keys.second == out_values);
}

C2H_TEST("Device radix descending sort works with bits of custom i128_t", "[radix][sort][device]")
{
  constexpr int max_items = 1 << 18;
  const int num_items     = GENERATE_COPY(take(1, random(max_items / 2, max_items)));

  c2h::device_vector<key> in_keys(num_items);
  c2h::device_vector<key> out_keys(num_items);
  c2h::gen(C2H_SEED(2), in_keys);

  const int begin_bit      = GENERATE_COPY(take(4, random(0, 120)));
  const int end_bit        = GENERATE_COPY(take(4, random(begin_bit, 128)));
  const bool is_descending = GENERATE(false, true);

  auto reference_keys = reference_sort_keys(in_keys, is_descending, begin_bit, end_bit);

  if (is_descending)
  {
    sort_keys_descending(
      thrust::raw_pointer_cast(in_keys.data()),
      thrust::raw_pointer_cast(out_keys.data()),
      num_items,
      pair_decomposer_t{},
      begin_bit,
      end_bit);
  }
  else
  {
    sort_keys(thrust::raw_pointer_cast(in_keys.data()),
              thrust::raw_pointer_cast(out_keys.data()),
              num_items,
              pair_decomposer_t{},
              begin_bit,
              end_bit);
  }

  REQUIRE(reference_keys == out_keys);
}

C2H_TEST("Device radix sort can sort pairs with bits of custom i128_t keys", "[radix][sort][device]")
{
  constexpr int max_items = 1 << 18;
  const int num_items     = GENERATE_COPY(take(1, random(max_items / 2, max_items)));

  c2h::device_vector<key> in_keys(num_items);
  c2h::device_vector<key> out_keys(num_items);

  c2h::device_vector<value> in_values(num_items);
  c2h::device_vector<value> out_values(num_items);
  c2h::gen(C2H_SEED(2), in_keys);
  c2h::gen(C2H_SEED(1), in_values);

  const int begin_bit      = GENERATE_COPY(take(4, random(0, 120)));
  const int end_bit        = GENERATE_COPY(take(4, random(begin_bit, 128)));
  const bool is_descending = GENERATE(false, true);

  auto reference = reference_sort_pairs(in_keys, in_values, is_descending, begin_bit, end_bit);

  if (is_descending)
  {
    sort_pairs_descending(
      thrust::raw_pointer_cast(in_keys.data()),
      thrust::raw_pointer_cast(out_keys.data()),
      thrust::raw_pointer_cast(in_values.data()),
      thrust::raw_pointer_cast(out_values.data()),
      num_items,
      pair_decomposer_t{},
      begin_bit,
      end_bit);
  }
  else
  {
    sort_pairs(
      thrust::raw_pointer_cast(in_keys.data()),
      thrust::raw_pointer_cast(out_keys.data()),
      thrust::raw_pointer_cast(in_values.data()),
      thrust::raw_pointer_cast(out_values.data()),
      num_items,
      pair_decomposer_t{},
      begin_bit,
      end_bit);
  }

  REQUIRE(reference.first == out_keys);
  REQUIRE(reference.second == out_values);
}

C2H_TEST("Device radix sort works with bits of custom i128_t (db)", "[radix][sort][device]")
{
  constexpr int max_items = 1 << 18;
  const int num_items     = GENERATE_COPY(take(4, random(max_items / 2, max_items)));

  c2h::device_vector<key> keys_1(num_items);
  c2h::device_vector<key> keys_2(num_items);
  c2h::gen(C2H_SEED(2), keys_1);

  key* d_keys_1 = thrust::raw_pointer_cast(keys_1.data());
  key* d_keys_2 = thrust::raw_pointer_cast(keys_2.data());

  cub::DoubleBuffer<key> keys(d_keys_1, d_keys_2);

  const int begin_bit      = GENERATE_COPY(take(4, random(0, 120)));
  const int end_bit        = GENERATE_COPY(take(4, random(begin_bit, 128)));
  const bool is_descending = GENERATE(false, true);

  auto reference_keys = reference_sort_keys(keys_1, is_descending, begin_bit, end_bit);

  double_buffer_sort_t action(is_descending);
  action.initialize();
  launch(action, keys, num_items, pair_decomposer_t{}, begin_bit, end_bit);

  keys.selector = action.selector();
  action.finalize();

  c2h::device_vector<key>& out_keys = keys.Current() == d_keys_1 ? keys_1 : keys_2;

  REQUIRE(reference_keys == out_keys);
}

C2H_TEST("Device radix sort works with bits of custom i128_t keys (db)", "[radix][sort][device]")
{
  constexpr int max_items = 1 << 18;
  const int num_items     = GENERATE_COPY(take(4, random(max_items / 2, max_items)));

  c2h::device_vector<key> keys_1(num_items);
  c2h::device_vector<key> keys_2(num_items);
  c2h::gen(C2H_SEED(2), keys_1);

  c2h::device_vector<value> values_1(num_items);
  c2h::device_vector<value> values_2(num_items);
  c2h::gen(C2H_SEED(1), values_1);

  key* d_keys_1 = thrust::raw_pointer_cast(keys_1.data());
  key* d_keys_2 = thrust::raw_pointer_cast(keys_2.data());

  value* d_values_1 = thrust::raw_pointer_cast(values_1.data());
  value* d_values_2 = thrust::raw_pointer_cast(values_2.data());

  cub::DoubleBuffer<key> keys(d_keys_1, d_keys_2);
  cub::DoubleBuffer<value> values(d_values_1, d_values_2);

  const int begin_bit      = GENERATE_COPY(take(4, random(0, 120)));
  const int end_bit        = GENERATE_COPY(take(4, random(begin_bit, 128)));
  const bool is_descending = GENERATE(false, true);

  auto reference_keys = reference_sort_pairs(keys_1, values_1, is_descending, begin_bit, end_bit);

  double_buffer_sort_t action(is_descending);
  action.initialize();
  launch(action, keys, values, num_items, pair_decomposer_t{}, begin_bit, end_bit);

  keys.selector   = action.selector();
  values.selector = action.selector();
  action.finalize();

  c2h::device_vector<key>& out_keys     = keys.Current() == d_keys_1 ? keys_1 : keys_2;
  c2h::device_vector<value>& out_values = values.Current() == d_values_1 ? values_1 : values_2;

  REQUIRE(reference_keys.first == out_keys);
  REQUIRE(reference_keys.second == out_values);
}

#if TEST_LAUNCH != 1

// example-begin custom-type
struct custom_t
{
  float f;
  int unused;
  long long int lli;

  custom_t() = default;
  custom_t(float f, long long int lli)
      : f(f)
      , unused(42)
      , lli(lli)
  {}
};

struct decomposer_t
{
  __host__ __device__ ::cuda::std::tuple<float&, long long int&> operator()(custom_t& key) const
  {
    return {key.f, key.lli};
  }
};
// example-end custom-type

static __host__ std::ostream& operator<<(std::ostream& os, const custom_t& self)
{
  return os << "{ " << self.f << ", " << self.lli << " }";
}

static __host__ __device__ bool operator==(const custom_t& lhs, const custom_t& rhs)
{
  return lhs.f == rhs.f && lhs.lli == rhs.lli;
}

C2H_TEST("Device radix sort works against some corner cases", "[radix][sort][device]")
{
  SECTION("Keys")
  {
    // example-begin keys
    constexpr int num_items = 6;

    thrust::device_vector<custom_t> in = {
      {+2.5f, 4}, //
      {-2.5f, 0}, //
      {+1.1f, 3}, //
      {+0.0f, 1}, //
      {-0.0f, 2}, //
      {+3.7f, 5} //
    };

    thrust::device_vector<custom_t> out(num_items);

    const custom_t* d_in = thrust::raw_pointer_cast(in.data());
    custom_t* d_out      = thrust::raw_pointer_cast(out.data());

    // 1) Get temp storage size
    std::uint8_t* d_temp_storage{};
    std::size_t temp_storage_bytes{};

    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, decomposer_t{});

    // 2) Allocate temp storage
    thrust::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
    d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

    // 3) Sort keys
    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, decomposer_t{});

    thrust::device_vector<custom_t> expected_output = {
      {-2.5f, 0}, //
      {+0.0f, 1}, //
      {-0.0f, 2}, //
      {+1.1f, 3}, //
      {+2.5f, 4}, //
      {+3.7f, 5} //
    };
    // example-end keys

    REQUIRE(expected_output == out);
  }

  SECTION("KeysDescending")
  {
    // example-begin keys-descending
    std::uint8_t* d_temp_storage{};
    std::size_t temp_storage_bytes{};

    constexpr int num_items = 6;

    thrust::device_vector<custom_t> in = {
      {+1.1f, 2}, //
      {+2.5f, 1}, //
      {-0.0f, 4}, //
      {+0.0f, 3}, //
      {-2.5f, 5}, //
      {+3.7f, 0} //
    };

    thrust::device_vector<custom_t> out(num_items);

    const custom_t* d_in = thrust::raw_pointer_cast(in.data());
    custom_t* d_out      = thrust::raw_pointer_cast(out.data());

    cub::DeviceRadixSort::SortKeysDescending(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, decomposer_t{});

    thrust::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
    d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

    cub::DeviceRadixSort::SortKeysDescending(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, decomposer_t{});

    thrust::device_vector<custom_t> expected_output = {
      {+3.7f, 0}, //
      {+2.5f, 1}, //
      {+1.1f, 2}, //
      {-0.0f, 4}, //
      {+0.0f, 3}, //
      {-2.5f, 5} //
    };
    // example-end keys-descending

    REQUIRE(expected_output == out);
  }

  SECTION("Pairs")
  {
    // example-begin pairs
    std::uint8_t* d_temp_storage{};
    std::size_t temp_storage_bytes{};

    constexpr int num_items = 6;

    thrust::device_vector<custom_t> keys_in = {
      {+2.5f, 4}, //
      {-2.5f, 0}, //
      {+1.1f, 3}, //
      {+0.0f, 1}, //
      {-0.0f, 2}, //
      {+3.7f, 5} //
    };

    thrust::device_vector<custom_t> keys_out(num_items);

    const custom_t* d_keys_in = thrust::raw_pointer_cast(keys_in.data());
    custom_t* d_keys_out      = thrust::raw_pointer_cast(keys_out.data());

    thrust::device_vector<int> vals_in = {4, 0, 3, 1, 2, 5};
    thrust::device_vector<int> vals_out(num_items);

    const int* d_vals_in = thrust::raw_pointer_cast(vals_in.data());
    int* d_vals_out      = thrust::raw_pointer_cast(vals_out.data());

    cub::DeviceRadixSort::SortPairs(
      d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_vals_in, d_vals_out, num_items, decomposer_t{});

    thrust::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
    d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

    cub::DeviceRadixSort::SortPairs(
      d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_vals_in, d_vals_out, num_items, decomposer_t{});

    thrust::device_vector<custom_t> expected_keys = {
      {-2.5f, 0}, //
      {+0.0f, 1}, //
      {-0.0f, 2}, //
      {+1.1f, 3}, //
      {+2.5f, 4}, //
      {+3.7f, 5} //
    };

    thrust::device_vector<int> expected_vals = {0, 1, 2, 3, 4, 5};
    // example-end pairs

    REQUIRE(expected_keys == keys_out);
    REQUIRE(expected_vals == vals_out);
  }

  SECTION("PairsDescending")
  {
    // example-begin pairs-descending
    std::uint8_t* d_temp_storage{};
    std::size_t temp_storage_bytes{};

    constexpr int num_items = 6;

    thrust::device_vector<custom_t> keys_in = {
      {+1.1f, 2}, //
      {+2.5f, 1}, //
      {-0.0f, 4}, //
      {+0.0f, 3}, //
      {-2.5f, 5}, //
      {+3.7f, 0} //
    };

    thrust::device_vector<custom_t> keys_out(num_items);

    const custom_t* d_keys_in = thrust::raw_pointer_cast(keys_in.data());
    custom_t* d_keys_out      = thrust::raw_pointer_cast(keys_out.data());

    thrust::device_vector<int> vals_in = {2, 1, 4, 3, 5, 0};
    thrust::device_vector<int> vals_out(num_items);

    const int* d_vals_in = thrust::raw_pointer_cast(vals_in.data());
    int* d_vals_out      = thrust::raw_pointer_cast(vals_out.data());

    cub::DeviceRadixSort::SortPairsDescending(
      d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_vals_in, d_vals_out, num_items, decomposer_t{});

    thrust::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
    d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

    cub::DeviceRadixSort::SortPairsDescending(
      d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_vals_in, d_vals_out, num_items, decomposer_t{});

    thrust::device_vector<custom_t> expected_keys = {
      {+3.7f, 0}, //
      {+2.5f, 1}, //
      {+1.1f, 2}, //
      {-0.0f, 4}, //
      {+0.0f, 3}, //
      {-2.5f, 5} //
    };

    thrust::device_vector<int> expected_vals = {0, 1, 2, 4, 3, 5};
    // example-end pairs-descending

    REQUIRE(expected_keys == keys_out);
    REQUIRE(expected_vals == vals_out);
  }
}

C2H_TEST("Device radix sort works against some corner cases (db)", "[radix][sort][device]")
{
  SECTION("Keys")
  {
    // example-begin keys-db
    std::uint8_t* d_temp_storage{};
    std::size_t temp_storage_bytes{};

    constexpr int num_items = 6;

    thrust::device_vector<custom_t> keys_buf = {
      {+2.5f, 4}, //
      {-2.5f, 0}, //
      {+1.1f, 3}, //
      {+0.0f, 1}, //
      {-0.0f, 2}, //
      {+3.7f, 5} //
    };

    thrust::device_vector<custom_t> keys_alt_buf(num_items);

    custom_t* d_keys_buf     = thrust::raw_pointer_cast(keys_buf.data());
    custom_t* d_keys_alt_buf = thrust::raw_pointer_cast(keys_alt_buf.data());

    cub::DoubleBuffer<custom_t> d_keys(d_keys_buf, d_keys_alt_buf);

    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys, num_items, decomposer_t{});

    thrust::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
    d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys, num_items, decomposer_t{});

    thrust::device_vector<custom_t>& current = //
      d_keys.Current() == d_keys_buf ? keys_buf : keys_alt_buf;

    thrust::device_vector<custom_t> expected_output = {
      {-2.5f, 0}, //
      {+0.0f, 1}, //
      {-0.0f, 2}, //
      {+1.1f, 3}, //
      {+2.5f, 4}, //
      {+3.7f, 5} //
    };
    // example-end keys-db

    REQUIRE(expected_output == current);
  }

  SECTION("KeysDescending")
  {
    // example-begin keys-descending-db
    std::uint8_t* d_temp_storage{};
    std::size_t temp_storage_bytes{};

    constexpr int num_items = 6;

    thrust::device_vector<custom_t> keys_buf = {
      {+1.1f, 2}, //
      {+2.5f, 1}, //
      {-0.0f, 4}, //
      {+0.0f, 3}, //
      {-2.5f, 5}, //
      {+3.7f, 0} //
    };

    thrust::device_vector<custom_t> keys_alt_buf(num_items);

    custom_t* d_keys_buf     = thrust::raw_pointer_cast(keys_buf.data());
    custom_t* d_keys_alt_buf = thrust::raw_pointer_cast(keys_alt_buf.data());

    cub::DoubleBuffer<custom_t> d_keys(d_keys_buf, d_keys_alt_buf);

    cub::DeviceRadixSort::SortKeysDescending(d_temp_storage, temp_storage_bytes, d_keys, num_items, decomposer_t{});

    thrust::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
    d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

    cub::DeviceRadixSort::SortKeysDescending(d_temp_storage, temp_storage_bytes, d_keys, num_items, decomposer_t{});

    thrust::device_vector<custom_t>& current = //
      d_keys.Current() == d_keys_buf ? keys_buf : keys_alt_buf;

    thrust::device_vector<custom_t> expected_output = {
      {+3.7f, 0}, //
      {+2.5f, 1}, //
      {+1.1f, 2}, //
      {-0.0f, 4}, //
      {+0.0f, 3}, //
      {-2.5f, 5} //
    };
    // example-end keys-descending-db

    REQUIRE(expected_output == current);
  }

  SECTION("Pairs")
  {
    // example-begin pairs-db
    std::uint8_t* d_temp_storage{};
    std::size_t temp_storage_bytes{};

    constexpr int num_items = 6;

    thrust::device_vector<custom_t> keys_buf = {
      {+2.5f, 4}, //
      {-2.5f, 0}, //
      {+1.1f, 3}, //
      {+0.0f, 1}, //
      {-0.0f, 2}, //
      {+3.7f, 5} //
    };

    thrust::device_vector<custom_t> keys_alt_buf(num_items);

    custom_t* d_keys_buf     = thrust::raw_pointer_cast(keys_buf.data());
    custom_t* d_keys_alt_buf = thrust::raw_pointer_cast(keys_alt_buf.data());

    thrust::device_vector<int> vals_buf = {4, 0, 3, 1, 2, 5};
    thrust::device_vector<int> vals_alt_buf(num_items);

    int* d_vals_buf     = thrust::raw_pointer_cast(vals_buf.data());
    int* d_vals_alt_buf = thrust::raw_pointer_cast(vals_alt_buf.data());

    cub::DoubleBuffer<custom_t> d_keys(d_keys_buf, d_keys_alt_buf);
    cub::DoubleBuffer<int> d_vals(d_vals_buf, d_vals_alt_buf);

    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_vals, num_items, decomposer_t{});

    thrust::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
    d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_vals, num_items, decomposer_t{});

    thrust::device_vector<custom_t>& current_keys = //
      d_keys.Current() == d_keys_buf ? keys_buf : keys_alt_buf;

    thrust::device_vector<int>& current_vals = //
      d_vals.Current() == d_vals_buf ? vals_buf : vals_alt_buf;

    thrust::device_vector<custom_t> expected_keys = {
      {-2.5f, 0}, //
      {+0.0f, 1}, //
      {-0.0f, 2}, //
      {+1.1f, 3}, //
      {+2.5f, 4}, //
      {+3.7f, 5} //
    };

    thrust::device_vector<int> expected_vals = {0, 1, 2, 3, 4, 5};
    // example-end pairs-db

    REQUIRE(expected_keys == current_keys);
    REQUIRE(expected_vals == current_vals);
  }

  SECTION("PairsDescending")
  {
    // example-begin pairs-descending-db
    std::uint8_t* d_temp_storage{};
    std::size_t temp_storage_bytes{};

    constexpr int num_items = 6;

    thrust::device_vector<custom_t> keys_buf = {
      {+1.1f, 2}, //
      {+2.5f, 1}, //
      {-0.0f, 4}, //
      {+0.0f, 3}, //
      {-2.5f, 5}, //
      {+3.7f, 0} //
    };

    thrust::device_vector<custom_t> keys_alt_buf(num_items);

    custom_t* d_keys_buf     = thrust::raw_pointer_cast(keys_buf.data());
    custom_t* d_keys_alt_buf = thrust::raw_pointer_cast(keys_alt_buf.data());

    thrust::device_vector<int> vals_buf = {2, 1, 4, 3, 5, 0};
    thrust::device_vector<int> vals_alt_buf(num_items);

    int* d_vals_buf     = thrust::raw_pointer_cast(vals_buf.data());
    int* d_vals_alt_buf = thrust::raw_pointer_cast(vals_alt_buf.data());

    cub::DoubleBuffer<custom_t> d_keys(d_keys_buf, d_keys_alt_buf);
    cub::DoubleBuffer<int> d_vals(d_vals_buf, d_vals_alt_buf);

    cub::DeviceRadixSort::SortPairsDescending(
      d_temp_storage, temp_storage_bytes, d_keys, d_vals, num_items, decomposer_t{});

    thrust::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
    d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

    cub::DeviceRadixSort::SortPairsDescending(
      d_temp_storage, temp_storage_bytes, d_keys, d_vals, num_items, decomposer_t{});

    thrust::device_vector<custom_t>& current_keys = //
      d_keys.Current() == d_keys_buf ? keys_buf : keys_alt_buf;

    thrust::device_vector<int>& current_vals = //
      d_vals.Current() == d_vals_buf ? vals_buf : vals_alt_buf;

    thrust::device_vector<custom_t> expected_keys = {
      {+3.7f, 0}, //
      {+2.5f, 1}, //
      {+1.1f, 2}, //
      {-0.0f, 4}, //
      {+0.0f, 3}, //
      {-2.5f, 5} //
    };

    thrust::device_vector<int> expected_vals = {0, 1, 2, 4, 3, 5};
    // example-end pairs-descending-db

    REQUIRE(expected_keys == current_keys);
    REQUIRE(expected_vals == current_vals);
  }
}

C2H_TEST("Device radix sort works against some corner cases (bits)", "[radix][sort][device]")
{
  SECTION("Keys")
  {
    // example-begin keys-bits
    constexpr int num_items            = 2;
    thrust::device_vector<custom_t> in = {
      {24.2f, 1ll << 61}, //
      {42.4f, 1ll << 60} //
    };

    constexpr int begin_bit = sizeof(long long int) * 8 - 4; // 60
    constexpr int end_bit   = sizeof(long long int) * 8 + 4; // 68

    // Decomposition orders the bits as follows:
    //
    //                    <------------- fp32 -----------> <------ int64 ------>
    // decompose(in[0]) = 01000001110000011001100110011010 00100000000000...0000
    // decompose(in[1]) = 01000010001010011001100110011010 00010000000000...0000
    //                    <-----------  higher bits  /  lower bits  ----------->
    //
    // The bit subrange `[60, 68)` specifies differentiating key bits:
    //
    //                    <------------- fp32 -----------> <------ int64 ------>
    // decompose(in[0]) = xxxxxxxxxxxxxxxxxxxxxxxxxxxx1010 0010xxxxxxxxxx...xxxx
    // decompose(in[1]) = xxxxxxxxxxxxxxxxxxxxxxxxxxxx1010 0001xxxxxxxxxx...xxxx
    //                    <-----------  higher bits  /  lower bits  ----------->

    thrust::device_vector<custom_t> out(num_items);

    const custom_t* d_in = thrust::raw_pointer_cast(in.data());
    custom_t* d_out      = thrust::raw_pointer_cast(out.data());

    // 1) Get temp storage size
    std::uint8_t* d_temp_storage{};
    std::size_t temp_storage_bytes{};

    cub::DeviceRadixSort::SortKeys(
      d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, decomposer_t{}, begin_bit, end_bit);

    // 2) Allocate temp storage
    thrust::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
    d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

    // 3) Sort keys
    cub::DeviceRadixSort::SortKeys(
      d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, decomposer_t{}, begin_bit, end_bit);

    thrust::device_vector<custom_t> expected_output = {
      {42.4f, 1ll << 60}, //
      {24.2f, 1ll << 61} //
    };
    // example-end keys-bits

    REQUIRE(expected_output == out);
  }

  SECTION("KeysDescending")
  {
    // example-begin keys-descending-bits
    constexpr int num_items            = 2;
    thrust::device_vector<custom_t> in = {{42.4f, 1ll << 60}, {24.2f, 1ll << 61}};

    constexpr int begin_bit = sizeof(long long int) * 8 - 4; // 60
    constexpr int end_bit   = sizeof(long long int) * 8 + 4; // 68

    // Decomposition orders the bits as follows:
    //
    //                    <------------- fp32 -----------> <------ int64 ------>
    // decompose(in[0]) = 01000010001010011001100110011010 00010000000000...0000
    // decompose(in[1]) = 01000001110000011001100110011010 00100000000000...0000
    //                    <-----------  higher bits  /  lower bits  ----------->
    //
    // The bit subrange `[60, 68)` specifies differentiating key bits:
    //
    //                    <------------- fp32 -----------> <------ int64 ------>
    // decompose(in[0]) = xxxxxxxxxxxxxxxxxxxxxxxxxxxx1010 0001xxxxxxxxxx...xxxx
    // decompose(in[1]) = xxxxxxxxxxxxxxxxxxxxxxxxxxxx1010 0010xxxxxxxxxx...xxxx
    //                    <-----------  higher bits  /  lower bits  ----------->

    thrust::device_vector<custom_t> out(num_items);

    const custom_t* d_in = thrust::raw_pointer_cast(in.data());
    custom_t* d_out      = thrust::raw_pointer_cast(out.data());

    // 1) Get temp storage size
    std::uint8_t* d_temp_storage{};
    std::size_t temp_storage_bytes{};

    cub::DeviceRadixSort::SortKeysDescending(
      d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, decomposer_t{}, begin_bit, end_bit);

    // 2) Allocate temp storage
    thrust::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
    d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

    // 3) Sort keys
    cub::DeviceRadixSort::SortKeysDescending(
      d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, decomposer_t{}, begin_bit, end_bit);

    thrust::device_vector<custom_t> expected_output = {
      {24.2f, 1ll << 61}, //
      {42.4f, 1ll << 60} //
    };
    // example-end keys-descending-bits

    REQUIRE(expected_output == out);
  }

  SECTION("Pairs")
  {
    // example-begin pairs-bits
    constexpr int num_items                 = 2;
    thrust::device_vector<custom_t> keys_in = {
      {24.2f, 1ll << 61}, //
      {42.4f, 1ll << 60} //
    };

    thrust::device_vector<int> vals_in = {1, 0};

    constexpr int begin_bit = sizeof(long long int) * 8 - 4; // 60
    constexpr int end_bit   = sizeof(long long int) * 8 + 4; // 68

    // Decomposition orders the bits as follows:
    //
    //                    <------------- fp32 -----------> <------ int64 ------>
    // decompose(in[0]) = 01000001110000011001100110011010 00100000000000...0000
    // decompose(in[1]) = 01000010001010011001100110011010 00010000000000...0000
    //                    <-----------  higher bits  /  lower bits  ----------->
    //
    // The bit subrange `[60, 68)` specifies differentiating key bits:
    //
    //                    <------------- fp32 -----------> <------ int64 ------>
    // decompose(in[0]) = xxxxxxxxxxxxxxxxxxxxxxxxxxxx1010 0010xxxxxxxxxx...xxxx
    // decompose(in[1]) = xxxxxxxxxxxxxxxxxxxxxxxxxxxx1010 0001xxxxxxxxxx...xxxx
    //                    <-----------  higher bits  /  lower bits  ----------->

    thrust::device_vector<custom_t> keys_out(num_items);
    thrust::device_vector<int> vals_out(num_items);

    const custom_t* d_keys_in = thrust::raw_pointer_cast(keys_in.data());
    custom_t* d_keys_out      = thrust::raw_pointer_cast(keys_out.data());
    const int* d_vals_in      = thrust::raw_pointer_cast(vals_in.data());
    int* d_vals_out           = thrust::raw_pointer_cast(vals_out.data());

    // 1) Get temp storage size
    std::uint8_t* d_temp_storage{};
    std::size_t temp_storage_bytes{};

    cub::DeviceRadixSort::SortPairs(
      d_temp_storage,
      temp_storage_bytes,
      d_keys_in,
      d_keys_out,
      d_vals_in,
      d_vals_out,
      num_items,
      decomposer_t{},
      begin_bit,
      end_bit);

    // 2) Allocate temp storage
    thrust::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
    d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

    // 3) Sort keys
    cub::DeviceRadixSort::SortPairs(
      d_temp_storage,
      temp_storage_bytes,
      d_keys_in,
      d_keys_out,
      d_vals_in,
      d_vals_out,
      num_items,
      decomposer_t{},
      begin_bit,
      end_bit);

    thrust::device_vector<custom_t> expected_keys = {
      {42.4f, 1ll << 60}, //
      {24.2f, 1ll << 61} //
    };

    thrust::device_vector<int> expected_vals = {0, 1};
    // example-end pairs-bits

    REQUIRE(expected_keys == keys_out);
    REQUIRE(expected_vals == vals_out);
  }

  SECTION("PairsDescending")
  {
    // example-begin pairs-descending-bits
    constexpr int num_items                 = 2;
    thrust::device_vector<custom_t> keys_in = {
      {42.4f, 1ll << 60}, //
      {24.2f, 1ll << 61} //
    };

    thrust::device_vector<int> vals_in = {1, 0};

    constexpr int begin_bit = sizeof(long long int) * 8 - 4; // 60
    constexpr int end_bit   = sizeof(long long int) * 8 + 4; // 68

    // Decomposition orders the bits as follows:
    //
    //                    <------------- fp32 -----------> <------ int64 ------>
    // decompose(in[0]) = 01000010001010011001100110011010 00010000000000...0000
    // decompose(in[1]) = 01000001110000011001100110011010 00100000000000...0000
    //                    <-----------  higher bits  /  lower bits  ----------->
    //
    // The bit subrange `[60, 68)` specifies differentiating key bits:
    //
    //                    <------------- fp32 -----------> <------ int64 ------>
    // decompose(in[0]) = xxxxxxxxxxxxxxxxxxxxxxxxxxxx1010 0001xxxxxxxxxx...xxxx
    // decompose(in[1]) = xxxxxxxxxxxxxxxxxxxxxxxxxxxx1010 0010xxxxxxxxxx...xxxx
    //                    <-----------  higher bits  /  lower bits  ----------->

    thrust::device_vector<custom_t> keys_out(num_items);
    thrust::device_vector<int> vals_out(num_items);

    const custom_t* d_keys_in = thrust::raw_pointer_cast(keys_in.data());
    custom_t* d_keys_out      = thrust::raw_pointer_cast(keys_out.data());
    const int* d_vals_in      = thrust::raw_pointer_cast(vals_in.data());
    int* d_vals_out           = thrust::raw_pointer_cast(vals_out.data());

    // 1) Get temp storage size
    std::uint8_t* d_temp_storage{};
    std::size_t temp_storage_bytes{};

    cub::DeviceRadixSort::SortPairsDescending(
      d_temp_storage,
      temp_storage_bytes,
      d_keys_in,
      d_keys_out,
      d_vals_in,
      d_vals_out,
      num_items,
      decomposer_t{},
      begin_bit,
      end_bit);

    // 2) Allocate temp storage
    thrust::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
    d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

    // 3) Sort keys
    cub::DeviceRadixSort::SortPairsDescending(
      d_temp_storage,
      temp_storage_bytes,
      d_keys_in,
      d_keys_out,
      d_vals_in,
      d_vals_out,
      num_items,
      decomposer_t{},
      begin_bit,
      end_bit);

    thrust::device_vector<custom_t> expected_keys = {
      {24.2f, 1ll << 61}, //
      {42.4f, 1ll << 60} //
    };

    thrust::device_vector<int> expected_vals = {0, 1};
    // example-end pairs-descending-bits

    REQUIRE(expected_keys == keys_out);
    REQUIRE(expected_vals == vals_out);
  }
}

C2H_TEST("Device radix sort works against some corner cases (bits) (db)", "[radix][sort][device]")
{
  SECTION("Keys")
  {
    // example-begin keys-bits-db
    constexpr int num_items = 2;

    thrust::device_vector<custom_t> keys_buf = {
      {24.2f, 1ll << 61}, //
      {42.4f, 1ll << 60} //
    };

    constexpr int begin_bit = sizeof(long long int) * 8 - 4; // 60
    constexpr int end_bit   = sizeof(long long int) * 8 + 4; // 68

    // Decomposition orders the bits as follows:
    //
    //                    <------------- fp32 -----------> <------ int64 ------>
    // decompose(in[0]) = 01000001110000011001100110011010 00100000000000...0000
    // decompose(in[1]) = 01000010001010011001100110011010 00010000000000...0000
    //                    <-----------  higher bits  /  lower bits  ----------->
    //
    // The bit subrange `[60, 68)` specifies differentiating key bits:
    //
    //                    <------------- fp32 -----------> <------ int64 ------>
    // decompose(in[0]) = xxxxxxxxxxxxxxxxxxxxxxxxxxxx1010 0010xxxxxxxxxx...xxxx
    // decompose(in[1]) = xxxxxxxxxxxxxxxxxxxxxxxxxxxx1010 0001xxxxxxxxxx...xxxx
    //                    <-----------  higher bits  /  lower bits  ----------->

    thrust::device_vector<custom_t> keys_alt_buf(num_items);

    custom_t* d_keys_buf     = thrust::raw_pointer_cast(keys_buf.data());
    custom_t* d_keys_alt_buf = thrust::raw_pointer_cast(keys_alt_buf.data());

    cub::DoubleBuffer<custom_t> d_keys(d_keys_buf, d_keys_alt_buf);

    // 1) Get temp storage size
    std::uint8_t* d_temp_storage{};
    std::size_t temp_storage_bytes{};

    cub::DeviceRadixSort::SortKeys(
      d_temp_storage, temp_storage_bytes, d_keys, num_items, decomposer_t{}, begin_bit, end_bit);

    // 2) Allocate temp storage
    thrust::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
    d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

    // 3) Sort keys
    cub::DeviceRadixSort::SortKeys(
      d_temp_storage, temp_storage_bytes, d_keys, num_items, decomposer_t{}, begin_bit, end_bit);

    thrust::device_vector<custom_t>& current_keys = //
      d_keys.Current() == d_keys_buf ? keys_buf : keys_alt_buf;

    thrust::device_vector<custom_t> expected_output = {
      {42.4f, 1ll << 60}, //
      {24.2f, 1ll << 61} //
    };
    // example-end keys-bits-db

    REQUIRE(expected_output == current_keys);
  }

  SECTION("KeysDescending")
  {
    // example-begin keys-descending-bits-db
    constexpr int num_items                  = 2;
    thrust::device_vector<custom_t> keys_buf = {
      //
      {42.4f, 1ll << 60}, //
      {24.2f, 1ll << 61} //
    };

    constexpr int begin_bit = sizeof(long long int) * 8 - 4; // 60
    constexpr int end_bit   = sizeof(long long int) * 8 + 4; // 68

    // Decomposition orders the bits as follows:
    //
    //                    <------------- fp32 -----------> <------ int64 ------>
    // decompose(in[0]) = 01000010001010011001100110011010 00010000000000...0000
    // decompose(in[1]) = 01000001110000011001100110011010 00100000000000...0000
    //                    <-----------  higher bits  /  lower bits  ----------->
    //
    // The bit subrange `[60, 68)` specifies differentiating key bits:
    //
    //                    <------------- fp32 -----------> <------ int64 ------>
    // decompose(in[0]) = xxxxxxxxxxxxxxxxxxxxxxxxxxxx1010 0001xxxxxxxxxx...xxxx
    // decompose(in[1]) = xxxxxxxxxxxxxxxxxxxxxxxxxxxx1010 0010xxxxxxxxxx...xxxx
    //                    <-----------  higher bits  /  lower bits  ----------->

    thrust::device_vector<custom_t> keys_alt_buf(num_items);

    custom_t* d_keys_buf     = thrust::raw_pointer_cast(keys_buf.data());
    custom_t* d_keys_alt_buf = thrust::raw_pointer_cast(keys_alt_buf.data());

    cub::DoubleBuffer<custom_t> d_keys(d_keys_buf, d_keys_alt_buf);

    // 1) Get temp storage size
    std::uint8_t* d_temp_storage{};
    std::size_t temp_storage_bytes{};

    cub::DeviceRadixSort::SortKeysDescending(
      d_temp_storage, temp_storage_bytes, d_keys, num_items, decomposer_t{}, begin_bit, end_bit);

    // 2) Allocate temp storage
    thrust::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
    d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

    // 3) Sort keys
    cub::DeviceRadixSort::SortKeysDescending(
      d_temp_storage, temp_storage_bytes, d_keys, num_items, decomposer_t{}, begin_bit, end_bit);

    thrust::device_vector<custom_t>& current_keys = //
      d_keys.Current() == d_keys_buf ? keys_buf : keys_alt_buf;

    thrust::device_vector<custom_t> expected_output = {
      {24.2f, 1ll << 61}, //
      {42.4f, 1ll << 60} //
    };
    // example-end keys-descending-bits-db

    REQUIRE(expected_output == current_keys);
  }

  SECTION("Pairs")
  {
    // example-begin pairs-bits-db
    constexpr int num_items                  = 2;
    thrust::device_vector<custom_t> keys_buf = {
      {24.2f, 1ll << 61}, //
      {42.4f, 1ll << 60} //
    };

    thrust::device_vector<int> vals_buf = {1, 0};

    constexpr int begin_bit = sizeof(long long int) * 8 - 4; // 60
    constexpr int end_bit   = sizeof(long long int) * 8 + 4; // 68

    // Decomposition orders the bits as follows:
    //
    //                    <------------- fp32 -----------> <------ int64 ------>
    // decompose(in[0]) = 01000001110000011001100110011010 00100000000000...0000
    // decompose(in[1]) = 01000010001010011001100110011010 00010000000000...0000
    //                    <-----------  higher bits  /  lower bits  ----------->
    //
    // The bit subrange `[60, 68)` specifies differentiating key bits:
    //
    //                    <------------- fp32 -----------> <------ int64 ------>
    // decompose(in[0]) = xxxxxxxxxxxxxxxxxxxxxxxxxxxx1010 0010xxxxxxxxxx...xxxx
    // decompose(in[1]) = xxxxxxxxxxxxxxxxxxxxxxxxxxxx1010 0001xxxxxxxxxx...xxxx
    //                    <-----------  higher bits  /  lower bits  ----------->

    thrust::device_vector<custom_t> keys_alt_buf(num_items);
    thrust::device_vector<int> vals_alt_buf(num_items);

    custom_t* d_keys_buf     = thrust::raw_pointer_cast(keys_buf.data());
    custom_t* d_keys_alt_buf = thrust::raw_pointer_cast(keys_alt_buf.data());
    int* d_vals_buf          = thrust::raw_pointer_cast(vals_buf.data());
    int* d_vals_alt_buf      = thrust::raw_pointer_cast(vals_alt_buf.data());

    cub::DoubleBuffer<custom_t> d_keys(d_keys_buf, d_keys_alt_buf);
    cub::DoubleBuffer<int> d_vals(d_vals_buf, d_vals_alt_buf);

    // 1) Get temp storage size
    std::uint8_t* d_temp_storage{};
    std::size_t temp_storage_bytes{};

    cub::DeviceRadixSort::SortPairs(
      d_temp_storage, temp_storage_bytes, d_keys, d_vals, num_items, decomposer_t{}, begin_bit, end_bit);

    // 2) Allocate temp storage
    thrust::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
    d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

    // 3) Sort keys
    cub::DeviceRadixSort::SortPairs(
      d_temp_storage, temp_storage_bytes, d_keys, d_vals, num_items, decomposer_t{}, begin_bit, end_bit);

    thrust::device_vector<custom_t>& current_keys = //
      d_keys.Current() == d_keys_buf ? keys_buf : keys_alt_buf;

    thrust::device_vector<int>& current_vals = //
      d_vals.Current() == d_vals_buf ? vals_buf : vals_alt_buf;

    thrust::device_vector<custom_t> expected_keys = {
      {42.4f, 1ll << 60}, //
      {24.2f, 1ll << 61} //
    };

    thrust::device_vector<int> expected_vals = {0, 1};
    // example-end pairs-bits-db

    REQUIRE(expected_keys == current_keys);
    REQUIRE(expected_vals == current_vals);
  }

  SECTION("PairsDescending")
  {
    // example-begin pairs-descending-bits-db
    constexpr int num_items = 2;

    thrust::device_vector<custom_t> keys_buf = {
      {42.4f, 1ll << 60}, //
      {24.2f, 1ll << 61} //
    };

    thrust::device_vector<int> vals_buf = {1, 0};

    constexpr int begin_bit = sizeof(long long int) * 8 - 4; // 60
    constexpr int end_bit   = sizeof(long long int) * 8 + 4; // 68

    // Decomposition orders the bits as follows:
    //
    //                    <------------- fp32 -----------> <------ int64 ------>
    // decompose(in[0]) = 01000010001010011001100110011010 00010000000000...0000
    // decompose(in[1]) = 01000001110000011001100110011010 00100000000000...0000
    //                    <-----------  higher bits  /  lower bits  ----------->
    //
    // The bit subrange `[60, 68)` specifies differentiating key bits:
    //
    //                    <------------- fp32 -----------> <------ int64 ------>
    // decompose(in[0]) = xxxxxxxxxxxxxxxxxxxxxxxxxxxx1010 0001xxxxxxxxxx...xxxx
    // decompose(in[1]) = xxxxxxxxxxxxxxxxxxxxxxxxxxxx1010 0010xxxxxxxxxx...xxxx
    //                    <-----------  higher bits  /  lower bits  ----------->

    thrust::device_vector<custom_t> keys_alt_buf(num_items);
    thrust::device_vector<int> vals_alt_buf(num_items);

    custom_t* d_keys_buf     = thrust::raw_pointer_cast(keys_buf.data());
    custom_t* d_keys_alt_buf = thrust::raw_pointer_cast(keys_alt_buf.data());
    int* d_vals_buf          = thrust::raw_pointer_cast(vals_buf.data());
    int* d_vals_alt_buf      = thrust::raw_pointer_cast(vals_alt_buf.data());

    cub::DoubleBuffer<custom_t> d_keys(d_keys_buf, d_keys_alt_buf);
    cub::DoubleBuffer<int> d_vals(d_vals_buf, d_vals_alt_buf);

    // 1) Get temp storage size
    std::uint8_t* d_temp_storage{};
    std::size_t temp_storage_bytes{};

    cub::DeviceRadixSort::SortPairsDescending(
      d_temp_storage, temp_storage_bytes, d_keys, d_vals, num_items, decomposer_t{}, begin_bit, end_bit);

    // 2) Allocate temp storage
    thrust::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
    d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

    // 3) Sort keys
    cub::DeviceRadixSort::SortPairsDescending(
      d_temp_storage, temp_storage_bytes, d_keys, d_vals, num_items, decomposer_t{}, begin_bit, end_bit);

    thrust::device_vector<custom_t>& current_keys = //
      d_keys.Current() == d_keys_buf ? keys_buf : keys_alt_buf;

    thrust::device_vector<int>& current_vals = //
      d_vals.Current() == d_vals_buf ? vals_buf : vals_alt_buf;

    thrust::device_vector<custom_t> expected_keys = {
      {24.2f, 1ll << 61}, //
      {42.4f, 1ll << 60} //
    };

    thrust::device_vector<int> expected_vals = {0, 1};
    // example-end pairs-descending-bits-db

    REQUIRE(expected_keys == current_keys);
    REQUIRE(expected_vals == current_vals);
  }
}
#endif
