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

#include <cub/device/device_merge_sort.cuh>

#include <thrust/copy.h>
#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/equal.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/reverse_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/random.h>
#include <thrust/sequence.h>
#include <thrust/shuffle.h>

#include <algorithm>

#include "catch2_large_array_sort_helper.cuh"
#include "catch2_test_device_merge_sort_common.cuh"
#include "catch2_test_launch_helper.h"
#include <c2h/catch2_test_helper.cuh>

// %PARAM% TEST_LAUNCH lid 0:1:2

DECLARE_LAUNCH_WRAPPER(cub::DeviceMergeSort::SortPairs, sort_pairs);
DECLARE_LAUNCH_WRAPPER(cub::DeviceMergeSort::SortPairsCopy, sort_pairs_copy);
DECLARE_LAUNCH_WRAPPER(cub::DeviceMergeSort::StableSortPairs, stable_sort_pairs);

DECLARE_LAUNCH_WRAPPER(cub::DeviceMergeSort::SortKeys, sort_keys);
DECLARE_LAUNCH_WRAPPER(cub::DeviceMergeSort::SortKeysCopy, sort_keys_copy);
DECLARE_LAUNCH_WRAPPER(cub::DeviceMergeSort::StableSortKeys, stable_sort_keys);
DECLARE_LAUNCH_WRAPPER(cub::DeviceMergeSort::StableSortKeysCopy, stable_sort_keys_copy);

using key_types =
  c2h::type_list<std::uint8_t,
                 std::int16_t,
                 std::uint32_t,
                 double,
                 c2h::custom_type_t<c2h::equal_comparable_t, c2h::less_comparable_t>>;
using wide_key_types = c2h::type_list<std::uint32_t, double>;

using value_types =
  c2h::type_list<std::uint8_t, float, c2h::custom_type_t<c2h::equal_comparable_t, c2h::less_comparable_t>>;

template <typename OffsetT, typename KeyT = std::uint8_t>
struct type_tuple
{
  using offset_t = OffsetT;
  using key_t    = KeyT;
};
using offset_types =
  c2h::type_list<type_tuple<std::int16_t>,
                 type_tuple<std::int32_t>,
                 type_tuple<std::int32_t, std::uint32_t>,
                 type_tuple<std::uint32_t>,
                 type_tuple<std::uint64_t>>;

/**
 * Function object that maps the targeted sorted rank of an item to a key.

 * E.g., `OffsetT` is `int32_t` and `KeyT` is `float`:
 * [  4,   2,   3,   1,   0] <= targeted key ranks
 * [4.0, 2.0, 3.0, 1.0, 0.0] <= corresponding keys
 */
template <typename OffsetT, typename KeyT>
struct rank_to_key_op_t
{
  __device__ __host__ KeyT operator()(const OffsetT& val)
  {
    return static_cast<KeyT>(val);
  }
};

template <typename OffsetT>
struct rank_to_key_op_t<OffsetT, c2h::custom_type_t<c2h::equal_comparable_t, c2h::less_comparable_t>>
{
  using custom_t = c2h::custom_type_t<c2h::equal_comparable_t, c2h::less_comparable_t>;
  __device__ __host__ custom_t operator()(const OffsetT& val)
  {
    custom_t custom_val{};
    custom_val.key = val;
    custom_val.val = val;
    return custom_val;
  }
};

/**
 * Helps initialize custom_type_t from a zip-iterator combination of sort-key and value
 */
template <typename CustomT>
struct tuple_to_custom_op_t
{
  template <typename KeyT, typename ValueT>
  __device__ __host__ CustomT operator()(const thrust::tuple<KeyT, ValueT>& val)
  {
    CustomT custom_val{};
    custom_val.key = static_cast<std::size_t>(thrust::get<0>(val));
    custom_val.val = static_cast<std::size_t>(thrust::get<1>(val));
    return custom_val;
  }
};

/**
 * @brief In combination with a counting iterator, this function object generates a sequence that wraps around after
 * reaching `UnsignedIntegralKeyT`'s maximum value. E.g., for a uint8_t this maps the sequence of indexes [0, ..., 254,
 * 255, 256, 256] -> [0, ..., 254, 255, 0, 1]
 */
template <typename UnsignedIntegralKeyT>
struct index_to_key_value_op
{
  static constexpr std::size_t max_key_value =
    static_cast<std::size_t>(::cuda::std::numeric_limits<UnsignedIntegralKeyT>::max());
  static constexpr std::size_t lowest_key_value =
    static_cast<std::size_t>(::cuda::std::numeric_limits<UnsignedIntegralKeyT>::lowest());
  static_assert(sizeof(UnsignedIntegralKeyT) < sizeof(std::size_t),
                "Calculation of num_distinct_key_values would overflow");
  static constexpr std::size_t num_distinct_key_values = (max_key_value - lowest_key_value + std::size_t{1ULL});

  __device__ __host__ UnsignedIntegralKeyT operator()(std::size_t index)
  {
    return static_cast<UnsignedIntegralKeyT>(index % num_distinct_key_values);
  }
};

/**
 * @brief In combination with a counting iterator, this function object helps generate the expected sorted order for a
 * sequence generated with `index_to_key_value_op`. It respects how many remainder items there are following the last
 * occurrence of `UnsignedIntegralKeyT`'s max value. E.g., when we use `num_total_items` of `260` with an `uint8_t`, the
 * input sequence was:
 * [0, ..., 254, 255, 256, 257, 258, 259] <= index
 * [0, ..., 254, 255,   0,   1,   2,   3] <= input
 * -----------------
 * [0, 0, 1, 1, 2, 2, 3, 3, 4, 5, 6, ..., 255] <= expected sorted order (note, [0, 3] occur twice)
 */
template <typename UnsignedIntegralKeyT>
class index_to_expected_key_op
{
private:
  static constexpr std::size_t max_key_value =
    static_cast<std::size_t>(::cuda::std::numeric_limits<UnsignedIntegralKeyT>::max());
  static constexpr std::size_t lowest_key_value =
    static_cast<std::size_t>(::cuda::std::numeric_limits<UnsignedIntegralKeyT>::lowest());
  static_assert(sizeof(UnsignedIntegralKeyT) < sizeof(std::size_t),
                "Calculation of num_distinct_key_values would overflow");
  static constexpr std::size_t num_distinct_key_values = (max_key_value - lowest_key_value + std::size_t{1ULL});

  // item_count / num_distinct_key_values
  std::size_t expected_count_per_item;
  // num remainder items: item_count%num_distinct_key_values
  std::size_t num_remainder_items;
  // remainder item_count: expected_count_per_item+1
  std::size_t remainder_item_count;

public:
  index_to_expected_key_op(std::size_t num_total_items)
      : expected_count_per_item(num_total_items / num_distinct_key_values)
      , num_remainder_items(num_total_items % num_distinct_key_values)
      , remainder_item_count(expected_count_per_item + std::size_t{1ULL})
  {}

  __device__ __host__ UnsignedIntegralKeyT operator()(std::size_t index)
  {
    // The first (num_remainder_items * remainder_item_count) are items that appear once more often than the items that
    // follow remainder_items_offset
    std::size_t remainder_items_offset = num_remainder_items * remainder_item_count;

    UnsignedIntegralKeyT target_item_index =
      (index <= remainder_items_offset)
        ?
        // This is one of the remainder items
        static_cast<UnsignedIntegralKeyT>(index / remainder_item_count)
        :
        // This is an item that appears exactly expected_count_per_item times
        static_cast<UnsignedIntegralKeyT>(
          num_remainder_items + ((index - remainder_items_offset) / expected_count_per_item));
    return target_item_index;
  }
};

/**
 * Generates a shuffled array of key ranks. E.g., for a vector of size 5: [4, 2, 3, 1, 0]
 */
template <typename OffsetT>
c2h::device_vector<OffsetT> make_shuffled_key_ranks_vector(OffsetT num_items, c2h::seed_t seed)
{
  c2h::device_vector<OffsetT> key_ranks(num_items);
  thrust::sequence(c2h::device_policy, key_ranks.begin(), key_ranks.end());
  thrust::shuffle(c2h::device_policy,
                  key_ranks.begin(),
                  key_ranks.end(),
                  thrust::default_random_engine{static_cast<unsigned int>(seed.get())});
  return key_ranks;
}

C2H_TEST("DeviceMergeSort::SortKeysCopy works", "[merge][sort][device]", wide_key_types)
{
  using key_t    = typename c2h::get<0, TestType>;
  using offset_t = std::int32_t;

  // Prepare input
  const offset_t num_items = GENERATE_COPY(take(2, random(1, 1000000)), values({500, 1000000, 2000000}));
  auto key_ranks           = make_shuffled_key_ranks_vector(num_items, C2H_SEED(2));
  c2h::device_vector<key_t> keys_in(num_items);
  thrust::transform(
    c2h::device_policy, key_ranks.begin(), key_ranks.end(), keys_in.begin(), rank_to_key_op_t<offset_t, key_t>{});

  // Perform sort
  c2h::device_vector<key_t> keys_out(num_items, static_cast<key_t>(42));
  sort_keys_copy(
    thrust::raw_pointer_cast(keys_in.data()), thrust::raw_pointer_cast(keys_out.data()), num_items, custom_less_op_t{});

  // Verify results
  auto key_ranks_it     = thrust::make_counting_iterator(offset_t{});
  auto keys_expected_it = thrust::make_transform_iterator(key_ranks_it, rank_to_key_op_t<offset_t, key_t>{});
  bool results_equal    = thrust::equal(c2h::device_policy, keys_out.cbegin(), keys_out.cend(), keys_expected_it);
  REQUIRE(results_equal == true);
}

C2H_TEST("DeviceMergeSort::SortKeys works", "[merge][sort][device]", wide_key_types)
{
  using key_t    = typename c2h::get<0, TestType>;
  using offset_t = std::int32_t;

  // Prepare input
  const offset_t num_items = GENERATE_COPY(take(2, random(1, 1000000)), values({500, 1000000, 2000000}));
  auto key_ranks           = make_shuffled_key_ranks_vector(num_items, C2H_SEED(2));
  c2h::device_vector<key_t> keys_in_out(num_items);
  thrust::transform(
    c2h::device_policy, key_ranks.begin(), key_ranks.end(), keys_in_out.begin(), rank_to_key_op_t<offset_t, key_t>{});

  // Perform sort
  sort_keys(thrust::raw_pointer_cast(keys_in_out.data()), num_items, custom_less_op_t{});

  // Verify results
  auto key_ranks_it     = thrust::make_counting_iterator(offset_t{});
  auto keys_expected_it = thrust::make_transform_iterator(key_ranks_it, rank_to_key_op_t<offset_t, key_t>{});
  bool results_equal    = thrust::equal(c2h::device_policy, keys_in_out.cbegin(), keys_in_out.cend(), keys_expected_it);
  REQUIRE(results_equal == true);
}

C2H_TEST("DeviceMergeSort::StableSortKeysCopy works and performs a stable sort when there are a lot sort-keys that "
         "compare equal",
         "[merge][sort][device]")
{
  using key_t    = c2h::custom_type_t<c2h::equal_comparable_t, c2h::less_comparable_t>;
  using offset_t = std::size_t;

  // Prepare input (generate a items that compare equally to check for stability of sort)
  const offset_t num_items = GENERATE_COPY(take(2, random(1, 1000000)), values({500, 1000000, 2000000}));
  c2h::device_vector<offset_t> key_ranks(num_items);
  c2h::gen(C2H_SEED(2), key_ranks, offset_t{}, static_cast<offset_t>(128));
  c2h::device_vector<key_t> keys_in(num_items);
  auto key_value_it = thrust::make_counting_iterator(offset_t{});
  auto key_init_it  = thrust::make_zip_iterator(key_ranks.begin(), key_value_it);
  thrust::transform(
    c2h::device_policy, key_init_it, key_init_it + num_items, keys_in.begin(), tuple_to_custom_op_t<key_t>{});

  // Perform sort
  c2h::device_vector<key_t> keys_out(num_items, rank_to_key_op_t<offset_t, key_t>{}(42));
  stable_sort_keys_copy(
    thrust::raw_pointer_cast(keys_in.data()), thrust::raw_pointer_cast(keys_out.data()), num_items, custom_less_op_t{});

  // Verify results
  c2h::host_vector<key_t> keys_expected(keys_in);
  std::stable_sort(keys_expected.begin(), keys_expected.end(), custom_less_op_t{});

  REQUIRE(keys_expected == keys_out);
}

C2H_TEST("DeviceMergeSort::StableSortKeys works", "[merge][sort][device]")
{
  using key_t    = c2h::custom_type_t<c2h::equal_comparable_t, c2h::less_comparable_t>;
  using offset_t = std::int32_t;

  // Prepare input
  const offset_t num_items = GENERATE_COPY(take(2, random(1, 1000000)), values({500, 1000000, 2000000}));
  c2h::device_vector<key_t> keys_in_out(num_items);
  c2h::gen(C2H_SEED(2), keys_in_out);

  // Perform sort
  stable_sort_keys(thrust::raw_pointer_cast(keys_in_out.data()), num_items, custom_less_op_t{});

  // Verify results
  c2h::host_vector<key_t> keys_expected(keys_in_out);
  std::stable_sort(keys_expected.begin(), keys_expected.end(), custom_less_op_t{});

  REQUIRE(keys_expected == keys_in_out);
}

C2H_TEST("DeviceMergeSort::SortPairsCopy works", "[merge][sort][device]", wide_key_types)
{
  using key_t    = typename c2h::get<0, TestType>;
  using offset_t = std::int32_t;

  // Prepare input
  const offset_t num_items = GENERATE_COPY(take(2, random(1, 1000000)), values({500, 1000000, 2000000}));
  auto key_ranks           = make_shuffled_key_ranks_vector(num_items, C2H_SEED(2));
  c2h::device_vector<key_t> keys_in(num_items);
  thrust::transform(
    c2h::device_policy, key_ranks.begin(), key_ranks.end(), keys_in.begin(), rank_to_key_op_t<offset_t, key_t>{});

  // Perform sort
  c2h::device_vector<key_t> keys_out(num_items, static_cast<key_t>(42));
  c2h::device_vector<offset_t> values_out(num_items, static_cast<offset_t>(42));
  sort_pairs_copy(
    thrust::raw_pointer_cast(keys_in.data()),
    thrust::raw_pointer_cast(key_ranks.data()),
    thrust::raw_pointer_cast(keys_out.data()),
    thrust::raw_pointer_cast(values_out.data()),
    num_items,
    custom_less_op_t{});

  // Verify results
  auto key_ranks_it       = thrust::make_counting_iterator(offset_t{});
  auto keys_expected_it   = thrust::make_transform_iterator(key_ranks_it, rank_to_key_op_t<offset_t, key_t>{});
  auto values_expected_it = thrust::make_counting_iterator(offset_t{});
  bool keys_equal         = thrust::equal(c2h::device_policy, keys_out.cbegin(), keys_out.cend(), keys_expected_it);
  bool values_equal = thrust::equal(c2h::device_policy, values_out.cbegin(), values_out.cend(), values_expected_it);
  REQUIRE(keys_equal == true);
  REQUIRE(values_equal == true);
}

C2H_TEST("DeviceMergeSort::SortPairs works", "[merge][sort][device]", wide_key_types)
{
  using key_t    = typename c2h::get<0, TestType>;
  using offset_t = std::int32_t;

  // Prepare input
  const offset_t num_items = GENERATE_COPY(take(2, random(1, 1000000)), values({500, 1000000, 2000000}));
  auto key_ranks           = make_shuffled_key_ranks_vector(num_items, C2H_SEED(2));
  c2h::device_vector<key_t> keys_in_out(num_items);
  thrust::transform(
    c2h::device_policy, key_ranks.begin(), key_ranks.end(), keys_in_out.begin(), rank_to_key_op_t<offset_t, key_t>{});

  // Perform sort
  sort_pairs(thrust::raw_pointer_cast(keys_in_out.data()),
             thrust::raw_pointer_cast(key_ranks.data()),
             num_items,
             custom_less_op_t{});

  // Verify results
  auto key_ranks_it       = thrust::make_counting_iterator(offset_t{});
  auto keys_expected_it   = thrust::make_transform_iterator(key_ranks_it, rank_to_key_op_t<offset_t, key_t>{});
  auto values_expected_it = thrust::make_counting_iterator(offset_t{});
  bool keys_equal   = thrust::equal(c2h::device_policy, keys_in_out.cbegin(), keys_in_out.cend(), keys_expected_it);
  bool values_equal = thrust::equal(c2h::device_policy, key_ranks.cbegin(), key_ranks.cend(), values_expected_it);
  REQUIRE(keys_equal == true);
  REQUIRE(values_equal == true);
}

C2H_TEST(
  "DeviceMergeSort::StableSortPairs works and performs a stable sort", "[merge][sort][device]", key_types, value_types)
{
  using key_t    = typename c2h::get<0, TestType>;
  using data_t   = typename c2h::get<1, TestType>;
  using offset_t = std::int32_t;

  // Prepare input
  const offset_t num_items = GENERATE_COPY(take(2, random(1, 1000000)), values({500, 1000000, 2000000}));
  c2h::device_vector<key_t> keys_in_out(num_items);
  c2h::device_vector<data_t> values_in_out(num_items);
  c2h::gen(C2H_SEED(2), keys_in_out);
  c2h::gen(C2H_SEED(1), values_in_out);

  // Prepare host data for verification
  c2h::host_vector<key_t> keys_expected(keys_in_out);
  c2h::host_vector<data_t> values_expected(values_in_out);
  auto zipped_expected_it = thrust::make_zip_iterator(keys_expected.begin(), values_expected.begin());
  std::stable_sort(zipped_expected_it, zipped_expected_it + num_items, compare_first_lt_op_t{});

  // Perform sort
  stable_sort_pairs(thrust::raw_pointer_cast(keys_in_out.data()),
                    thrust::raw_pointer_cast(values_in_out.data()),
                    num_items,
                    custom_less_op_t{});

  REQUIRE(keys_expected == keys_in_out);
  REQUIRE(values_expected == values_in_out);
}

C2H_TEST("DeviceMergeSort::StableSortPairs works for large inputs", "[merge][sort][device]", offset_types)
{
  using testing_types_tuple = c2h::get<0, TestType>;
  using key_t               = typename testing_types_tuple::key_t;
  using offset_t            = typename testing_types_tuple::offset_t;

  // Clamp 64-bit offset type problem sizes to just slightly larger than 2^32 items
  auto num_items_ull =
    std::min(static_cast<std::size_t>(::cuda::std::numeric_limits<offset_t>::max()) - 1,
             ::cuda::std::numeric_limits<std::uint32_t>::max() + static_cast<std::size_t>(2000000ULL));
  offset_t num_items = static_cast<offset_t>(num_items_ull);

  SECTION("Random")
  {
    try
    {
      // Initialize random input data
      large_array_sort_helper<key_t> arrays;
      constexpr bool is_descending = false;
      arrays.initialize_for_unstable_key_sort(C2H_SEED(1), num_items, is_descending);

      // Free extra data buffer used during initialization, but not needed for the "in-place" merge sort
      arrays.deallocate_outputs();

      // Perform sort
      stable_sort_keys(thrust::raw_pointer_cast(arrays.keys_in.data()), num_items, custom_less_op_t{});

      // Verify results
      arrays.verify_unstable_key_sort(num_items, is_descending, arrays.keys_in);
    }
    catch (std::bad_alloc& e)
    {
      const std::size_t num_bytes = num_items * sizeof(key_t);
      std::cerr << "Skipping merge sort test with " << num_items << " elements (" << num_bytes
                << " bytes): " << e.what() << "\n";
    }
  }

  SECTION("Pre-sorted input")
  {
    try
    {
      c2h::device_vector<key_t> keys_in_out(num_items);

      // Pre-populated array with a constant value
      auto counting_it = thrust::make_counting_iterator(std::size_t{0});
      thrust::copy(counting_it, counting_it + num_items, keys_in_out.begin());

      // Perform sort
      stable_sort_keys(thrust::raw_pointer_cast(keys_in_out.data()), num_items, custom_less_op_t{});

      // Perform comparison
      auto expected_result_it = thrust::make_transform_iterator(
        thrust::make_counting_iterator(std::size_t{}), index_to_expected_key_op<key_t>(num_items));
      bool is_correct = thrust::equal(expected_result_it, expected_result_it + num_items, keys_in_out.begin());
      REQUIRE(is_correct == true);
    }
    catch (std::bad_alloc& e)
    {
      const std::size_t num_bytes = num_items * sizeof(key_t);
      std::cerr << "Skipping merge sort test with " << num_items << " elements (" << num_bytes
                << " bytes): " << e.what() << "\n";
    }
  }

  SECTION("Reverse-sorted input")
  {
    try
    {
      c2h::device_vector<key_t> keys_in_out(num_items);

      auto counting_it   = thrust::make_counting_iterator(std::size_t{0});
      auto key_value_it  = thrust::make_transform_iterator(counting_it, index_to_key_value_op<key_t>{});
      auto rev_sorted_it = thrust::make_reverse_iterator(key_value_it + num_items);
      thrust::copy(rev_sorted_it, rev_sorted_it + num_items, keys_in_out.begin());

      // Perform sort
      stable_sort_keys(thrust::raw_pointer_cast(keys_in_out.data()), num_items, custom_less_op_t{});

      // Perform comparison
      auto expected_result_it = thrust::make_transform_iterator(
        thrust::make_counting_iterator(std::size_t{}), index_to_expected_key_op<key_t>(num_items));
      bool is_correct = thrust::equal(expected_result_it, expected_result_it + num_items, keys_in_out.cbegin());
      REQUIRE(is_correct == true);
    }
    catch (std::bad_alloc& e)
    {
      const std::size_t num_bytes = num_items * sizeof(key_t);
      std::cerr << "Skipping merge sort test with " << num_items << " elements (" << num_bytes
                << " bytes): " << e.what() << "\n";
    }
  }
}
