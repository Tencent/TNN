/******************************************************************************
 * Copyright (c) 2011-2024, NVIDIA CORPORATION.  All rights reserved.
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

#pragma once

#include <cub/util_type.cuh>

#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/memory.h>
#include <thrust/random.h>
#include <thrust/shuffle.h>
#include <thrust/tabulate.h>

#include <cuda/std/iterator>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

#include <algorithm>
#include <cstdint>
#include <numeric>

#include <c2h/catch2_test_helper.cuh>
#include <c2h/cpu_timer.cuh>
#include <c2h/device_policy.cuh>
#include <c2h/generators.cuh> // seed_t
#include <c2h/vector.cuh>

// #define DEBUG_TIMING

#ifdef DEBUG_TIMING
#  define TIME(expr) expr
#else
#  define TIME(expr) /* no op */ []() {}()
#endif

namespace detail
{

template <typename KeyType>
class key_sort_ref_key_transform
{
  static constexpr double max_key = static_cast<double>(::cuda::std::numeric_limits<KeyType>::max());
  const double m_conversion;
  std::size_t m_num_items;
  bool m_is_descending;

public:
  key_sort_ref_key_transform(std::size_t num_items, bool is_descending)
      : m_conversion(max_key / num_items)
      , m_num_items(num_items)
      , m_is_descending(is_descending)
  {}

  _CCCL_HOST_DEVICE KeyType operator()(std::size_t idx) const
  {
    return m_is_descending ? static_cast<KeyType>((m_num_items - 1 - idx) * m_conversion)
                           : static_cast<KeyType>(idx * m_conversion);
  }
};

template <typename KeyType>
struct summary
{
  std::size_t index;
  std::size_t count;
  KeyType key;
};

template <typename KeyType>
struct index_to_summary
{
  using summary_t = summary<KeyType>;

  std::size_t num_items;
  std::size_t num_summaries;
  bool is_descending;

  template <typename index_type>
  _CCCL_HOST_DEVICE summary_t operator()(index_type idx) const
  {
    constexpr KeyType max_key = ::cuda::std::numeric_limits<KeyType>::max();

    const double key_conversion = static_cast<double>(max_key) / static_cast<double>(num_summaries);
    const KeyType key           = is_descending ? static_cast<KeyType>((num_summaries - 1 - idx) * key_conversion)
                                                : static_cast<KeyType>(idx * key_conversion);

    const std::size_t elements_per_summary = num_items / num_summaries;
    const std::size_t run_index            = idx * elements_per_summary;
    const std::size_t run_size = idx == (num_summaries - 1) ? (num_items - run_index) : elements_per_summary;

    return summary_t{run_index, run_size, key};
  }
};

template <typename KeyType>
class pair_sort_ref_key_transform
{
  static constexpr KeyType max_key = ::cuda::std::numeric_limits<KeyType>::max();

  double m_key_conversion; // Converts summary index to key
  std::size_t m_num_summaries;
  std::size_t m_unpadded_run_size; // typical run size
  bool m_is_descending;

public:
  pair_sort_ref_key_transform(std::size_t num_items, std::size_t num_summaries, bool is_descending)
      : m_key_conversion(static_cast<double>(max_key) / static_cast<double>(num_summaries))
      , m_num_summaries(num_summaries)
      , m_unpadded_run_size(num_items / num_summaries)
      , m_is_descending(is_descending)
  {}

  _CCCL_HOST_DEVICE KeyType operator()(std::size_t idx) const
  {
    // The final summary may be padded, so truncate the summary_idx at the last valid idx:
    const std::size_t summary_idx = thrust::min(m_num_summaries - 1, idx / m_unpadded_run_size);
    const KeyType key = m_is_descending ? static_cast<KeyType>((m_num_summaries - 1 - summary_idx) * m_key_conversion)
                                        : static_cast<KeyType>(summary_idx * m_key_conversion);

    return key;
  }
};

template <typename ValueType>
struct index_to_value
{
  template <typename index_type>
  _CCCL_HOST_DEVICE ValueType operator()(index_type index)
  {
    return static_cast<ValueType>(index);
  }
};

} // namespace detail

template <typename KeyType, typename ValueType = cub::NullType>
struct large_array_sort_helper
{
  // Sorted keys/values in host memory
  // (May be unused if results can be verified with fancy iterators)
  c2h::host_vector<KeyType> keys_ref;
  c2h::host_vector<ValueType> values_ref;

  // Unsorted keys/values in device memory
  c2h::device_vector<KeyType> keys_in;
  c2h::device_vector<ValueType> values_in;

  // Allocated device memory for output keys/values
  c2h::device_vector<KeyType> keys_out;
  c2h::device_vector<ValueType> values_out;

  // Double buffer for keys/values. Aliases the in/out arrays.
  cub::DoubleBuffer<KeyType> keys_buffer;
  cub::DoubleBuffer<ValueType> values_buffer;

  // By default, both input and output arrays are allocated to ensure that 2 * num_items * (sizeof(KeyType) +
  // sizeof(ValueType)) device memory is available at the start of the initialize_* methods. This ensures that we'll
  // fail quickly if the problem size exceeds the necessary storage required for sorting. If the output arrays are not
  // being used (e.g. in-place merge sort API with temporary storage allocation), these may be freed easily by calling
  // this method:
  void deallocate_outputs()
  {
    keys_out.clear();
    keys_out.shrink_to_fit();
    values_out.clear();
    values_out.shrink_to_fit();
  }

  // Populates keys_in with random KeyTypes. Allocates keys_out and configures keys_buffer appropriately.
  // Allocates a total of 2 * num_items * sizeof(KeyType) device memory and no host memory.
  // Shuffle will allocate some additional device memory overhead for scan temp storage.
  // Pass the sorted output to verify_unstable_key_sort to validate.
  void initialize_for_unstable_key_sort(c2h::seed_t seed, std::size_t num_items, bool is_descending)
  {
    TIME(c2h::cpu_timer timer);

    // Preallocate device memory ASAP so we fail quickly on bad_alloc
    keys_in.resize(num_items);
    keys_out.resize(num_items);
    keys_buffer =
      cub::DoubleBuffer<KeyType>(thrust::raw_pointer_cast(keys_in.data()), thrust::raw_pointer_cast(keys_out.data()));

    TIME(timer.print_elapsed_seconds_and_reset("Device Alloc"));

    { // Place the sorted keys into keys_out
      auto key_iter = thrust::make_transform_iterator(
        thrust::make_counting_iterator(std::size_t{0}),
        detail::key_sort_ref_key_transform<KeyType>(num_items, is_descending));
      thrust::copy(c2h::device_policy, key_iter, key_iter + num_items, keys_out.begin());
    }

    TIME(timer.print_elapsed_seconds_and_reset("Generate sorted keys"));

    // shuffle random keys into keys_in
    thrust::shuffle_copy(
      c2h::device_policy,
      keys_out.cbegin(),
      keys_out.cend(),
      keys_in.begin(),
      thrust::default_random_engine(static_cast<std::uint32_t>(seed.get())));

    TIME(timer.print_elapsed_seconds_and_reset("Shuffle"));

    // Reset keys_out to remove the valid sorted keys:
    thrust::fill(c2h::device_policy, keys_out.begin(), keys_out.end(), KeyType{});

    TIME(timer.print_elapsed_seconds_and_reset("Reset Output"));
  }

  // Verify the results of sorting the keys_in produced by initialize_for_unstable_key_sort.
  void verify_unstable_key_sort(std::size_t num_items, bool is_descending, const c2h::device_vector<KeyType>& keys)
  {
    TIME(c2h::cpu_timer timer);
    auto key_iter = thrust::make_transform_iterator(
      thrust::make_counting_iterator(std::size_t{0}),
      detail::key_sort_ref_key_transform<KeyType>{num_items, is_descending});
    REQUIRE(thrust::equal(c2h::device_policy, keys.cbegin(), keys.cend(), key_iter));
    TIME(timer.print_elapsed_seconds_and_reset("Validate keys"));
  }

  // Populates keys_in with random KeyTypes and values_in with sequential ValueTypes.
  // Allocates keys_out and values_out and configures keys_buffer and values_buffer appropriately.
  // values_ref will contain the expected stable sorted values.
  // Allocates 2 * num_items * (sizeof(KeyType) + sizeof(ValueType)) device memory.
  // May allocate up to 2 * num_items * (sizeof(KeyType) + sizeof(ValueType)) on the host.
  // Pass the sorted outputs to verify_stable_pair_sort to validate.
  void initialize_for_stable_pair_sort(c2h::seed_t seed, std::size_t num_items, bool is_descending)
  {
    static_assert(!::cuda::std::is_same<ValueType, cub::NullType>::value, "ValueType must be valid.");
    using summary_t = detail::summary<KeyType>;

    const std::size_t max_summaries = this->compute_max_summaries(num_items);
    const std::size_t num_summaries = this->compute_num_summaries(num_items, max_summaries);

    TIME(c2h::cpu_timer timer);

    c2h::device_vector<summary_t> d_summaries;
    // Overallocate -- if this fails, there won't be be enough free device memory for the input/output arrays.
    // Better to fail now before spending time computing the inputs/outputs.
    d_summaries.reserve(2 * max_summaries);
    d_summaries.resize(num_summaries);

    TIME(timer.print_elapsed_seconds_and_reset("Device allocate"));

    // Populate the summaries using evenly spaced keys and constant sized runs, padding the last run to fill.
    thrust::tabulate(c2h::device_policy,
                     d_summaries.begin(),
                     d_summaries.end(),
                     detail::index_to_summary<KeyType>{num_items, num_summaries, is_descending});

    TIME(timer.print_elapsed_seconds_and_reset("idx -> summary"));

    // Copy the summaries to host memory and release device summary memory.
    c2h::host_vector<summary_t> h_summaries = d_summaries;

    TIME(timer.print_elapsed_seconds_and_reset("D->H Summaries"));

    d_summaries.clear();
    d_summaries.shrink_to_fit();

    TIME(timer.print_elapsed_seconds_and_reset("Free device summaries"));

    // Build the unsorted key and reference value arrays on host:
    c2h::host_vector<KeyType> h_unsorted_keys(num_items);
    c2h::host_vector<ValueType> h_sorted_values(num_items);

    TIME(timer.print_elapsed_seconds_and_reset("Host allocate"));

    {
      using range_t = typename thrust::random::uniform_int_distribution<std::size_t>::param_type;
      constexpr range_t run_range{1, 256};

      thrust::default_random_engine rng(static_cast<std::uint32_t>(seed.get()));
      thrust::random::uniform_int_distribution<std::size_t> dist;
      range_t summary_range{0, num_summaries - 1};
      for (std::size_t i = 0; i < num_items; /*inc in loop*/)
      {
        const std::size_t summ_idx = dist(rng, summary_range);
        summary_t& summary         = h_summaries[summ_idx];
        const std::size_t run_size = std::min(summary.count, dist(rng, run_range));

        std::fill(h_unsorted_keys.begin() + i, // formatting
                  h_unsorted_keys.begin() + i + run_size,
                  summary.key);
        std::iota(h_sorted_values.begin() + summary.index, // formatting
                  h_sorted_values.begin() + summary.index + run_size,
                  static_cast<ValueType>(i));

        i += run_size;
        summary.index += run_size;
        summary.count -= run_size;
        if (summary.count == 0)
        {
          using std::swap;
          swap(summary, h_summaries.back());
          h_summaries.pop_back();
          summary_range.second -= 1;
        }
      }
    }

    TIME(timer.print_elapsed_seconds_and_reset("Host-side summary processing"));

    // Release the host summary memory.
    REQUIRE(h_summaries.empty());
    h_summaries.shrink_to_fit();

    TIME(timer.print_elapsed_seconds_and_reset("Host summaries free"));

    // Copy the unsorted keys to device
    keys_in = h_unsorted_keys;
    h_unsorted_keys.clear();
    h_unsorted_keys.shrink_to_fit();

    TIME(timer.print_elapsed_seconds_and_reset("Unsorted keys H->D"));

    // Unsorted values are just a sequence
    values_in.resize(num_items);
    thrust::tabulate(c2h::device_policy, values_in.begin(), values_in.end(), detail::index_to_value<ValueType>{});

    TIME(timer.print_elapsed_seconds_and_reset("Unsorted value gen"));

    // Copy the sorted values to the member array.
    // Sorted keys are verified using a fancy iterator.
    values_ref = std::move(h_sorted_values); // Same memory space, just move.

    TIME(timer.print_elapsed_seconds_and_reset("Copy/move refs"));

    keys_out.resize(num_items);
    values_out.resize(num_items);

    TIME(timer.print_elapsed_seconds_and_reset("Prep device outputs"));

    keys_buffer =
      cub::DoubleBuffer<KeyType>(thrust::raw_pointer_cast(keys_in.data()), thrust::raw_pointer_cast(keys_out.data()));
    values_buffer = cub::DoubleBuffer<ValueType>(
      thrust::raw_pointer_cast(values_in.data()), thrust::raw_pointer_cast(values_out.data()));
  }

  // Verify the results of sorting the keys_in produced by initialize_for_stable_pair_sort.
  void verify_stable_pair_sort(
    std::size_t num_items,
    bool is_descending,
    const c2h::device_vector<KeyType>& keys,
    const c2h::device_vector<ValueType>& values)
  {
    static_assert(!::cuda::std::is_same<ValueType, cub::NullType>::value, "ValueType must be valid.");

    const std::size_t max_summaries = this->compute_max_summaries(num_items);
    const std::size_t num_summaries = this->compute_num_summaries(num_items, max_summaries);

    TIME(c2h::cpu_timer timer);

    auto ref_key_begin = thrust::make_transform_iterator(
      thrust::make_counting_iterator(std::size_t{0}),
      detail::pair_sort_ref_key_transform<KeyType>(num_items, num_summaries, is_descending));

    REQUIRE(thrust::equal(c2h::device_policy, keys.cbegin(), keys.cend(), ref_key_begin));

    TIME(timer.print_elapsed_seconds_and_reset("Validate keys"));

    REQUIRE((values == this->values_ref) == true);

    TIME(timer.print_elapsed_seconds_and_reset("Validate values"));
  }

private:
  // The maximum number of summaries that will fill the target memory footprint of one full set of key/value pairs.
  static std::size_t compute_max_summaries(std::size_t num_items)
  {
    using summary_t = detail::summary<KeyType>;

    const std::size_t max_summary_mem = num_items * (sizeof(KeyType) + sizeof(ValueType));
    const std::size_t max_summaries   = ::cuda::ceil_div(max_summary_mem, sizeof(summary_t));
    return max_summaries;
  }

  // The actual number of summaries to use, considering memory, key type, and number of items.
  static std::size_t compute_num_summaries(std::size_t num_items, std::size_t max_summaries)
  {
    constexpr KeyType max_key       = ::cuda::std::numeric_limits<KeyType>::max();
    const std::size_t num_summaries = std::min(std::min(max_summaries, num_items), static_cast<std::size_t>(max_key));
    return num_summaries;
  }
};
