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

#include <cub/device/device_segmented_radix_sort.cuh>
#include <cub/device/dispatch/dispatch_radix_sort.cuh> // DispatchSegmentedRadixSort
#include <cub/util_type.cuh>

#include <thrust/iterator/constant_iterator.h>
#include <thrust/memory.h>
#include <thrust/scatter.h>

#include <algorithm>
#include <limits>

#include "catch2_radix_sort_helper.cuh"
#include "catch2_test_launch_helper.h"
#include "thrust/detail/raw_pointer_cast.h"
#include <c2h/catch2_test_helper.cuh>

// TODO replace with DeviceSegmentedRadixSort::SortPairs interface once https://github.com/NVIDIA/cccl/issues/50 is
// addressed Temporary wrapper that allows specializing the DeviceSegmentedRadixSort algorithm for different offset
// types
template <bool IS_DESCENDING,
          typename KeyT,
          typename ValueT,
          typename NumItemsT,
          typename BeginOffsetIteratorT,
          typename EndOffsetIteratorT>
CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t dispatch_segmented_radix_sort_pairs_wrapper(
  void* d_temp_storage,
  size_t& temp_storage_bytes,
  const KeyT* d_keys_in,
  KeyT* d_keys_out,
  const ValueT* d_values_in,
  ValueT* d_values_out,
  NumItemsT num_items,
  NumItemsT num_segments,
  BeginOffsetIteratorT d_begin_offsets,
  EndOffsetIteratorT d_end_offsets,
  bool* selector,
  int begin_bit       = 0,
  int end_bit         = sizeof(KeyT) * 8,
  bool is_overwrite   = true,
  cudaStream_t stream = 0)
{
  cub::DoubleBuffer<KeyT> d_keys(const_cast<KeyT*>(d_keys_in), d_keys_out);
  cub::DoubleBuffer<ValueT> d_values(const_cast<ValueT*>(d_values_in), d_values_out);
  auto status = cub::DispatchSegmentedRadixSort<
    IS_DESCENDING,
    KeyT,
    ValueT,
    BeginOffsetIteratorT,
    EndOffsetIteratorT, //
    NumItemsT>::Dispatch(d_temp_storage,
                         temp_storage_bytes,
                         d_keys,
                         d_values,
                         num_items,
                         num_segments,
                         d_begin_offsets,
                         d_end_offsets,
                         begin_bit,
                         end_bit,
                         is_overwrite,
                         stream);
  if (status != cudaSuccess)
  {
    return status;
  }
  if (is_overwrite)
  {
    // Only write to selector in the DoubleBuffer invocation
    *selector = d_keys.Current() != d_keys_out;
  }
  return cudaSuccess;
}

// %PARAM% TEST_LAUNCH lid 0:1:2

DECLARE_LAUNCH_WRAPPER(cub::DeviceSegmentedRadixSort::SortPairs, sort_pairs);
DECLARE_LAUNCH_WRAPPER(cub::DeviceSegmentedRadixSort::SortPairsDescending, sort_pairs_descending);
DECLARE_LAUNCH_WRAPPER(dispatch_segmented_radix_sort_pairs_wrapper<true>,
                       dispatch_segmented_radix_sort_pairs_descending);
DECLARE_LAUNCH_WRAPPER(dispatch_segmented_radix_sort_pairs_wrapper<false>, dispatch_segmented_radix_sort_pairs);

using custom_value_t = c2h::custom_type_t<c2h::equal_comparable_t>;
using value_types    = c2h::type_list<cuda::std::uint8_t, cuda::std::uint64_t, custom_value_t>;

// Index types used for OffsetsT testing
C2H_TEST("DeviceSegmentedRadixSort::SortPairs: Basic testing",
         "[pairs][segmented][radix][sort][device]",
         value_types,
         offset_types)
{
  using key_t    = cuda::std::uint32_t;
  using value_t  = c2h::get<0, TestType>;
  using offset_t = c2h::get<1, TestType>;

  constexpr std::size_t min_num_items = 1 << 5;
  constexpr std::size_t max_num_items = 1 << 20;
  const std::size_t num_items         = GENERATE_COPY(take(3, random(min_num_items, max_num_items)));
  const std::size_t num_segments      = GENERATE_COPY(take(2, random(std::size_t{2}, num_items / 2)));

  c2h::device_vector<key_t> in_keys(num_items);
  const int num_key_seeds = 1;
  c2h::gen(C2H_SEED(num_key_seeds), in_keys);

  c2h::device_vector<value_t> in_values(num_items);
  const int num_value_seeds = 1;
  c2h::gen(C2H_SEED(num_value_seeds), in_values);

  c2h::device_vector<offset_t> offsets(num_segments + 1);
  const int num_segment_seeds = 1;
  generate_segment_offsets(C2H_SEED(num_segment_seeds), offsets, static_cast<offset_t>(num_items));

  // Initialize the output vectors by copying the inputs since not all items
  // may belong to a segment.
  c2h::device_vector<key_t> out_keys(in_keys);
  c2h::device_vector<value_t> out_values(in_values);

  const bool is_descending = GENERATE(false, true);

  CAPTURE(num_items, num_segments, is_descending);

  if (is_descending)
  {
    sort_pairs_descending(
      thrust::raw_pointer_cast(in_keys.data()),
      thrust::raw_pointer_cast(out_keys.data()),
      thrust::raw_pointer_cast(in_values.data()),
      thrust::raw_pointer_cast(out_values.data()),
      static_cast<int>(num_items),
      static_cast<int>(num_segments),
      // Mix pointers/iterators for segment info to test using different iterable types:
      thrust::raw_pointer_cast(offsets.data()),
      offsets.cbegin() + 1,
      begin_bit<key_t>(),
      end_bit<key_t>());
  }
  else
  {
    sort_pairs(
      thrust::raw_pointer_cast(in_keys.data()),
      thrust::raw_pointer_cast(out_keys.data()),
      thrust::raw_pointer_cast(in_values.data()),
      thrust::raw_pointer_cast(out_values.data()),
      static_cast<int>(num_items),
      static_cast<int>(num_segments),
      // Mix pointers/iterators for segment info to test using different iterable types:
      thrust::raw_pointer_cast(offsets.data()),
      offsets.cbegin() + 1,
      begin_bit<key_t>(),
      end_bit<key_t>());
  }

  auto refs        = segmented_radix_sort_reference(in_keys, in_values, is_descending, offsets);
  auto& ref_keys   = refs.first;
  auto& ref_values = refs.second;

  REQUIRE(ref_keys == out_keys);
  REQUIRE(ref_values == out_values);
}

C2H_TEST("DeviceSegmentedRadixSort::SortPairs: DoubleBuffer API", "[pairs][segmented][radix][sort][device]", value_types)
{
  using key_t    = cuda::std::uint32_t;
  using value_t  = c2h::get<0, TestType>;
  using offset_t = cuda::std::int32_t;

  constexpr std::size_t max_num_items = 1 << 18;
  const std::size_t num_items         = GENERATE_COPY(take(1, random(max_num_items / 2, max_num_items)));
  const std::size_t num_segments      = GENERATE_COPY(take(1, random(std::size_t{2}, num_items / 2)));

  c2h::device_vector<key_t> in_keys(num_items);
  const int num_key_seeds = 1;
  c2h::gen(C2H_SEED(num_key_seeds), in_keys);

  c2h::device_vector<value_t> in_values(num_items);
  const int num_value_seeds = 1;
  c2h::gen(C2H_SEED(num_value_seeds), in_values);

  c2h::device_vector<offset_t> offsets(num_segments + 1);
  const int num_segment_seeds = 1;
  generate_segment_offsets(C2H_SEED(num_segment_seeds), offsets, static_cast<offset_t>(num_items));

  // Initialize the output vectors by copying the inputs since not all items
  // may belong to a segment.
  c2h::device_vector<key_t> out_keys(in_keys);
  c2h::device_vector<value_t> out_values(in_values);

  const bool is_descending = GENERATE(false, true);

  CAPTURE(num_items, num_segments, is_descending);

  cub::DoubleBuffer<key_t> key_buffer(
    thrust::raw_pointer_cast(in_keys.data()), thrust::raw_pointer_cast(out_keys.data()));
  cub::DoubleBuffer<value_t> value_buffer(
    thrust::raw_pointer_cast(in_values.data()), thrust::raw_pointer_cast(out_values.data()));

  double_buffer_segmented_sort_t action(is_descending);
  action.initialize();
  launch(action,
         key_buffer,
         value_buffer,
         static_cast<int>(num_items),
         static_cast<int>(num_segments),
         // Mix pointers/iterators for segment info to test using different iterable types:
         thrust::raw_pointer_cast(offsets.data()),
         offsets.cbegin() + 1,
         begin_bit<key_t>(),
         end_bit<key_t>());

  key_buffer.selector   = action.selector();
  value_buffer.selector = action.selector();
  action.finalize();

  auto refs        = segmented_radix_sort_reference(in_keys, in_values, is_descending, offsets);
  auto& ref_keys   = refs.first;
  auto& ref_values = refs.second;

  auto& keys   = key_buffer.selector == 0 ? in_keys : out_keys;
  auto& values = value_buffer.selector == 0 ? in_values : out_values;

  REQUIRE(ref_keys == keys);
  REQUIRE(ref_values == values);
}

C2H_TEST("DeviceSegmentedRadixSort::SortPairs: unspecified ranges",
         "[pairs][segmented][radix][sort][device]",
         value_types)
{
  using key_t    = cuda::std::uint32_t;
  using value_t  = c2h::get<0, TestType>;
  using offset_t = cuda::std::int32_t;

  constexpr std::size_t max_num_items = 1 << 18;
  const std::size_t num_items         = GENERATE_COPY(take(1, random(max_num_items / 2, max_num_items)));
  const std::size_t num_segments      = GENERATE_COPY(take(1, random(std::size_t{2}, num_items / 2)));

  c2h::device_vector<key_t> in_keys(num_items);
  const int num_key_seeds = 1;
  c2h::gen(C2H_SEED(num_key_seeds), in_keys);

  c2h::device_vector<value_t> in_values(num_items);
  const int num_value_seeds = 1;
  c2h::gen(C2H_SEED(num_value_seeds), in_values);

  // Initialize the output vectors by copying the inputs since not all items
  // may belong to a segment.
  c2h::device_vector<key_t> out_keys(in_keys);
  c2h::device_vector<value_t> out_values(in_values);

  c2h::device_vector<offset_t> begin_offsets(num_segments + 1);
  const int num_segment_seeds = 1;
  generate_segment_offsets(C2H_SEED(num_segment_seeds), begin_offsets, static_cast<offset_t>(num_items));

  // Create separate begin/end offsets arrays and remove some of the segments by
  // setting both offsets to 0.
  c2h::device_vector<offset_t> end_offsets(begin_offsets.cbegin() + 1, begin_offsets.cend());
  begin_offsets.pop_back();

  {
    std::size_t num_empty_segments = num_segments / 16;
    c2h::device_vector<std::size_t> indices(num_empty_segments);
    c2h::gen(C2H_SEED(1), indices, std::size_t{0}, num_segments - 1);
    auto begin = thrust::make_constant_iterator(key_t{0});
    auto end   = begin + num_empty_segments;
    thrust::scatter(c2h::device_policy, begin, end, indices.cbegin(), begin_offsets.begin());
    thrust::scatter(c2h::device_policy, begin, end, indices.cbegin(), end_offsets.begin());
  }

  const bool is_descending = GENERATE(false, true);

  CAPTURE(num_items, num_segments, is_descending);

  if (is_descending)
  {
    sort_pairs_descending(
      thrust::raw_pointer_cast(in_keys.data()),
      thrust::raw_pointer_cast(out_keys.data()),
      thrust::raw_pointer_cast(in_values.data()),
      thrust::raw_pointer_cast(out_values.data()),
      static_cast<int>(num_items),
      static_cast<int>(num_segments),
      // Mix pointers/iterators for segment info to test using different iterable types:
      thrust::raw_pointer_cast(begin_offsets.data()),
      end_offsets.cbegin(),
      begin_bit<key_t>(),
      end_bit<key_t>());
  }
  else
  {
    sort_pairs(
      thrust::raw_pointer_cast(in_keys.data()),
      thrust::raw_pointer_cast(out_keys.data()),
      thrust::raw_pointer_cast(in_values.data()),
      thrust::raw_pointer_cast(out_values.data()),
      static_cast<int>(num_items),
      static_cast<int>(num_segments),
      // Mix pointers/iterators for segment info to test using different iterable types:
      thrust::raw_pointer_cast(begin_offsets.data()),
      end_offsets.cbegin(),
      begin_bit<key_t>(),
      end_bit<key_t>());
  }

  auto refs        = segmented_radix_sort_reference(in_keys, in_values, is_descending, begin_offsets, end_offsets);
  auto& ref_keys   = refs.first;
  auto& ref_values = refs.second;

  REQUIRE((ref_keys == out_keys) == true);
  REQUIRE((ref_values == out_values) == true);
}

#if defined(CCCL_TEST_ENABLE_LARGE_SEGMENTED_SORT)

C2H_TEST("DeviceSegmentedRadixSort::SortPairs: very large num. items and num. segments",
         "[pairs][segmented][radix][sort][device]",
         all_offset_types)
try
{
  using key_t                      = cuda::std::uint8_t; // minimize memory footprint to support a wider range of GPUs
  using value_t                    = cuda::std::uint8_t;
  using offset_t                   = c2h::get<0, TestType>;
  constexpr std::size_t Step       = 500;
  using segment_iterator_t         = segment_iterator<offset_t, Step>;
  constexpr std::size_t uint32_max = ::cuda::std::numeric_limits<std::uint32_t>::max();
  constexpr int num_key_seeds      = 1;
  constexpr int num_value_seeds    = 1;
  const bool is_descending         = GENERATE(false, true);
  const bool is_overwrite          = GENERATE(false, true);
  constexpr std::size_t num_items =
    (sizeof(offset_t) == 8) ? uint32_max + (1 << 20) : ::cuda::std::numeric_limits<offset_t>::max();
  const std::size_t num_segments = ::cuda::ceil_div(num_items, Step);
  CAPTURE(c2h::type_name<offset_t>(), num_items, num_segments, is_descending, is_overwrite);

  c2h::device_vector<key_t> in_keys(num_items);
  c2h::device_vector<value_t> in_values(num_items);
  c2h::gen(C2H_SEED(num_key_seeds), in_keys);
  c2h::gen(C2H_SEED(num_value_seeds), in_values);
  c2h::device_vector<key_t> out_keys(num_items);
  c2h::device_vector<value_t> out_values(num_items);
  auto offsets =
    thrust::make_transform_iterator(thrust::make_counting_iterator(std::size_t{0}), segment_iterator_t{num_items});
  auto offsets_plus_1 = offsets + 1;
  // Allocate host/device-accessible memory to communicate the selected output buffer
  bool* selector_ptr = nullptr;
  if (is_overwrite)
  {
    REQUIRE(cudaSuccess == cudaMallocHost(&selector_ptr, sizeof(*selector_ptr)));
  }

  auto refs = segmented_radix_sort_reference(in_keys, in_values, is_descending, num_segments, offsets, offsets_plus_1);
  auto& ref_keys      = refs.first;
  auto& ref_values    = refs.second;
  auto out_keys_ptr   = thrust::raw_pointer_cast(out_keys.data());
  auto out_values_ptr = thrust::raw_pointer_cast(out_values.data());
  if (is_descending)
  {
    dispatch_segmented_radix_sort_pairs_descending(
      thrust::raw_pointer_cast(in_keys.data()),
      out_keys_ptr,
      thrust::raw_pointer_cast(in_values.data()),
      out_values_ptr,
      static_cast<offset_t>(num_items),
      static_cast<offset_t>(num_segments),
      offsets,
      offsets_plus_1,
      selector_ptr,
      begin_bit<key_t>(),
      end_bit<key_t>(),
      is_overwrite);
  }
  else
  {
    dispatch_segmented_radix_sort_pairs(
      thrust::raw_pointer_cast(in_keys.data()),
      out_keys_ptr,
      thrust::raw_pointer_cast(in_values.data()),
      out_values_ptr,
      static_cast<offset_t>(num_items),
      static_cast<offset_t>(num_segments),
      // Mix pointers/iterators for segment info to test using different iterable types:
      offsets,
      offsets_plus_1,
      selector_ptr,
      begin_bit<key_t>(),
      end_bit<key_t>(),
      is_overwrite);
  }
  if (is_overwrite)
  {
    if (*selector_ptr)
    {
      std::swap(out_keys, in_keys);
      std::swap(out_values, in_values);
    }
    REQUIRE(cudaSuccess == cudaFreeHost(selector_ptr));
  }
  REQUIRE(ref_keys == out_keys);
  REQUIRE(ref_values == out_values);
}
catch (std::bad_alloc& e)
{
  std::cerr << "Skipping segmented radix sort test, unsufficient GPU memory. " << e.what() << "\n";
}

C2H_TEST("DeviceSegmentedRadixSort::SortPairs: very large segments",
         "[pairs][segmented][radix][sort][device]",
         all_offset_types)
try
{
  using key_t                      = cuda::std::uint8_t; // minimize memory footprint to support a wider range of GPUs
  using value_t                    = cuda::std::uint8_t;
  using offset_t                   = c2h::get<0, TestType>;
  constexpr std::size_t uint32_max = ::cuda::std::numeric_limits<std::uint32_t>::max();
  constexpr int num_key_seeds      = 1;
  constexpr int num_value_seeds    = 1;
  const bool is_descending         = GENERATE(false, true);
  const bool is_overwrite          = GENERATE(false, true);
  constexpr std::size_t num_items = (sizeof(offset_t) == 8) ? uint32_max : ::cuda::std::numeric_limits<offset_t>::max();
  constexpr std::size_t num_segments = 2;
  CAPTURE(c2h::type_name<offset_t>(), num_items, is_descending, is_overwrite);

  c2h::device_vector<key_t> in_keys(num_items);
  c2h::device_vector<value_t> in_values(num_items);
  c2h::device_vector<key_t> out_keys(num_items);
  c2h::gen(C2H_SEED(num_key_seeds), in_keys);
  c2h::gen(C2H_SEED(num_value_seeds), in_values);
  c2h::device_vector<value_t> out_values(num_items);
  c2h::device_vector<offset_t> offsets(num_segments + 1);
  offsets[0] = 0;
  offsets[1] = static_cast<offset_t>(num_items);
  offsets[2] = static_cast<offset_t>(num_items);
  // Allocate host/device-accessible memory to communicate the selected output buffer
  bool* selector_ptr = nullptr;
  if (is_overwrite)
  {
    REQUIRE(cudaSuccess == cudaMallocHost(&selector_ptr, sizeof(*selector_ptr)));
  }

  auto refs = segmented_radix_sort_reference(
    in_keys, in_values, is_descending, num_segments, offsets.cbegin(), offsets.cbegin() + 1);
  auto& ref_keys      = refs.first;
  auto& ref_values    = refs.second;
  auto out_keys_ptr   = thrust::raw_pointer_cast(out_keys.data());
  auto out_values_ptr = thrust::raw_pointer_cast(out_values.data());
  if (is_descending)
  {
    dispatch_segmented_radix_sort_pairs_descending(
      thrust::raw_pointer_cast(in_keys.data()),
      out_keys_ptr,
      thrust::raw_pointer_cast(in_values.data()),
      out_values_ptr,
      static_cast<offset_t>(num_items),
      static_cast<offset_t>(num_segments),
      thrust::raw_pointer_cast(offsets.data()),
      offsets.cbegin() + 1,
      selector_ptr,
      begin_bit<key_t>(),
      end_bit<key_t>(),
      is_overwrite);
  }
  else
  {
    dispatch_segmented_radix_sort_pairs(
      thrust::raw_pointer_cast(in_keys.data()),
      out_keys_ptr,
      thrust::raw_pointer_cast(in_values.data()),
      out_values_ptr,
      static_cast<offset_t>(num_items),
      static_cast<offset_t>(num_segments),
      thrust::raw_pointer_cast(offsets.data()),
      offsets.cbegin() + 1,
      selector_ptr,
      begin_bit<key_t>(),
      end_bit<key_t>(),
      is_overwrite);
  }
  if (out_keys_ptr != thrust::raw_pointer_cast(out_keys.data()))
  {
    std::swap(out_keys, in_keys);
    std::swap(out_values, in_values);
  }
  if (is_overwrite)
  {
    if (*selector_ptr)
    {
      std::swap(out_keys, in_keys);
      std::swap(out_values, in_values);
    }
    REQUIRE(cudaSuccess == cudaFreeHost(selector_ptr));
  }
  REQUIRE(ref_keys == out_keys);
  REQUIRE(ref_values == out_values);
}
catch (std::bad_alloc& e)
{
  std::cerr << "Skipping segmented radix sort test, unsufficient GPU memory. " << e.what() << "\n";
}

#endif // defined(CCCL_TEST_ENABLE_LARGE_SEGMENTED_SORT)
