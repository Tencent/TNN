/******************************************************************************
 * Copyright (c) 2011-2023, NVIDIA CORPORATION.  All rights reserved.
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

// #define CCCL_TEST_ENABLE_LARGE_SEGMENTED_SORT
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_segmented_radix_sort.cuh>
#include <cub/util_macro.cuh>
#include <cub/util_math.cuh>
#include <cub/util_type.cuh>

#include <thrust/gather.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/memory.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>

#include <cuda/std/bit>

#include <array>
#include <climits>
#include <cstdint>

#include <c2h/catch2_test_helper.cuh>
#include <c2h/generators.cuh>
#include <c2h/utility.cuh>
#include <c2h/vector.cuh>

// Index types used for OffsetsT testing
using offset_types = c2h::type_list<cuda::std::int32_t, cuda::std::uint64_t>;
using all_offset_types =
  c2h::type_list<cuda::std::int32_t, cuda::std::uint32_t, cuda::std::int64_t, cuda::std::uint64_t>;

// Create a segment iterator that returns the next multiple of Step except for a few cases. This allows to save memory
template <typename OffsetT, OffsetT Step>
struct segment_iterator
{
  OffsetT last = 0;

  segment_iterator(OffsetT last1)
      : last{last1}
  {}

  __host__ __device__ OffsetT operator()(OffsetT x) const
  {
    switch (x)
    {
      case Step * 100:
        return Step * 100 + Step / 2;
      case Step * 200:
        return Step * 200 + Step / 2;
      case Step * 300:
        return Step * 300 + Step / 2;
      case Step * 400:
        return Step * 400 + Step / 2;
      case Step * 500:
        return Step * 500 + Step / 2;
      case Step * 600:
        return Step * 600 + Step / 2;
      case Step * 700:
        return Step * 700 + Step / 2;
      case Step * 800:
        return Step * 800 + Step / 2;
      case Step * 900:
        return Step * 900 + Step / 2;
      default:
        return (x >= last) ? last : x * Step;
    }
  }
};

// The launchers defined in catch2_test_launch_helper.h do not support
// passing objects by reference since the device-launch tests cannot
// pass references to a __global__ function. The DoubleBuffer object
// must be passed by reference to the radix sort APIs so that the selector
// can be updated appropriately for the caller. This wrapper allows the
// selector to be updated in a way that's compatible with the launch helpers.
// Call initialize() before using to allocate temporary memory, and finalize()
// when finished to release.
struct double_buffer_sort_t
{
private:
  bool m_is_descending;
  int* m_selector;

public:
  explicit double_buffer_sort_t(bool is_descending)
      : m_is_descending(is_descending)
      , m_selector(nullptr)
  {}

  void initialize()
  {
    REQUIRE(cudaSuccess == cudaMallocHost(&m_selector, sizeof(int)));
  }

  void finalize()
  {
    REQUIRE(cudaSuccess == cudaFreeHost(m_selector));
    m_selector = nullptr;
  }

  int selector() const
  {
    return *m_selector;
  }

  template <class KeyT, class... As>
  CUB_RUNTIME_FUNCTION cudaError_t
  operator()(std::uint8_t* d_temp_storage, std::size_t& temp_storage_bytes, cub::DoubleBuffer<KeyT> keys, As... as)
  {
    const cudaError_t status =
      m_is_descending ? cub::DeviceRadixSort::SortKeysDescending(d_temp_storage, temp_storage_bytes, keys, as...)
                      : cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, keys, as...);

    *m_selector = keys.selector;
    return status;
  }

  template <class KeyT, class ValueT, class... As>
  CUB_RUNTIME_FUNCTION cudaError_t operator()(
    std::uint8_t* d_temp_storage,
    std::size_t& temp_storage_bytes,
    cub::DoubleBuffer<KeyT> keys,
    cub::DoubleBuffer<ValueT> values,
    As... as)
  {
    const cudaError_t status =
      m_is_descending
        ? cub::DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes, keys, values, as...)
        : cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, keys, values, as...);

    *m_selector = keys.selector;
    return status;
  }
};

struct double_buffer_segmented_sort_t
{
private:
  bool m_is_descending;
  int* m_selector;

public:
  explicit double_buffer_segmented_sort_t(bool is_descending)
      : m_is_descending(is_descending)
      , m_selector(nullptr)
  {}

  void initialize()
  {
    REQUIRE(cudaSuccess == cudaMallocHost(&m_selector, sizeof(int)));
  }

  void finalize()
  {
    REQUIRE(cudaSuccess == cudaFreeHost(m_selector));
    m_selector = nullptr;
  }

  int selector() const
  {
    return *m_selector;
  }

  template <class KeyT, class... As>
  CUB_RUNTIME_FUNCTION cudaError_t
  operator()(std::uint8_t* d_temp_storage, std::size_t& temp_storage_bytes, cub::DoubleBuffer<KeyT> keys, As... as)
  {
    const cudaError_t status =
      m_is_descending
        ? cub::DeviceSegmentedRadixSort::SortKeysDescending(d_temp_storage, temp_storage_bytes, keys, as...)
        : cub::DeviceSegmentedRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, keys, as...);

    *m_selector = keys.selector;
    return status;
  }

  template <class KeyT, class ValueT, class... As>
  CUB_RUNTIME_FUNCTION cudaError_t operator()(
    std::uint8_t* d_temp_storage,
    std::size_t& temp_storage_bytes,
    cub::DoubleBuffer<KeyT> keys,
    cub::DoubleBuffer<ValueT> values,
    As... as)
  {
    const cudaError_t status =
      m_is_descending
        ? cub::DeviceSegmentedRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes, keys, values, as...)
        : cub::DeviceSegmentedRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, keys, values, as...);

    *m_selector = keys.selector;
    return status;
  }
};

// Helpers to assist with specifying default args to DeviceRadixSort API:
template <typename T>
constexpr int begin_bit()
{
  return 0;
}

template <typename T>
constexpr int end_bit()
{
  return static_cast<int>(sizeof(T) * CHAR_BIT);
}

template <class KeyT>
c2h::host_vector<KeyT> get_striped_keys(const c2h::host_vector<KeyT>& h_keys, int begin_bit, int end_bit)
{
  c2h::host_vector<KeyT> h_striped_keys(h_keys);
  KeyT* h_striped_keys_data = thrust::raw_pointer_cast(h_striped_keys.data());

  using traits_t      = cub::Traits<KeyT>;
  using bit_ordered_t = typename traits_t::UnsignedBits;

  const int num_bits = end_bit - begin_bit;

  for (std::size_t i = 0; i < h_keys.size(); i++)
  {
    bit_ordered_t key = ::cuda::std::bit_cast<bit_ordered_t>(h_keys[i]);

    _CCCL_IF_CONSTEXPR (traits_t::CATEGORY == cub::FLOATING_POINT)
    {
      const bit_ordered_t negative_zero = bit_ordered_t(1) << bit_ordered_t(sizeof(bit_ordered_t) * 8 - 1);

      if (key == negative_zero)
      {
        key = 0;
      }
    }

    key = traits_t::TwiddleIn(key);

    if ((begin_bit > 0) || (end_bit < static_cast<int>(sizeof(KeyT) * 8)))
    {
      key &= ((bit_ordered_t{1} << num_bits) - 1) << begin_bit;
    }

    // striped keys are used to compare bit ordered representation of keys,
    // so we do not twiddle-out the key here:
    // key = traits_t::TwiddleOut(key);

    memcpy(h_striped_keys_data + i, &key, sizeof(KeyT));
  }

  return h_striped_keys;
}

template <class T>
struct indirect_binary_comparator_t
{
  const T* h_ptr{};
  bool is_descending{};

  indirect_binary_comparator_t(const T* h_ptr, bool is_descending)
      : h_ptr(h_ptr)
      , is_descending(is_descending)
  {}

  bool operator()(std::size_t a, std::size_t b)
  {
    if (is_descending)
    {
      return h_ptr[a] > h_ptr[b];
    }

    return h_ptr[a] < h_ptr[b];
  }
};

template <class KeyT, class SegBeginIterT, class SegEndIterT>
c2h::host_vector<std::size_t> get_permutation(
  const c2h::host_vector<KeyT>& h_keys,
  bool is_descending,
  std::size_t num_segments,
  SegBeginIterT h_seg_begin_it,
  SegEndIterT h_seg_end_it,
  int begin_bit,
  int end_bit)
{
  c2h::host_vector<KeyT> h_striped_keys = get_striped_keys(h_keys, begin_bit, end_bit);

  c2h::host_vector<std::size_t> h_permutation(h_keys.size());
  thrust::sequence(h_permutation.begin(), h_permutation.end());

  using traits_t      = cub::Traits<KeyT>;
  using bit_ordered_t = typename traits_t::UnsignedBits;

  auto bit_ordered_striped_keys =
    reinterpret_cast<const bit_ordered_t*>(thrust::raw_pointer_cast(h_striped_keys.data()));

  indirect_binary_comparator_t<bit_ordered_t> comp{bit_ordered_striped_keys, is_descending};

  for (std::size_t segment = 0; segment < num_segments; ++segment)
  {
    std::stable_sort(
      h_permutation.begin() + h_seg_begin_it[segment], h_permutation.begin() + h_seg_end_it[segment], comp);
  }

  return h_permutation;
}

template <class KeyT>
c2h::host_vector<KeyT> radix_sort_reference(
  const c2h::device_vector<KeyT>& d_keys,
  bool is_descending,
  int begin_bit = 0,
  int end_bit   = static_cast<int>(sizeof(KeyT) * CHAR_BIT))
{
  c2h::host_vector<KeyT> h_keys(d_keys);
  std::array<std::size_t, 2> segments{0, d_keys.size()};
  c2h::host_vector<std::size_t> h_permutation =
    get_permutation(h_keys, is_descending, 1, segments.cbegin(), segments.cbegin() + 1, begin_bit, end_bit);
  c2h::host_vector<KeyT> result(d_keys.size());
  thrust::gather(h_permutation.cbegin(), h_permutation.cend(), h_keys.cbegin(), result.begin());

  return result;
}

template <class KeyT, class ValueT>
std::pair<c2h::host_vector<KeyT>, c2h::host_vector<ValueT>> radix_sort_reference(
  const c2h::device_vector<KeyT>& d_keys,
  const c2h::device_vector<ValueT>& d_values,
  bool is_descending,
  int begin_bit = 0,
  int end_bit   = static_cast<int>(sizeof(KeyT) * CHAR_BIT))
{
  std::pair<c2h::host_vector<KeyT>, c2h::host_vector<ValueT>> result;
  result.first.resize(d_keys.size());
  result.second.resize(d_keys.size());

  std::array<std::size_t, 2> segments{0, d_keys.size()};

  c2h::host_vector<KeyT> h_keys(d_keys);
  c2h::host_vector<std::size_t> h_permutation =
    get_permutation(h_keys, is_descending, 1, segments.cbegin(), segments.cbegin() + 1, begin_bit, end_bit);

  c2h::host_vector<ValueT> h_values(d_values);
  thrust::gather(h_permutation.cbegin(),
                 h_permutation.cend(),
                 thrust::make_zip_iterator(h_keys.cbegin(), h_values.cbegin()),
                 thrust::make_zip_iterator(result.first.begin(), result.second.begin()));

  return result;
}

template <class KeyT, class SegBeginIterT, class SegEndIterT>
c2h::host_vector<KeyT> segmented_radix_sort_reference(
  const c2h::device_vector<KeyT>& d_keys,
  bool is_descending,
  std::size_t num_segments,
  SegBeginIterT h_seg_begin_it,
  SegEndIterT h_seg_end_it,
  int begin_bit = 0,
  int end_bit   = static_cast<int>(sizeof(KeyT) * CHAR_BIT))
{
  c2h::host_vector<KeyT> h_keys(d_keys);
  c2h::host_vector<std::size_t> h_permutation =
    get_permutation(h_keys, is_descending, num_segments, h_seg_begin_it, h_seg_end_it, begin_bit, end_bit);
  c2h::host_vector<KeyT> result(d_keys.size());
  thrust::gather(h_permutation.cbegin(), h_permutation.cend(), h_keys.cbegin(), result.begin());

  return result;
}

template <class KeyT, class ValueT, class SegBeginIterT, class SegEndIterT>
std::pair<c2h::host_vector<KeyT>, c2h::host_vector<ValueT>> segmented_radix_sort_reference(
  const c2h::device_vector<KeyT>& d_keys,
  const c2h::device_vector<ValueT>& d_values,
  bool is_descending,
  std::size_t num_segments,
  SegBeginIterT h_seg_begin_it,
  SegEndIterT h_seg_end_it,
  int begin_bit = 0,
  int end_bit   = static_cast<int>(sizeof(KeyT) * CHAR_BIT))
{
  std::pair<c2h::host_vector<KeyT>, c2h::host_vector<ValueT>> result;
  result.first.resize(d_keys.size());
  result.second.resize(d_keys.size());

  c2h::host_vector<KeyT> h_keys(d_keys);
  c2h::host_vector<std::size_t> h_permutation =
    get_permutation(h_keys, is_descending, num_segments, h_seg_begin_it, h_seg_end_it, begin_bit, end_bit);

  c2h::host_vector<ValueT> h_values(d_values);
  thrust::gather(h_permutation.cbegin(),
                 h_permutation.cend(),
                 thrust::make_zip_iterator(h_keys.cbegin(), h_values.cbegin()),
                 thrust::make_zip_iterator(result.first.begin(), result.second.begin()));

  return result;
}

template <class KeyT, class OffsetT>
c2h::host_vector<KeyT> segmented_radix_sort_reference(
  const c2h::device_vector<KeyT>& d_keys,
  bool is_descending,
  const c2h::device_vector<OffsetT>& d_offsets,
  int begin_bit = 0,
  int end_bit   = static_cast<int>(sizeof(KeyT) * CHAR_BIT))
{
  const c2h::host_vector<OffsetT> h_offsets(d_offsets);
  const std::size_t num_segments = h_offsets.size() - 1;
  auto h_seg_begin_it            = h_offsets.cbegin();
  auto h_seg_end_it              = h_offsets.cbegin() + 1;
  return segmented_radix_sort_reference(
    d_keys, is_descending, num_segments, h_seg_begin_it, h_seg_end_it, begin_bit, end_bit);
}

template <class KeyT, class OffsetT>
c2h::host_vector<KeyT> segmented_radix_sort_reference(
  const c2h::device_vector<KeyT>& d_keys,
  bool is_descending,
  const c2h::device_vector<OffsetT>& d_offsets_begin,
  const c2h::device_vector<OffsetT>& d_offsets_end,
  int begin_bit = 0,
  int end_bit   = static_cast<int>(sizeof(KeyT) * CHAR_BIT))
{
  const c2h::host_vector<OffsetT> h_offsets_begin(d_offsets_begin);
  const c2h::host_vector<OffsetT> h_offsets_end(d_offsets_end);
  const std::size_t num_segments = h_offsets_begin.size();
  auto h_seg_begin_it            = h_offsets_begin.cbegin();
  auto h_seg_end_it              = h_offsets_end.cbegin();
  return segmented_radix_sort_reference(
    d_keys, is_descending, num_segments, h_seg_begin_it, h_seg_end_it, begin_bit, end_bit);
}

template <class KeyT, class ValueT, class OffsetT>
std::pair<c2h::host_vector<KeyT>, c2h::host_vector<ValueT>> segmented_radix_sort_reference(
  const c2h::device_vector<KeyT>& d_keys,
  const c2h::device_vector<ValueT>& d_values,
  bool is_descending,
  const c2h::device_vector<OffsetT>& d_offsets,
  int begin_bit = 0,
  int end_bit   = static_cast<int>(sizeof(KeyT) * CHAR_BIT))
{
  const c2h::host_vector<OffsetT> h_offsets(d_offsets);
  const std::size_t num_segments = h_offsets.size() - 1;
  auto h_seg_begin_it            = h_offsets.cbegin();
  auto h_seg_end_it              = h_offsets.cbegin() + 1;
  return segmented_radix_sort_reference(
    d_keys, d_values, is_descending, num_segments, h_seg_begin_it, h_seg_end_it, begin_bit, end_bit);
}

template <class KeyT, class ValueT, class OffsetT>
std::pair<c2h::host_vector<KeyT>, c2h::host_vector<ValueT>> segmented_radix_sort_reference(
  const c2h::device_vector<KeyT>& d_keys,
  const c2h::device_vector<ValueT>& d_values,
  bool is_descending,
  const c2h::device_vector<OffsetT>& d_offsets_begin,
  const c2h::device_vector<OffsetT>& d_offsets_end,
  int begin_bit = 0,
  int end_bit   = static_cast<int>(sizeof(KeyT) * CHAR_BIT))
{
  const c2h::host_vector<OffsetT> h_offsets_begin(d_offsets_begin);
  const c2h::host_vector<OffsetT> h_offsets_end(d_offsets_end);
  const std::size_t num_segments = h_offsets_begin.size();
  auto h_seg_begin_it            = h_offsets_begin.cbegin();
  auto h_seg_end_it              = h_offsets_end.cbegin();
  return segmented_radix_sort_reference(
    d_keys, d_values, is_descending, num_segments, h_seg_begin_it, h_seg_end_it, begin_bit, end_bit);
}

template <typename OffsetT>
struct radix_offset_scan_op_t
{
  OffsetT num_items;

  __host__ __device__ OffsetT operator()(OffsetT a, OffsetT b) const
  {
    const OffsetT sum = a + b;
    return CUB_MIN(sum, num_items);
  }
};

template <class OffsetT>
void generate_segment_offsets(c2h::seed_t seed, c2h::device_vector<OffsetT>& offsets, std::size_t num_items)
{
  const std::size_t num_segments        = offsets.size() - 1;
  const OffsetT expected_segment_length = static_cast<OffsetT>(::cuda::ceil_div(num_items, num_segments));
  const OffsetT max_segment_length      = (expected_segment_length * 2) + 1;
  c2h::gen(seed, offsets, OffsetT{0}, max_segment_length);
  thrust::exclusive_scan(
    c2h::device_policy,
    offsets.begin(),
    offsets.end(),
    offsets.begin(),
    OffsetT{0},
    radix_offset_scan_op_t<OffsetT>{static_cast<OffsetT>(num_items)});
}
