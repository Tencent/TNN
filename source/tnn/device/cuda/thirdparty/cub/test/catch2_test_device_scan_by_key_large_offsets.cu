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

#include <cub/device/device_scan.cuh>

#include <cstdint>

#include "catch2_test_launch_helper.h"
#include <c2h/catch2_test_helper.cuh>

DECLARE_LAUNCH_WRAPPER(cub::DeviceScan::ExclusiveScanByKey, device_exclusive_scan_by_key);
DECLARE_LAUNCH_WRAPPER(cub::DeviceScan::InclusiveScanByKey, device_inclusive_scan_by_key);

// %PARAM% TEST_LAUNCH lid 0:1:2

// List of offset types to be used for testing large number of items
// using offset_types = c2h::type_list<std::uint32_t, std::uint64_t>;
using offset_types = c2h::type_list<std::uint64_t>;

template <typename ItemT, bool IsExclusive>
struct expected_sum_op
{
  uint64_t segment_size;
  ItemT init_value;

  __host__ __device__ __forceinline__ ItemT operator()(const uint64_t index) const
  {
    uint64_t index_within_segment = index % segment_size;
    uint64_t full_segments        = index / segment_size;

    uint64_t sum_within_partial_segment{};
    if (IsExclusive)
    {
      sum_within_partial_segment = (index_within_segment * (index_within_segment - 1)) / 2;
    }
    else
    {
      sum_within_partial_segment = (index_within_segment * (index_within_segment + 1)) / 2;
    }
    return index_within_segment == 0
           ? (IsExclusive ? init_value : init_value + full_segments)
           : static_cast<ItemT>(sum_within_partial_segment + full_segments) + init_value;
  }
};

template <typename ItemT>
struct mod_op
{
  uint64_t segment_size;

  __host__ __device__ __forceinline__ uint64_t operator()(const uint64_t index) const
  {
    auto mod = static_cast<ItemT>(index % segment_size);
    auto div = static_cast<ItemT>(index / segment_size);
    return mod == 0 ? div : mod;
  }
};

template <typename KeyT>
struct div_op
{
  uint64_t segment_size;

  __host__ __device__ __forceinline__ KeyT operator()(const uint64_t index) const
  {
    return static_cast<KeyT>(index / segment_size);
  }
};

C2H_TEST("DeviceScan::ScanByKey works for very large number of items", "[by_key][scan][device]", offset_types)
try
{
  using op_t     = cub::Sum;
  using item_t   = std::uint32_t;
  using key_t    = std::uint64_t;
  using index_t  = std::uint64_t;
  using offset_t = typename c2h::get<0, TestType>;

  // Clamp 64-bit offset type problem sizes to just slightly larger than 2^32 items
  auto num_items_max_ull =
    std::min(static_cast<std::size_t>(::cuda::std::numeric_limits<offset_t>::max()),
             ::cuda::std::numeric_limits<std::uint32_t>::max() + static_cast<std::size_t>(2000000ULL));
  offset_t num_items_max = static_cast<offset_t>(num_items_max_ull);
  offset_t num_items_min =
    num_items_max_ull > 10000 ? static_cast<offset_t>(num_items_max_ull - 10000ULL) : offset_t{0};
  offset_t num_items = GENERATE_COPY(
    values(
      {num_items_max, static_cast<offset_t>(num_items_max - 1), static_cast<offset_t>(1), static_cast<offset_t>(3)}),
    take(2, random(num_items_min, num_items_max)));

  // Prepare input (generate a series of: 0, 1, 2, ..., <segment_size-1>,  1, 1, 2, ..., <segment_size-1>, 2, 1, ...)
  const index_t segment_size = GENERATE_COPY(values({offset_t{1000}, offset_t{1}}));
  auto index_it              = thrust::make_counting_iterator(index_t{});
  auto keys_it               = thrust::make_transform_iterator(index_it, div_op<key_t>{segment_size});
  auto items_it              = thrust::make_transform_iterator(index_it, mod_op<item_t>{segment_size});

  // Output memory allocation
  c2h::device_vector<item_t> d_items_out(num_items);
  auto d_items_out_it = thrust::raw_pointer_cast(d_items_out.data());

  // Run test
  SECTION("ExclusiveScanByKey")
  {
    constexpr bool is_exclusive = true;
    auto initial_value          = item_t{42};
    device_exclusive_scan_by_key(keys_it, items_it, d_items_out_it, op_t{}, initial_value, num_items, cub::Equality{});

    // Ensure that we created the correct output
    auto expected_out_it = thrust::make_transform_iterator(
      index_it, expected_sum_op<item_t, is_exclusive>{static_cast<index_t>(segment_size), initial_value});
    bool all_results_correct = thrust::equal(d_items_out.cbegin(), d_items_out.cend(), expected_out_it);
    REQUIRE(all_results_correct == true);
  }
  SECTION("InclusiveScanByKey")
  {
    constexpr bool is_exclusive = false;
    auto initial_value          = item_t{0};
    device_inclusive_scan_by_key(keys_it, items_it, d_items_out_it, op_t{}, num_items, cub::Equality{});

    // Ensure that we created the correct output
    auto expected_out_it = thrust::make_transform_iterator(
      index_it, expected_sum_op<item_t, is_exclusive>{static_cast<index_t>(segment_size), initial_value});
    bool all_results_correct = thrust::equal(d_items_out.cbegin(), d_items_out.cend(), expected_out_it);
    REQUIRE(all_results_correct == true);
  }
}
catch (std::bad_alloc&)
{
  // Exceeding memory is not a failure.
}
