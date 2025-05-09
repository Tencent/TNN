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

#include <cub/device/device_segmented_reduce.cuh>

#include <thrust/iterator/constant_iterator.h>

#include <cstdint>

#include "catch2/catch.hpp"
#include "catch2_test_launch_helper.h"
#include <c2h/catch2_test_helper.cuh>

DECLARE_LAUNCH_WRAPPER(cub::DeviceSegmentedReduce::Reduce, device_segmented_reduce);
DECLARE_LAUNCH_WRAPPER(cub::DeviceSegmentedReduce::Sum, device_segmented_sum);

// %PARAM% TEST_LAUNCH lid 0:1

// List of types to test
using offsets = c2h::type_list<std::ptrdiff_t, std::size_t>;

C2H_TEST("Device segmented reduce works with fancy input iterators and 64-bit offsets", "[reduce][device]", offsets)
{
  using offset_t = typename c2h::get<0, TestType>;
  using op_t     = cub::Sum;

  constexpr offset_t offset_zero           = 0;
  constexpr offset_t offset_one            = 1;
  constexpr offset_t iterator_value        = 2;
  constexpr offset_t min_items_per_segment = offset_one << 31;
  constexpr offset_t max_items_per_segment = offset_one << 33;

  constexpr int num_segments = 2;

  // generate individual segment lengths and store cumulative sum in segment_offsets
  const offset_t num_items_in_first_segment =
    GENERATE_COPY(take(2, random(min_items_per_segment, max_items_per_segment)));
  const offset_t num_items_in_second_segment =
    GENERATE_COPY(take(2, random(min_items_per_segment, max_items_per_segment)));
  c2h::device_vector<offset_t> segment_offsets = {
    offset_zero, num_items_in_first_segment, num_items_in_first_segment + num_items_in_second_segment};

  // store expected result and initialize device output container
  c2h::host_vector<offset_t> expected_result = {
    iterator_value * num_items_in_first_segment, iterator_value * num_items_in_second_segment};
  c2h::device_vector<offset_t> device_result(num_segments);

  // prepare device iterators
  auto in_it        = thrust::make_constant_iterator(iterator_value);
  auto d_offsets_it = thrust::raw_pointer_cast(segment_offsets.data());
  auto d_out_it     = thrust::raw_pointer_cast(device_result.data());

  // reduce
  device_segmented_reduce(in_it, d_out_it, num_segments, d_offsets_it, d_offsets_it + 1, op_t{}, offset_t{});

  // verify result
  REQUIRE(expected_result == device_result);
}
