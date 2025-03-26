/******************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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

#include <thrust/device_vector.h>
#include <thrust/equal.h>

#include <climits>
#include <cstddef>

#include "thrust/detail/raw_pointer_cast.h"
#include <c2h/catch2_test_helper.cuh>

// example-begin segmented-reduce-custommin
struct CustomMin
{
  template <typename T>
  __device__ __forceinline__ T operator()(const T& a, const T& b) const
  {
    return (b < a) ? b : a;
  }
};

// example-end segmented-reduce-custommin

struct is_equal
{
  __device__ bool operator()(cub::KeyValuePair<int, int> lhs, cub::KeyValuePair<int, int> rhs)
  {
    return !(lhs != rhs);
  }
};

C2H_TEST("cub::DeviceSegmentedReduce::Reduce works with int data elements", "[segmented_reduce][device]")
{
  // example-begin segmented-reduce-reduce
  int num_segments                     = 3;
  thrust::device_vector<int> d_offsets = {0, 3, 3, 7};
  auto d_offsets_it                    = thrust::raw_pointer_cast(d_offsets.data());
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9};
  thrust::device_vector<int> d_out(3);
  CustomMin min_op;
  int initial_value{INT_MAX};

  // Determine temporary device storage requirements
  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceSegmentedReduce::Reduce(
    d_temp_storage,
    temp_storage_bytes,
    d_in.begin(),
    d_out.begin(),
    num_segments,
    d_offsets_it,
    d_offsets_it + 1,
    min_op,
    initial_value);

  thrust::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
  d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

  // Run reduction
  cub::DeviceSegmentedReduce::Reduce(
    d_temp_storage,
    temp_storage_bytes,
    d_in.begin(),
    d_out.begin(),
    num_segments,
    d_offsets_it,
    d_offsets_it + 1,
    min_op,
    initial_value);

  thrust::device_vector<int> expected{6, INT_MAX, 0};
  // example-end segmented-reduce-reduce

  REQUIRE(d_out == expected);
}

C2H_TEST("cub::DeviceSegmentedReduce::Sum works with int data elements", "[segmented_reduce][device]")
{
  // example-begin segmented-reduce-sum
  int num_segments                     = 3;
  thrust::device_vector<int> d_offsets = {0, 3, 3, 7};
  auto d_offsets_it                    = thrust::raw_pointer_cast(d_offsets.data());
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9};
  thrust::device_vector<int> d_out(3);

  // Determine temporary device storage requirements
  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceSegmentedReduce::Sum(
    d_temp_storage, temp_storage_bytes, d_in.begin(), d_out.begin(), num_segments, d_offsets_it, d_offsets_it + 1);

  thrust::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
  d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

  // Run reduction
  cub::DeviceSegmentedReduce::Sum(
    d_temp_storage, temp_storage_bytes, d_in.begin(), d_out.begin(), num_segments, d_offsets_it, d_offsets_it + 1);

  thrust::device_vector<int> expected{21, 0, 17};
  // example-end segmented-reduce-sum

  REQUIRE(d_out == expected);
}

C2H_TEST("cub::DeviceSegmentedReduce::Min works with int data elements", "[segmented_reduce][device]")
{
  // example-begin segmented-reduce-min
  int num_segments                     = 3;
  thrust::device_vector<int> d_offsets = {0, 3, 3, 7};
  auto d_offsets_it                    = thrust::raw_pointer_cast(d_offsets.data());
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9};
  thrust::device_vector<int> d_out(3);

  // Determine temporary device storage requirements
  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceSegmentedReduce::Min(
    d_temp_storage, temp_storage_bytes, d_in.begin(), d_out.begin(), num_segments, d_offsets_it, d_offsets_it + 1);

  thrust::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
  d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

  // Run reduction
  cub::DeviceSegmentedReduce::Min(
    d_temp_storage, temp_storage_bytes, d_in.begin(), d_out.begin(), num_segments, d_offsets_it, d_offsets_it + 1);

  thrust::device_vector<int> expected{6, INT_MAX, 0};
  // example-end segmented-reduce-min

  REQUIRE(d_out == expected);
}

C2H_TEST("cub::DeviceSegmentedReduce::ArgMin works with int data elements", "[segmented_reduce][device]")
{
  // example-begin segmented-reduce-argmin
  int num_segments                     = 3;
  thrust::device_vector<int> d_offsets = {0, 3, 3, 7};
  auto d_offsets_it                    = thrust::raw_pointer_cast(d_offsets.data());
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9};
  thrust::device_vector<cub::KeyValuePair<int, int>> d_out(3);

  // Determine temporary device storage requirements
  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceSegmentedReduce::ArgMin(
    d_temp_storage, temp_storage_bytes, d_in.begin(), d_out.begin(), num_segments, d_offsets_it, d_offsets_it + 1);

  thrust::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
  d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

  // Run reduction
  cub::DeviceSegmentedReduce::ArgMin(
    d_temp_storage, temp_storage_bytes, d_in.begin(), d_out.begin(), num_segments, d_offsets_it, d_offsets_it + 1);

  thrust::device_vector<cub::KeyValuePair<int, int>> expected{{1, 6}, {1, INT_MAX}, {2, 0}};
  // example-end segmented-reduce-argmin

  REQUIRE(thrust::equal(d_out.begin(), d_out.end(), expected.begin(), is_equal()));
}

C2H_TEST("cub::DeviceSegmentedReduce::Max works with int data elements", "[segmented_reduce][device]")
{
  // example-begin segmented-reduce-max
  int num_segments                     = 3;
  thrust::device_vector<int> d_offsets = {0, 3, 3, 7};
  auto d_offsets_it                    = thrust::raw_pointer_cast(d_offsets.data());
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9};
  thrust::device_vector<int> d_out(3);

  // Determine temporary device storage requirements
  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceSegmentedReduce::Max(
    d_temp_storage, temp_storage_bytes, d_in.begin(), d_out.begin(), num_segments, d_offsets_it, d_offsets_it + 1);

  thrust::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
  d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

  // Run reduction
  cub::DeviceSegmentedReduce::Max(
    d_temp_storage, temp_storage_bytes, d_in.begin(), d_out.begin(), num_segments, d_offsets_it, d_offsets_it + 1);

  thrust::device_vector<int> expected{8, INT_MIN, 9};
  // example-end segmented-reduce-max

  REQUIRE(d_out == expected);
}

C2H_TEST("cub::DeviceSegmentedReduce::ArgMax works with int data elements", "[segmented_reduce][device]")
{
  // example-begin segmented-reduce-argmax
  int num_segments                     = 3;
  thrust::device_vector<int> d_offsets = {0, 3, 3, 7};
  auto d_offsets_it                    = thrust::raw_pointer_cast(d_offsets.data());
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9};
  thrust::device_vector<cub::KeyValuePair<int, int>> d_out(3);

  // Determine temporary device storage requirements
  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceSegmentedReduce::ArgMax(
    d_temp_storage, temp_storage_bytes, d_in.begin(), d_out.begin(), num_segments, d_offsets_it, d_offsets_it + 1);

  thrust::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
  d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

  // Run reduction
  cub::DeviceSegmentedReduce::ArgMax(
    d_temp_storage, temp_storage_bytes, d_in.begin(), d_out.begin(), num_segments, d_offsets_it, d_offsets_it + 1);

  thrust::device_vector<cub::KeyValuePair<int, int>> expected{{0, 8}, {1, INT_MIN}, {3, 9}};
  // example-end segmented-reduce-argmax

  REQUIRE(thrust::equal(d_out.begin(), d_out.end(), expected.begin(), is_equal()));
}
