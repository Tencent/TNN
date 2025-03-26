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

#include <cub/device/device_run_length_encode.cuh>

#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

#include <algorithm>
#include <array>

#include <catch2/catch.hpp>
#include <nvbench_helper.cuh>

template <class T>
double get_expected_entropy(bit_entropy in_entropy)
{
  if (in_entropy == bit_entropy::_0_000)
  {
    return 0.0;
  }

  if (in_entropy == bit_entropy::_1_000)
  {
    return sizeof(T) * 8;
  }

  const int samples    = static_cast<int>(in_entropy) + 1;
  const double p1      = std::pow(0.5, samples);
  const double p2      = 1 - p1;
  const double entropy = (-p1 * std::log2(p1)) + (-p2 * std::log2(p2));
  return sizeof(T) * 8 * entropy;
}

template <class T>
double compute_actual_entropy(thrust::device_vector<T> in)
{
  const int n = static_cast<int>(in.size());

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
  thrust::device_vector<T> unique(n);
  thrust::device_vector<int> counts(n);
  thrust::device_vector<int> num_runs(1);
  thrust::sort(in.begin(), in.end(), less_t{});

  // RLE
  void* d_temp_storage           = nullptr;
  std::size_t temp_storage_bytes = 0;

  T* d_in             = thrust::raw_pointer_cast(in.data());
  T* d_unique_out     = thrust::raw_pointer_cast(unique.data());
  int* d_counts_out   = thrust::raw_pointer_cast(counts.data());
  int* d_num_runs_out = thrust::raw_pointer_cast(num_runs.data());

  cub::DeviceRunLengthEncode::Encode(
    d_temp_storage, temp_storage_bytes, d_in, d_unique_out, d_counts_out, d_num_runs_out, n);

  thrust::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
  d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

  cub::DeviceRunLengthEncode::Encode(
    d_temp_storage, temp_storage_bytes, d_in, d_unique_out, d_counts_out, d_num_runs_out, n);

  thrust::host_vector<int> h_counts   = counts;
  thrust::host_vector<int> h_num_runs = num_runs;
#else
  std::vector<T> h_in(in.begin(), in.end());
  std::sort(h_in.begin(), h_in.end(), less_t{});
  thrust::host_vector<int> h_counts;
  T prev     = h_in[0];
  int length = 1;

  for (std::size_t i = 1; i < h_in.size(); i++)
  {
    const T next = h_in[i];
    if (next == prev)
    {
      length++;
    }
    else
    {
      h_counts.push_back(length);
      prev   = next;
      length = 1;
    }
  }
  h_counts.push_back(length);

  thrust::host_vector<int> h_num_runs(1, h_counts.size());
#endif

  // normalize counts
  thrust::host_vector<double> ps(h_num_runs[0]);
  for (std::size_t i = 0; i < ps.size(); i++)
  {
    ps[i] = static_cast<double>(h_counts[i]) / n;
  }

  double entropy = 0.0;

  if (ps.size())
  {
    for (double p : ps)
    {
      entropy -= p * std::log2(p);
    }
  }

  return entropy;
}

TEMPLATE_LIST_TEST_CASE("Generators produce data with given entropy", "[gen]", fundamental_types)
{
  constexpr int num_entropy_levels = 6;
  std::array<bit_entropy, num_entropy_levels> entropy_levels{
    bit_entropy::_0_000,
    bit_entropy::_0_201,
    bit_entropy::_0_337,
    bit_entropy::_0_544,
    bit_entropy::_0_811,
    bit_entropy::_1_000};

  std::vector<double> entropy(num_entropy_levels);
  std::transform(entropy_levels.cbegin(), entropy_levels.cend(), entropy.begin(), [](bit_entropy entropy) {
    const thrust::device_vector<TestType> data = generate(1 << 24, entropy);
    return compute_actual_entropy(data);
  });

  REQUIRE(std::is_sorted(entropy.begin(), entropy.end(), less_t{}));
  REQUIRE(std::unique(entropy.begin(), entropy.end()) == entropy.end());
}

TEST_CASE("Generators support bool", "[gen]")
{
  constexpr int num_entropy_levels = 6;
  std::array<bit_entropy, num_entropy_levels> entropy_levels{
    bit_entropy::_0_000,
    bit_entropy::_0_201,
    bit_entropy::_0_337,
    bit_entropy::_0_544,
    bit_entropy::_0_811,
    bit_entropy::_1_000};

  std::vector<std::size_t> number_of_set(num_entropy_levels);
  std::transform(entropy_levels.cbegin(), entropy_levels.cend(), number_of_set.begin(), [](bit_entropy entropy) {
    const thrust::device_vector<bool> data = generate(1 << 24, entropy);
    return thrust::count(data.begin(), data.end(), true);
  });

  REQUIRE(std::is_sorted(number_of_set.begin(), number_of_set.end()));
  REQUIRE(std::unique(number_of_set.begin(), number_of_set.end()) == number_of_set.end());
}
