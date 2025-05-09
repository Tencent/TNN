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

#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cmath>
#include <limits>
#include <map>

#include <boost/math/distributions/chi_squared.hpp>
#include <catch2/catch.hpp>
#include <nvbench_helper.cuh>

template <typename T>
bool is_uniform(thrust::host_vector<T> data, T min, T max)
{
  const double value_range = static_cast<double>(max) - min;
  const bool exact_binning = value_range < (1 << 20);
  const int number_of_bins = exact_binning ? static_cast<int>(max - min + 1) : static_cast<int>(std::sqrt(data.size()));
  thrust::host_vector<int> bins(number_of_bins, 0);

  const double interval       = value_range / static_cast<double>(number_of_bins);
  const double expected_count = static_cast<double>(data.size()) / number_of_bins;

  for (T val : data)
  {
    int bin_index = exact_binning ? val - min : (val - static_cast<double>(min)) / interval;

    if (bin_index >= 0 && bin_index < number_of_bins)
    {
      bins[bin_index]++;
    }
  }

  double chi_square = 0.0;
  for (const auto& count : bins)
  {
    chi_square += std::pow(count - expected_count, 2) / expected_count;
  }

  boost::math::chi_squared_distribution<double> chi_squared_dist(number_of_bins - 1);

  const double confidence     = 0.95;
  const double critical_value = boost::math::quantile(chi_squared_dist, confidence);

  return chi_square <= critical_value;
}

using types =
  nvbench::type_list<int8_t,
                     int16_t,
                     int32_t,
                     int64_t,
#if NVBENCH_HELPER_HAS_I128
                     int128_t,
#endif
                     float,
                     double>;

TEMPLATE_LIST_TEST_CASE("Generators produce uniformly distributed data", "[gen][uniform]", types)
{
  const std::size_t elements = 1 << GENERATE_COPY(16, 20, 24, 28);
  const TestType min         = std::numeric_limits<TestType>::min();
  const TestType max         = std::numeric_limits<TestType>::max();

  const thrust::device_vector<TestType> data = generate(elements, bit_entropy::_1_000, min, max);

  REQUIRE(is_uniform<TestType>(data, min, max));
}

struct complex_to_real_t
{
  __host__ __device__ float operator()(const complex& c) const
  {
    return c.real();
  }
};

struct complex_to_imag_t
{
  __host__ __device__ float operator()(const complex& c) const
  {
    return c.imag();
  }
};

TEST_CASE("Generators produce uniformly distributed complex", "[gen]")
{
  const float min = std::numeric_limits<float>::min();
  const float max = std::numeric_limits<float>::max();

  const thrust::device_vector<complex> data = generate(1 << 16);

  thrust::device_vector<float> component(data.size());
  thrust::transform(data.begin(), data.end(), component.begin(), complex_to_real_t());
  REQUIRE(is_uniform<float>(component, min, max));

  thrust::transform(data.begin(), data.end(), component.begin(), complex_to_imag_t());
  REQUIRE(is_uniform<float>(component, min, max));
}

TEST_CASE("Generators produce uniformly distributed bools", "[gen]")
{
  const thrust::device_vector<bool> data = generate(1 << 24, bit_entropy::_0_544);

  const std::size_t falses = thrust::count(data.begin(), data.end(), false);
  const std::size_t trues  = thrust::count(data.begin(), data.end(), true);

  REQUIRE(falses > 0);
  REQUIRE(trues > 0);
  REQUIRE(falses + trues == data.size());

  const double ratio = static_cast<double>(falses) / trues;
  REQUIRE(ratio > 0.7);
}

using offsets = nvbench::type_list<uint32_t, uint64_t>;

TEMPLATE_LIST_TEST_CASE("Generators produce uniformly distributed offsets", "[gen]", offsets)
{
  const std::size_t min_segment_size = 1;
  const std::size_t max_segment_size = 256;
  const std::size_t elements         = 1 << GENERATE_COPY(16, 20, 24, 28);
  const thrust::device_vector<TestType> d_segments =
    generate.uniform.segment_offsets(elements, min_segment_size, max_segment_size);
  const thrust::host_vector<TestType> h_segments = d_segments;
  const std::size_t num_segments                 = h_segments.size() - 1;

  std::size_t actual_elements = 0;
  thrust::host_vector<int> segment_sizes(num_segments);
  for (std::size_t sid = 0; sid < num_segments; sid++)
  {
    const TestType begin = h_segments[sid];
    const TestType end   = h_segments[sid + 1];
    REQUIRE(begin <= end);

    const TestType size = end - begin;
    REQUIRE(size >= min_segment_size);
    REQUIRE(size <= max_segment_size);

    segment_sizes[sid] = size;
    actual_elements += size;
  }

  REQUIRE(actual_elements == elements);
  REQUIRE(is_uniform<int>(std::move(segment_sizes), min_segment_size, max_segment_size));
}

TEMPLATE_LIST_TEST_CASE("Generators produce uniformly distributed key segments", "[gen]", types)
{
  const std::size_t min_segment_size = 1;
  const std::size_t max_segment_size = 128;
  const std::size_t elements         = 1 << GENERATE_COPY(16, 20, 24, 28);
  const thrust::device_vector<TestType> d_keys =
    generate.uniform.key_segments(elements, min_segment_size, max_segment_size);
  REQUIRE(d_keys.size() == elements);

  const thrust::host_vector<TestType> h_keys = d_keys;

  thrust::host_vector<int> segment_sizes;

  TestType prev = h_keys[0];
  int length    = 1;

  for (std::size_t kid = 1; kid < elements; kid++)
  {
    TestType next = h_keys[kid];

    if (next == prev)
    {
      length++;
    }
    else
    {
      REQUIRE(length >= min_segment_size);
      REQUIRE(length <= max_segment_size);

      segment_sizes.push_back(length);

      prev   = next;
      length = 1;
    }
  }
  REQUIRE(length >= min_segment_size);
  REQUIRE(length <= max_segment_size);
  segment_sizes.push_back(length);

  REQUIRE(is_uniform<int>(std::move(segment_sizes), min_segment_size, max_segment_size));
}
