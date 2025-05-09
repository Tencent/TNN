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

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <algorithm>
#include <cmath>

#include <boost/math/statistics/anderson_darling.hpp>
#include <boost/math/statistics/univariate_statistics.hpp>
#include <catch2/catch.hpp>
#include <nvbench_helper.cuh>

bool is_normal(thrust::host_vector<double> data)
{
  std::sort(data.begin(), data.end());
  const double A2 = boost::math::statistics::anderson_darling_normality_statistic(data);
  return A2 / data.size() < 0.05;
}

using types = nvbench::type_list<uint32_t, uint64_t>;

TEMPLATE_LIST_TEST_CASE("Generators produce power law distributed data", "[gen][power-law]", types)
{
  const std::size_t elements                              = 1 << 28;
  const std::size_t segments                              = 4 * 1024;
  const thrust::device_vector<TestType> d_segment_offsets = generate.power_law.segment_offsets(elements, segments);
  REQUIRE(d_segment_offsets.size() == segments + 1);

  std::size_t actual_elements = 0;
  thrust::host_vector<double> log_sizes(segments);
  const thrust::host_vector<TestType> h_segment_offsets = d_segment_offsets;
  for (int i = 0; i < segments; ++i)
  {
    const TestType begin = h_segment_offsets[i];
    const TestType end   = h_segment_offsets[i + 1];
    REQUIRE(begin <= end);

    const std::size_t size = end - begin;
    actual_elements += size;
    log_sizes[i] = std::log(size);
  }

  REQUIRE(actual_elements == elements);
  REQUIRE(is_normal(std::move(log_sizes)));
}
