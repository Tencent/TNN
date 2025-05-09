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

#include <cub/device/device_select.cuh>

#include <algorithm>

#include "catch2_test_launch_helper.h"
#include <c2h/catch2_test_helper.cuh>
#include <c2h/vector.cuh>

// %PARAM% TEST_LAUNCH lid 0:1:2

DECLARE_LAUNCH_WRAPPER(cub::DeviceSelect::If, select_if);

using types = c2h::type_list<
  // Type large enough to dispatch to the fallback policy
  c2h::custom_type_t<c2h::equal_comparable_t, c2h::less_comparable_t, c2h::huge_data<256>::type>,
  // Type large enough to require virtual shared memory
  c2h::custom_type_t<c2h::equal_comparable_t, c2h::less_comparable_t, c2h::huge_data<512>::type>>;

template <typename T>
struct less_than_t
{
  T compare;

  explicit __host__ less_than_t(T compare)
      : compare(compare)
  {}

  __host__ __device__ bool operator()(const T& a) const
  {
    return a < compare;
  }
};

C2H_TEST("DeviceSelect::If works for large types", "[select_if][vsmem][device]", types)
{
  using type = typename c2h::get<0, TestType>;

  const int num_items = GENERATE_COPY(take(2, random(1, 10000)));
  c2h::device_vector<type> in(num_items);
  c2h::device_vector<type> out(num_items);
  c2h::gen(C2H_SEED(2), in);

  // Just pick one of the input elements as boundary
  less_than_t<type> le{in[num_items / 2]};

  // Run test
  c2h::device_vector<int> num_selected_out(1, 0);
  select_if(in.begin(), out.begin(), num_selected_out.begin(), num_items, le);

  // Ensure that we create the same output as std
  c2h::host_vector<type> reference = in;
  std::stable_partition(reference.begin(), reference.end(), le);

  out.resize(num_selected_out[0]);
  reference.resize(num_selected_out[0]);
  REQUIRE(reference == out);
}
