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

#include <cub/device/device_run_length_encode.cuh>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/sequence.h>

#include <algorithm>
#include <limits>
#include <numeric>

#include "catch2_test_launch_helper.h"
#include <c2h/catch2_test_helper.cuh>

DECLARE_LAUNCH_WRAPPER(cub::DeviceRunLengthEncode::Encode, run_length_encode);

// %PARAM% TEST_LAUNCH lid 0:1:2

using all_types =
  c2h::type_list<std::uint8_t,
                 std::uint64_t,
                 std::int8_t,
                 std::int64_t,
                 ulonglong2,
                 c2h::custom_type_t<c2h::equal_comparable_t>>;

using types = c2h::type_list<std::uint32_t, std::int8_t>;

#if 0 // DeviceRunLengthEncode::Encode cannot handle empty inputs
      // https://github.com/NVIDIA/cccl/issues/426
C2H_TEST("DeviceRunLengthEncode::Encode can handle empty input", "[device][run_length_encode]")
{
  constexpr int num_items = 0;
  c2h::device_vector<int> in(num_items);
  c2h::device_vector<int> out_num_runs(1, 42);

  // Note intentionally no discard_iterator as we want to ensure nothing is written to the output arrays
  run_length_encode(in.begin(),
                    static_cast<int*>(nullptr),
                    static_cast<int*>(nullptr),
                    thrust::raw_pointer_cast(out_num_runs.data()),
                    num_items);

  REQUIRE(out_num_runs.front() == num_items);
}
#endif

C2H_TEST("DeviceRunLengthEncode::Encode can handle a single element", "[device][run_length_encode]")
{
  constexpr int num_items = 1;
  c2h::device_vector<int> in(num_items, 42);
  c2h::device_vector<int> out_unique(num_items);
  c2h::device_vector<int> out_counts(num_items);
  c2h::device_vector<int> out_num_runs(num_items, -1);

  run_length_encode(in.begin(), out_unique.begin(), out_counts.begin(), out_num_runs.begin(), num_items);

  REQUIRE(out_unique.front() == 42);
  REQUIRE(out_counts.front() == 1);
  REQUIRE(out_num_runs.front() == num_items);
}

C2H_TEST("DeviceRunLengthEncode::Encode can handle different counting types", "[device][run_length_encode]")
{
  constexpr int num_items = 1;
  c2h::device_vector<int> in(num_items, 42);
  c2h::device_vector<int> out_unique(num_items);
  c2h::device_vector<cuda::std::size_t> out_counts(num_items);
  c2h::device_vector<std::int16_t> out_num_runs(num_items);

  run_length_encode(in.begin(), out_unique.begin(), out_counts.begin(), out_num_runs.begin(), num_items);

  REQUIRE(out_unique.front() == 42);
  REQUIRE(out_counts.front() == 1);
  REQUIRE(out_num_runs.front() == static_cast<std::int16_t>(num_items));
}

C2H_TEST("DeviceRunLengthEncode::Encode can handle all unique", "[device][run_length_encode]", types)
{
  using type = typename c2h::get<0, TestType>;

  constexpr int num_items = 10;
  c2h::device_vector<type> out_unique(num_items);
  c2h::device_vector<int> out_counts(num_items);
  c2h::device_vector<int> out_num_runs(1);

  run_length_encode(
    thrust::make_counting_iterator(type{}), out_unique.begin(), out_counts.begin(), out_num_runs.begin(), num_items);

  c2h::device_vector<type> reference_unique(num_items);
  thrust::sequence(c2h::device_policy, reference_unique.begin(), reference_unique.end(), type{}); // [0, 1, 2, ...,
                                                                                                  // num_items -1]
  c2h::device_vector<int> reference_counts(num_items, 1); // [1, 1, ..., 1]
  c2h::device_vector<int> reference_num_runs(1, num_items); // [num_items]

  REQUIRE(out_unique == reference_unique);
  REQUIRE(out_counts == reference_counts);
  REQUIRE(out_num_runs == reference_num_runs);
}

C2H_TEST("DeviceRunLengthEncode::Encode can handle all equal", "[device][run_length_encode]", types)
{
  using type = typename c2h::get<0, TestType>;

  constexpr int num_items = 10;
  c2h::device_vector<type> in(num_items, type{1});
  c2h::device_vector<type> out_unique(1);
  c2h::device_vector<int> out_counts(1);
  c2h::device_vector<int> out_num_runs(1);

  run_length_encode(in.begin(), out_unique.begin(), out_counts.begin(), out_num_runs.begin(), num_items);

  c2h::device_vector<type> reference_unique(1, type{1}); // [1]
  c2h::device_vector<int> reference_counts(1, num_items); // [num_items]
  c2h::device_vector<int> reference_num_runs(1, 1); // [1]

  REQUIRE(out_unique == reference_unique);
  REQUIRE(out_counts == reference_counts);
  REQUIRE(out_num_runs == reference_num_runs);
}

C2H_TEST("DeviceRunLengthEncode::Encode can handle iterators", "[device][run_length_encode]", all_types)
{
  using type = typename c2h::get<0, TestType>;

  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)));
  c2h::device_vector<type> in(num_items);
  c2h::device_vector<type> out_unique(num_items);
  c2h::device_vector<int> out_counts(num_items);
  c2h::device_vector<int> out_num_runs(num_items);
  c2h::gen(C2H_SEED(2), in);

  run_length_encode(in.begin(), out_unique.begin(), out_counts.begin(), out_num_runs.begin(), num_items);

  // trim output
  out_unique.resize(out_num_runs.front());
  out_counts.resize(out_num_runs.front());

  c2h::host_vector<type> reference_out = in;
  reference_out.erase(std::unique(reference_out.begin(), reference_out.end()), reference_out.end());
  REQUIRE(out_unique == reference_out);
}

C2H_TEST("DeviceRunLengthEncode::Encode can handle pointers", "[device][run_length_encode]", types)
{
  using type = typename c2h::get<0, TestType>;

  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)));
  c2h::device_vector<type> in(num_items);
  c2h::device_vector<type> out_unique(num_items);
  c2h::device_vector<int> out_counts(num_items);
  c2h::device_vector<int> out_num_runs(num_items);
  c2h::gen(C2H_SEED(2), in);

  run_length_encode(
    thrust::raw_pointer_cast(in.data()),
    thrust::raw_pointer_cast(out_unique.data()),
    thrust::raw_pointer_cast(out_counts.data()),
    thrust::raw_pointer_cast(out_num_runs.data()),
    num_items);

  // trim output
  out_unique.resize(out_num_runs.front());
  out_counts.resize(out_num_runs.front());

  c2h::host_vector<type> reference_out = in;
  reference_out.erase(std::unique(reference_out.begin(), reference_out.end()), reference_out.end());
  REQUIRE(out_unique == reference_out);
}

#if 0 // https://github.com/NVIDIA/cccl/issues/400
template<class T>
struct convertible_from_T {
  T val_;

  convertible_from_T() = default;
  __host__ __device__ convertible_from_T(const T& val) noexcept : val_(val) {}
  __host__ __device__ convertible_from_T& operator=(const T& val) noexcept {
    val_ = val;
  }
  // Converting back to T helps satisfy all the machinery that T supports
  __host__ __device__ operator T() const noexcept { return val_; }
};

C2H_TEST("DeviceRunLengthEncode::Encode works with a different output type", "[device][run_length_encode]")
{
  using type = c2h::custom_type_t<c2h::equal_comparable_t>;

  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)));
  c2h::device_vector<type> in(num_items);
  c2h::device_vector<convertible_from_T<type>> out_unique(num_items);
  c2h::device_vector<int>  out_counts(num_items);
  c2h::device_vector<int>  out_num_runs(num_items);
  c2h::gen(C2H_SEED(2), in);

  run_length_encode(in.begin(),
                    out_unique.begin(),
                    out_counts.begin(),
                    out_num_runs.begin(),
                    num_items);

  // trim output
  out_unique.resize(out_num_runs.front());

  c2h::host_vector<convertible_from_T<type>> reference_out = in;
  reference_out.erase(std::unique(reference_out.begin(), reference_out.end()), reference_out.end());
  REQUIRE(out_unique == reference_out);
}
#endif // https://github.com/NVIDIA/cccl/issues/400

C2H_TEST("DeviceRunLengthEncode::Encode can handle leading NaN", "[device][run_length_encode]")
{
  using type = double;

  constexpr int num_items = 10;
  c2h::device_vector<type> in(num_items);
  thrust::sequence(c2h::device_policy, in.begin(), in.end(), 0.0);
  c2h::device_vector<type> out_unique(num_items);
  c2h::device_vector<int> out_counts(num_items);
  c2h::device_vector<int> out_num_runs(1);

  c2h::device_vector<type> reference_unique = in;
  in.front()                                = std::numeric_limits<type>::quiet_NaN();

  run_length_encode(in.begin(), out_unique.begin(), out_counts.begin(), out_num_runs.begin(), num_items);

  c2h::device_vector<int> reference_counts(num_items, 1); // [1, 1, ..., 1]
  c2h::device_vector<int> reference_num_runs(1, num_items); // [num_items]

  // turn the NaN into something else to make it comparable
  out_unique.front()       = 42.0;
  reference_unique.front() = 42.0;

  REQUIRE(out_unique == reference_unique);
  REQUIRE(out_counts == reference_counts);
  REQUIRE(out_num_runs == reference_num_runs);
}
