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
#include <thrust/logical.h>
#include <thrust/sequence.h>

#include <algorithm>
#include <limits>
#include <numeric>

#include "catch2_test_launch_helper.h"
#include <c2h/catch2_test_helper.cuh>

DECLARE_LAUNCH_WRAPPER(cub::DeviceRunLengthEncode::NonTrivialRuns, run_length_encode);

// %PARAM% TEST_LAUNCH lid 0:1:2

using all_types =
  c2h::type_list<std::uint8_t,
                 std::uint64_t,
                 std::int8_t,
                 std::int64_t,
                 ulonglong2,
                 c2h::custom_type_t<c2h::equal_comparable_t>>;

using types = c2h::type_list<std::uint32_t, std::int8_t>;

#if 0 // DeviceRunLengthEncode::NonTrivialRuns cannot handle inputs with one or less elements
      // https://github.com/NVIDIA/cccl/issues/426
C2H_TEST("DeviceRunLengthEncode::NonTrivialRuns can handle empty input", "[device][run_length_encode]")
{
  constexpr int num_items = 0;
  c2h::device_vector<int> out_num_runs(1, 42);

  // Note intentionally no discard_iterator as we want to ensure nothing is written to the output arrays
  run_length_encode(static_cast<int*>(nullptr),
                    static_cast<int*>(nullptr),
                    static_cast<int*>(nullptr),
                    thrust::raw_pointer_cast(out_num_runs.data()),
                    num_items);

  REQUIRE(out_num_runs.front() == 0);
}

C2H_TEST("DeviceRunLengthEncode::NonTrivialRuns can handle a single element", "[device][run_length_encode]")
{
  constexpr int num_items = 1;
  c2h::device_vector<int> out_num_runs(1, 42);

  // Note intentionally no discard_iterator as we want to ensure nothing is written to the output arrays
  run_length_encode(static_cast<int*>(nullptr),
                    static_cast<int*>(nullptr),
                    static_cast<int*>(nullptr),
                    thrust::raw_pointer_cast(out_num_runs.data()),
                    num_items);

  REQUIRE(out_num_runs.front() == 0);
}
#endif

#if 0 // DeviceRunLengthEncode::NonTrivialRuns cannot handle inputs larger than INT32_MAX
C2H_TEST("DeviceRunLengthEncode::NonTrivialRuns can handle large indexes", "[device][run_length_encode]")
{
  constexpr cuda::std::size_t num_items = 1ull << 33;
  c2h::device_vector<cuda::std::size_t> out_num_runs(1, -1);

  // Note intentionally no discard_iterator as we want to ensure nothing is written to the output arrays
  run_length_encode(thrust::make_counting_iterator(cuda::std::size_t{0}),
                    static_cast<cuda::std::size_t*>(nullptr),
                    static_cast<cuda::std::size_t*>(nullptr),
                    out_num_runs.begin(),
                    num_items);

  REQUIRE(out_num_runs.front() == 0);
}
#endif

C2H_TEST("DeviceRunLengthEncode::NonTrivialRuns can handle different counting types", "[device][run_length_encode]")
{
  constexpr int num_items = 1;
  c2h::device_vector<int> in(num_items, 42);
  c2h::device_vector<int> out_num_runs(1, 42);

  // Note intentionally no discard_iterator as we want to ensure nothing is written to the output
  // arrays
  run_length_encode(
    in.begin(),
    static_cast<cuda::std::size_t*>(nullptr),
    static_cast<std::uint16_t*>(nullptr),
    out_num_runs.begin(),
    num_items);

  REQUIRE(out_num_runs.front() == 0);
}

C2H_TEST("DeviceRunLengthEncode::NonTrivialRuns can handle all unique", "[device][run_length_encode]", types)
{
  using type = typename c2h::get<0, TestType>;

  constexpr int num_items = 10;
  c2h::device_vector<int> out_num_runs(1, -1);

  run_length_encode(
    thrust::make_counting_iterator(type{}),
    static_cast<int*>(nullptr),
    static_cast<int*>(nullptr),
    out_num_runs.begin(),
    num_items);

  REQUIRE(out_num_runs.front() == 0);
}

C2H_TEST("DeviceRunLengthEncode::NonTrivialRuns can handle all equal", "[device][run_length_encode]", types)
{
  using type = typename c2h::get<0, TestType>;

  constexpr int num_items = 10;
  c2h::device_vector<type> in(num_items);
  c2h::device_vector<int> out_offsets(1, -1);
  c2h::device_vector<int> out_lengths(1, -1);
  c2h::device_vector<int> out_num_runs(1, -1);
  c2h::gen(C2H_SEED(2), in);
  thrust::fill(c2h::device_policy, in.begin(), in.end(), in.front());

  run_length_encode(in.begin(), out_offsets.begin(), out_lengths.begin(), out_num_runs.begin(), num_items);

  REQUIRE(out_offsets.front() == 0);
  REQUIRE(out_lengths.front() == num_items);
  REQUIRE(out_num_runs.front() == 1);
}

template <class T, class Index>
bool validate_results(
  const c2h::device_vector<T>& in,
  const c2h::device_vector<Index>& out_offsets,
  const c2h::device_vector<Index>& out_lengths,
  const c2h::device_vector<Index>& out_num_runs,
  const int num_items)
{
  const c2h::host_vector<T>& h_in               = in;
  const c2h::host_vector<Index>& h_out_offsets  = out_offsets;
  const c2h::host_vector<Index>& h_out_lengths  = out_lengths;
  const c2h::host_vector<Index>& h_out_num_runs = out_num_runs;

  const cuda::std::size_t num_runs = static_cast<cuda::std::size_t>(h_out_num_runs.front());
  for (cuda::std::size_t run = 0; run < num_runs; ++run)
  {
    const cuda::std::size_t first_index = static_cast<cuda::std::size_t>(h_out_offsets[run]);
    const cuda::std::size_t final_index = first_index + static_cast<cuda::std::size_t>(h_out_lengths[run]);

    // Ensure we started a new run
    if (first_index > 0)
    {
      if (h_in[first_index] == h_in[first_index - 1])
      {
        return false;
      }
    }

    // Ensure the run is valid
    const auto first_elem = h_in[first_index];
    const auto all_equal  = [first_elem](const T& elem) -> bool {
      return first_elem == elem;
    };
    if (!std::all_of(h_in.begin() + first_index + 1, h_in.begin() + final_index, all_equal))
    {
      return false;
    }

    // Ensure the run is of maximal length
    if (final_index < static_cast<cuda::std::size_t>(num_items))
    {
      if (h_in[first_index] == h_in[final_index])
      {
        return false;
      }
    }
  }
  return true;
}

C2H_TEST("DeviceRunLengthEncode::NonTrivialRuns can handle iterators", "[device][run_length_encode]", all_types)
{
  using type = typename c2h::get<0, TestType>;

  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)));
  c2h::device_vector<type> in(num_items);
  c2h::device_vector<int> out_offsets(num_items, -1);
  c2h::device_vector<int> out_lengths(num_items, -1);
  c2h::device_vector<int> out_num_runs(1, -1);
  c2h::gen(C2H_SEED(2), in);

  run_length_encode(in.begin(), out_offsets.begin(), out_lengths.begin(), out_num_runs.begin(), num_items);

  out_offsets.resize(out_num_runs.front());
  out_lengths.resize(out_num_runs.front());
  REQUIRE(validate_results(in, out_offsets, out_lengths, out_num_runs, num_items));
}

C2H_TEST("DeviceRunLengthEncode::NonTrivialRuns can handle pointers", "[device][run_length_encode]", types)
{
  using type = typename c2h::get<0, TestType>;

  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)));
  c2h::device_vector<type> in(num_items);
  c2h::device_vector<int> out_offsets(num_items, -1);
  c2h::device_vector<int> out_lengths(num_items, -1);
  c2h::device_vector<int> out_num_runs(1, -1);
  c2h::gen(C2H_SEED(2), in);

  run_length_encode(
    thrust::raw_pointer_cast(in.data()),
    thrust::raw_pointer_cast(out_offsets.data()),
    thrust::raw_pointer_cast(out_lengths.data()),
    thrust::raw_pointer_cast(out_num_runs.data()),
    num_items);

  out_offsets.resize(out_num_runs.front());
  out_lengths.resize(out_num_runs.front());
  REQUIRE(validate_results(in, out_offsets, out_lengths, out_num_runs, num_items));
}

// Guard against #293
template <bool TimeSlicing>
struct device_rle_policy_hub
{
  static constexpr int threads = 96;
  static constexpr int items   = 15;

  struct Policy350 : cub::ChainedPolicy<350, Policy350, Policy350>
  {
    using RleSweepPolicyT = cub::
      AgentRlePolicy<threads, items, cub::BLOCK_LOAD_DIRECT, cub::LOAD_DEFAULT, TimeSlicing, cub::BLOCK_SCAN_WARP_SCANS>;
  };

  using MaxPolicy = Policy350;
};

struct CustomDeviceRunLengthEncode
{
  template <bool TimeSlicing,
            typename InputIteratorT,
            typename OffsetsOutputIteratorT,
            typename LengthsOutputIteratorT,
            typename NumRunsOutputIteratorT>
  CUB_RUNTIME_FUNCTION __forceinline__ static cudaError_t NonTrivialRuns(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OffsetsOutputIteratorT d_offsets_out,
    LengthsOutputIteratorT d_lengths_out,
    NumRunsOutputIteratorT d_num_runs_out,
    int num_items,
    cudaStream_t stream = 0)
  {
    using OffsetT    = int; // Signed integer type for global offsets
    using EqualityOp = cub::Equality; // Default == operator

    return cub::DeviceRleDispatch<InputIteratorT,
                                  OffsetsOutputIteratorT,
                                  LengthsOutputIteratorT,
                                  NumRunsOutputIteratorT,
                                  EqualityOp,
                                  OffsetT,
                                  device_rle_policy_hub<TimeSlicing>>::
      Dispatch(d_temp_storage,
               temp_storage_bytes,
               d_in,
               d_offsets_out,
               d_lengths_out,
               d_num_runs_out,
               EqualityOp(),
               num_items,
               stream);
  }
};

DECLARE_LAUNCH_WRAPPER(CustomDeviceRunLengthEncode::NonTrivialRuns<true>, run_length_encode_293_true);
DECLARE_LAUNCH_WRAPPER(CustomDeviceRunLengthEncode::NonTrivialRuns<false>, run_length_encode_293_false);

using time_slicing = c2h::type_list<std::true_type, std::false_type>;

C2H_TEST("DeviceRunLengthEncode::NonTrivialRuns does not run out of memory", "[device][run_length_encode]", time_slicing)
{
  using type         = typename c2h::get<0, TestType>;
  using policy_hub_t = device_rle_policy_hub<type::value>;

  constexpr int tile_size    = policy_hub_t::threads * policy_hub_t::items;
  constexpr int num_items    = 2 * tile_size;
  constexpr int magic_number = num_items + 1;

  c2h::host_vector<int> h_keys(num_items);
  thrust::sequence(h_keys.begin(), h_keys.begin() + tile_size);

  int expected_non_trivial_runs = 0;
  int value                     = tile_size;
  int large_group_size          = 3;
  for (int i = 0; i < tile_size; i++)
  {
    int j = 0;
    for (; j < large_group_size && i < tile_size; ++j, ++i)
    {
      h_keys[tile_size + i] = value;
    }
    if (j == large_group_size)
    {
      ++expected_non_trivial_runs;
    }
    ++value;

    if (i < tile_size)
    {
      h_keys[tile_size + i] = value;
    }
    ++value;
  }

  // in #293 we were writing before the output arrays. So add a sentinel element in front to check
  // against OOB writes
  c2h::device_vector<int> in = h_keys;
  c2h::device_vector<int> out_offsets(num_items + 1, -1);
  c2h::device_vector<int> out_lengths(num_items + 1, -1);
  c2h::device_vector<int> out_num_runs(1, -1);
  out_offsets.front() = magic_number;
  out_lengths.front() = magic_number;

  if (type::value)
  {
    run_length_encode_293_true(
      in.begin(), out_offsets.begin() + 1, out_lengths.begin() + 1, out_num_runs.begin(), num_items);
  }
  else
  {
    run_length_encode_293_false(
      in.begin(), out_offsets.begin() + 1, out_lengths.begin() + 1, out_num_runs.begin(), num_items);
  }

  REQUIRE(out_num_runs.front() == expected_non_trivial_runs);
  REQUIRE(out_lengths.front() == magic_number);
  REQUIRE(out_offsets.front() == magic_number);
}
