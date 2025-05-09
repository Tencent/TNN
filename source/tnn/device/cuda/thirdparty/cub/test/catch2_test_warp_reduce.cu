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

#include <cub/util_macro.cuh>
#include <cub/warp/warp_reduce.cuh>

#include <thrust/iterator/constant_iterator.h>

#include <cuda/std/functional>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

#include <c2h/catch2_test_helper.cuh>
#include <c2h/custom_type.cuh>

template <int LOGICAL_WARP_THREADS, int TOTAL_WARPS, typename T, typename ActionT>
__global__ void warp_reduce_kernel(T* in, T* out, ActionT action)
{
  using warp_reduce_t = cub::WarpReduce<T, LOGICAL_WARP_THREADS>;
  using storage_t     = typename warp_reduce_t::TempStorage;

  __shared__ storage_t storage[TOTAL_WARPS];

  const int tid = threadIdx.x;

  // Get warp index
  int warp_id = tid / LOGICAL_WARP_THREADS;

  // Load data
  T thread_data = in[tid];
  // Instantiate and run warp reduction
  warp_reduce_t warp_reduce(storage[warp_id]);
  auto result = action(tid, warp_reduce, thread_data);

  // Write warp aggregate
  out[tid] = result;
}

/**
 * @brief Delegate wrapper for WarpReduce::Sum
 */
template <typename T>
struct warp_sum_t
{
  template <int LOGICAL_WARP_THREADS>
  __device__ T operator()(int linear_tid, cub::WarpReduce<T, LOGICAL_WARP_THREADS>& warp_reduce, T& thread_data) const
  {
    auto result = warp_reduce.Sum(thread_data);
    return ((linear_tid % LOGICAL_WARP_THREADS) == 0) ? result : thread_data;
  }
};

/**
 * @brief Delegate wrapper for partial WarpReduce::Sum
 */
template <typename T>
struct warp_sum_partial_t
{
  int num_valid;
  template <int LOGICAL_WARP_THREADS>
  __device__ __forceinline__ T
  operator()(int linear_tid, cub::WarpReduce<T, LOGICAL_WARP_THREADS>& warp_reduce, T& thread_data) const
  {
    auto result = warp_reduce.Sum(thread_data, num_valid);
    return ((linear_tid % LOGICAL_WARP_THREADS) == 0) ? result : thread_data;
  }
};

/**
 * @brief Delegate wrapper for WarpReduce::Reduce
 */
template <typename T, typename ReductionOpT>
struct warp_reduce_t
{
  ReductionOpT reduction_op;
  template <int LOGICAL_WARP_THREADS>
  __device__ __forceinline__ T
  operator()(int linear_tid, cub::WarpReduce<T, LOGICAL_WARP_THREADS>& warp_reduce, T& thread_data) const
  {
    auto result = warp_reduce.Reduce(thread_data, reduction_op);
    return ((linear_tid % LOGICAL_WARP_THREADS) == 0) ? result : thread_data;
  }
};

/**
 * @brief Delegate wrapper for partial WarpReduce::Reduce
 */
template <typename T, typename ReductionOpT>
struct warp_reduce_partial_t
{
  int num_valid;
  ReductionOpT reduction_op;
  template <int LOGICAL_WARP_THREADS>
  __device__ T operator()(int linear_tid, cub::WarpReduce<T, LOGICAL_WARP_THREADS>& warp_reduce, T& thread_data) const
  {
    auto result = warp_reduce.Reduce(thread_data, reduction_op, num_valid);
    return ((linear_tid % LOGICAL_WARP_THREADS) == 0) ? result : thread_data;
  }
};

/**
 * @brief Delegate wrapper for WarpReduce::TailSegmentedSum
 */
template <typename T>
struct warp_seg_sum_tail_t
{
  uint8_t* d_flags;
  template <int LOGICAL_WARP_THREADS>
  __device__ T operator()(int linear_tid, cub::WarpReduce<T, LOGICAL_WARP_THREADS>& warp_reduce, T& thread_data) const
  {
    const bool has_agg = (linear_tid % LOGICAL_WARP_THREADS == 0) || ((linear_tid == 0) ? 0 : d_flags[linear_tid - 1]);
    auto result        = warp_reduce.TailSegmentedSum(thread_data, d_flags[linear_tid]);
    return has_agg ? result : thread_data;
  }
};

/**
 * @brief Delegate wrapper for WarpReduce::HeadSegmentedSum
 */
template <typename T>
struct warp_seg_sum_head_t
{
  uint8_t* d_flags;
  template <int LOGICAL_WARP_THREADS>
  __device__ T operator()(int linear_tid, cub::WarpReduce<T, LOGICAL_WARP_THREADS>& warp_reduce, T& thread_data) const
  {
    const bool has_agg = ((linear_tid % LOGICAL_WARP_THREADS == 0) || d_flags[linear_tid]);
    auto result        = warp_reduce.HeadSegmentedSum(thread_data, d_flags[linear_tid]);
    return (has_agg) ? result : thread_data;
  }
};

/**
 * @brief Delegate wrapper for WarpReduce::TailSegmentedReduce
 */
template <typename T, typename ReductionOpT>
struct warp_seg_reduce_tail_t
{
  uint8_t* d_flags;
  ReductionOpT reduction_op;
  template <int LOGICAL_WARP_THREADS>
  __device__ T operator()(int linear_tid, cub::WarpReduce<T, LOGICAL_WARP_THREADS>& warp_reduce, T& thread_data) const
  {
    const bool has_agg = (linear_tid % LOGICAL_WARP_THREADS == 0) || ((linear_tid == 0) ? 0 : d_flags[linear_tid - 1]);
    auto result        = warp_reduce.TailSegmentedReduce(thread_data, d_flags[linear_tid], reduction_op);
    return has_agg ? result : thread_data;
  }
};

/**
 * @brief Delegate wrapper for WarpReduce::HeadSegmentedReduce
 */
template <typename T, typename ReductionOpT>
struct warp_seg_reduce_head_t
{
  uint8_t* d_flags;
  ReductionOpT reduction_op;
  template <int LOGICAL_WARP_THREADS>
  __device__ T operator()(int linear_tid, cub::WarpReduce<T, LOGICAL_WARP_THREADS>& warp_reduce, T& thread_data) const
  {
    const bool has_agg = ((linear_tid % LOGICAL_WARP_THREADS == 0) || d_flags[linear_tid]);
    auto result        = warp_reduce.HeadSegmentedReduce(thread_data, d_flags[linear_tid], reduction_op);
    return (has_agg) ? result : thread_data;
  }
};

/**
 * @brief Dispatch helper function
 */
template <int LOGICAL_WARP_THREADS, int TOTAL_WARPS, typename T, typename ActionT>
void warp_reduce(c2h::device_vector<T>& in, c2h::device_vector<T>& out, ActionT action)
{
  warp_reduce_kernel<LOGICAL_WARP_THREADS, TOTAL_WARPS, T, ActionT><<<1, LOGICAL_WARP_THREADS * TOTAL_WARPS>>>(
    thrust::raw_pointer_cast(in.data()), thrust::raw_pointer_cast(out.data()), action);

  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
}

/**
 * @brief Compares the results returned from system under test against the expected results.
 */
template <typename T, typename ::cuda::std::enable_if<::cuda::std::is_floating_point<T>::value, int>::type = 0>
void verify_results(const c2h::host_vector<T>& expected_data, const c2h::device_vector<T>& test_results)
{
  REQUIRE_APPROX_EQ(expected_data, test_results);
}

/**
 * @brief Compares the results returned from system under test against the expected results.
 */
template <typename T, typename ::cuda::std::enable_if<!::cuda::std::is_floating_point<T>::value, int>::type = 0>
void verify_results(const c2h::host_vector<T>& expected_data, const c2h::device_vector<T>& test_results)
{
  REQUIRE(expected_data == test_results);
}

enum class reduce_mode
{
  all,
  partial,
  head_flags,
  tail_flags,
};

template <typename InputItT, typename FlagInputItT, typename ReductionOp, typename ResultOutItT>
void compute_host_reference(
  reduce_mode mode,
  InputItT h_in,
  FlagInputItT h_flags,
  int warps,
  int logical_warp_threads,
  int valid_warp_threads,
  ReductionOp reduction_op,
  ResultOutItT h_data_out)
{
  // Accumulate segments (lane 0 of each warp is implicitly a segment head)
  for (int warp = 0; warp < warps; ++warp)
  {
    int warp_offset = warp * logical_warp_threads;
    int item_offset = warp_offset + valid_warp_threads - 1;

    // Last item in warp
    auto head_aggregate = h_in[item_offset];
    auto tail_aggregate = h_in[item_offset];

    if (mode != reduce_mode::tail_flags && h_flags[item_offset])
    {
      h_data_out[item_offset] = head_aggregate;
    }
    item_offset--;

    // Work backwards
    while (item_offset >= warp_offset)
    {
      if (h_flags[item_offset + 1])
      {
        head_aggregate = h_in[item_offset];
      }
      else
      {
        head_aggregate = reduction_op(head_aggregate, h_in[item_offset]);
      }

      if (h_flags[item_offset])
      {
        if (mode == reduce_mode::head_flags)
        {
          h_data_out[item_offset] = head_aggregate;
        }
        else if (mode == reduce_mode::tail_flags)
        {
          h_data_out[item_offset + 1] = tail_aggregate;
          tail_aggregate              = h_in[item_offset];
        }
      }
      else
      {
        tail_aggregate = reduction_op(tail_aggregate, h_in[item_offset]);
      }

      item_offset--;
    }

    // Record last segment aggregate
    if (mode == reduce_mode::tail_flags)
    {
      h_data_out[warp_offset] = tail_aggregate;
    }
    else
    {
      h_data_out[warp_offset] = head_aggregate;
    }
  }
}

// List of types to test
using custom_t =
  c2h::custom_type_t<c2h::accumulateable_t, c2h::equal_comparable_t, c2h::lexicographical_less_comparable_t>;
using full_type_list =
  c2h::type_list<std::uint8_t, std::uint16_t, std::int32_t, std::int64_t, custom_t, ulonglong4, uchar3, short2>;

using builtin_type_list = c2h::type_list<std::uint8_t, std::uint16_t, std::int32_t, std::int64_t>;

// Logical warp sizes to test
using logical_warp_threads = c2h::enum_type_list<int, 32, 16, 9, 7, 1>;

using segmented_modes = c2h::enum_type_list<reduce_mode, reduce_mode::head_flags, reduce_mode::tail_flags>;

template <int logical_warp_threads>
struct total_warps_t
{
private:
  static constexpr int max_warps      = 2;
  static constexpr bool is_arch_warp  = (logical_warp_threads == CUB_WARP_THREADS(0));
  static constexpr bool is_pow_of_two = ((logical_warp_threads & (logical_warp_threads - 1)) == 0);
  static constexpr int total_warps    = (is_arch_warp || is_pow_of_two) ? max_warps : 1;

public:
  static constexpr int value()
  {
    return total_warps;
  }
};

template <typename TestType>
struct params_t
{
  using type = typename c2h::get<0, TestType>;

  static constexpr int logical_warp_threads = c2h::get<1, TestType>::value;
  static constexpr int total_warps          = total_warps_t<logical_warp_threads>::value();
  static constexpr int tile_size            = total_warps * logical_warp_threads;
};

C2H_TEST("Warp sum works", "[reduce][warp]", full_type_list, logical_warp_threads)
{
  using params = params_t<TestType>;
  using type   = typename params::type;

  // Prepare test data
  c2h::device_vector<type> d_in(params::tile_size);
  c2h::device_vector<type> d_out(params::tile_size);
  constexpr auto valid_items = params::logical_warp_threads;
  c2h::gen(C2H_SEED(10), d_in);

  // Run test
  warp_reduce<params::logical_warp_threads, params::total_warps>(d_in, d_out, warp_sum_t<type>{});

  // Prepare verification data
  c2h::host_vector<type> h_in  = d_in;
  c2h::host_vector<type> h_out = h_in;
  auto h_flags                 = thrust::make_constant_iterator(false);
  compute_host_reference(
    reduce_mode::all,
    h_in,
    h_flags,
    params::total_warps,
    params::logical_warp_threads,
    valid_items,
    ::cuda::std::plus<type>{},
    h_out.begin());

  // Verify results
  verify_results(h_out, d_out);
}

C2H_TEST("Warp reduce works", "[reduce][warp]", builtin_type_list, logical_warp_threads)
{
  using params   = params_t<TestType>;
  using type     = typename params::type;
  using red_op_t = cub::Min;

  // Prepare test data
  c2h::device_vector<type> d_in(params::tile_size);
  c2h::device_vector<type> d_out(params::tile_size);
  constexpr auto valid_items = params::logical_warp_threads;
  c2h::gen(C2H_SEED(10), d_in);

  // Run test
  warp_reduce<params::logical_warp_threads, params::total_warps>(d_in, d_out, warp_reduce_t<type, red_op_t>{red_op_t{}});

  // Prepare verification data
  c2h::host_vector<type> h_in  = d_in;
  c2h::host_vector<type> h_out = h_in;
  auto h_flags                 = thrust::make_constant_iterator(false);
  compute_host_reference(
    reduce_mode::all,
    h_in,
    h_flags,
    params::total_warps,
    params::logical_warp_threads,
    valid_items,
    red_op_t{},
    h_out.begin());

  // Verify results
  verify_results(h_out, d_out);
}

C2H_TEST("Warp sum on partial warp works", "[reduce][warp]", full_type_list, logical_warp_threads)
{
  using params = params_t<TestType>;
  using type   = typename params::type;

  // Prepare test data
  c2h::device_vector<type> d_in(params::tile_size);
  c2h::device_vector<type> d_out(params::tile_size);
  const int valid_items = GENERATE_COPY(take(2, random(1, params::logical_warp_threads)));
  c2h::gen(C2H_SEED(10), d_in);

  // Run test
  warp_reduce<params::logical_warp_threads, params::total_warps>(d_in, d_out, warp_sum_partial_t<type>{valid_items});

  // Prepare verification data
  c2h::host_vector<type> h_in  = d_in;
  c2h::host_vector<type> h_out = h_in;
  auto h_flags                 = thrust::make_constant_iterator(false);
  compute_host_reference(
    reduce_mode::all,
    h_in,
    h_flags,
    params::total_warps,
    params::logical_warp_threads,
    valid_items,
    ::cuda::std::plus<type>{},
    h_out.begin());

  // Verify results
  verify_results(h_out, d_out);
}

C2H_TEST("Warp reduce on partial warp works", "[reduce][warp]", builtin_type_list, logical_warp_threads)
{
  using params   = params_t<TestType>;
  using type     = typename params::type;
  using red_op_t = cub::Min;

  // Prepare test data
  c2h::device_vector<type> d_in(params::tile_size);
  c2h::device_vector<type> d_out(params::tile_size);
  const int valid_items = GENERATE_COPY(take(2, random(1, params::logical_warp_threads)));
  c2h::gen(C2H_SEED(10), d_in);

  // Run test
  warp_reduce<params::logical_warp_threads, params::total_warps>(
    d_in, d_out, warp_reduce_partial_t<type, red_op_t>{valid_items, red_op_t{}});

  // Prepare verification data
  c2h::host_vector<type> h_in  = d_in;
  c2h::host_vector<type> h_out = h_in;
  auto h_flags                 = thrust::make_constant_iterator(false);
  compute_host_reference(
    reduce_mode::all,
    h_in,
    h_flags,
    params::total_warps,
    params::logical_warp_threads,
    valid_items,
    red_op_t{},
    h_out.begin());

  // Verify results
  verify_results(h_out, d_out);
}

C2H_TEST("Warp segmented sum works", "[reduce][warp]", full_type_list, logical_warp_threads, segmented_modes)
{
  using params = params_t<TestType>;
  using type   = typename params::type;

  constexpr auto segmented_mod = c2h::get<2, TestType>::value;
  static_assert(segmented_mod == reduce_mode::tail_flags || segmented_mod == reduce_mode::head_flags,
                "Segmented tests must either be head or tail flags");
  using warp_seg_sum_t =
    ::cuda::std::_If<(segmented_mod == reduce_mode::tail_flags), warp_seg_sum_tail_t<type>, warp_seg_sum_head_t<type>>;

  // Prepare test data
  c2h::device_vector<type> d_in(params::tile_size);
  c2h::device_vector<uint8_t> d_flags(params::tile_size);
  c2h::device_vector<type> d_out(params::tile_size);
  constexpr auto valid_items = params::logical_warp_threads;
  constexpr uint8_t min      = 0;
  constexpr uint8_t max      = 2;
  c2h::gen(C2H_SEED(5), d_in);
  c2h::gen(C2H_SEED(5), d_flags, min, max);

  // Run test
  warp_reduce<params::logical_warp_threads, params::total_warps>(
    d_in, d_out, warp_seg_sum_t{thrust::raw_pointer_cast(d_flags.data())});

  // Prepare verification data
  c2h::host_vector<type> h_in       = d_in;
  c2h::host_vector<uint8_t> h_flags = d_flags;
  c2h::host_vector<type> h_out      = h_in;
  compute_host_reference(
    segmented_mod,
    h_in,
    h_flags,
    params::total_warps,
    params::logical_warp_threads,
    valid_items,
    ::cuda::std::plus<type>{},
    h_out.begin());

  // Verify results
  verify_results(h_out, d_out);
}

C2H_TEST("Warp segmented reduction works", "[reduce][warp]", builtin_type_list, logical_warp_threads, segmented_modes)
{
  using params   = params_t<TestType>;
  using type     = typename params::type;
  using red_op_t = cub::Min;

  constexpr auto segmented_mod = c2h::get<2, TestType>::value;
  static_assert(segmented_mod == reduce_mode::tail_flags || segmented_mod == reduce_mode::head_flags,
                "Segmented tests must either be head or tail flags");
  using warp_seg_reduction_t =
    ::cuda::std::_If<(segmented_mod == reduce_mode::tail_flags),
                     warp_seg_reduce_tail_t<type, red_op_t>,
                     warp_seg_reduce_head_t<type, red_op_t>>;

  // Prepare test data
  c2h::device_vector<type> d_in(params::tile_size);
  c2h::device_vector<uint8_t> d_flags(params::tile_size);
  c2h::device_vector<type> d_out(params::tile_size);
  constexpr auto valid_items = params::logical_warp_threads;
  constexpr uint8_t min      = 0;
  constexpr uint8_t max      = 2;
  c2h::gen(C2H_SEED(5), d_in);
  c2h::gen(C2H_SEED(5), d_flags, min, max);

  // Run test
  warp_reduce<params::logical_warp_threads, params::total_warps>(
    d_in, d_out, warp_seg_reduction_t{thrust::raw_pointer_cast(d_flags.data()), red_op_t{}});

  // Prepare verification data
  c2h::host_vector<type> h_in       = d_in;
  c2h::host_vector<uint8_t> h_flags = d_flags;
  c2h::host_vector<type> h_out      = h_in;
  compute_host_reference(
    segmented_mod,
    h_in,
    h_flags,
    params::total_warps,
    params::logical_warp_threads,
    valid_items,
    red_op_t{},
    h_out.begin());

  // Verify results
  verify_results(h_out, d_out);
}
