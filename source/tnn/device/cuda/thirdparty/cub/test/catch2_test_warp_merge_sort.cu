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
#include <cub/util_ptx.cuh>
#include <cub/warp/warp_merge_sort.cuh>

#include <thrust/iterator/constant_iterator.h>

#include <cuda/std/type_traits>

#include <algorithm>

#include <c2h/catch2_test_helper.cuh>
#include <c2h/custom_type.cuh>

struct CustomLess
{
  template <typename T>
  __device__ __host__ bool operator()(const T& lhs, const T& rhs)
  {
    return lhs < rhs;
  }
};

/**
 * @brief Kernel to dispatch to the appropriate WarpMergeSort member function, sorting keys-only.
 */
template <int ITEMS_PER_THREAD,
          int LOGICAL_WARP_THREADS,
          int TOTAL_WARPS,
          typename T,
          typename SegmentSizeItT,
          typename ActionT>
__global__ void warp_merge_sort_kernel(T* in, T* out, SegmentSizeItT segment_sizes, T oob_default, ActionT action)
{
  using warp_merge_sort_t = cub::WarpMergeSort<T, ITEMS_PER_THREAD, LOGICAL_WARP_THREADS>;
  using storage_t         = typename warp_merge_sort_t::TempStorage;

  // Get linear thread and warp index
  const int tid     = threadIdx.x;
  const int warp_id = tid / LOGICAL_WARP_THREADS;

  // Test case of partially finished CTA
  if (warp_id >= TOTAL_WARPS)
  {
    return;
  }

  // Thread-local storage & warp-scope temporary storage allocation
  T thread_data[ITEMS_PER_THREAD];
  __shared__ storage_t storage[TOTAL_WARPS];

  // Instantiate warp-scope algorithm
  warp_merge_sort_t warp_sort(storage[warp_id]);

  const int warp_offset   = LOGICAL_WARP_THREADS * ITEMS_PER_THREAD * warp_id;
  const int thread_offset = warp_offset + warp_sort.get_linear_tid() * ITEMS_PER_THREAD;
  const int valid_items   = segment_sizes[warp_id];

  // Load data
  for (int item = 0; item < ITEMS_PER_THREAD; item++)
  {
    const int idx     = thread_offset + item;
    thread_data[item] = in[idx];
  }
  cub::WARP_SYNC(warp_sort.get_member_mask());

  // Run merge sort test
  action(warp_sort, thread_data, valid_items, oob_default);

  // Store data
  for (int item = 0; item < ITEMS_PER_THREAD; item++)
  {
    const int idx = thread_offset + item;
    out[idx]      = (idx - warp_offset) >= valid_items ? oob_default : thread_data[item];
  }
}

/**
 * @brief Kernel to dispatch to the appropriate WarpMergeSort member function, sorting key-value
 * pairs.
 */
template <int ITEMS_PER_THREAD,
          int LOGICAL_WARP_THREADS,
          int TOTAL_WARPS,
          typename KeyT,
          typename ValueT,
          typename SegmentSizeItT,
          typename ActionT>
__global__ void warp_merge_sort_kernel(
  KeyT* keys_in,
  KeyT* keys_out,
  ValueT* values_in,
  ValueT* values_out,
  SegmentSizeItT segment_sizes,
  KeyT oob_default,
  ActionT action)
{
  using warp_merge_sort_t = cub::WarpMergeSort<KeyT, ITEMS_PER_THREAD, LOGICAL_WARP_THREADS, ValueT>;
  using storage_t         = typename warp_merge_sort_t::TempStorage;

  // Get linear thread and warp index
  const int tid     = cub::RowMajorTid(blockDim.x, blockDim.y, blockDim.z);
  const int warp_id = tid / LOGICAL_WARP_THREADS;

  // Test case of partially finished CTA
  if (warp_id >= TOTAL_WARPS)
  {
    return;
  }

  // Thread-local storage & warp-scope temporary storage allocation
  KeyT keys[ITEMS_PER_THREAD];
  ValueT values[ITEMS_PER_THREAD];
  __shared__ storage_t storage[TOTAL_WARPS];

  // Instantiate warp-scope algorithm
  warp_merge_sort_t warp_sort(storage[warp_id]);

  const int warp_offset   = LOGICAL_WARP_THREADS * ITEMS_PER_THREAD * warp_id;
  const int thread_offset = warp_offset + warp_sort.get_linear_tid() * ITEMS_PER_THREAD;
  const int valid_items   = segment_sizes[warp_id];

  // Load data
  for (int item = 0; item < ITEMS_PER_THREAD; item++)
  {
    const int idx = thread_offset + item;
    keys[item]    = keys_in[idx];
    values[item]  = values_in[idx];
  }
  cub::WARP_SYNC(warp_sort.get_member_mask());

  // Run merge sort test
  action(warp_sort, keys, values, valid_items, oob_default);

  // Store data
  for (int item = 0; item < ITEMS_PER_THREAD; item++)
  {
    const int idx   = thread_offset + item;
    keys_out[idx]   = (idx - warp_offset) >= valid_items ? oob_default : keys[item];
    values_out[idx] = (idx - warp_offset) >= valid_items ? ValueT{} : values[item];
  }
}

// -----------------------------------------------------------
// Dimensions being instantiated:
// {full,partial} x {stable, 'unstable'} x {keys, kv-pairs}
// -----------------------------------------------------------

/**
 * @brief Delegate wrapper for WarpMergeSort::StableSort on keys-only
 */
struct warp_stable_sort_keys_t
{
  template <typename T, int ITEMS_PER_THREAD, typename WarpSortT>
  __device__ void
  operator()(WarpSortT& warp_sort, T (&thread_data)[ITEMS_PER_THREAD], int /*valid_items*/, T /*oob_default*/) const
  {
    warp_sort.StableSort(thread_data, CustomLess{});
  }
};

/**
 * @brief Delegate wrapper for partial WarpMergeSort::StableSort keys-only
 */
struct warp_partial_stable_sort_keys_t
{
  template <typename T, int ITEMS_PER_THREAD, typename WarpSortT>
  __device__ void
  operator()(WarpSortT& warp_sort, T (&thread_data)[ITEMS_PER_THREAD], int valid_items, T oob_default) const
  {
    warp_sort.StableSort(thread_data, CustomLess{}, valid_items, oob_default);
  }
};

/**
 * @brief Delegate wrapper for WarpMergeSort::Sort on keys-only
 */
struct warp_sort_keys_t
{
  template <typename T, int ITEMS_PER_THREAD, typename WarpSortT>
  __device__ void
  operator()(WarpSortT& warp_sort, T (&thread_data)[ITEMS_PER_THREAD], int /*valid_items*/, T /*oob_default*/) const
  {
    warp_sort.Sort(thread_data, CustomLess{});
  }
};

/**
 * @brief Delegate wrapper for partial WarpMergeSort::StableSort keys-only
 */
struct warp_partial_sort_keys_t
{
  template <typename T, int ITEMS_PER_THREAD, typename WarpSortT>
  __device__ void
  operator()(WarpSortT& warp_sort, T (&thread_data)[ITEMS_PER_THREAD], int valid_items, T oob_default) const
  {
    warp_sort.Sort(thread_data, CustomLess{}, valid_items, oob_default);
  }
};

/**
 * @brief Delegate wrapper for WarpMergeSort::StableSort on key-value pairs
 */
struct warp_stable_sort_pairs_t
{
  template <typename KeyT, typename ValueT, int ITEMS_PER_THREAD, typename WarpSortT>
  __device__ void operator()(
    WarpSortT& warp_sort,
    KeyT (&keys)[ITEMS_PER_THREAD],
    ValueT (&values)[ITEMS_PER_THREAD],
    int /*valid_items*/,
    KeyT /*oob_default*/) const
  {
    warp_sort.StableSort(keys, values, CustomLess{});
  }
};

/**
 * @brief Delegate wrapper for partial WarpMergeSort::StableSort key-value pairs
 */
struct warp_partial_stable_sort_pairs_t
{
  template <typename KeyT, typename ValueT, int ITEMS_PER_THREAD, typename WarpSortT>
  __device__ void operator()(
    WarpSortT& warp_sort,
    KeyT (&keys)[ITEMS_PER_THREAD],
    ValueT (&values)[ITEMS_PER_THREAD],
    int valid_items,
    KeyT oob_default) const
  {
    warp_sort.StableSort(keys, values, CustomLess{}, valid_items, oob_default);
  }
};

/**
 * @brief Delegate wrapper for WarpMergeSort::Sort on key-value pairs
 */
struct warp_sort_pairs_t
{
  template <typename KeyT, typename ValueT, int ITEMS_PER_THREAD, typename WarpSortT>
  __device__ void operator()(
    WarpSortT& warp_sort,
    KeyT (&keys)[ITEMS_PER_THREAD],
    ValueT (&values)[ITEMS_PER_THREAD],
    int /*valid_items*/,
    KeyT /*oob_default*/) const
  {
    warp_sort.Sort(keys, values, CustomLess{});
  }
};

/**
 * @brief Delegate wrapper for partial WarpMergeSort::StableSort key-value pairs
 */
struct warp_partial_sort_pairs_t
{
  template <typename KeyT, typename ValueT, int ITEMS_PER_THREAD, typename WarpSortT>
  __device__ void operator()(
    WarpSortT& warp_sort,
    KeyT (&keys)[ITEMS_PER_THREAD],
    ValueT (&values)[ITEMS_PER_THREAD],
    int valid_items,
    KeyT oob_default) const
  {
    warp_sort.Sort(keys, values, CustomLess{}, valid_items, oob_default);
  }
};

/**
 * @brief Dispatch helper function for sorting keys
 */
template <int ITEMS_PER_THREAD,
          int LOGICAL_WARP_THREADS,
          int TOTAL_WARPS,
          typename T,
          typename SegmentSizesItT,
          typename ActionT>
void warp_merge_sort(
  c2h::device_vector<T>& in, c2h::device_vector<T>& out, SegmentSizesItT segment_sizes, T oob_default, ActionT action)
{
  warp_merge_sort_kernel<ITEMS_PER_THREAD, LOGICAL_WARP_THREADS, TOTAL_WARPS>
    <<<1, LOGICAL_WARP_THREADS * TOTAL_WARPS>>>(
      thrust::raw_pointer_cast(in.data()), thrust::raw_pointer_cast(out.data()), segment_sizes, oob_default, action);

  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
}

/**
 * @brief Dispatch helper function for sorting key-value pairs
 */
template <int ITEMS_PER_THREAD,
          int LOGICAL_WARP_THREADS,
          int TOTAL_WARPS,
          typename KeyT,
          typename ValueT,
          typename SegmentSizesItT,
          typename ActionT>
void warp_merge_sort(
  c2h::device_vector<KeyT>& keys_in,
  c2h::device_vector<KeyT>& keys_out,
  c2h::device_vector<ValueT>& values_in,
  c2h::device_vector<ValueT>& values_out,
  SegmentSizesItT segment_sizes,
  KeyT oob_default,
  ActionT action)
{
  warp_merge_sort_kernel<ITEMS_PER_THREAD, LOGICAL_WARP_THREADS, TOTAL_WARPS>
    <<<1, LOGICAL_WARP_THREADS * TOTAL_WARPS>>>(
      thrust::raw_pointer_cast(keys_in.data()),
      thrust::raw_pointer_cast(keys_out.data()),
      thrust::raw_pointer_cast(values_in.data()),
      thrust::raw_pointer_cast(values_out.data()),
      segment_sizes,
      oob_default,
      action);

  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
}

/**
 * @brief Performs a stable sort on per-warp segments of data and assigns oob_default to items that
 * are out-of-bounds.
 */
template <typename RandomItT, typename SegmentSizeItT, typename T>
void compute_host_reference(
  RandomItT h_data, SegmentSizeItT segment_sizes, unsigned int num_segments, T oob_default, int logical_warp_items)
{
  for (unsigned int segment_id = 0; segment_id < num_segments; segment_id++)
  {
    unsigned int segment_size = segment_sizes[segment_id];
    std::stable_sort(h_data, h_data + segment_size);
    std::fill(h_data + segment_size, h_data + logical_warp_items, oob_default);
    h_data += logical_warp_items;
  }
}

/**
 * @brief Stability requirement of the sorting algorithm
 */
enum class stability
{
  stable,
  unstable
};

// List of key types to test
using custom_t  = c2h::custom_type_t<c2h::equal_comparable_t, c2h::lexicographical_less_comparable_t>;
using key_types = c2h::type_list<std::uint8_t, std::int32_t, std::int64_t, custom_t>;

// List of value types
using value_types = c2h::type_list<std::int32_t, custom_t>;

// Logical warp sizes to test
using logical_warp_threads = c2h::enum_type_list<int, 32, 4>;

// Number of items per thread to test
using items_per_thread_list = c2h::enum_type_list<int, 1, 4, 7>;

// Whether the sort is required to be stable or not
using stability_list = c2h::enum_type_list<stability, stability::stable, stability::unstable>;

template <typename TestType>
struct params_t
{
  using type = typename c2h::get<0, TestType>;

  static constexpr int logical_warp_threads = c2h::get<1, TestType>::value;
  static constexpr int items_per_thread     = c2h::get<2, TestType>::value;
  static constexpr int logical_warp_items   = logical_warp_threads * items_per_thread;
  static constexpr int total_warps          = 2;
  static constexpr int tile_size            = items_per_thread * total_warps * logical_warp_threads;
  static constexpr bool is_stable           = c2h::get<3, TestType>::value == stability::stable;
};

C2H_TEST(
  "Warp sort on keys-only works", "[sort][warp]", key_types, logical_warp_threads, items_per_thread_list, stability_list)
{
  using params             = params_t<TestType>;
  using type               = typename params::type;
  using warp_sort_delegate = ::cuda::std::_If<params::is_stable, warp_stable_sort_keys_t, warp_sort_keys_t>;

  // Prepare test data
  c2h::device_vector<type> d_in(params::tile_size);
  c2h::device_vector<type> d_out(params::tile_size);
  auto segment_sizes     = thrust::make_constant_iterator(params::logical_warp_items);
  const auto oob_default = std::numeric_limits<type>::max();
  c2h::gen(C2H_SEED(10), d_in);

  // Run test
  warp_merge_sort<params::items_per_thread, params::logical_warp_threads, params::total_warps>(
    d_in, d_out, segment_sizes, oob_default, warp_sort_delegate{});

  // Prepare verification data
  c2h::host_vector<type> h_in_out = d_in;
  compute_host_reference(h_in_out.begin(), segment_sizes, params::total_warps, oob_default, params::logical_warp_items);

  // Verify results
  REQUIRE(h_in_out == d_out);
}

C2H_TEST("Warp sort keys-only on partial warp-tile works",
         "[sort][warp]",
         key_types,
         logical_warp_threads,
         items_per_thread_list,
         stability_list)
{
  using params = params_t<TestType>;
  using type   = typename params::type;
  using warp_sort_delegate =
    ::cuda::std::_If<params::is_stable, warp_partial_stable_sort_keys_t, warp_partial_sort_keys_t>;

  // Prepare test data
  c2h::device_vector<type> d_in(params::tile_size);
  c2h::device_vector<type> d_out(params::tile_size);
  c2h::device_vector<int> d_segment_sizes(params::total_warps);
  const auto oob_default = std::numeric_limits<type>::max();
  c2h::gen(C2H_SEED(5), d_in);
  c2h::gen(C2H_SEED(5), d_segment_sizes, 0, params::logical_warp_items);

  // Run test
  warp_merge_sort<params::items_per_thread, params::logical_warp_threads, params::total_warps>(
    d_in, d_out, d_segment_sizes.cbegin(), oob_default, warp_sort_delegate{});

  // Prepare verification data
  c2h::host_vector<type> h_in_out     = d_in;
  c2h::host_vector<int> segment_sizes = d_segment_sizes;
  compute_host_reference(h_in_out.begin(), segment_sizes, params::total_warps, oob_default, params::logical_warp_items);

  // Verify results
  REQUIRE(h_in_out == d_out);
}

C2H_TEST("Warp sort on keys-value pairs works",
         "[sort][warp]",
         key_types,
         logical_warp_threads,
         items_per_thread_list,
         stability_list,
         value_types)
{
  using params             = params_t<TestType>;
  using key_type           = typename params::type;
  using value_type         = typename c2h::get<4, TestType>;
  using warp_sort_delegate = ::cuda::std::_If<params::is_stable, warp_stable_sort_pairs_t, warp_sort_pairs_t>;

  // Prepare test data
  c2h::device_vector<key_type> d_keys_in(params::tile_size);
  c2h::device_vector<key_type> d_keys_out(params::tile_size);
  c2h::device_vector<value_type> d_values_in(params::tile_size);
  c2h::device_vector<value_type> d_values_out(params::tile_size);
  auto segment_sizes     = thrust::make_constant_iterator(params::logical_warp_items);
  const auto oob_default = std::numeric_limits<key_type>::max();
  c2h::gen(C2H_SEED(10), d_keys_in);

  // Run test
  warp_merge_sort<params::items_per_thread, params::logical_warp_threads, params::total_warps>(
    d_keys_in, d_keys_out, d_values_in, d_values_out, segment_sizes, oob_default, warp_stable_sort_pairs_t{});

  // Prepare verification data
  c2h::host_vector<key_type> h_keys_in_out     = d_keys_in;
  c2h::host_vector<value_type> h_values_in_out = d_values_in;
  auto cpu_kv_pairs = thrust::make_zip_iterator(h_keys_in_out.begin(), h_values_in_out.begin());
  compute_host_reference(
    cpu_kv_pairs,
    segment_sizes,
    params::total_warps,
    thrust::make_tuple(oob_default, value_type{}),
    params::logical_warp_items);

  // Verify results
  REQUIRE(h_keys_in_out == d_keys_out);
  REQUIRE(h_values_in_out == d_values_out);
}

C2H_TEST("Warp sort on key-value pairs of a partial warp-tile works",
         "[sort][warp]",
         key_types,
         logical_warp_threads,
         items_per_thread_list,
         stability_list,
         value_types)
{
  using params     = params_t<TestType>;
  using key_type   = typename params::type;
  using value_type = typename c2h::get<4, TestType>;
  using warp_sort_delegate =
    ::cuda::std::_If<params::is_stable, warp_partial_stable_sort_pairs_t, warp_partial_sort_pairs_t>;

  // Prepare test data
  c2h::device_vector<key_type> d_keys_in(params::tile_size);
  c2h::device_vector<key_type> d_keys_out(params::tile_size);
  c2h::device_vector<value_type> d_values_in(params::tile_size);
  c2h::device_vector<value_type> d_values_out(params::tile_size);
  c2h::device_vector<int> d_segment_sizes(params::total_warps);
  const auto oob_default = std::numeric_limits<key_type>::max();
  c2h::gen(C2H_SEED(5), d_keys_in);
  c2h::gen(C2H_SEED(5), d_segment_sizes, 0, params::logical_warp_items);

  // Run test
  warp_merge_sort<params::items_per_thread, params::logical_warp_threads, params::total_warps>(
    d_keys_in, d_keys_out, d_values_in, d_values_out, d_segment_sizes.cbegin(), oob_default, warp_sort_delegate{});

  // Prepare verification data
  c2h::host_vector<key_type> h_keys_in_out     = d_keys_in;
  c2h::host_vector<value_type> h_values_in_out = d_values_in;
  c2h::host_vector<int> segment_sizes          = d_segment_sizes;
  auto cpu_kv_pairs = thrust::make_zip_iterator(h_keys_in_out.begin(), h_values_in_out.begin());
  compute_host_reference(
    cpu_kv_pairs,
    segment_sizes,
    params::total_warps,
    thrust::make_tuple(oob_default, value_type{}),
    params::logical_warp_items);

  // Verify results
  REQUIRE(h_keys_in_out == d_keys_out);
  REQUIRE(h_values_in_out == d_values_out);
}
