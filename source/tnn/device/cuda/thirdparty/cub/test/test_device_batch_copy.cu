/******************************************************************************
 * Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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

#include <cub/device/device_copy.cuh>
#include <cub/util_ptx.cuh>

#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/logical.h>
#include <thrust/sequence.h>
#include <thrust/tuple.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "test_util.h"
#include <c2h/vector.cuh>

/**
 * @brief Host-side random data generation
 */
template <typename T>
void GenerateRandomData(
  T* rand_out,
  const std::size_t num_items,
  const T min_rand_val                                                           = std::numeric_limits<T>::min(),
  const T max_rand_val                                                           = std::numeric_limits<T>::max(),
  const std::uint_fast32_t seed                                                  = 320981U,
  typename std::enable_if<std::is_integral<T>::value && (sizeof(T) >= 2)>::type* = nullptr)
{
  // initialize random number generator
  std::mt19937 rng(seed);
  std::uniform_int_distribution<T> uni_dist(min_rand_val, max_rand_val);

  // generate random numbers
  for (std::size_t i = 0; i < num_items; ++i)
  {
    rand_out[i] = uni_dist(rng);
  }
}

/**
 * @brief Used for generating a shuffled but cohesive sequence of output-range offsets for the
 * sequence of input-ranges.
 */
template <typename RangeOffsetT, typename ByteOffsetT, typename RangeSizeT>
c2h::host_vector<ByteOffsetT>
GetShuffledRangeOffsets(const c2h::host_vector<RangeSizeT>& range_sizes, const std::uint_fast32_t seed = 320981U)
{
  RangeOffsetT num_ranges = static_cast<RangeOffsetT>(range_sizes.size());

  // We're remapping the i-th range to pmt_idxs[i]
  std::mt19937 rng(seed);
  c2h::host_vector<RangeOffsetT> pmt_idxs(num_ranges);
  std::iota(pmt_idxs.begin(), pmt_idxs.end(), static_cast<RangeOffsetT>(0));
  std::shuffle(std::begin(pmt_idxs), std::end(pmt_idxs), rng);

  // Compute the offsets using the new mapping
  ByteOffsetT running_offset = {};
  c2h::host_vector<ByteOffsetT> permuted_offsets;
  permuted_offsets.reserve(num_ranges);
  for (auto permuted_range_idx : pmt_idxs)
  {
    permuted_offsets.push_back(running_offset);
    running_offset += range_sizes[permuted_range_idx];
  }

  // Generate the scatter indexes that identify where each range was mapped to
  c2h::host_vector<RangeOffsetT> scatter_idxs(num_ranges);
  for (RangeOffsetT i = 0; i < num_ranges; i++)
  {
    scatter_idxs[pmt_idxs[i]] = i;
  }

  c2h::host_vector<ByteOffsetT> new_offsets(num_ranges);
  for (RangeOffsetT i = 0; i < num_ranges; i++)
  {
    new_offsets[i] = permuted_offsets[scatter_idxs[i]];
  }

  return new_offsets;
}

template <size_t n, typename... T>
typename std::enable_if<n >= thrust::tuple_size<thrust::tuple<T...>>::value>::type
print_tuple(std::ostream&, const thrust::tuple<T...>&)
{}

template <size_t n, typename... T>
typename std::enable_if<n + 1 <= thrust::tuple_size<thrust::tuple<T...>>::value>::type
print_tuple(std::ostream& os, const thrust::tuple<T...>& tup)
{
  _CCCL_IF_CONSTEXPR (n != 0)
  {
    os << ", ";
  }
  os << thrust::get<n>(tup);
  print_tuple<n + 1>(os, tup);
}

template <typename... T>
std::ostream& operator<<(std::ostream& os, const thrust::tuple<T...>& tup)
{
  os << "[";
  print_tuple<0>(os, tup);
  return os << "]";
}

struct Identity
{
  template <typename T>
  __host__ __device__ __forceinline__ T operator()(T x)
  {
    return x;
  }
};

/**
 * @brief Function object class template that takes an offset and returns an iterator at the given
 * offset relative to a fixed base iterator.
 *
 * @tparam IteratorT The random-access iterator type to be returned
 */
template <typename IteratorT>
struct OffsetToIteratorOp
{
  template <typename OffsetT>
  __host__ __device__ __forceinline__ thrust::transform_output_iterator<Identity, IteratorT>
  operator()(OffsetT offset) const
  {
    return thrust::make_transform_output_iterator(base_it + offset, Identity{});
  }
  IteratorT base_it;
};

template <typename AtomicT>
struct RepeatIndex
{
  template <typename OffsetT>
  __host__ __device__ __forceinline__ thrust::constant_iterator<AtomicT> operator()(OffsetT i)
  {
    return thrust::constant_iterator<AtomicT>(static_cast<AtomicT>(i));
  }
};

enum class TestDataGen
{
  // Random offsets into a data segment
  RANDOM,

  // Ranges cohesively reside next to each other
  CONSECUTIVE
};

std::string TestDataGenToString(TestDataGen gen)
{
  switch (gen)
  {
    case TestDataGen::RANDOM:
      return "TestDataGen::RANDOM";
    case TestDataGen::CONSECUTIVE:
      return "TestDataGen::CONSECUTIVE";
    default:
      return "Unknown";
  }
}

/**
 * @brief
 *
 * @tparam AtomicT The type of the elements being copied
 * @tparam RangeOffsetT Type used for indexing into the array of ranges
 * @tparam RangeSizeT Type used for indexing into individual elements of a range (large enough to
 * cover the max range size)
 * @tparam ByteOffsetT Type used for indexing into elements over *all* the ranges' sizes
 */
template <typename AtomicT, typename RangeOffsetT, typename RangeSizeT, typename ByteOffsetT>
void RunTest(RangeOffsetT num_ranges, RangeSizeT min_range_size, RangeSizeT max_range_size, TestDataGen output_gen)
try
{
  // Range segment data (their offsets and sizes)
  c2h::host_vector<RangeSizeT> h_range_sizes(num_ranges);
  thrust::counting_iterator<RangeOffsetT> iota(0);
  auto d_range_srcs = thrust::make_transform_iterator(iota, RepeatIndex<AtomicT>{});
  c2h::host_vector<ByteOffsetT> h_offsets(num_ranges + 1);

  // Generate the range sizes
  GenerateRandomData(h_range_sizes.data(), h_range_sizes.size(), min_range_size, max_range_size);

  // Compute the total bytes to be copied
  std::partial_sum(h_range_sizes.begin(), h_range_sizes.end(), h_offsets.begin() + 1);
  const ByteOffsetT num_total_items = h_offsets.back();
  h_offsets.pop_back();

  constexpr int32_t shuffle_seed = 123241;

  // Shuffle output range source-offsets
  if (output_gen == TestDataGen::RANDOM)
  {
    h_offsets = GetShuffledRangeOffsets<RangeOffsetT, ByteOffsetT>(h_range_sizes, shuffle_seed);
  }

  // Device-side resources
  c2h::device_vector<AtomicT> d_out(num_total_items);
  c2h::device_vector<ByteOffsetT> d_offsets(h_offsets);
  c2h::device_vector<RangeSizeT> d_range_sizes(h_range_sizes);

  // Prepare d_range_dsts
  using AtomicIterT = typename c2h::device_vector<AtomicT>::iterator;
  OffsetToIteratorOp<AtomicIterT> dst_transform_op{d_out.begin()};
  auto d_range_dsts = thrust::make_transform_iterator(d_offsets.begin(), dst_transform_op);

  // Get temporary storage requirements
  size_t temp_storage_bytes = 0;
  CubDebugExit(cub::DeviceCopy::Batched(
    nullptr, temp_storage_bytes, d_range_srcs, d_range_dsts, d_range_sizes.cbegin(), num_ranges));

  c2h::device_vector<std::uint8_t> d_temp_storage(temp_storage_bytes);

  c2h::host_vector<AtomicT> h_out(num_total_items);
  c2h::host_vector<AtomicT> h_gpu_results(num_total_items);

  // Invoke device-side algorithm being under test
  CubDebugExit(cub::DeviceCopy::Batched(
    thrust::raw_pointer_cast(d_temp_storage.data()),
    temp_storage_bytes,
    d_range_srcs,
    d_range_dsts,
    d_range_sizes.cbegin(),
    num_ranges));

  // Copy back the output range
  h_gpu_results = d_out;

  // CPU-side result generation for verification
  for (RangeOffsetT i = 0; i < num_ranges; i++)
  {
    std::copy(d_range_srcs[i], d_range_srcs[i] + h_range_sizes[i], h_out.begin() + h_offsets[i]);
  }

  const auto it_pair = std::mismatch(h_gpu_results.cbegin(), h_gpu_results.cend(), h_out.cbegin());

  if (it_pair.first != h_gpu_results.cend())
  {
    std::cout << "Mismatch at index " << std::distance(h_gpu_results.cbegin(), it_pair.first)
              << ", CPU vs. GPU: " << *it_pair.second << ", " << *it_pair.first << "\n";
  }
  AssertEquals(it_pair.first, h_gpu_results.cend());
}
catch (std::bad_alloc& e)
{
  (void) e;
#ifdef DEBUG_CHECKED_ALLOC_FAILURE
  std::cout
    << "Skipping test 'RunTest(" << num_ranges << ", " //
    << min_range_size << ", " //
    << max_range_size << ", " //
    << TestDataGenToString(output_gen) << ")" //
    << "' due to insufficient memory: " << e.what() << "\n";
#endif // DEBUG_CHECKED_ALLOC_FAILURE
}

struct object_with_non_trivial_ctor
{
  static constexpr int MAGIC = 923390;

  int field;
  int magic;

  __host__ __device__ object_with_non_trivial_ctor()
  {
    magic = MAGIC;
    field = 0;
  }
  __host__ __device__ object_with_non_trivial_ctor(int f)
  {
    magic = MAGIC;
    field = f;
  }

  object_with_non_trivial_ctor(const object_with_non_trivial_ctor& x) = default;

  __host__ __device__ object_with_non_trivial_ctor& operator=(const object_with_non_trivial_ctor& x)
  {
    if (magic == MAGIC)
    {
      field = x.field;
    }
    return *this;
  }
};

void nontrivial_constructor_test()
{
  constexpr int num_buffers = 3;
  c2h::device_vector<object_with_non_trivial_ctor> a(num_buffers, object_with_non_trivial_ctor(99));
  c2h::device_vector<object_with_non_trivial_ctor> b(num_buffers);
  using iterator = c2h::device_vector<object_with_non_trivial_ctor>::iterator;

  c2h::device_vector<iterator> a_iter{a.begin(), a.begin() + 1, a.begin() + 2};

  c2h::device_vector<iterator> b_iter{b.begin(), b.begin() + 1, b.begin() + 2};

  auto sizes = thrust::make_constant_iterator(1);

  std::uint8_t* d_temp_storage{};
  std::size_t temp_storage_bytes{};

  cub::DeviceCopy::Batched(d_temp_storage, temp_storage_bytes, a_iter.begin(), b_iter.begin(), sizes, num_buffers);

  c2h::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
  d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

  cub::DeviceCopy::Batched(d_temp_storage, temp_storage_bytes, a_iter.begin(), b_iter.begin(), sizes, num_buffers);

  for (int i = 0; i < 10; i++)
  {
    object_with_non_trivial_ctor ha(a[i]);
    object_with_non_trivial_ctor hb(b[i]);
    int ia = ha.field;
    int ib = hb.field;

    if (ia != ib)
    {
      std::cerr << "error: " << ia << " != " << ib << "\n";
    }
  }
}

int main(int argc, char** argv)
{
  CommandLineArgs args(argc, argv);

  // Initialize device
  CubDebugExit(args.DeviceInit());

  //---------------------------------------------------------------------
  // DeviceCopy::Batched tests
  //---------------------------------------------------------------------
  // Run the nontrivial constructor test suggested by senior-zero
  nontrivial_constructor_test();
  // Type used for indexing into the array of ranges
  using RangeOffsetT = uint32_t;

  // Type used for indexing into individual elements of a range (large enough to cover the max range
  using RangeSizeT = uint32_t;

  // Type used for indexing into bytes over *all* the ranges' sizes
  using ByteOffsetT = uint32_t;

  // Total number of bytes that are targeted to be copied on each run
  constexpr RangeOffsetT target_copy_size = 64U << 20;

  // The number of randomly
  constexpr std::size_t num_rnd_range_tests = 32;

  // Each range's size will be random within this interval
  c2h::host_vector<std::pair<std::size_t, std::size_t>> size_ranges = {
    {0, 1},
    {1, 2},
    {0, 16},
    {1, 32},
    {1, 1024},
    {1, 32 * 1024},
    {128 * 1024, 256 * 1024},
    {target_copy_size, target_copy_size}};

  std::mt19937 rng(0);
  std::uniform_int_distribution<std::size_t> size_dist(1, 1000000);
  for (std::size_t i = 0; i < num_rnd_range_tests; i++)
  {
    auto range_begin = size_dist(rng);
    auto range_end   = size_dist(rng);
    if (range_begin > range_end)
    {
      std::swap(range_begin, range_end);
    }
    size_ranges.push_back({range_begin, range_end});
  }

  for (const auto& size_range : size_ranges)
  {
    // The most granular type being copied.
    using AtomicCopyT         = int64_t;
    RangeSizeT min_range_size = static_cast<RangeSizeT>(CUB_ROUND_UP_NEAREST(size_range.first, sizeof(AtomicCopyT)));
    RangeSizeT max_range_size =
      static_cast<RangeSizeT>(CUB_ROUND_UP_NEAREST(size_range.second, static_cast<RangeSizeT>(sizeof(AtomicCopyT))));
    double average_range_size      = (min_range_size + max_range_size) / 2.0;
    RangeOffsetT target_num_ranges = static_cast<RangeOffsetT>(target_copy_size / average_range_size);

    // Run tests with output ranges being consecutive
    RunTest<AtomicCopyT, RangeOffsetT, RangeSizeT, ByteOffsetT>(
      target_num_ranges, min_range_size, max_range_size, TestDataGen::CONSECUTIVE);

    // Run tests with output ranges being randomly shuffled
    RunTest<AtomicCopyT, RangeOffsetT, RangeSizeT, ByteOffsetT>(
      target_num_ranges, min_range_size, max_range_size, TestDataGen::RANDOM);
  }

  for (const auto& size_range : size_ranges)
  {
    // The most granular type being copied.
    using AtomicCopyT         = thrust::tuple<int64_t, int32_t, int16_t, char, char>;
    RangeSizeT min_range_size = static_cast<RangeSizeT>(CUB_ROUND_UP_NEAREST(size_range.first, sizeof(AtomicCopyT)));
    RangeSizeT max_range_size =
      static_cast<RangeSizeT>(CUB_ROUND_UP_NEAREST(size_range.second, static_cast<RangeSizeT>(sizeof(AtomicCopyT))));
    double average_range_size      = (min_range_size + max_range_size) / 2.0;
    RangeOffsetT target_num_ranges = static_cast<RangeOffsetT>(target_copy_size / average_range_size);

    // Run tests with output ranges being consecutive
    RunTest<AtomicCopyT, RangeOffsetT, RangeSizeT, ByteOffsetT>(
      target_num_ranges, min_range_size, max_range_size, TestDataGen::CONSECUTIVE);

    // Run tests with output ranges being randomly shuffled
    RunTest<AtomicCopyT, RangeOffsetT, RangeSizeT, ByteOffsetT>(
      target_num_ranges, min_range_size, max_range_size, TestDataGen::RANDOM);
  }

  //---------------------------------------------------------------------
  // DeviceCopy::Batched test with 64-bit offsets
  //---------------------------------------------------------------------
  using ByteOffset64T = uint64_t;
  using RangeSize64T  = uint64_t;
  ByteOffset64T large_target_copy_size =
    static_cast<ByteOffset64T>(std::numeric_limits<uint32_t>::max()) + (128ULL * 1024ULL * 1024ULL);
  // Make sure min_range_size is in fact smaller than max range size
  constexpr RangeOffsetT single_range = 1;

  // Run tests with output ranges being consecutive
  RunTest<uint8_t, RangeOffsetT, RangeSize64T, ByteOffset64T>(
    single_range, large_target_copy_size, large_target_copy_size, TestDataGen::CONSECUTIVE);
}
