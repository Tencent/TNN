/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2022, NVIDIA CORPORATION.  All rights reserved.
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

#include <cub/device/device_histogram.cuh>
#include <cub/iterator/counting_input_iterator.cuh>

#include <cuda/std/__algorithm_>
#include <cuda/std/array>
#include <cuda/std/bit>
#include <cuda/std/type_traits>

#include <algorithm>
#include <limits>
#include <tuple>

#include "catch2_test_launch_helper.h"
#include <c2h/catch2_test_helper.cuh>
#include <c2h/extended_types.cuh>
#include <c2h/vector.cuh>

// %PARAM% TEST_LAUNCH lid 0:1:2

DECLARE_LAUNCH_WRAPPER(cub::DeviceHistogram::HistogramEven, histogram_even);
DECLARE_LAUNCH_WRAPPER(cub::DeviceHistogram::HistogramRange, histogram_range);

DECLARE_TMPL_LAUNCH_WRAPPER(cub::DeviceHistogram::MultiHistogramEven,
                            multi_histogram_even,
                            ESCAPE_LIST(int Channels, int ActiveChannels),
                            ESCAPE_LIST(Channels, ActiveChannels));

DECLARE_TMPL_LAUNCH_WRAPPER(cub::DeviceHistogram::MultiHistogramRange,
                            multi_histogram_range,
                            ESCAPE_LIST(int Channels, int ActiveChannels),
                            ESCAPE_LIST(Channels, ActiveChannels));

namespace cs = cuda::std;
using cs::array;
using cs::size_t;

template <typename T>
auto cast_if_half_pointer(T* p) -> T*
{
  return p;
}

#if TEST_HALF_T
auto cast_if_half_pointer(half_t* p) -> __half*
{
  return reinterpret_cast<__half*>(p);
}

auto cast_if_half_pointer(const half_t* p) -> const __half*
{
  return reinterpret_cast<const __half*>(p);
}
#endif

template <typename T>
using caller_vector = c2h::
#if TEST_LAUNCH == 1
  device_vector<T>;
#else
  host_vector<T>;
#endif

template <typename T, size_t N>
auto to_caller_vector_of_ptrs(array<c2h::device_vector<T>, N>& in)
  -> caller_vector<decltype(cast_if_half_pointer(cs::declval<T*>()))>
{
  c2h::host_vector<decltype(cast_if_half_pointer(cs::declval<T*>()))> r(N);
  for (size_t i = 0; i < N; i++)
  {
    r[i] = cast_if_half_pointer(thrust::raw_pointer_cast(in[i].data()));
  }
  return r;
}

template <int Channels, typename CounterT, size_t ActiveChannels, typename SampleT, typename TransformOp, typename OffsetT>
auto compute_reference_result(
  const c2h::host_vector<SampleT>& h_samples,
  const TransformOp& sample_to_bin_index,
  const array<int, ActiveChannels>& num_levels,
  OffsetT width,
  OffsetT height,
  OffsetT row_pitch) -> array<c2h::host_vector<CounterT>, ActiveChannels>
{
  auto h_histogram = array<c2h::host_vector<CounterT>, ActiveChannels>{};
  for (size_t c = 0; c < ActiveChannels; ++c)
  {
    h_histogram[c].resize(num_levels[c] - 1);
  }
  for (OffsetT row = 0; row < height; ++row)
  {
    for (OffsetT pixel = 0; pixel < width; ++pixel)
    {
      for (size_t c = 0; c < ActiveChannels; ++c)
      {
        // TODO(bgruber): use an mdspan to access h_samples
        const auto offset = row * (row_pitch / sizeof(SampleT)) + pixel * Channels + c;
        const int bin     = sample_to_bin_index(static_cast<int>(c), h_samples[offset]);
        if (bin >= 0 && bin < static_cast<int>(h_histogram[c].size())) // if bin is valid
        {
          ++h_histogram[c][bin];
        }
      }
    }
  }
  return h_histogram;
}

template <size_t ActiveChannels, typename LevelT>
auto setup_bin_levels_for_even(const array<int, ActiveChannels>& num_levels, LevelT max_level, int max_level_count)
  -> array<array<LevelT, ActiveChannels>, 2>
{
  array<array<LevelT, ActiveChannels>, 2> levels;
  auto& lower_level = levels[0];
  auto& upper_level = levels[1];

  // Create upper and lower levels between between [0:max_level], getting narrower with each channel. Example:
  //    max_level = 256
  //   num_levels = { 257, 129,  65 }
  //  lower_level = {   0,  64,  96 }
  //  upper_level = { 256, 192, 160 }

  // TODO(bgruber): eventually, we could just pick a random lower/upper bound for each channel

  const auto min_bin_width = max_level / (max_level_count - 1);
  REQUIRE(min_bin_width > 0);

  for (size_t c = 0; c < ActiveChannels; ++c)
  {
    const int num_bins        = num_levels[c] - 1;
    const auto min_hist_width = num_bins * min_bin_width;
    lower_level[c]            = static_cast<LevelT>(max_level / 2 - min_hist_width / 2);
    upper_level[c]            = static_cast<LevelT>(max_level / 2 + min_hist_width / 2);
    CAPTURE(c, num_levels[c]);
    REQUIRE(lower_level[c] < upper_level[c]);
  }
  return levels;
}

template <size_t ActiveChannels, typename LevelT>
auto setup_bin_levels_for_range(const array<int, ActiveChannels>& num_levels, LevelT max_level, int max_level_count)
  -> array<c2h::host_vector<LevelT>, ActiveChannels>
{
  // TODO(bgruber): eventually, we could just pick random levels for each channel

  const auto min_bin_width = max_level / (max_level_count - 1);
  REQUIRE(min_bin_width > 0);

  array<c2h::host_vector<LevelT>, ActiveChannels> levels;
  for (size_t c = 0; c < ActiveChannels; ++c)
  {
    levels[c].resize(num_levels[c]);
    const int num_bins        = num_levels[c] - 1;
    const auto min_hist_width = num_bins * min_bin_width;
    const auto lower_level    = (max_level / 2 - min_hist_width / 2);
    for (int l = 0; l < num_levels[c]; ++l)
    {
      levels[c][l] = static_cast<LevelT>(lower_level + l * min_bin_width);
      if (l > 0)
      {
        REQUIRE(levels[c][l - 1] < levels[c][l]);
      }
    }
  }
  return levels;
}

template <size_t ActiveChannels>
auto generate_level_counts_to_test(int max_level_count) -> array<int, ActiveChannels>
{
  // TODO(bgruber): eventually, just pick a random number of levels per channel

  // first channel tests maximum number of levels, later channels less and less
  array<int, ActiveChannels> r{max_level_count};
  for (size_t c = 1; c < ActiveChannels; ++c)
  {
    r[c] = r[c - 1] / 2 + 1;
  }
  return r;
}

struct bit_and_anything
{
  template <typename T>
  _CCCL_HOST_DEVICE auto operator()(const T& a, const T& b) const -> T
  {
    using U = typename cub::Traits<T>::UnsignedBits;
    return ::cuda::std::bit_cast<T>(static_cast<U>(::cuda::std::bit_cast<U>(a) & ::cuda::std::bit_cast<U>(b)));
  }
};

template <typename SampleT, int Channels, size_t ActiveChannels, typename CounterT, typename LevelT, typename OffsetT>
void test_even_and_range(LevelT max_level, int max_level_count, OffsetT width, OffsetT height, int entropy_reduction = 0)
{
  const auto padding_bytes = static_cast<OffsetT>(GENERATE(size_t{0}, 13 * sizeof(SampleT)));
  CAPTURE(
    c2h::type_name<SampleT>(),
    c2h::type_name<CounterT>(),
    c2h::type_name<LevelT>(),
    c2h::type_name<OffsetT>(),
    Channels,
    ActiveChannels,
    CoutCast(max_level),
    max_level_count,
    width,
    height,
    padding_bytes,
    entropy_reduction);

  // Prepare input image (samples)
  const OffsetT row_pitch = width * Channels * sizeof(SampleT) + padding_bytes;
  const auto num_levels   = generate_level_counts_to_test<ActiveChannels>(max_level_count);

  const OffsetT total_samples = height * (row_pitch / sizeof(SampleT));
  c2h::device_vector<SampleT> d_samples;
  d_samples.resize(total_samples);

  if (entropy_reduction >= 0)
  {
    c2h::gen(C2H_SEED(1), d_samples, SampleT{0}, static_cast<SampleT>(max_level));
    if (entropy_reduction > 0)
    {
      c2h::device_vector<SampleT> tmp(d_samples.size());
      for (int i = 0; i < entropy_reduction; ++i)
      {
        c2h::gen(C2H_SEED(1), tmp);
        thrust::transform(
          c2h::device_policy, d_samples.cbegin(), d_samples.cend(), tmp.cbegin(), d_samples.begin(), bit_and_anything{});
      }
    }
  }

  auto h_samples = c2h::host_vector<SampleT>(d_samples);

  // Allocate output histogram
  auto d_histogram = array<c2h::device_vector<CounterT>, ActiveChannels>();
  for (size_t c = 0; c < ActiveChannels; ++c)
  {
    d_histogram[c].resize(num_levels[c] - 1);
  }

  SECTION("HistogramEven")
  {
    // Setup levels
    const auto levels       = setup_bin_levels_for_even(num_levels, max_level, max_level_count);
    const auto& lower_level = levels[0]; // TODO(bgruber): use structured bindings in C++17
    const auto& upper_level = levels[1];
    CAPTURE(lower_level, upper_level);

    // Compute reference result
    auto fp_scales = array<LevelT, ActiveChannels>{}; // only used when LevelT is floating point
    std::ignore    = fp_scales; // casting to void was insufficient. TODO(bgruber): use [[maybe_unsued]] in C++17
    for (size_t c = 0; c < ActiveChannels; ++c)
    {
      _CCCL_IF_CONSTEXPR (!cs::is_integral<LevelT>::value)
      {
        fp_scales[c] = static_cast<LevelT>(num_levels[c] - 1) / static_cast<LevelT>(upper_level[c] - lower_level[c]);
      }
    }

    auto sample_to_bin_index = [&](int channel, SampleT sample) {
      using common_t             = typename cs::common_type<LevelT, SampleT>::type;
      const auto n               = num_levels[channel];
      const auto max             = static_cast<common_t>(upper_level[channel]);
      const auto min             = static_cast<common_t>(lower_level[channel]);
      const auto promoted_sample = static_cast<common_t>(sample);
      if (promoted_sample < min || promoted_sample >= max)
      {
        return n; // out of range
      }
      _CCCL_IF_CONSTEXPR (cs::is_integral<LevelT>::value)
      {
        // Accurate bin computation following the arithmetic we guarantee in the HistoEven docs
        return static_cast<int>(static_cast<uint64_t>(promoted_sample - min) * static_cast<uint64_t>(n - 1)
                                / static_cast<uint64_t>(max - min));
      }
      else
      {
        return static_cast<int>((sample - min) * fp_scales[channel]);
      }
      _CCCL_UNREACHABLE();
    };
    auto h_histogram = compute_reference_result<Channels, CounterT>(
      h_samples, sample_to_bin_index, num_levels, width, height, row_pitch);

    // Compute result and verify
    {
      const auto* sample_ptr = cast_if_half_pointer(thrust::raw_pointer_cast(d_samples.data()));
      _CCCL_IF_CONSTEXPR (ActiveChannels == 1 && Channels == 1)
      {
        histogram_even(
          sample_ptr,
          cast_if_half_pointer(thrust::raw_pointer_cast(d_histogram[0].data())),
          num_levels[0],
          cast_if_half_pointer(lower_level.data())[0],
          cast_if_half_pointer(upper_level.data())[0],
          width,
          height,
          row_pitch);
      }
      else
      {
        auto d_histogram_ptrs    = to_caller_vector_of_ptrs(d_histogram);
        const auto d_num_levels  = caller_vector<int>(num_levels.begin(), num_levels.end());
        const auto d_lower_level = caller_vector<LevelT>(lower_level.begin(), lower_level.end());
        const auto d_upper_level = caller_vector<LevelT>(upper_level.begin(), upper_level.end());
        multi_histogram_even<Channels, ActiveChannels>(
          sample_ptr,
          cast_if_half_pointer(thrust::raw_pointer_cast(d_histogram_ptrs.data())),
          thrust::raw_pointer_cast(d_num_levels.data()),
          cast_if_half_pointer(thrust::raw_pointer_cast(d_lower_level.data())),
          cast_if_half_pointer(thrust::raw_pointer_cast(d_upper_level.data())),
          width,
          height,
          row_pitch);
      }
    }
    for (size_t c = 0; c < ActiveChannels; ++c)
    {
      CHECK(h_histogram[c] == d_histogram[c]);
    }
  }

  SECTION("HistogramRange")
  {
    // Setup levels
    const auto h_levels = setup_bin_levels_for_range(num_levels, max_level, max_level_count);
    CAPTURE(h_levels);

    // Compute reference result
    const auto sample_to_bin_index = [&](int channel, SampleT sample) {
      const auto* l  = h_levels[channel].data();
      const auto n   = static_cast<int>(h_levels[channel].size());
      const auto* ub = std::upper_bound(l, l + n, static_cast<LevelT>(sample));
      return ub == l /* sample smaller than first bin */ ? n : static_cast<int>(std::distance(l, ub) - 1);
    };
    auto h_histogram = compute_reference_result<Channels, CounterT>(
      h_samples, sample_to_bin_index, num_levels, width, height, row_pitch);

    // Compute result and verify
    {
      const auto* sample_ptr = cast_if_half_pointer(thrust::raw_pointer_cast(d_samples.data()));
      auto d_levels          = array<c2h::device_vector<LevelT>, ActiveChannels>{};
      std::copy(h_levels.begin(), h_levels.end(), d_levels.begin());
      _CCCL_IF_CONSTEXPR (ActiveChannels == 1 && Channels == 1)
      {
        histogram_range(
          sample_ptr,
          cast_if_half_pointer(thrust::raw_pointer_cast(d_histogram[0].data())),
          num_levels[0],
          cast_if_half_pointer(thrust::raw_pointer_cast(d_levels[0].data())),
          width,
          height,
          row_pitch);
      }
      else
      {
        auto d_histogram_ptrs   = to_caller_vector_of_ptrs(d_histogram);
        const auto d_num_levels = caller_vector<int>(num_levels.begin(), num_levels.end());
        const auto level_ptrs   = to_caller_vector_of_ptrs(d_levels);
        multi_histogram_range<Channels, ActiveChannels>(
          sample_ptr,
          cast_if_half_pointer(thrust::raw_pointer_cast(d_histogram_ptrs.data())),
          thrust::raw_pointer_cast(d_num_levels.data()),
          cast_if_half_pointer(thrust::raw_pointer_cast(level_ptrs.data())),
          width,
          height,
          row_pitch);
      }
    }
    for (size_t c = 0; c < ActiveChannels; ++c)
    {
      CHECK(h_histogram[c] == d_histogram[c]);
    }
  }
}

using types =
  c2h::type_list<std::int8_t,
                 std::uint8_t,
                 std::int16_t,
                 std::uint16_t,
                 std::int32_t,
                 std::uint32_t,
                 std::int64_t,
                 std::uint64_t,
#if TEST_HALF_T
                 half_t,
#endif
                 float,
                 double>;

C2H_TEST("DeviceHistogram::Histogram* basic use", "[histogram][device]", types)
{
  using sample_t = c2h::get<0, TestType>;
  using level_t =
    typename cs::conditional<cub::NumericTraits<sample_t>::CATEGORY == cub::FLOATING_POINT, sample_t, int>::type;
  // Max for int8/uint8 is 2^8, for half_t is 2^10. Beyond, we would need a different level generation
  const auto max_level       = level_t{sizeof(sample_t) == 1 ? 126 : 1024};
  const auto max_level_count = (sizeof(sample_t) == 1 ? 126 : 1024) + 1;
  test_even_and_range<sample_t, 4, 3, int>(max_level, max_level_count, 1920, 1080);
}

// TODO(bgruber): float produces INFs in the HistogramRange test setup AND the HistogramEven implementation
// This test covers int32 and int64 arithmetic for bin computation
C2H_TEST("DeviceHistogram::Histogram* large levels", "[histogram][device]", c2h::remove<types, float>)
{
  using sample_t             = c2h::get<0, TestType>;
  using level_t              = sample_t;
  const auto max_level_count = 128;
  auto max_level             = cub::NumericTraits<level_t>::Max();
  _CCCL_IF_CONSTEXPR (sizeof(sample_t) > sizeof(int))
  {
    max_level /= static_cast<level_t>(max_level_count - 1); // cf. overflow detection in ScaleTransform::MayOverflow
  }
  test_even_and_range<sample_t, 4, 3, int>(max_level, max_level_count, 1920, 1080);
}

C2H_TEST("DeviceHistogram::Histogram* odd image sizes", "[histogram][device]")
{
  using sample_t                = int;
  using level_t                 = int;
  constexpr sample_t max_level  = 256;
  constexpr int max_level_count = 256 + 1;

  using P      = cs::pair<int, int>;
  const auto p = GENERATE(P{1920, 0}, P{0, 0}, P{0, 1080}, P{1, 1}, P{15, 1}, P{1, 15}, P{10000, 1}, P{1, 10000});
  test_even_and_range<sample_t, 4, 3, int, level_t, int>(max_level, max_level_count, p.first, p.second);
}

C2H_TEST("DeviceHistogram::Histogram* entropy", "[histogram][device]")
{
  const int entropy_reduction = GENERATE(-1, 3, 5); // entropy_reduction = -1 -> all samples == 0
  test_even_and_range<int, 4, 3, int>(256, 256 + 1, 1920, 1080, entropy_reduction);
}

template <int Channels, int ActiveChannels>
struct ChannelConfig
{
  static constexpr auto channels        = Channels;
  static constexpr auto active_channels = ActiveChannels;
};

C2H_TEST_LIST("DeviceHistogram::Histogram* channel configs",
              "[histogram][device]",
              ChannelConfig<1, 1>,
              ChannelConfig<3, 3>,
              ChannelConfig<4, 3>,
              ChannelConfig<4, 4>)
{
  test_even_and_range<int, TestType::channels, TestType::active_channels, int, int, int>(256, 256 + 1, 128, 32);
}

// Testing only HistogramEven is fine, because HistogramRange shares the loading logic and the different binning
// implementations are not affected by the iterator.
C2H_TEST("DeviceHistogram::HistogramEven sample iterator", "[histogram_even][device]")
{
  using sample_t                 = int;
  const auto width               = 100;
  const auto padding             = 13; // in elements
  const auto height              = 30;
  constexpr auto channels        = 4;
  constexpr auto active_channels = 3;
  const auto row_pitch           = (width + padding) * channels * static_cast<int>(sizeof(sample_t));
  const auto total_values        = (width + padding) * channels * height;

  const auto num_levels  = array<int, active_channels>{11, 3, 2};
  const auto lower_level = caller_vector<int>{0, -10, cs::numeric_limits<int>::lowest()};
  const auto upper_level = caller_vector<int>{total_values, 10, cs::numeric_limits<int>::max()};

  auto sample_iterator = cub::CountingInputIterator<sample_t>(0);

  // Channel #0: 0, 4,  8, 12
  // Channel #1: 1, 5,  9, 13
  // Channel #2: 2, 6, 10, 14
  // unused:     3, 7, 11, 15

  auto d_histogram = array<c2h::device_vector<int>, active_channels>();
  for (int c = 0; c < active_channels; ++c)
  {
    d_histogram[c].resize(num_levels[c] - 1);
  }

  multi_histogram_even<channels, active_channels>(
    sample_iterator,
    thrust::raw_pointer_cast(to_caller_vector_of_ptrs(d_histogram).data()),
    thrust::raw_pointer_cast(caller_vector<int>(num_levels.begin(), num_levels.end()).data()),
    thrust::raw_pointer_cast(lower_level.data()),
    thrust::raw_pointer_cast(upper_level.data()),
    width,
    height,
    row_pitch);

  CHECK(d_histogram[0] == c2h::host_vector<int>(10, (width * height) / 10));
  CHECK(d_histogram[1] == c2h::host_vector<int>{0, 3});
  CHECK(d_histogram[2] == c2h::host_vector<int>{width * height});
}

// Regression: https://github.com/NVIDIA/cub/issues/479
C2H_TEST("DeviceHistogram::Histogram* regression NVIDIA/cub#479", "[histogram][device]")
{
  test_even_and_range<float, 4, 3, int>(12, 7, 1920, 1080);
}

C2H_TEST("DeviceHistogram::Histogram* down-conversion size_t to int", "[histogram][device]")
{
  _CCCL_IF_CONSTEXPR (sizeof(size_t) != sizeof(int))
  {
    using offset_t = cs::make_signed<size_t>::type;
    test_even_and_range<unsigned char, 4, 3, int>(256, 256 + 1, offset_t{1920}, offset_t{1080});
  }
}

C2H_TEST("DeviceHistogram::HistogramRange levels/samples aliasing", "[histogram_range][device]")
{
  constexpr int num_levels = 7;
  constexpr int h_samples[]{
    0,  2,  4,  6,  8,  10, 12, // levels
    1, // bin 0
    3,  3, // bin 1
    5,  5,  5, // bin 2
    7,  7,  7,  7, // bin 3
    9,  9,  9,  9,  9, // bin 4
    11, 11, 11, 11, 11, 11 // bin 5
  };

  auto d_histogram = c2h::device_vector<int>(num_levels - 1);
  auto d_samples   = c2h::device_vector<int>(cs::begin(h_samples), cs::end(h_samples));
  histogram_range(
    thrust::raw_pointer_cast(d_samples.data()),
    thrust::raw_pointer_cast(d_histogram.data()),
    num_levels,
    thrust::raw_pointer_cast(d_samples.data()), // Alias levels with samples (fancy way to `d_histogram[bin]++`).
    static_cast<int>(d_samples.size()));

  auto h_histogram = c2h::host_vector<int>(d_histogram);
  for (int bin = 0; bin < num_levels - 1; bin++)
  {
    // Each bin should contain `bin + 1` samples, plus one extra, since samples also contain levels.
    CHECK(h_histogram[bin] == bin + 2);
  }
}

// We cannot use launch wrappers for this test, since it checks error codes explicitly.
#if TEST_LAUNCH == 0
// Our bin computation for HistogramEven is guaranteed only for when (max_level - min_level) * num_bins does not
// overflow using uint64_t arithmetic. In case of overflow, we expect cudaErrorInvalidValue to be returned.
C2H_TEST_LIST("DeviceHistogram::HistogramEven bin computation does not overflow",
              "[histogram_even][device]",
              uint8_t,
              uint16_t,
              uint32_t,
              uint64_t)
{
  using sample_t                 = TestType;
  using counter_t                = uint32_t;
  constexpr sample_t lower_level = 0;
  constexpr sample_t upper_level = cs::numeric_limits<sample_t>::max();
  constexpr auto num_samples     = 1000;
  auto d_samples                 = cub::CountingInputIterator<sample_t>{0UL};
  auto d_histo_out               = c2h::device_vector<counter_t>(1024);
  const auto num_bins            = GENERATE(1, 2);

  // Verify we always initializes temp_storage_bytes
  constexpr size_t canary_bytes = 3;
  size_t temp_storage_bytes     = canary_bytes;
  const auto error1             = cub::DeviceHistogram::HistogramEven(
    nullptr,
    temp_storage_bytes,
    d_samples,
    raw_pointer_cast(d_histo_out.data()),
    num_bins + 1,
    lower_level,
    upper_level,
    num_samples);
  // CHECK(error1 == ???); // TODO(bgruber): add a new check? what is expected? It's neither 0 or 1.
  std::ignore = error1;
  CHECK(temp_storage_bytes != canary_bytes);

  auto temp_storage = c2h::device_vector<char>(temp_storage_bytes);
  const auto error2 = cub::DeviceHistogram::HistogramEven(
    raw_pointer_cast(temp_storage.data()),
    temp_storage_bytes,
    d_samples,
    raw_pointer_cast(d_histo_out.data()),
    num_bins + 1,
    lower_level,
    upper_level,
    num_samples);

  // Since test #1 is just a single bin, we expect it to succeed
  // Since we promote up to 64-bit integer arithmetic we expect tests to not overflow for types of
  // up to 4 bytes. For 64-bit and wider types, we do not perform further promotion to even wider
  // types, hence we expect cudaErrorInvalidValue to be returned to indicate of a potential overflow
  // Ensure we do not return an error on querying temporary storage requirements
  CHECK(error2 == (num_bins == 1 || sizeof(sample_t) <= 4UL ? cudaSuccess : cudaErrorInvalidValue));
}
#endif // TEST_LAUNCH == 0

// Regression test for https://github.com/NVIDIA/cub/issues/489: integer rounding errors lead to incorrect bin detection
C2H_TEST("DeviceHistogram::HistogramEven bin calculation regression", "[histogram_even][device]")
{
  constexpr int num_levels   = 8;
  const auto h_histogram_ref = c2h::host_vector<int>{1, 5, 0, 2, 1, 0, 0};
  const auto d_samples       = c2h::device_vector<int>{2, 6, 7, 2, 3, 0, 2, 2, 6, 999};
  constexpr int lower_level  = 0;
  constexpr int upper_level  = 12;

  auto d_histogram = c2h::device_vector<int>(h_histogram_ref.size());
  histogram_even(
    thrust::raw_pointer_cast(d_samples.data()),
    thrust::raw_pointer_cast(d_histogram.data()),
    num_levels,
    lower_level,
    upper_level,
    static_cast<int>(d_samples.size()));
  CHECK(h_histogram_ref == d_histogram);
}
