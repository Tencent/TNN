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

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/agent/agent_histogram.cuh>
#include <cub/block/block_load.cuh>
#include <cub/util_device.cuh>
#include <cub/util_type.cuh>

CUB_NAMESPACE_BEGIN

namespace detail
{

namespace histogram
{

enum class primitive_sample
{
  no,
  yes
};

enum class sample_size
{
  _1,
  _2,
  unknown
};

enum class counter_size
{
  _4,
  unknown
};

template <class T>
constexpr primitive_sample is_primitive_sample()
{
  return Traits<T>::PRIMITIVE ? primitive_sample::yes : primitive_sample::no;
}

template <class CounterT>
constexpr counter_size classify_counter_size()
{
  return sizeof(CounterT) == 4 ? counter_size::_4 : counter_size::unknown;
}

template <class SampleT>
constexpr sample_size classify_sample_size()
{
  return sizeof(SampleT) == 1 ? sample_size::_1 : sizeof(SampleT) == 2 ? sample_size::_2 : sample_size::unknown;
}

template <class SampleT>
constexpr int v_scale()
{
  return (sizeof(SampleT) + sizeof(int) - 1) / sizeof(int);
}

template <class SampleT, int NumActiveChannels, int NominalItemsPerThread>
constexpr int t_scale()
{
  return CUB_MAX((NominalItemsPerThread / NumActiveChannels / v_scale<SampleT>()), 1);
}

template <class SampleT,
          int NumChannels,
          int NumActiveChannels,
          counter_size CounterSize,
          primitive_sample PrimitiveSample = is_primitive_sample<SampleT>(),
          sample_size SampleSize           = classify_sample_size<SampleT>()>
struct sm90_tuning
{
  static constexpr int threads = 384;
  static constexpr int items   = t_scale<SampleT, NumActiveChannels, 16>();

  static constexpr CacheLoadModifier load_modifier               = LOAD_LDG;
  static constexpr BlockHistogramMemoryPreference mem_preference = SMEM;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;

  static constexpr bool rle_compress  = true;
  static constexpr bool work_stealing = false;
};

template <class SampleT>
struct sm90_tuning<SampleT, 1, 1, counter_size::_4, primitive_sample::yes, sample_size::_1>
{
  static constexpr int threads = 768;
  static constexpr int items   = 12;

  static constexpr CacheLoadModifier load_modifier               = LOAD_LDG;
  static constexpr BlockHistogramMemoryPreference mem_preference = SMEM;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;

  static constexpr bool rle_compress  = false;
  static constexpr bool work_stealing = false;
};

template <class SampleT>
struct sm90_tuning<SampleT, 1, 1, counter_size::_4, primitive_sample::yes, sample_size::_2>
{
  static constexpr int threads = 960;
  static constexpr int items   = 10;

  static constexpr CacheLoadModifier load_modifier               = LOAD_DEFAULT;
  static constexpr BlockHistogramMemoryPreference mem_preference = SMEM;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;

  static constexpr bool rle_compress  = true;
  static constexpr bool work_stealing = false;
};

} // namespace histogram

template <class SampleT, class CounterT, int NumChannels, int NumActiveChannels>
struct device_histogram_policy_hub
{
  template <int NOMINAL_ITEMS_PER_THREAD>
  struct TScale
  {
    enum
    {
      V_SCALE = (sizeof(SampleT) + sizeof(int) - 1) / sizeof(int),
      VALUE   = CUB_MAX((NOMINAL_ITEMS_PER_THREAD / NumActiveChannels / V_SCALE), 1)
    };
  };

  /// SM35
  struct Policy350 : ChainedPolicy<350, Policy350, Policy350>
  {
    // TODO This might be worth it to separate usual histogram and the multi one
    using AgentHistogramPolicyT =
      AgentHistogramPolicy<128, TScale<8>::VALUE, BLOCK_LOAD_DIRECT, LOAD_LDG, true, BLEND, true>;
  };

  /// SM50
  struct Policy500 : ChainedPolicy<500, Policy500, Policy350>
  {
    // TODO This might be worth it to separate usual histogram and the multi one
    using AgentHistogramPolicyT =
      AgentHistogramPolicy<384, TScale<16>::VALUE, cub::BLOCK_LOAD_DIRECT, LOAD_LDG, true, SMEM, false>;
  };

  /// SM900
  struct Policy900 : ChainedPolicy<900, Policy900, Policy500>
  {
    using tuning = detail::histogram::
      sm90_tuning<SampleT, NumChannels, NumActiveChannels, histogram::classify_counter_size<CounterT>()>;

    using AgentHistogramPolicyT =
      AgentHistogramPolicy<tuning::threads,
                           tuning::items,
                           tuning::load_algorithm,
                           tuning::load_modifier,
                           tuning::rle_compress,
                           tuning::mem_preference,
                           tuning::work_stealing>;
  };

  using MaxPolicy = Policy900;
};

} // namespace detail

CUB_NAMESPACE_END
