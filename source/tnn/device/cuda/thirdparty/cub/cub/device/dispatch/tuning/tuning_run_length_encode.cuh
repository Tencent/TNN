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

#include <cub/agent/agent_reduce_by_key.cuh>
#include <cub/agent/agent_rle.cuh>
#include <cub/agent/single_pass_scan_operators.cuh>
#include <cub/block/block_load.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_store.cuh>
#include <cub/util_device.cuh>
#include <cub/util_type.cuh>

CUB_NAMESPACE_BEGIN

namespace detail
{

namespace rle
{

enum class primitive_key
{
  no,
  yes
};
enum class primitive_length
{
  no,
  yes
};
enum class key_size
{
  _1,
  _2,
  _4,
  _8,
  _16,
  unknown
};
enum class length_size
{
  _4,
  unknown
};

template <class T>
constexpr primitive_key is_primitive_key()
{
  return Traits<T>::PRIMITIVE ? primitive_key::yes : primitive_key::no;
}

template <class T>
constexpr primitive_length is_primitive_length()
{
  return Traits<T>::PRIMITIVE ? primitive_length::yes : primitive_length::no;
}

template <class KeyT>
constexpr key_size classify_key_size()
{
  return sizeof(KeyT) == 1 ? key_size::_1
       : sizeof(KeyT) == 2 ? key_size::_2
       : sizeof(KeyT) == 4 ? key_size::_4
       : sizeof(KeyT) == 8 ? key_size::_8
       : sizeof(KeyT) == 16
         ? key_size::_16
         : key_size::unknown;
}

template <class LengthT>
constexpr length_size classify_length_size()
{
  return sizeof(LengthT) == 4 ? length_size::_4 : length_size::unknown;
}

namespace encode
{

template <class LengthT,
          class KeyT,
          primitive_length PrimitiveLength = is_primitive_length<LengthT>(),
          primitive_key PrimitiveKey       = is_primitive_key<KeyT>(),
          length_size LengthSize           = classify_length_size<LengthT>(),
          key_size KeySize                 = classify_key_size<KeyT>()>
struct sm90_tuning
{
  static constexpr int max_input_bytes      = CUB_MAX(sizeof(KeyT), sizeof(LengthT));
  static constexpr int combined_input_bytes = sizeof(KeyT) + sizeof(LengthT);

  static constexpr int threads = 128;

  static constexpr int nominal_4b_items_per_thread = 6;

  static constexpr int items =
    (max_input_bytes <= 8)
      ? 6
      : CUB_MIN(nominal_4b_items_per_thread,
                CUB_MAX(1, ((nominal_4b_items_per_thread * 8) + combined_input_bytes - 1) / combined_input_bytes));

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;

  using delay_constructor = detail::default_reduce_by_key_delay_constructor_t<LengthT, int>;
};

template <class LengthT, class KeyT>
struct sm90_tuning<LengthT, KeyT, primitive_length::yes, primitive_key::yes, length_size::_4, key_size::_1>
{
  static constexpr int threads = 256;

  static constexpr int items = 13;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;

  using delay_constructor = detail::no_delay_constructor_t<620>;
};

template <class LengthT, class KeyT>
struct sm90_tuning<LengthT, KeyT, primitive_length::yes, primitive_key::yes, length_size::_4, key_size::_2>
{
  static constexpr int threads = 128;

  static constexpr int items = 22;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;

  using delay_constructor = detail::no_delay_constructor_t<775>;
};

template <class LengthT, class KeyT>
struct sm90_tuning<LengthT, KeyT, primitive_length::yes, primitive_key::yes, length_size::_4, key_size::_4>
{
  static constexpr int threads = 192;

  static constexpr int items = 14;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;

  using delay_constructor = detail::fixed_delay_constructor_t<284, 480>;
};

template <class LengthT, class KeyT>
struct sm90_tuning<LengthT, KeyT, primitive_length::yes, primitive_key::yes, length_size::_4, key_size::_8>
{
  static constexpr int threads = 128;

  static constexpr int items = 19;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;

  using delay_constructor = detail::no_delay_constructor_t<515>;
};

#if CUB_IS_INT128_ENABLED
template <class LengthT>
struct sm90_tuning<LengthT, __int128_t, primitive_length::yes, primitive_key::no, length_size::_4, key_size::_16>
{
  static constexpr int threads = 128;

  static constexpr int items = 11;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;

  using delay_constructor = detail::fixed_delay_constructor_t<428, 930>;
};

template <class LengthT>
struct sm90_tuning<LengthT, __uint128_t, primitive_length::yes, primitive_key::no, length_size::_4, key_size::_16>
{
  static constexpr int threads = 128;

  static constexpr int items = 11;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;

  using delay_constructor = detail::fixed_delay_constructor_t<428, 930>;
};
#endif

template <class LengthT,
          class KeyT,
          primitive_length PrimitiveLength = is_primitive_length<LengthT>(),
          primitive_key PrimitiveKey       = is_primitive_key<KeyT>(),
          length_size LengthSize           = classify_length_size<LengthT>(),
          key_size KeySize                 = classify_key_size<KeyT>()>
struct sm80_tuning
{
  static constexpr int max_input_bytes      = CUB_MAX(sizeof(KeyT), sizeof(LengthT));
  static constexpr int combined_input_bytes = sizeof(KeyT) + sizeof(LengthT);

  static constexpr int threads = 128;

  static constexpr int nominal_4b_items_per_thread = 6;

  static constexpr int items =
    (max_input_bytes <= 8)
      ? 6
      : CUB_MIN(nominal_4b_items_per_thread,
                CUB_MAX(1, ((nominal_4b_items_per_thread * 8) + combined_input_bytes - 1) / combined_input_bytes));

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;

  using delay_constructor = detail::default_reduce_by_key_delay_constructor_t<LengthT, int>;
};

template <class LengthT, class KeyT>
struct sm80_tuning<LengthT, KeyT, primitive_length::yes, primitive_key::yes, length_size::_4, key_size::_1>
{
  static constexpr int threads = 256;

  static constexpr int items = 14;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;

  using delay_constructor = detail::no_delay_constructor_t<640>;
};

template <class LengthT, class KeyT>
struct sm80_tuning<LengthT, KeyT, primitive_length::yes, primitive_key::yes, length_size::_4, key_size::_2>
{
  static constexpr int threads = 256;

  static constexpr int items = 13;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;

  using delay_constructor = detail::no_delay_constructor_t<900>;
};

template <class LengthT, class KeyT>
struct sm80_tuning<LengthT, KeyT, primitive_length::yes, primitive_key::yes, length_size::_4, key_size::_4>
{
  static constexpr int threads = 256;

  static constexpr int items = 13;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;

  using delay_constructor = detail::no_delay_constructor_t<1080>;
};

template <class LengthT, class KeyT>
struct sm80_tuning<LengthT, KeyT, primitive_length::yes, primitive_key::yes, length_size::_4, key_size::_8>
{
  static constexpr int threads = 224;

  static constexpr int items = 9;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;

  using delay_constructor = detail::no_delay_constructor_t<1075>;
};

#if CUB_IS_INT128_ENABLED
template <class LengthT>
struct sm80_tuning<LengthT, __int128_t, primitive_length::yes, primitive_key::no, length_size::_4, key_size::_16>
{
  static constexpr int threads = 128;

  static constexpr int items = 7;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;

  using delay_constructor = detail::no_delay_constructor_t<630>;
};

template <class LengthT>
struct sm80_tuning<LengthT, __uint128_t, primitive_length::yes, primitive_key::no, length_size::_4, key_size::_16>
{
  static constexpr int threads = 128;

  static constexpr int items = 7;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;

  using delay_constructor = detail::no_delay_constructor_t<630>;
};
#endif

} // namespace encode

namespace non_trivial_runs
{

template <class LengthT,
          class KeyT,
          primitive_length PrimitiveLength = is_primitive_length<LengthT>(),
          primitive_key PrimitiveKey       = is_primitive_key<KeyT>(),
          length_size LengthSize           = classify_length_size<LengthT>(),
          key_size KeySize                 = classify_key_size<KeyT>()>
struct sm90_tuning
{
  static constexpr int threads = 96;

  static constexpr int nominal_4b_items_per_thread = 15;

  static constexpr int items =
    CUB_MIN(nominal_4b_items_per_thread, CUB_MAX(1, (nominal_4b_items_per_thread * 4 / sizeof(KeyT))));

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;

  static constexpr bool store_with_time_slicing = true;

  using delay_constructor = detail::default_reduce_by_key_delay_constructor_t<LengthT, int>;
};

template <class LengthT, class KeyT>
struct sm90_tuning<LengthT, KeyT, primitive_length::yes, primitive_key::yes, length_size::_4, key_size::_1>
{
  static constexpr int threads = 256;

  static constexpr int items = 18;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;

  static constexpr bool store_with_time_slicing = false;

  using delay_constructor = detail::no_delay_constructor_t<385>;
};

template <class LengthT, class KeyT>
struct sm90_tuning<LengthT, KeyT, primitive_length::yes, primitive_key::yes, length_size::_4, key_size::_2>
{
  static constexpr int threads = 224;

  static constexpr int items = 20;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;

  static constexpr bool store_with_time_slicing = false;

  using delay_constructor = detail::no_delay_constructor_t<675>;
};

template <class LengthT, class KeyT>
struct sm90_tuning<LengthT, KeyT, primitive_length::yes, primitive_key::yes, length_size::_4, key_size::_4>
{
  static constexpr int threads = 256;

  static constexpr int items = 18;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;

  static constexpr bool store_with_time_slicing = false;

  using delay_constructor = detail::no_delay_constructor_t<695>;
};

template <class LengthT, class KeyT>
struct sm90_tuning<LengthT, KeyT, primitive_length::yes, primitive_key::yes, length_size::_4, key_size::_8>
{
  static constexpr int threads = 224;

  static constexpr int items = 14;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;

  static constexpr bool store_with_time_slicing = false;

  using delay_constructor = detail::no_delay_constructor_t<840>;
};

#if CUB_IS_INT128_ENABLED
template <class LengthT>
struct sm90_tuning<LengthT, __int128_t, primitive_length::yes, primitive_key::no, length_size::_4, key_size::_16>
{
  static constexpr int threads = 288;

  static constexpr int items = 9;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;

  static constexpr bool store_with_time_slicing = false;

  using delay_constructor = detail::fixed_delay_constructor_t<484, 1150>;
};

template <class LengthT>
struct sm90_tuning<LengthT, __uint128_t, primitive_length::yes, primitive_key::no, length_size::_4, key_size::_16>
{
  static constexpr int threads = 288;

  static constexpr int items = 9;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;

  static constexpr bool store_with_time_slicing = false;

  using delay_constructor = detail::fixed_delay_constructor_t<484, 1150>;
};
#endif

template <class LengthT,
          class KeyT,
          primitive_length PrimitiveLength = is_primitive_length<LengthT>(),
          primitive_key PrimitiveKey       = is_primitive_key<KeyT>(),
          length_size LengthSize           = classify_length_size<LengthT>(),
          key_size KeySize                 = classify_key_size<KeyT>()>
struct sm80_tuning
{
  static constexpr int threads = 96;

  static constexpr int nominal_4b_items_per_thread = 15;

  static constexpr int items =
    CUB_MIN(nominal_4b_items_per_thread, CUB_MAX(1, (nominal_4b_items_per_thread * 4 / sizeof(KeyT))));

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;

  static constexpr bool store_with_time_slicing = true;

  using delay_constructor = detail::default_reduce_by_key_delay_constructor_t<LengthT, int>;
};

template <class LengthT, class KeyT>
struct sm80_tuning<LengthT, KeyT, primitive_length::yes, primitive_key::yes, length_size::_4, key_size::_1>
{
  static constexpr int threads = 192;

  static constexpr int items = 20;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;

  static constexpr bool store_with_time_slicing = false;

  using delay_constructor = detail::no_delay_constructor_t<630>;
};

template <class LengthT, class KeyT>
struct sm80_tuning<LengthT, KeyT, primitive_length::yes, primitive_key::yes, length_size::_4, key_size::_2>
{
  static constexpr int threads = 192;

  static constexpr int items = 20;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;

  static constexpr bool store_with_time_slicing = false;

  using delay_constructor = detail::no_delay_constructor_t<1015>;
};

template <class LengthT, class KeyT>
struct sm80_tuning<LengthT, KeyT, primitive_length::yes, primitive_key::yes, length_size::_4, key_size::_4>
{
  static constexpr int threads = 224;

  static constexpr int items = 15;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;

  static constexpr bool store_with_time_slicing = false;

  using delay_constructor = detail::no_delay_constructor_t<915>;
};

template <class LengthT, class KeyT>
struct sm80_tuning<LengthT, KeyT, primitive_length::yes, primitive_key::yes, length_size::_4, key_size::_8>
{
  static constexpr int threads = 256;

  static constexpr int items = 13;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;

  static constexpr bool store_with_time_slicing = false;

  using delay_constructor = detail::no_delay_constructor_t<1065>;
};

#if CUB_IS_INT128_ENABLED
template <class LengthT>
struct sm80_tuning<LengthT, __int128_t, primitive_length::yes, primitive_key::no, length_size::_4, key_size::_16>
{
  static constexpr int threads = 192;

  static constexpr int items = 13;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;

  static constexpr bool store_with_time_slicing = false;

  using delay_constructor = detail::no_delay_constructor_t<1050>;
};

template <class LengthT>
struct sm80_tuning<LengthT, __uint128_t, primitive_length::yes, primitive_key::no, length_size::_4, key_size::_16>
{
  static constexpr int threads = 192;

  static constexpr int items = 13;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;

  static constexpr bool store_with_time_slicing = false;

  using delay_constructor = detail::no_delay_constructor_t<1050>;
};
#endif

} // namespace non_trivial_runs

} // namespace rle

template <class LengthT, class KeyT>
struct device_run_length_encode_policy_hub
{
  static constexpr int MAX_INPUT_BYTES      = CUB_MAX(sizeof(KeyT), sizeof(LengthT));
  static constexpr int COMBINED_INPUT_BYTES = sizeof(KeyT) + sizeof(LengthT);

  struct DefaultTuning
  {
    static constexpr int NOMINAL_4B_ITEMS_PER_THREAD = 6;
    static constexpr int ITEMS_PER_THREAD =
      (MAX_INPUT_BYTES <= 8)
        ? 6
        : CUB_MIN(NOMINAL_4B_ITEMS_PER_THREAD,
                  CUB_MAX(1, ((NOMINAL_4B_ITEMS_PER_THREAD * 8) + COMBINED_INPUT_BYTES - 1) / COMBINED_INPUT_BYTES));

    using ReduceByKeyPolicyT =
      AgentReduceByKeyPolicy<128,
                             ITEMS_PER_THREAD,
                             BLOCK_LOAD_DIRECT,
                             LOAD_LDG,
                             BLOCK_SCAN_WARP_SCANS,
                             detail::default_reduce_by_key_delay_constructor_t<LengthT, int>>;
  };

  /// SM35
  struct Policy350
      : DefaultTuning
      , ChainedPolicy<350, Policy350, Policy350>
  {};

  /// SM80
  struct Policy800 : ChainedPolicy<800, Policy800, Policy350>
  {
    using tuning = detail::rle::encode::sm80_tuning<LengthT, KeyT>;

    using ReduceByKeyPolicyT =
      AgentReduceByKeyPolicy<tuning::threads,
                             tuning::items,
                             tuning::load_algorithm,
                             LOAD_DEFAULT,
                             BLOCK_SCAN_WARP_SCANS,
                             typename tuning::delay_constructor>;
  };

  // SM86
  struct Policy860
      : DefaultTuning
      , ChainedPolicy<860, Policy860, Policy800>
  {};

  /// SM90
  struct Policy900 : ChainedPolicy<900, Policy900, Policy860>
  {
    using tuning = detail::rle::encode::sm90_tuning<LengthT, KeyT>;

    using ReduceByKeyPolicyT =
      AgentReduceByKeyPolicy<tuning::threads,
                             tuning::items,
                             tuning::load_algorithm,
                             LOAD_DEFAULT,
                             BLOCK_SCAN_WARP_SCANS,
                             typename tuning::delay_constructor>;
  };

  using MaxPolicy = Policy900;
};

template <class LengthT, class KeyT>
struct device_non_trivial_runs_policy_hub
{
  struct DefaultTuning
  {
    enum
    {
      NOMINAL_4B_ITEMS_PER_THREAD = 15,

      ITEMS_PER_THREAD =
        CUB_MIN(NOMINAL_4B_ITEMS_PER_THREAD, CUB_MAX(1, (NOMINAL_4B_ITEMS_PER_THREAD * 4 / sizeof(KeyT)))),
    };

    using RleSweepPolicyT =
      AgentRlePolicy<96,
                     ITEMS_PER_THREAD,
                     BLOCK_LOAD_DIRECT,
                     LOAD_LDG,
                     true,
                     BLOCK_SCAN_WARP_SCANS,
                     detail::default_reduce_by_key_delay_constructor_t<int, int>>;
  };

  /// SM35
  struct Policy350
      : DefaultTuning
      , ChainedPolicy<350, Policy350, Policy350>
  {};

  // SM80
  struct Policy800 : ChainedPolicy<800, Policy800, Policy350>
  {
    using tuning = detail::rle::non_trivial_runs::sm80_tuning<LengthT, KeyT>;

    using RleSweepPolicyT =
      AgentRlePolicy<tuning::threads,
                     tuning::items,
                     tuning::load_algorithm,
                     LOAD_DEFAULT,
                     tuning::store_with_time_slicing,
                     BLOCK_SCAN_WARP_SCANS,
                     typename tuning::delay_constructor>;
  };

  // SM86
  struct Policy860
      : DefaultTuning
      , ChainedPolicy<860, Policy860, Policy800>
  {};

  // SM90
  struct Policy900 : ChainedPolicy<900, Policy900, Policy860>
  {
    using tuning = detail::rle::non_trivial_runs::sm90_tuning<LengthT, KeyT>;

    using RleSweepPolicyT =
      AgentRlePolicy<tuning::threads,
                     tuning::items,
                     tuning::load_algorithm,
                     LOAD_DEFAULT,
                     tuning::store_with_time_slicing,
                     BLOCK_SCAN_WARP_SCANS,
                     typename tuning::delay_constructor>;
  };

  using MaxPolicy = Policy900;
};

} // namespace detail

CUB_NAMESPACE_END
