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
#include <cub/agent/single_pass_scan_operators.cuh>
#include <cub/block/block_load.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_store.cuh>
#include <cub/util_device.cuh>
#include <cub/util_type.cuh>

CUB_NAMESPACE_BEGIN

namespace detail
{

namespace reduce_by_key
{

enum class primitive_key
{
  no,
  yes
};
enum class primitive_accum
{
  no,
  yes
};
enum class primitive_op
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
enum class accum_size
{
  _1,
  _2,
  _4,
  _8,
  _16,
  unknown
};

template <class T>
constexpr primitive_key is_primitive_key()
{
  return Traits<T>::PRIMITIVE ? primitive_key::yes : primitive_key::no;
}

template <class T>
constexpr primitive_accum is_primitive_accum()
{
  return Traits<T>::PRIMITIVE ? primitive_accum::yes : primitive_accum::no;
}

template <class ReductionOpT>
constexpr primitive_op is_primitive_op()
{
  return basic_binary_op_t<ReductionOpT>::value ? primitive_op::yes : primitive_op::no;
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

template <class AccumT>
constexpr accum_size classify_accum_size()
{
  return sizeof(AccumT) == 1 ? accum_size::_1
       : sizeof(AccumT) == 2 ? accum_size::_2
       : sizeof(AccumT) == 4 ? accum_size::_4
       : sizeof(AccumT) == 8 ? accum_size::_8
       : sizeof(AccumT) == 16
         ? accum_size::_16
         : accum_size::unknown;
}

template <class KeyT,
          class AccumT,
          primitive_op PrimitiveOp,
          primitive_key PrimitiveKey     = is_primitive_key<KeyT>(),
          primitive_accum PrimitiveAccum = is_primitive_accum<AccumT>(),
          key_size KeySize               = classify_key_size<KeyT>(),
          accum_size AccumSize           = classify_accum_size<AccumT>()>
struct sm90_tuning
{
  static constexpr int max_input_bytes      = CUB_MAX(sizeof(KeyT), sizeof(AccumT));
  static constexpr int combined_input_bytes = sizeof(KeyT) + sizeof(AccumT);

  static constexpr int threads = 128;

  static constexpr int nominal_4b_items_per_thread = 6;
  static constexpr int items =
    (max_input_bytes <= 8)
      ? 6
      : CUB_MIN(nominal_4b_items_per_thread,
                CUB_MAX(1, ((nominal_4b_items_per_thread * 8) + combined_input_bytes - 1) / combined_input_bytes));

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;

  using delay_constructor = detail::default_reduce_by_key_delay_constructor_t<AccumT, int>;
};

// 8-bit key
template <class KeyT, class AccumT>
struct sm90_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_1, accum_size::_1>
{
  static constexpr int threads = 256;

  static constexpr int items = 13;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;

  using delay_constructor = detail::no_delay_constructor_t<720>;
};

template <class KeyT, class AccumT>
struct sm90_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_1, accum_size::_2>
{
  static constexpr int threads = 320;

  static constexpr int items = 23;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;

  using delay_constructor = detail::no_delay_constructor_t<865>;
};

template <class KeyT, class AccumT>
struct sm90_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_1, accum_size::_4>
{
  static constexpr int threads = 192;

  static constexpr int items = 14;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;

  using delay_constructor = detail::no_delay_constructor_t<735>;
};

template <class KeyT, class AccumT>
struct sm90_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_1, accum_size::_8>
{
  static constexpr int threads = 128;

  static constexpr int items = 13;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;

  using delay_constructor = detail::no_delay_constructor_t<580>;
};

template <class KeyT, class AccumT>
struct sm90_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::no, key_size::_1, accum_size::_16>
{
  static constexpr int threads = 128;

  static constexpr int items = 11;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;

  using delay_constructor = detail::no_delay_constructor_t<1100>;
};

// 16-bit key
template <class KeyT, class AccumT>
struct sm90_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_2, accum_size::_1>
{
  static constexpr int threads = 128;

  static constexpr int items = 23;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;

  using delay_constructor = detail::no_delay_constructor_t<985>;
};

template <class KeyT, class AccumT>
struct sm90_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_2, accum_size::_2>
{
  static constexpr int threads = 256;

  static constexpr int items = 11;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;

  using delay_constructor = detail::fixed_delay_constructor_t<276, 650>;
};

template <class KeyT, class AccumT>
struct sm90_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_2, accum_size::_4>
{
  static constexpr int threads = 256;

  static constexpr int items = 14;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;

  using delay_constructor = detail::fixed_delay_constructor_t<240, 765>;
};

template <class KeyT, class AccumT>
struct sm90_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_2, accum_size::_8>
{
  static constexpr int threads = 128;

  static constexpr int items = 19;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;

  using delay_constructor = detail::no_delay_constructor_t<1190>;
};

template <class KeyT, class AccumT>
struct sm90_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::no, key_size::_2, accum_size::_16>
{
  static constexpr int threads = 128;

  static constexpr int items = 11;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;

  using delay_constructor = detail::no_delay_constructor_t<1175>;
};

// 32-bit key
template <class KeyT, class AccumT>
struct sm90_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_4, accum_size::_1>
{
  static constexpr int threads = 256;

  static constexpr int items = 13;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;

  using delay_constructor = detail::fixed_delay_constructor_t<404, 645>;
};

template <class KeyT, class AccumT>
struct sm90_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_4, accum_size::_2>
{
  static constexpr int threads = 256;

  static constexpr int items = 18;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;

  using delay_constructor = detail::no_delay_constructor_t<1160>;
};

template <class KeyT, class AccumT>
struct sm90_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_4, accum_size::_4>
{
  static constexpr int threads = 256;

  static constexpr int items = 18;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;

  using delay_constructor = detail::no_delay_constructor_t<1170>;
};

template <class KeyT, class AccumT>
struct sm90_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_4, accum_size::_8>
{
  static constexpr int threads = 128;

  static constexpr int items = 13;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;

  using delay_constructor = detail::no_delay_constructor_t<1055>;
};

template <class KeyT, class AccumT>
struct sm90_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::no, key_size::_4, accum_size::_16>
{
  static constexpr int threads = 128;

  static constexpr int items = 11;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;

  using delay_constructor = detail::no_delay_constructor_t<1195>;
};

// 64-bit key
template <class KeyT, class AccumT>
struct sm90_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_8, accum_size::_1>
{
  static constexpr int threads = 256;

  static constexpr int items = 10;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;

  using delay_constructor = detail::no_delay_constructor_t<1170>;
};

template <class KeyT, class AccumT>
struct sm90_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_8, accum_size::_2>
{
  static constexpr int threads = 256;

  static constexpr int items = 9;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;

  using delay_constructor = detail::fixed_delay_constructor_t<236, 1030>;
};

template <class KeyT, class AccumT>
struct sm90_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_8, accum_size::_4>
{
  static constexpr int threads = 128;

  static constexpr int items = 13;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;

  using delay_constructor = detail::fixed_delay_constructor_t<152, 560>;
};

template <class KeyT, class AccumT>
struct sm90_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_8, accum_size::_8>
{
  static constexpr int threads = 128;

  static constexpr int items = 23;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;

  using delay_constructor = detail::no_delay_constructor_t<1030>;
};

template <class KeyT, class AccumT>
struct sm90_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::no, key_size::_8, accum_size::_16>
{
  static constexpr int threads = 128;

  static constexpr int items = 11;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;

  using delay_constructor = detail::no_delay_constructor_t<1125>;
};

// 128-bit key
template <class KeyT, class AccumT>
struct sm90_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::no, primitive_accum::yes, key_size::_16, accum_size::_1>
{
  static constexpr int threads = 128;

  static constexpr int items = 11;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;

  using delay_constructor = detail::no_delay_constructor_t<1080>;
};

template <class KeyT, class AccumT>
struct sm90_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::no, primitive_accum::yes, key_size::_16, accum_size::_2>
{
  static constexpr int threads = 128;

  static constexpr int items = 11;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;

  using delay_constructor = detail::fixed_delay_constructor_t<320, 1005>;
};

template <class KeyT, class AccumT>
struct sm90_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::no, primitive_accum::yes, key_size::_16, accum_size::_4>
{
  static constexpr int threads = 128;

  static constexpr int items = 11;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;

  using delay_constructor = detail::fixed_delay_constructor_t<232, 1100>;
};

template <class KeyT, class AccumT>
struct sm90_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::no, primitive_accum::yes, key_size::_16, accum_size::_8>
{
  static constexpr int threads = 128;

  static constexpr int items = 11;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;

  using delay_constructor = detail::no_delay_constructor_t<1195>;
};

template <class KeyT, class AccumT>
struct sm90_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::no, primitive_accum::no, key_size::_16, accum_size::_16>
{
  static constexpr int threads = 128;

  static constexpr int items = 11;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;

  using delay_constructor = detail::no_delay_constructor_t<1150>;
};

template <class KeyT,
          class AccumT,
          primitive_op PrimitiveOp,
          primitive_key PrimitiveKey     = is_primitive_key<KeyT>(),
          primitive_accum PrimitiveAccum = is_primitive_accum<AccumT>(),
          key_size KeySize               = classify_key_size<KeyT>(),
          accum_size AccumSize           = classify_accum_size<AccumT>()>
struct sm80_tuning
{
  static constexpr int max_input_bytes      = CUB_MAX(sizeof(KeyT), sizeof(AccumT));
  static constexpr int combined_input_bytes = sizeof(KeyT) + sizeof(AccumT);

  static constexpr int threads = 128;

  static constexpr int nominal_4b_items_per_thread = 6;
  static constexpr int items =
    (max_input_bytes <= 8)
      ? 6
      : CUB_MIN(nominal_4b_items_per_thread,
                CUB_MAX(1, ((nominal_4b_items_per_thread * 8) + combined_input_bytes - 1) / combined_input_bytes));

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;

  using delay_constructor = detail::default_reduce_by_key_delay_constructor_t<AccumT, int>;
};

// 8-bit key
template <class KeyT, class AccumT>
struct sm80_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_1, accum_size::_1>
{
  static constexpr int threads = 256;

  static constexpr int items = 13;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;

  using delay_constructor = detail::no_delay_constructor_t<975>;
};

template <class KeyT, class AccumT>
struct sm80_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_1, accum_size::_2>
{
  static constexpr int threads = 224;

  static constexpr int items = 12;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;

  using delay_constructor = detail::no_delay_constructor_t<840>;
};

template <class KeyT, class AccumT>
struct sm80_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_1, accum_size::_4>
{
  static constexpr int threads = 256;

  static constexpr int items = 15;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;

  using delay_constructor = detail::no_delay_constructor_t<760>;
};

template <class KeyT, class AccumT>
struct sm80_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_1, accum_size::_8>
{
  static constexpr int threads = 224;

  static constexpr int items = 7;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;

  using delay_constructor = detail::no_delay_constructor_t<1070>;
};

template <class KeyT, class AccumT>
struct sm80_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::no, key_size::_1, accum_size::_16>
{
  static constexpr int threads = 128;

  static constexpr int items = 9;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;

  using delay_constructor = detail::no_delay_constructor_t<1175>;
};

// 16-bit key
template <class KeyT, class AccumT>
struct sm80_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_2, accum_size::_1>
{
  static constexpr int threads = 256;

  static constexpr int items = 11;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;

  using delay_constructor = detail::no_delay_constructor_t<620>;
};

template <class KeyT, class AccumT>
struct sm80_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_2, accum_size::_2>
{
  static constexpr int threads = 224;

  static constexpr int items = 14;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;

  using delay_constructor = detail::no_delay_constructor_t<640>;
};

template <class KeyT, class AccumT>
struct sm80_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_2, accum_size::_4>
{
  static constexpr int threads = 256;

  static constexpr int items = 14;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;

  using delay_constructor = detail::no_delay_constructor_t<905>;
};

template <class KeyT, class AccumT>
struct sm80_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_2, accum_size::_8>
{
  static constexpr int threads = 224;

  static constexpr int items = 9;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;

  using delay_constructor = detail::no_delay_constructor_t<810>;
};

template <class KeyT, class AccumT>
struct sm80_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::no, key_size::_2, accum_size::_16>
{
  static constexpr int threads = 160;

  static constexpr int items = 9;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;

  using delay_constructor = detail::no_delay_constructor_t<1115>;
};

// 32-bit key
template <class KeyT, class AccumT>
struct sm80_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_4, accum_size::_1>
{
  static constexpr int threads = 288;

  static constexpr int items = 11;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;

  using delay_constructor = detail::no_delay_constructor_t<1110>;
};

template <class KeyT, class AccumT>
struct sm80_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_4, accum_size::_2>
{
  static constexpr int threads = 192;

  static constexpr int items = 15;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;

  using delay_constructor = detail::no_delay_constructor_t<1200>;
};

template <class KeyT, class AccumT>
struct sm80_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_4, accum_size::_4>
{
  static constexpr int threads = 256;

  static constexpr int items = 15;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;

  using delay_constructor = detail::no_delay_constructor_t<1110>;
};

template <class KeyT, class AccumT>
struct sm80_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_4, accum_size::_8>
{
  static constexpr int threads = 224;

  static constexpr int items = 9;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;

  using delay_constructor = detail::no_delay_constructor_t<1165>;
};

template <class KeyT, class AccumT>
struct sm80_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::no, key_size::_4, accum_size::_16>
{
  static constexpr int threads = 160;

  static constexpr int items = 9;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;

  using delay_constructor = detail::no_delay_constructor_t<1100>;
};

// 64-bit key
template <class KeyT, class AccumT>
struct sm80_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_8, accum_size::_1>
{
  static constexpr int threads = 192;

  static constexpr int items = 10;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;

  using delay_constructor = detail::no_delay_constructor_t<1175>;
};

template <class KeyT, class AccumT>
struct sm80_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_8, accum_size::_2>
{
  static constexpr int threads = 224;

  static constexpr int items = 7;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;

  using delay_constructor = detail::no_delay_constructor_t<1075>;
};

template <class KeyT, class AccumT>
struct sm80_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_8, accum_size::_4>
{
  static constexpr int threads = 384;

  static constexpr int items = 7;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;

  using delay_constructor = detail::no_delay_constructor_t<1040>;
};

template <class KeyT, class AccumT>
struct sm80_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_8, accum_size::_8>
{
  static constexpr int threads = 128;

  static constexpr int items = 14;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;

  using delay_constructor = detail::no_delay_constructor_t<1080>;
};

template <class KeyT, class AccumT>
struct sm80_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::no, key_size::_8, accum_size::_16>
{
  static constexpr int threads = 128;

  static constexpr int items = 11;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;

  using delay_constructor = detail::no_delay_constructor_t<430>;
};

// 128-bit key
template <class KeyT, class AccumT>
struct sm80_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::no, primitive_accum::yes, key_size::_16, accum_size::_1>
{
  static constexpr int threads = 192;

  static constexpr int items = 7;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;

  using delay_constructor = detail::no_delay_constructor_t<1105>;
};

template <class KeyT, class AccumT>
struct sm80_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::no, primitive_accum::yes, key_size::_16, accum_size::_2>
{
  static constexpr int threads = 192;

  static constexpr int items = 7;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;

  using delay_constructor = detail::no_delay_constructor_t<755>;
};

template <class KeyT, class AccumT>
struct sm80_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::no, primitive_accum::yes, key_size::_16, accum_size::_4>
{
  static constexpr int threads = 192;

  static constexpr int items = 7;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;

  using delay_constructor = detail::no_delay_constructor_t<535>;
};

template <class KeyT, class AccumT>
struct sm80_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::no, primitive_accum::yes, key_size::_16, accum_size::_8>
{
  static constexpr int threads = 192;

  static constexpr int items = 7;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;

  using delay_constructor = detail::no_delay_constructor_t<1035>;
};

template <class KeyT, class AccumT>
struct sm80_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::no, primitive_accum::no, key_size::_16, accum_size::_16>
{
  static constexpr int threads = 128;

  static constexpr int items = 11;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;

  using delay_constructor = detail::no_delay_constructor_t<1090>;
};
} // namespace reduce_by_key

template <class ReductionOpT, class AccumT, class KeyT>
struct device_reduce_by_key_policy_hub
{
  static constexpr int MAX_INPUT_BYTES      = CUB_MAX(sizeof(KeyT), sizeof(AccumT));
  static constexpr int COMBINED_INPUT_BYTES = sizeof(KeyT) + sizeof(AccumT);

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
                             detail::default_reduce_by_key_delay_constructor_t<AccumT, int>>;
  };

  /// SM35
  struct Policy350
      : DefaultTuning
      , ChainedPolicy<350, Policy350, Policy350>
  {};

  /// SM80
  struct Policy800 : ChainedPolicy<800, Policy800, Policy350>
  {
    using tuning =
      detail::reduce_by_key::sm80_tuning<KeyT, AccumT, detail::reduce_by_key::is_primitive_op<ReductionOpT>()>;

    using ReduceByKeyPolicyT =
      AgentReduceByKeyPolicy<tuning::threads,
                             tuning::items,
                             tuning::load_algorithm,
                             LOAD_DEFAULT,
                             BLOCK_SCAN_WARP_SCANS,
                             typename tuning::delay_constructor>;
  };

  /// SM86
  struct Policy860
      : DefaultTuning
      , ChainedPolicy<860, Policy860, Policy800>
  {};

  /// SM90
  struct Policy900 : ChainedPolicy<900, Policy900, Policy860>
  {
    using tuning =
      detail::reduce_by_key::sm90_tuning<KeyT, AccumT, detail::reduce_by_key::is_primitive_op<ReductionOpT>()>;

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

} // namespace detail

CUB_NAMESPACE_END
