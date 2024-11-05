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

#include <cub/detail/type_traits.cuh>
#include <cub/thread/thread_operators.cuh>

#include <c2h/catch2_test_helper.cuh>

/**
 * @brief Helper class template to facilitate specifying input/output type pairs along with the key
 * type for *-by-key algorithms, and an equality operator type.
 */
template <typename InputT, typename OutputT = InputT, typename KeyT = std::int32_t, typename EqualityOpT = cub::Equality>
struct type_quad
{
  using input_t  = InputT;
  using output_t = OutputT;
  using key_t    = KeyT;
  using eq_op_t  = EqualityOpT;
};

/**
 * @brief Mod2Equality (used for integral keys, making keys more likely to equal each other)
 */
struct Mod2Equality
{
  template <typename T>
  __host__ __device__ __forceinline__ T operator()(const T& a, const T& b) const
  {
    return (a % 2) == (b % 2);
  }
};

template <typename InputIt, typename OutputIt, typename InitT, typename BinaryOp>
void compute_exclusive_scan_reference(InputIt first, InputIt last, OutputIt result, InitT init, BinaryOp op)
{
  using value_t  = cub::detail::value_t<InputIt>;
  using accum_t  = ::cuda::std::__accumulator_t<BinaryOp, value_t, InitT>;
  using output_t = cub::detail::value_t<OutputIt>;
  accum_t acc    = static_cast<accum_t>(init);
  for (; first != last; ++first)
  {
    *result++ = static_cast<output_t>(acc);
    acc       = op(acc, *first);
  }
}

template <typename InputIt, typename OutputIt, typename BinaryOp, typename InitT>
void compute_inclusive_scan_reference(InputIt first, InputIt last, OutputIt result, BinaryOp op, InitT init)
{
  using value_t  = cub::detail::value_t<InputIt>;
  using accum_t  = ::cuda::std::__accumulator_t<BinaryOp, value_t, InitT>;
  using output_t = cub::detail::value_t<OutputIt>;
  accum_t acc    = static_cast<accum_t>(init);
  for (; first != last; ++first)
  {
    acc       = op(acc, *first);
    *result++ = static_cast<output_t>(acc);
  }
}

template <typename ValueInItT,
          typename KeyInItT,
          typename ValuesOutItT,
          typename ScanOpT,
          typename EqualityOpT,
          typename InitT>
void compute_exclusive_scan_by_key_reference(
  ValueInItT h_values_it,
  KeyInItT h_keys_it,
  ValuesOutItT result_out_it,
  ScanOpT scan_op,
  EqualityOpT equality_op,
  InitT init,
  std::size_t num_items)
{
  using value_t  = cub::detail::value_t<ValueInItT>;
  using accum_t  = ::cuda::std::__accumulator_t<ScanOpT, value_t, InitT>;
  using output_t = cub::detail::value_t<ValuesOutItT>;

  if (num_items > 0)
  {
    for (std::size_t i = 0; i < num_items;)
    {
      accum_t val       = static_cast<accum_t>(h_values_it[i]);
      result_out_it[i]  = init;
      accum_t inclusive = static_cast<accum_t>(scan_op(init, val));

      ++i;

      for (; i < num_items && equality_op(h_keys_it[i - 1], h_keys_it[i]); ++i)
      {
        val              = static_cast<accum_t>(h_values_it[i]);
        result_out_it[i] = static_cast<output_t>(inclusive);
        inclusive        = static_cast<accum_t>(scan_op(inclusive, val));
      }
    }
  }
}

template <typename ValueT, typename KeyT, typename ValuesOutItT, typename ScanOpT, typename EqualityOpT, typename InitT>
void compute_exclusive_scan_by_key_reference(
  const c2h::device_vector<ValueT>& d_values,
  const c2h::device_vector<KeyT>& d_keys,
  ValuesOutItT result_out_it,
  ScanOpT scan_op,
  EqualityOpT equality_op,
  InitT init)
{
  c2h::host_vector<ValueT> host_values(d_values);
  c2h::host_vector<KeyT> host_keys(d_keys);

  std::size_t num_items = host_values.size();

  compute_exclusive_scan_by_key_reference(
    host_values.cbegin(), host_keys.cbegin(), result_out_it, scan_op, equality_op, init, num_items);
}

template <typename ValueInItT, typename KeyInItT, typename ValuesOutItT, typename ScanOpT, typename EqualityOpT>
void compute_inclusive_scan_by_key_reference(
  ValueInItT h_values_it,
  KeyInItT h_keys_it,
  ValuesOutItT result_out_it,
  ScanOpT scan_op,
  EqualityOpT equality_op,
  std::size_t num_items)
{
  using value_t  = cub::detail::value_t<ValueInItT>;
  using accum_t  = ::cuda::std::__accumulator_t<ScanOpT, value_t, value_t>;
  using output_t = cub::detail::value_t<ValuesOutItT>;

  for (std::size_t i = 0; i < num_items;)
  {
    accum_t inclusive = h_values_it[i];
    result_out_it[i]  = static_cast<output_t>(inclusive);

    ++i;

    for (; i < num_items && equality_op(h_keys_it[i - 1], h_keys_it[i]); ++i)
    {
      accum_t val      = h_values_it[i];
      inclusive        = static_cast<accum_t>(scan_op(inclusive, val));
      result_out_it[i] = static_cast<output_t>(inclusive);
    }
  }
}

template <typename ValueT, typename KeyT, typename ValuesOutItT, typename ScanOpT, typename EqualityOpT>
void compute_inclusive_scan_by_key_reference(
  const c2h::device_vector<ValueT>& d_values,
  const c2h::device_vector<KeyT>& d_keys,
  ValuesOutItT result_out_it,
  ScanOpT scan_op,
  EqualityOpT equality_op)
{
  c2h::host_vector<ValueT> host_values(d_values);
  c2h::host_vector<KeyT> host_keys(d_keys);

  std::size_t num_items = host_values.size();

  compute_inclusive_scan_by_key_reference(
    host_values.cbegin(), host_keys.cbegin(), result_out_it, scan_op, equality_op, num_items);
}
