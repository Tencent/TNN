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

#include <cub/thread/thread_operators.cuh>
#include <cub/util_namespace.cuh>
#include <cub/util_type.cuh>

#include <thrust/iterator/constant_iterator.h>

#include <iostream>
#include <numeric>
#include <type_traits>

#include <c2h/catch2_test_helper.cuh>
#include <c2h/custom_type.cuh>
#include <c2h/extended_types.cuh>
#include <c2h/test_util_vec.cuh>
#include <nv/target>

CUB_NAMESPACE_BEGIN

#if TEST_HALF_T
// Half support is provided by SM53+. We currently test against a few older architectures.
// The specializations below can be removed once we drop these architectures.
template <>
__host__ __device__ __forceinline__ //
  __half
  Min::operator()(__half& a, __half& b) const
{
  NV_IF_TARGET(NV_PROVIDES_SM_53, (return CUB_MIN(a, b);), (return CUB_MIN(__half2float(a), __half2float(b));));
}

template <>
__host__ __device__ __forceinline__ //
  KeyValuePair<int, __half>
  ArgMin::operator()(const KeyValuePair<int, __half>& a, const KeyValuePair<int, __half>& b) const
{
  const float av = __half2float(a.value);
  const float bv = __half2float(b.value);

  if ((bv < av) || ((av == bv) && (b.key < a.key)))
  {
    return b;
  }

  return a;
}

template <>
__host__ __device__ __forceinline__ //
  __half
  Max::operator()(__half& a, __half& b) const
{
  NV_IF_TARGET(NV_PROVIDES_SM_53, (return CUB_MAX(a, b);), (return CUB_MAX(__half2float(a), __half2float(b));));
}

template <>
__host__ __device__ __forceinline__ //
  KeyValuePair<int, __half>
  ArgMax::operator()(const KeyValuePair<int, __half>& a, const KeyValuePair<int, __half>& b) const
{
  const float av = __half2float(a.value);
  const float bv = __half2float(b.value);

  if ((bv > av) || ((av == bv) && (b.key < a.key)))
  {
    return b;
  }

  return a;
}
#endif // TEST_HALF_T

/**
 * @brief Introduces the required NumericTraits for `c2h::custom_type_t`.
 */
template <template <typename> class... Policies>
struct NumericTraits<c2h::custom_type_t<Policies...>>
{
  using custom_t                     = c2h::custom_type_t<Policies...>;
  static constexpr Category CATEGORY = NOT_A_NUMBER;
  enum
  {
    PRIMITIVE = false,
    NULL_TYPE = false,
  };
  __host__ __device__ static custom_t Max()
  {
    custom_t val{};
    val.key = NumericTraits<decltype(std::declval<custom_t>().key)>::Max();
    val.val = NumericTraits<decltype(std::declval<custom_t>().val)>::Max();
    return val;
  }

  __host__ __device__ static custom_t Lowest()
  {
    custom_t val{};
    val.key = NumericTraits<decltype(std::declval<custom_t>().key)>::Lowest();
    val.val = NumericTraits<decltype(std::declval<custom_t>().val)>::Lowest();
    return val;
  }
};

template <typename Key, typename Value>
static std::ostream& operator<<(std::ostream& os, const KeyValuePair<Key, Value>& val)
{
  os << '(' << val.key << ',' << val.value << ')';
  return os;
}

template <typename Key, typename Value>
__host__ __device__ __forceinline__ bool
operator==(const KeyValuePair<Key, Value>& lhs, const KeyValuePair<Key, Value>& rhs)
{
  return lhs.key == rhs.key && lhs.value == rhs.value;
}

CUB_NAMESPACE_END

// Comparing results computed on CPU and GPU for extended floating point types is impossible.
// For instance, when used with a constant iterator of two, the accumulator in sequential reference
// computation (CPU) bumps into the 4096 limits, which will never change (`4096 + 2 = 4096`).
// Meanwhile, per-thread aggregates (`2 * 16 = 32`) are accumulated within and among thread blocks,
// yielding `inf` as a result. No reasonable epsilon can be selected to compare `inf` with `4096`.
// To make `__half` and `__nv_bfloat16` arithmetic associative, the function object below raises
// extended floating points to the area of unsigned short integers. This allows us to test large
// inputs with few code-path differences in device algorithms.
struct ExtendedFloatSum
{
  template <class T>
  __host__ __device__ T operator()(T a, T b) const
  {
    T result{};
    result.__x = a.raw() + b.raw();
    return result;
  }

#if TEST_HALF_T
  __host__ __device__ __half operator()(__half a, __half b) const
  {
    uint16_t result = this->operator()(half_t{a}, half_t(b)).raw();
    return reinterpret_cast<__half&>(result);
  }
#endif

#if TEST_BF_T
  __device__ __nv_bfloat16 operator()(__nv_bfloat16 a, __nv_bfloat16 b) const
  {
    uint16_t result = this->operator()(bfloat16_t{a}, bfloat16_t(b)).raw();
    return reinterpret_cast<__nv_bfloat16&>(result);
  }
#endif
};

template <class It>
inline It unwrap_it(It it)
{
  return it;
}

#if TEST_HALF_T
inline __half* unwrap_it(half_t* it)
{
  return reinterpret_cast<__half*>(it);
}

template <class OffsetT>
inline cub::ConstantInputIterator<__half, OffsetT> unwrap_it(cub::ConstantInputIterator<half_t, OffsetT> it)
{
  half_t wrapped_val = *it;
  __half val         = wrapped_val.operator __half();
  return cub::ConstantInputIterator<__half, OffsetT>(val);
}
#endif

#if TEST_BF_T
inline __nv_bfloat16* unwrap_it(bfloat16_t* it)
{
  return reinterpret_cast<__nv_bfloat16*>(it);
}

template <class OffsetT>
cub::ConstantInputIterator<__nv_bfloat16, OffsetT> inline unwrap_it(cub::ConstantInputIterator<bfloat16_t, OffsetT> it)
{
  bfloat16_t wrapped_val = *it;
  __nv_bfloat16 val      = wrapped_val.operator __nv_bfloat16();
  return cub::ConstantInputIterator<__nv_bfloat16, OffsetT>(val);
}
#endif

template <typename T>
using unwrap_value_t = typename std::remove_reference<decltype(*unwrap_it(std::declval<T*>()))>::type;

template <class WrappedItT, //
          class ItT = decltype(unwrap_it(std::declval<WrappedItT>()))>
std::integral_constant<bool, !std::is_same<WrappedItT, ItT>::value> //
  inline reference_extended_fp(WrappedItT)
{
  return {};
}

inline ExtendedFloatSum unwrap_op(std::true_type /* extended float */, cub::Sum) //
{
  return {};
}

template <bool V, class OpT>
inline OpT unwrap_op(std::integral_constant<bool, V> /* base case */, OpT op)
{
  return op;
}

/**
 * @brief Initializes the given item type with a constant non-zero value.
 */
template <typename T>
inline void init_default_constant(T& val)
{
  val = T{2};
}

template <template <typename> class... Policies>
inline void init_default_constant(c2h::custom_type_t<Policies...>& val)
{
  val.key = 2;
  val.val = 2;
}

inline void init_default_constant(uchar3& val)
{
  val = uchar3{2, 2, 2};
}

inline void init_default_constant(ulonglong4& val)
{
  val = ulonglong4{2, 2, 2, 2};
}

template <typename InputItT,
          typename OffsetItT,
          typename SizeItT,
          typename ReductionOpT,
          typename InitT,
          typename ResultOutItT>
inline void compute_host_reference(
  InputItT h_in,
  OffsetItT h_offsets,
  SizeItT h_sizes_begin,
  std::size_t num_segments,
  ReductionOpT reduction_op,
  InitT init,
  ResultOutItT h_data_out)
{
  for (std::size_t segment = 0; segment < num_segments; segment++)
  {
    auto seg_begin = h_in + h_offsets[segment];
    auto seg_end   = seg_begin + h_sizes_begin[segment];
    // TODO Should this be using cub accumulator t?
    h_data_out[segment] =
      static_cast<cub::detail::value_t<ResultOutItT>>(std::accumulate(seg_begin, seg_end, init, reduction_op));
  }
}

/**
 * @brief Helper function to compute the reference solution for result verification taking an
 * arbitrary host-accessible input iterator.
 */
template <typename InputItT, typename ReductionOpT, typename AccumulatorT>
inline AccumulatorT
compute_single_problem_reference(InputItT h_in_begin, InputItT h_in_end, ReductionOpT reduction_op, AccumulatorT init)
{
  constexpr std::size_t num_segments = 1;
  c2h::host_vector<AccumulatorT> h_results(num_segments);

  compute_host_reference(
    h_in_begin,
    thrust::make_constant_iterator(0),
    thrust::make_constant_iterator(thrust::distance(h_in_begin, h_in_end)),
    num_segments,
    reduction_op,
    init,
    h_results.begin());

  return *h_results.begin();
}

/**
 * @brief Helper function to compute the reference solution for result verification, taking a
 * c2h::device_vector.
 */
template <typename ItemT, typename ReductionOpT, typename AccumulatorT>
inline AccumulatorT
compute_single_problem_reference(const c2h::device_vector<ItemT>& d_in, ReductionOpT reduction_op, AccumulatorT init)
{
  constexpr std::size_t num_segments = 1;
  c2h::host_vector<ItemT> h_items(d_in);
  c2h::host_vector<AccumulatorT> h_results(num_segments);

  return compute_single_problem_reference(h_items.cbegin(), h_items.cend(), reduction_op, init);
}

/**
 * @brief Helper function to compute the reference solution for result verification, taking a
 * c2h::device_vector of input items and a c2h::device_vector of offsets into the segments.
 */
template <typename ItemT, typename OffsetT, typename ReductionOpT, typename AccumulatorT, typename ResultItT>
void compute_segmented_problem_reference(
  const c2h::device_vector<ItemT>& d_in,
  const c2h::device_vector<OffsetT>& d_offsets,
  ReductionOpT reduction_op,
  AccumulatorT init,
  ResultItT h_results)
{
  c2h::host_vector<ItemT> h_items(d_in);
  c2h::host_vector<OffsetT> h_offsets(d_offsets);
  auto offsets_it = h_offsets.cbegin();
  auto seg_sizes_it =
    thrust::make_transform_iterator(thrust::make_counting_iterator(std::size_t{0}), [offsets_it](std::size_t i) {
      return offsets_it[i + 1] - offsets_it[i];
    });
  std::size_t num_segments = h_offsets.size() - 1;

  compute_host_reference(
    h_items.cbegin(), h_offsets.cbegin(), seg_sizes_it, num_segments, reduction_op, init, h_results);
}

/**
 * @brief Helper function to compute the reference solution for result verification, taking a
 * host-accessible input iterator and a c2h::device_vector of offsets into the segments.
 */
template <typename InputItT, typename OffsetT, typename ReductionOpT, typename AccumulatorT, typename ResultItT>
void compute_segmented_problem_reference(
  InputItT in_it,
  const c2h::device_vector<OffsetT>& d_offsets,
  ReductionOpT reduction_op,
  AccumulatorT init,
  ResultItT h_results)
{
  c2h::host_vector<OffsetT> h_offsets(d_offsets);
  auto offsets_it = h_offsets.cbegin();
  auto seg_sizes_it =
    thrust::make_transform_iterator(thrust::make_counting_iterator(std::size_t{0}), [offsets_it](std::size_t i) {
      return offsets_it[i + 1] - offsets_it[i];
    });
  std::size_t num_segments = h_offsets.size() - 1;

  compute_host_reference(in_it, h_offsets.cbegin(), seg_sizes_it, num_segments, reduction_op, init, h_results);
}

/**
 * @brief Helper function to compute the reference solution for result verification, taking a
 * c2h::device_vector of input items and a c2h::device_vector of offsets into the segments.
 */
template <typename ItemT, typename OffsetT, typename ResultItT>
void compute_segmented_argmin_reference(
  const c2h::device_vector<ItemT>& d_in, const c2h::device_vector<OffsetT>& d_offsets, ResultItT h_results)
{
  c2h::host_vector<ItemT> h_items(d_in);
  c2h::host_vector<OffsetT> h_offsets(d_offsets);
  const auto num_segments = h_offsets.size() - 1;
  for (std::size_t seg = 0; seg < num_segments; seg++)
  {
    if (h_offsets[seg] >= h_offsets[seg + 1])
    {
      h_results[seg] = {1, cub::Traits<ItemT>::Max()};
    }
    else
    {
      auto expected_result_it =
        std::min_element(h_items.cbegin() + h_offsets[seg], h_items.cbegin() + h_offsets[seg + 1]);
      int result_offset = static_cast<int>(thrust::distance((h_items.cbegin() + h_offsets[seg]), expected_result_it));
      h_results[seg]    = {result_offset, *expected_result_it};
    }
  }
}

/**
 * @brief Helper function to compute the reference solution for result verification, taking a
 * c2h::device_vector of input items and a c2h::device_vector of offsets into the segments.
 */
template <typename ItemT, typename OffsetT, typename ResultItT>
void compute_segmented_argmax_reference(
  const c2h::device_vector<ItemT>& d_in, const c2h::device_vector<OffsetT>& d_offsets, ResultItT h_results)
{
  c2h::host_vector<ItemT> h_items(d_in);
  c2h::host_vector<OffsetT> h_offsets(d_offsets);
  const auto num_segments = h_offsets.size() - 1;
  for (std::size_t seg = 0; seg < num_segments; seg++)
  {
    if (h_offsets[seg] >= h_offsets[seg + 1])
    {
      h_results[seg] = {1, cub::Traits<ItemT>::Lowest()};
    }
    else
    {
      auto expected_result_it =
        std::max_element(h_items.cbegin() + h_offsets[seg], h_items.cbegin() + h_offsets[seg + 1]);
      int result_offset = static_cast<int>(thrust::distance((h_items.cbegin() + h_offsets[seg]), expected_result_it));
      h_results[seg]    = {result_offset, *expected_result_it};
    }
  }
}

/**
 * @brief Helper function to compute the reference solution for unique keys (i.e., collapsing each
 * run of equal keys into a single key).
 */
template <typename InputItT, typename OutputItT>
inline OutputItT compute_unique_keys_reference(InputItT h_in_begin, std::size_t num_keys, OutputItT h_out_it)
{
  if (num_keys == 0)
  {
    return h_out_it;
  }
  *h_out_it++ = h_in_begin[0];
  for (std::size_t i = 1; i < num_keys; i++)
  {
    if (!(h_in_begin[i - 1] == h_in_begin[i]))
    {
      *h_out_it = h_in_begin[i];
      h_out_it++;
    }
  }
  return h_out_it;
}

/**
 * @brief Helper function to compute the reference solution for unique keys (i.e., collapsing each
 * run of equal keys into a single key).
 */
template <typename ItemT>
inline c2h::host_vector<ItemT> compute_unique_keys_reference(const c2h::device_vector<ItemT>& d_keys)
{
  c2h::host_vector<ItemT> h_keys(d_keys);
  c2h::host_vector<ItemT> h_unique_keys_out(d_keys.size());

  auto end_it = compute_unique_keys_reference(h_keys.cbegin(), h_keys.size(), h_unique_keys_out.begin());
  h_unique_keys_out.resize(thrust::distance(h_unique_keys_out.begin(), end_it));
  return h_unique_keys_out;
}

/**
 * @brief Helper class template to facilitate specifying input/output type pairs along with the key
 * type for reduce-by-key algorithms.
 */
template <typename InputT, typename OutputT = InputT, typename KeyT = std::int32_t>
struct type_triple
{
  using input_t  = InputT;
  using output_t = OutputT;
  using key_t    = KeyT;
};

/**
 * @brief Helper class template to facilitate specifying input/output type pairs.
 */
template <typename InputT, typename OutputT = InputT>
struct type_pair
{
  using input_t  = InputT;
  using output_t = OutputT;
};

/**
 * @brief Helper class template to facilitate accessing types specified by type-parameterized tests.
 */
template <typename TestType>
struct params_t
{
  using type_pair_t = typename c2h::get<0, TestType>;
  using item_t      = typename type_pair_t::input_t;
  using output_t    = typename type_pair_t::output_t;
};
