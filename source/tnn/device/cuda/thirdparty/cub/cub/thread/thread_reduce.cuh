/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
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

/**
 * @file
 * Thread reduction over statically-sized array-like types
 */

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/detail/type_traits.cuh> // are_same()
#include <cub/thread/thread_operators.cuh> // cub_operator_to_dpx_t
#include <cub/util_namespace.cuh>
#include <cub/util_type.cuh>

#include <cuda/std/bit> // bit_cast
#include <cuda/std/cstdint> // uint16_t
#include <cuda/std/functional> // cuda::std::plus
#include <cuda/std/utility> // pair

// #include <functional> // std::plus

CUB_NAMESPACE_BEGIN

/// Internal namespace (to prevent ADL mishaps between static functions when mixing different CUB installations)
namespace internal
{

#ifndef DOXYGEN_SHOULD_SKIP_THIS // Do not document

/// DPX instructions compute min, max, and sum for up to three 16 and 32-bit signed or unsigned integer parameters
/// see DPX documetation https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#dpx
/// NOTE: The compiler is able to automatically vectorize all cases with 3 operands
///       However, all other cases with per-halfword comparison need to be explicitly vectorized
/// TODO: Remove DPX specilization when nvbug 4823237 is fixed
///
/// DPX reduction is enabled if the following conditions are met:
/// - Hopper+ architectures. DPX instructions are emulated before Hopper
/// - The number of elements must be large enough for performance reasons (see below)
/// - All types must be the same
/// - Only works with integral types of 2 bytes
/// - DPX instructions provide Min, Max, and Sum SIMD operations
/// If the number of instructions is the same, we favor the compiler

template <typename Input, typename ReductionOp, typename AccumT>
_CCCL_NODISCARD _CCCL_DEVICE _CCCL_FORCEINLINE // clang-format off
constexpr bool enable_dpx_reduction()
{
  using T = decltype(::cuda::std::declval<Input>()[0]);
  // TODO: use constexpr variable in C++14+
  using Lenght = ::cuda::std::integral_constant<int, detail::static_size<Input>()>;
  return ((Lenght{} >= 9 && detail::are_same<ReductionOp, cub::Sum/*, std::plus<T>*/>()) || Lenght{} >= 10)
            && detail::are_same<T, AccumT>()
            && detail::is_one_of<T, int16_t, uint16_t>()
            && detail::is_one_of<ReductionOp, cub::Min, cub::Max, cub::Sum/*, std::plus<T>*/>();
}
// clang-format on

// Considering compiler vectorization with 3-way comparison, the number of SASS instructions is
// Standard: ceil((L - 3) / 2) + 1
//   replacing L with L/2 for SIMD
// DPX:      ceil((L/2 - 3) / 2) + 1 + 2 [for halfword comparison: PRMT, VIMNMX] + L % 2 [for last element]
//   finally, the last two comparision operations are vectorized in a 3-way reduction
//           ceil((L/2 - 3) / 2) + 3
//
// length | Standard |  DPX
//  2     |    1     |  NA
//  3     |    1     |  NA
//  4     |    2     |  3
//  5     |    2     |  3
//  6     |    3     |  3
//  7     |    3     |  3
//  8     |    4     |  4
//  9     |    4     |  4
// 10     |    5     |  4 // ***
// 11     |    5     |  4 // ***
// 12     |    6     |  5 // ***
// 13     |    6     |  5 // ***
// 14     |    7     |  5 // ***
// 15     |    7     |  5 // ***
// 16     |    8     |  6 // ***

template <typename AccumT, typename Input, typename ReductionOp>
_CCCL_NODISCARD _CCCL_DEVICE _CCCL_FORCEINLINE AccumT
ThreadReduceSequential(const Input& input, ReductionOp reduction_op)
{
  AccumT retval = input[0];
#  pragma unroll
  for (int i = 1; i < detail::static_size<Input>(); ++i)
  {
    retval = reduction_op(retval, input[i]);
  }
  return retval;
}

/// Specialization for DPX reduction
template <typename Input, typename ReductionOp>
_CCCL_NODISCARD _CCCL_DEVICE _CCCL_FORCEINLINE auto
ThreadReduceDpx(const Input& input, ReductionOp reduction_op) -> ::cuda::std::__remove_cvref_t<decltype(input[0])>
{
  using T              = ::cuda::std::__remove_cvref_t<decltype(input[0])>;
  constexpr int length = detail::static_size<Input>();
  T array[length];
#  pragma unroll
  for (int i = 0; i < length; ++i)
  {
    array[i] = input[i];
  }
  using DpxReduceOp   = cub_operator_to_dpx_t<ReductionOp, T>;
  using SimdType      = ::cuda::std::pair<T, T>;
  auto unsigned_input = reinterpret_cast<const unsigned*>(array);
  auto simd_reduction = ThreadReduceSequential<length / 2>(unsigned_input, DpxReduceOp{});
  auto simd_values    = ::cuda::std::bit_cast<SimdType>(simd_reduction);
  auto ret_value      = reduction_op(simd_values.first, simd_values.second);
  return (length % 2 == 0) ? ret_value : reduction_op(ret_value, input[length - 1]);
}

// DPX/Sequential dispatch
template <typename Input,
          typename ReductionOp,
          typename ValueT = ::cuda::std::__remove_cvref_t<decltype(::cuda::std::declval<Input>()[0])>,
          typename AccumT = ::cuda::std::__accumulator_t<ReductionOp, ValueT>,
          _CUB_TEMPLATE_REQUIRES(enable_dpx_reduction<Input, ReductionOp, AccumT>())>
_CCCL_NODISCARD _CCCL_DEVICE _CCCL_FORCEINLINE AccumT ThreadReduce(const Input& input, ReductionOp reduction_op)
{
  static_assert(sizeof(Input) != sizeof(Input), "a");
  static_assert(detail::has_subscript<Input>::value, "Input must support the subscript operator[]");
  static_assert(detail::has_size<Input>::value, "Input must have the size() method");
  static_assert(detail::has_binary_call_operator<ReductionOp, ValueT>::value,
                "ReductionOp must have the binary call operator: operator(ValueT, ValueT)");
  NV_IF_TARGET(NV_PROVIDES_SM_90,
               (return ThreadReduceDpx(input, reduction_op);),
               (return ThreadReduceSequential<AccumT>(input, reduction_op);))
}

template <typename Input,
          typename ReductionOp,
          typename ValueT = ::cuda::std::__remove_cvref_t<decltype(::cuda::std::declval<Input>()[0])>,
          typename AccumT = ::cuda::std::__accumulator_t<ReductionOp, ValueT>,
          _CUB_TEMPLATE_REQUIRES(!enable_dpx_reduction<Input, ReductionOp, AccumT>())>
_CCCL_NODISCARD _CCCL_DEVICE _CCCL_FORCEINLINE AccumT ThreadReduce(const Input& input, ReductionOp reduction_op)
{
  static_assert(detail::has_subscript<Input>::value, "Input must support the subscript operator[]");
  static_assert(detail::has_size<Input>::value, "Input must have the size() method");
  static_assert(detail::has_binary_call_operator<ReductionOp, ValueT>::value,
                "ReductionOp must have the binary call operator: operator(ValueT, ValueT)");
  return ThreadReduceSequential<AccumT>(input, reduction_op);
}

#endif // !DOXYGEN_SHOULD_SKIP_THIS

/**
 * @brief Reduction over statically-sized array-like types, seeded with the specified @p prefix.
 *
 * @tparam Input
 *   <b>[inferred]</b> The data type to be reduced having member
 *   <tt>operator[](int i)</tt> and must be statically-sized (size() method or static array)
 *
 * @tparam ReductionOp
 *   <b>[inferred]</b> Binary reduction operator type having member
 *   <tt>T operator()(const T &a, const T &b)</tt>
 *
 * @tparam PrefixT
 *   <b>[inferred]</b> The prefix type
 *
 * @param[in] input
 *   Input array
 *
 * @param[in] reduction_op
 *   Binary reduction operator
 *
 * @param[in] prefix
 *   Prefix to seed reduction with
 *
 * @return Aggregate of type <tt>cuda::std::__accumulator_t<ReductionOp, ValueT, PrefixT></tt>
 */
template <typename Input,
          typename ReductionOp,
          typename PrefixT,
#ifndef DOXYGEN_SHOULD_SKIP_THIS // Do not document
          typename ValueT = ::cuda::std::__remove_cvref_t<decltype(::cuda::std::declval<Input>()[0])>,
#endif // !DOXYGEN_SHOULD_SKIP_THIS
          typename AccumT = ::cuda::std::__accumulator_t<ReductionOp, ValueT, PrefixT>>
_CCCL_NODISCARD _CCCL_DEVICE _CCCL_FORCEINLINE AccumT
ThreadReduce(const Input& input, ReductionOp reduction_op, PrefixT prefix)
{
  static_assert(detail::has_subscript<Input>::value, "Input must support the subscript operator[]");
  static_assert(detail::has_size<Input>::value, "Input must have the size() method");
  static_assert(detail::has_binary_call_operator<ReductionOp, ValueT>::value,
                "ReductionOp must have the binary call operator: operator(ValueT, ValueT)");
  constexpr int length = detail::static_size<Input>();
  // copy to a temporary array of type AccumT
  AccumT array[length + 1];
  array[0] = prefix;
#pragma unroll
  for (int i = 0; i < length; ++i)
  {
    array[i + 1] = input[i];
  }
  return ThreadReduce<decltype(array), ReductionOp, AccumT, AccumT>(array, reduction_op);
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS // Do not document

/**
 * @remark The pointer interface adds little value and requires Length to be explicit.
 *         Prefer using the array-like interface
 *
 * @brief Perform a sequential reduction over @p length elements of the @p input pointer. The aggregate is returned.
 *
 * @tparam T
 *   <b>[inferred]</b> The data type to be reduced
 *
 * @tparam ReductionOp
 *   <b>[inferred]</b> Binary reduction operator type having member
 *   <tt>T operator()(const T &a, const T &b)</tt>
 *
 * @param[in] input
 *   Input pointer
 *
 * @param[in] reduction_op
 *   Binary reduction operator
 *
 * @return Aggregate of type <tt>cuda::std::__accumulator_t<ReductionOp, T></tt>
 */
template <int Length, typename T, typename ReductionOp, typename AccumT = ::cuda::std::__accumulator_t<ReductionOp, T>>
_CCCL_NODISCARD _CCCL_DEVICE _CCCL_FORCEINLINE AccumT ThreadReduce(const T* input, ReductionOp reduction_op)
{
  static_assert(Length > 0, "Length must be greater than 0");
  static_assert(detail::has_binary_call_operator<ReductionOp, T>::value,
                "ReductionOp must have the binary call operator: operator(V1, V2)");
  using ArrayT = T[Length];
  auto array   = reinterpret_cast<const T(*)[Length]>(input);
  return ThreadReduce(*array, reduction_op);
}

/**
 * @remark The pointer interface adds little value and requires Length to be explicit.
 *         Prefer using the array-like interface
 *
 * @brief Perform a sequential reduction over @p length elements of the @p input pointer, seeded with the specified @p
 *        prefix. The aggregate is returned.
 *
 * @tparam length
 *   Length of input pointer
 *
 * @tparam T
 *   <b>[inferred]</b> The data type to be reduced
 *
 * @tparam ReductionOp
 *   <b>[inferred]</b> Binary reduction operator type having member
 *   <tt>T operator()(const T &a, const T &b)</tt>
 *
 * @tparam PrefixT
 *   <b>[inferred]</b> The prefix type
 *
 * @param[in] input
 *   Input pointer
 *
 * @param[in] reduction_op
 *   Binary reduction operator
 *
 * @param[in] prefix
 *   Prefix to seed reduction with
 *
 * @return Aggregate of type <tt>cuda::std::__accumulator_t<ReductionOp, T, PrefixT></tt>
 */
template <int Length,
          typename T,
          typename ReductionOp,
          typename PrefixT,
          typename AccumT = ::cuda::std::__accumulator_t<ReductionOp, T, PrefixT>,
          _CUB_TEMPLATE_REQUIRES(Length > 0)>
_CCCL_NODISCARD _CCCL_DEVICE _CCCL_FORCEINLINE AccumT
ThreadReduce(const T* input, ReductionOp reduction_op, PrefixT prefix)
{
  static_assert(detail::has_binary_call_operator<ReductionOp, T>::value,
                "ReductionOp must have the binary call operator: operator(V1, V2)");
  auto array = reinterpret_cast<const T(*)[Length]>(input);
  return ThreadReduce(*array, reduction_op, prefix);
}

template <int Length, typename T, typename ReductionOp, typename PrefixT, _CUB_TEMPLATE_REQUIRES(Length == 0)>
_CCCL_NODISCARD _CCCL_DEVICE _CCCL_FORCEINLINE T ThreadReduce(const T*, ReductionOp, PrefixT prefix)
{
  return prefix;
}

#endif // !DOXYGEN_SHOULD_SKIP_THIS

} // namespace internal
CUB_NAMESPACE_END
