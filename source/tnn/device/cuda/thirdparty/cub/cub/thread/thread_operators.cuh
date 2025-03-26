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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
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
 * Simple binary operator functor types
 */

/******************************************************************************
 * Simple functor operators
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

#include <cub/detail/type_traits.cuh> // always_false
#include <cub/util_cpp_dialect.cuh>
#include <cub/util_type.cuh>

#include <cuda/std/functional> // cuda::std::plus
#include <cuda/std/type_traits> // cuda::std::common_type
#include <cuda/std/utility> // cuda::std::forward

// #include <functional> // std::plus

CUB_NAMESPACE_BEGIN

/// @brief Inequality functor (wraps equality functor)
template <typename EqualityOp>
struct InequalityWrapper
{
  /// Wrapped equality operator
  EqualityOp op;

  /// Constructor
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE InequalityWrapper(EqualityOp op)
      : op(op)
  {}

  /// Boolean inequality operator, returns `t != u`
  template <typename T, typename U>
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE bool operator()(T&& t, U&& u)
  {
    return !op(::cuda::std::forward<T>(t), ::cuda::std::forward<U>(u));
  }
};

#if _CCCL_STD_VER > 2011
using Equality   = ::cuda::std::equal_to<>;
using Inequality = ::cuda::std::not_equal_to<>;
using Sum        = ::cuda::std::plus<>;
using Difference = ::cuda::std::minus<>;
using Division   = ::cuda::std::divides<>;
#else
/// @brief Default equality functor
struct Equality
{
  /// Boolean equality operator, returns `t == u`
  template <typename T, typename U>
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE bool operator()(T&& t, U&& u) const
  {
    return ::cuda::std::forward<T>(t) == ::cuda::std::forward<U>(u);
  }
};

/// @brief Default inequality functor
struct Inequality
{
  /// Boolean inequality operator, returns `t != u`
  template <typename T, typename U>
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE bool operator()(T&& t, U&& u) const
  {
    return ::cuda::std::forward<T>(t) != ::cuda::std::forward<U>(u);
  }
};

/// @brief Default sum functor
struct Sum
{
  /// Binary sum operator, returns `t + u`
  template <typename T, typename U>
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE auto
  operator()(T&& t, U&& u) const -> decltype(::cuda::std::forward<T>(t) + ::cuda::std::forward<U>(u))
  {
    return ::cuda::std::forward<T>(t) + ::cuda::std::forward<U>(u);
  }
};

/// @brief Default difference functor
struct Difference
{
  /// Binary difference operator, returns `t - u`
  template <typename T, typename U>
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE auto
  operator()(T&& t, U&& u) const -> decltype(::cuda::std::forward<T>(t) - ::cuda::std::forward<U>(u))
  {
    return ::cuda::std::forward<T>(t) - ::cuda::std::forward<U>(u);
  }
};

/// @brief Default division functor
struct Division
{
  /// Binary division operator, returns `t / u`
  template <typename T, typename U>
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE auto
  operator()(T&& t, U&& u) const -> decltype(::cuda::std::forward<T>(t) / ::cuda::std::forward<U>(u))
  {
    return ::cuda::std::forward<T>(t) / ::cuda::std::forward<U>(u);
  }
};
#endif

/// @brief Default max functor
struct Max
{
  /// Boolean max operator, returns `(t > u) ? t : u`
  template <typename T, typename U>
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE typename ::cuda::std::common_type<T, U>::type operator()(T&& t, U&& u) const
  {
    return CUB_MAX(t, u);
  }
};

/// @brief Arg max functor (keeps the value and offset of the first occurrence
///        of the larger item)
struct ArgMax
{
  /// Boolean max operator, preferring the item having the smaller offset in
  /// case of ties
  template <typename T, typename OffsetT>
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE KeyValuePair<OffsetT, T>
  operator()(const KeyValuePair<OffsetT, T>& a, const KeyValuePair<OffsetT, T>& b) const
  {
    // Mooch BUG (device reduce argmax gk110 3.2 million random fp32)
    // return ((b.value > a.value) ||
    //         ((a.value == b.value) && (b.key < a.key)))
    //      ? b : a;

    if ((b.value > a.value) || ((a.value == b.value) && (b.key < a.key)))
    {
      return b;
    }

    return a;
  }
};

/// @brief Default min functor
struct Min
{
  /// Boolean min operator, returns `(t < u) ? t : u`
  template <typename T, typename U>
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE typename ::cuda::std::common_type<T, U>::type operator()(T&& t, U&& u) const
  {
    return CUB_MIN(t, u);
  }
};

/// @brief Arg min functor (keeps the value and offset of the first occurrence
///        of the smallest item)
struct ArgMin
{
  /// Boolean min operator, preferring the item having the smaller offset in
  /// case of ties
  template <typename T, typename OffsetT>
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE KeyValuePair<OffsetT, T>
  operator()(const KeyValuePair<OffsetT, T>& a, const KeyValuePair<OffsetT, T>& b) const
  {
    // Mooch BUG (device reduce argmax gk110 3.2 million random fp32)
    // return ((b.value < a.value) ||
    //         ((a.value == b.value) && (b.key < a.key)))
    //      ? b : a;

    if ((b.value < a.value) || ((a.value == b.value) && (b.key < a.key)))
    {
      return b;
    }

    return a;
  }
};

namespace detail
{
template <typename ScanOpT>
struct ScanBySegmentOp
{
  /// Wrapped operator
  ScanOpT op;

  /// Constructor
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE ScanBySegmentOp() {}

  /// Constructor
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE ScanBySegmentOp(ScanOpT op)
      : op(op)
  {}

  /**
   * @brief Scan operator
   *
   * @tparam KeyValuePairT
   *   KeyValuePair pairing of T (value) and int (head flag)
   *
   * @param[in] first
   *   First partial reduction
   *
   * @param[in] second
   *   Second partial reduction
   */
  template <typename KeyValuePairT>
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE KeyValuePairT operator()(const KeyValuePairT& first, const KeyValuePairT& second)
  {
    KeyValuePairT retval;
    retval.key = first.key | second.key;
#ifdef _NVHPC_CUDA // WAR bug on nvc++
    if (second.key)
    {
      retval.value = second.value;
    }
    else
    {
      // If second.value isn't copied into a temporary here, nvc++ will
      // crash while compiling the TestScanByKeyWithLargeTypes test in
      // thrust/testing/scan_by_key.cu:
      auto v2      = second.value;
      retval.value = op(first.value, v2);
    }
#else // not nvc++:
    // if (second.key) {
    //   The second partial reduction spans a segment reset, so it's value
    //   aggregate becomes the running aggregate
    // else {
    //   The second partial reduction does not span a reset, so accumulate both
    //   into the running aggregate
    // }
    retval.value = (second.key) ? second.value : op(first.value, second.value);
#endif
    return retval;
  }
};

template <class OpT>
struct basic_binary_op_t
{
  static constexpr bool value = false;
};

template <>
struct basic_binary_op_t<Sum>
{
  static constexpr bool value = true;
};

template <>
struct basic_binary_op_t<Min>
{
  static constexpr bool value = true;
};

template <>
struct basic_binary_op_t<Max>
{
  static constexpr bool value = true;
};
} // namespace detail

/// @brief Default cast functor
template <typename B>
struct CastOp
{
  /// Cast operator, returns `(B) a`
  template <typename A>
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE B operator()(A&& a) const
  {
    return (B) a;
  }
};

/// @brief Binary operator wrapper for switching non-commutative scan arguments
template <typename ScanOp>
class SwizzleScanOp
{
private:
  /// Wrapped scan operator
  ScanOp scan_op;

public:
  /// Constructor
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE SwizzleScanOp(ScanOp scan_op)
      : scan_op(scan_op)
  {}

  /// Switch the scan arguments
  template <typename T>
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE T operator()(const T& a, const T& b)
  {
    T _a(a);
    T _b(b);

    return scan_op(_b, _a);
  }
};

/**
 * @brief Reduce-by-segment functor.
 *
 * Given two cub::KeyValuePair inputs `a` and `b` and a binary associative
 * combining operator `f(const T &x, const T &y)`, an instance of this functor
 * returns a cub::KeyValuePair whose `key` field is `a.key + b.key`, and whose
 * `value` field is either `b.value` if `b.key` is non-zero, or
 * `f(a.value, b.value)` otherwise.
 *
 * ReduceBySegmentOp is an associative, non-commutative binary combining
 * operator for input sequences of cub::KeyValuePair pairings. Such sequences
 * are typically used to represent a segmented set of values to be reduced
 * and a corresponding set of {0,1}-valued integer "head flags" demarcating the
 * first value of each segment.
 *
 * @tparam ReductionOpT Binary reduction operator to apply to values
 */
template <typename ReductionOpT>
struct ReduceBySegmentOp
{
  /// Wrapped reduction operator
  ReductionOpT op;

  /// Constructor
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE ReduceBySegmentOp() {}

  /// Constructor
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE ReduceBySegmentOp(ReductionOpT op)
      : op(op)
  {}

  /**
   * @brief Scan operator
   *
   * @tparam KeyValuePairT
   *   KeyValuePair pairing of T (value) and OffsetT (head flag)
   *
   * @param[in] first
   *   First partial reduction
   *
   * @param[in] second
   *   Second partial reduction
   */
  template <typename KeyValuePairT>
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE KeyValuePairT operator()(const KeyValuePairT& first, const KeyValuePairT& second)
  {
    KeyValuePairT retval;
    retval.key = first.key + second.key;
#ifdef _NVHPC_CUDA // WAR bug on nvc++
    if (second.key)
    {
      retval.value = second.value;
    }
    else
    {
      // If second.value isn't copied into a temporary here, nvc++ will
      // crash while compiling the TestScanByKeyWithLargeTypes test in
      // thrust/testing/scan_by_key.cu:
      auto v2      = second.value;
      retval.value = op(first.value, v2);
    }
#else // not nvc++:
    // if (second.key) {
    //   The second partial reduction spans a segment reset, so it's value
    //   aggregate becomes the running aggregate
    // else {
    //   The second partial reduction does not span a reset, so accumulate both
    //   into the running aggregate
    // }
    retval.value = (second.key) ? second.value : op(first.value, second.value);
#endif
    return retval;
  }
};

/**
 * @tparam ReductionOpT Binary reduction operator to apply to values
 */
template <typename ReductionOpT>
struct ReduceByKeyOp
{
  /// Wrapped reduction operator
  ReductionOpT op;

  /// Constructor
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE ReduceByKeyOp() {}

  /// Constructor
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE ReduceByKeyOp(ReductionOpT op)
      : op(op)
  {}

  /**
   * @brief Scan operator
   *
   * @param[in] first First partial reduction
   * @param[in] second Second partial reduction
   */
  template <typename KeyValuePairT>
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE KeyValuePairT operator()(const KeyValuePairT& first, const KeyValuePairT& second)
  {
    KeyValuePairT retval = second;

    if (first.key == second.key)
    {
      retval.value = op(first.value, retval.value);
    }

    return retval;
  }
};

template <typename BinaryOpT>
struct BinaryFlip
{
  BinaryOpT binary_op;

  _CCCL_HOST_DEVICE explicit BinaryFlip(BinaryOpT binary_op)
      : binary_op(binary_op)
  {}

  template <typename T, typename U>
  _CCCL_DEVICE auto
  operator()(T&& t, U&& u) -> decltype(binary_op(::cuda::std::forward<U>(u), ::cuda::std::forward<T>(t)))
  {
    return binary_op(::cuda::std::forward<U>(u), ::cuda::std::forward<T>(t));
  }
};

template <typename BinaryOpT>
_CCCL_HOST_DEVICE BinaryFlip<BinaryOpT> MakeBinaryFlip(BinaryOpT binary_op)
{
  return BinaryFlip<BinaryOpT>(binary_op);
}

namespace internal
{
// TODO: Remove DPX specilization when nvbug 4823237 is fixed

template <typename T>
struct DpxMin
{
  static_assert(detail::always_false<T>(), "DpxMin is not supported for this type");
};

template <>
struct DpxMin<::cuda::std::int16_t>
{
  _CCCL_NODISCARD _CCCL_DEVICE _CCCL_FORCEINLINE unsigned operator()(unsigned a, unsigned b) const
  {
    return __vmins2(a, b);
  }
};

template <>
struct DpxMin<::cuda::std::uint16_t>
{
  _CCCL_NODISCARD _CCCL_DEVICE _CCCL_FORCEINLINE unsigned operator()(unsigned a, unsigned b) const
  {
    return __vminu2(a, b);
  }
};

//----------------------------------------------------------------------------------------------------------------------

template <typename T>
struct DpxMax
{
  static_assert(detail::always_false<T>(), "DpxMax is not supported for this type");
};

template <>
struct DpxMax<::cuda::std::int16_t>
{
  _CCCL_NODISCARD _CCCL_DEVICE _CCCL_FORCEINLINE unsigned operator()(unsigned a, unsigned b) const
  {
    return __vmaxs2(a, b);
  }
};

template <>
struct DpxMax<::cuda::std::uint16_t>
{
  _CCCL_NODISCARD _CCCL_DEVICE _CCCL_FORCEINLINE unsigned operator()(unsigned a, unsigned b) const
  {
    return __vmaxu2(a, b);
  }
};

//----------------------------------------------------------------------------------------------------------------------

template <typename T>
struct DpxSum
{
  static_assert(detail::always_false<T>(), "DpxSum is not supported for this type");
};

template <>
struct DpxSum<::cuda::std::int16_t>
{
  _CCCL_NODISCARD _CCCL_DEVICE _CCCL_FORCEINLINE unsigned operator()(unsigned a, unsigned b) const
  {
    return __vadd2(a, b);
  }
};

template <>
struct DpxSum<::cuda::std::uint16_t>
{
  _CCCL_NODISCARD _CCCL_DEVICE _CCCL_FORCEINLINE unsigned operator()(unsigned a, unsigned b) const
  {
    return __vadd2(a, b);
  }
};

//----------------------------------------------------------------------------------------------------------------------

template <typename ReduceOp, typename T>
struct CubOperatorToDpx
{
  static_assert(detail::always_false<T>(), "Dpx is not supported for this operator");
};

template <typename T>
struct CubOperatorToDpx<cub::Min, T>
{
  using type = DpxMin<T>;
};

template <typename T>
struct CubOperatorToDpx<cub::Max, T>
{
  using type = DpxMax<T>;
};

template <typename T>
struct CubOperatorToDpx<cub::Sum, T>
{
  using type = DpxSum<T>;
};

// template <typename T>
// struct CubOperatorToDpx<std::plus<T>, T>
//{
//   using type = DpxSum<T>;
// };

template <typename ReduceOp, typename T>
using cub_operator_to_dpx_t = CubOperatorToDpx<ReduceOp, T>;

} // namespace internal

CUB_NAMESPACE_END
