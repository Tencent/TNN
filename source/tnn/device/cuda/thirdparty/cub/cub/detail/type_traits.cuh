/******************************************************************************
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
 * \file
 * Wrappers and extensions around <type_traits> utilities.
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

#include <cub/util_cpp_dialect.cuh>
#include <cub/util_namespace.cuh>

_CCCL_SUPPRESS_DEPRECATED_PUSH
#include <cuda/std/functional>
_CCCL_SUPPRESS_DEPRECATED_POP
#include <cuda/std/type_traits>

#define _CUB_TEMPLATE_REQUIRES(...) ::cuda::std::__enable_if_t<(__VA_ARGS__)>* = nullptr

CUB_NAMESPACE_BEGIN
namespace detail
{

template <typename Invokable, typename... Args>
using invoke_result_t =
#if _CCCL_STD_VER < 2017
  typename ::cuda::std::result_of<Invokable(Args...)>::type;
#else // 2017+
  ::cuda::std::invoke_result_t<Invokable, Args...>;
#endif

template <typename T, typename... TArgs>
_CCCL_NODISCARD _CCCL_HOST_DEVICE _CCCL_FORCEINLINE constexpr bool are_same()
{
  return ::cuda::std::conjunction<::cuda::std::is_same<T, TArgs>...>::value;
}

template <typename T, typename... TArgs>
_CCCL_NODISCARD _CCCL_HOST_DEVICE _CCCL_FORCEINLINE constexpr bool is_one_of()
{
  return ::cuda::std::disjunction<::cuda::std::is_same<T, TArgs>...>::value;
}

template <typename...>
_CCCL_NODISCARD _CCCL_HOST_DEVICE _CCCL_FORCEINLINE constexpr bool always_false()
{
  return false;
}

template <typename T, typename V, typename = void>
struct has_binary_call_operator : ::cuda::std::false_type
{};

template <typename T, typename V>
struct has_binary_call_operator<
  T,
  V,
  ::cuda::std::void_t<decltype(::cuda::std::declval<T>()(::cuda::std::declval<V>(), ::cuda::std::declval<V>()))>>
    : ::cuda::std::true_type
{};

/***********************************************************************************************************************
 * Array like type traits
 **********************************************************************************************************************/

template <typename T, typename = void>
struct has_subscript : ::cuda::std::false_type
{};

template <typename T>
struct has_subscript<T, ::cuda::std::void_t<decltype(::cuda::std::declval<T>()[0])>> : ::cuda::std::true_type
{};

template <typename T>
using has_subscript_t = typename has_subscript<T>::type;

template <typename T, typename = void>
struct has_size : ::cuda::std::false_type
{};

// TODO: use ::cuda::std::size(::cuda::std::declval<T>()) when std::size will be available in libcu++
template <typename T>
struct has_size<T, ::cuda::std::void_t<decltype(::cuda::std::declval<T>().size())>> : ::cuda::std::true_type
{};

template <typename T, ::cuda::std::size_t N>
struct has_size<T[N], void> : ::cuda::std::true_type
{};

template <typename T>
using has_size_t = typename has_size<T>::type;

/***********************************************************************************************************************
 * StaticSize: a type trait that returns the number of elements in an Array-like type
 **********************************************************************************************************************/
// StaticSize is useful where size(obj) cannot be checked at compile time
// e.g.
// using Array = NonTriviallyConstructible[8];
// std::size(Array{})   // compile error
// static_size<Array>() // ok

template <typename T, typename = void>
struct StaticSize
{
  static_assert(detail::always_false<T>(), "StaticSize not supported for this type");
};

template <typename T>
struct StaticSize<T,
                  ::cuda::std::void_t<decltype(::cuda::std::integral_constant<int, ::cuda::std::declval<T>().size()>{})>>
{
  static_assert(::cuda::std::is_trivially_constructible<T>::value, "T must be trivially constructible");
  static constexpr auto value = T{}.size();
};

template <typename T, ::cuda::std::size_t N>
struct StaticSize<T[N], void>
{
  static constexpr auto value = N;
};

template <typename T>
_CCCL_NODISCARD _CCCL_HOST_DEVICE _CCCL_FORCEINLINE constexpr ::cuda::std::size_t static_size()
{
  return StaticSize<T>::value;
}

} // namespace detail
CUB_NAMESPACE_END
