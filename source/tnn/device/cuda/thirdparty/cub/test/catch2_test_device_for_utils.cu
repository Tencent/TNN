/******************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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

#include "insert_nested_NVTX_range_guard.h"
// above header needs to be included first

#include <cub/device/device_for.cuh>

#include <c2h/catch2_test_helper.cuh>

template <class T>
struct value_t
{
  __device__ void operator()(T) {}
};

template <class T>
struct const_ref_t
{
  __device__ void operator()(const T&) {}
};

template <class T>
struct rref_t
{
  __device__ void operator()(T&&) {}
};

template <class T>
struct value_ret_t
{
  __device__ T operator()(T v)
  {
    return v;
  }
};

template <class T>
struct ref_t
{
  __device__ void operator()(T&) {}
};

struct tpl_value_t
{
  template <class T>
  __device__ void operator()(T)
  {}
};

template <class T>
struct overload_value_t
{
  __device__ void operator()(T) {}
  __device__ void operator()(T) const {}
};

template <class T>
struct value_const_t
{
  __device__ void operator()(T) const {}
};

template <class T>
void test()
{
  STATIC_REQUIRE(cub::detail::for_each::has_unique_value_overload<T, value_t<T>>::value);
  STATIC_REQUIRE(cub::detail::for_each::has_unique_value_overload<T, value_const_t<T>>::value);
  STATIC_REQUIRE(cub::detail::for_each::has_unique_value_overload<T, value_ret_t<T>>::value);
  STATIC_REQUIRE(!cub::detail::for_each::has_unique_value_overload<T, rref_t<T>>::value);
  STATIC_REQUIRE(!cub::detail::for_each::has_unique_value_overload<T, ref_t<T>>::value);
  STATIC_REQUIRE(!cub::detail::for_each::has_unique_value_overload<T, const_ref_t<T>>::value);
  STATIC_REQUIRE(!cub::detail::for_each::has_unique_value_overload<T, overload_value_t<T>>::value);
  STATIC_REQUIRE(!cub::detail::for_each::has_unique_value_overload<T, tpl_value_t>::value);
}

C2H_TEST("Device for utils correctly detect value overloads", "[for][device]")
{
  test<int>();
  test<double>();

  // conversions do not work ;(
  STATIC_REQUIRE(cub::detail::for_each::has_unique_value_overload<char, value_t<int>>::value);
}
