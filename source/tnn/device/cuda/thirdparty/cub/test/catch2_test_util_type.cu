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

#include <cub/iterator/counting_input_iterator.cuh>
#include <cub/iterator/discard_output_iterator.cuh>
#include <cub/util_type.cuh>

#include <cuda/std/type_traits>

#include <c2h/catch2_test_helper.cuh>

C2H_TEST("Tests non_void_value_t", "[util][type]")
{
  using fallback_t        = float;
  using void_fancy_it     = cub::DiscardOutputIterator<std::size_t>;
  using non_void_fancy_it = cub::CountingInputIterator<int>;

  // falls back for const void*
  STATIC_REQUIRE(::cuda::std::is_same<fallback_t, //
                                      cub::detail::non_void_value_t<const void*, fallback_t>>::value);
  // falls back for const volatile void*
  STATIC_REQUIRE(::cuda::std::is_same<fallback_t, //
                                      cub::detail::non_void_value_t<const volatile void*, fallback_t>>::value);
  // falls back for volatile void*
  STATIC_REQUIRE(::cuda::std::is_same<fallback_t, //
                                      cub::detail::non_void_value_t<volatile void*, fallback_t>>::value);
  // falls back for void*
  STATIC_REQUIRE(::cuda::std::is_same<fallback_t, //
                                      cub::detail::non_void_value_t<void*, fallback_t>>::value);
  // works for int*
  STATIC_REQUIRE(::cuda::std::is_same<int, //
                                      cub::detail::non_void_value_t<int*, void>>::value);
  // falls back for fancy iterator with a void value type
  STATIC_REQUIRE(::cuda::std::is_same<fallback_t, //
                                      cub::detail::non_void_value_t<void_fancy_it, fallback_t>>::value);
  // works for a fancy iterator that has int as value type
  STATIC_REQUIRE(::cuda::std::is_same<int, //
                                      cub::detail::non_void_value_t<non_void_fancy_it, fallback_t>>::value);
}

CUB_DEFINE_DETECT_NESTED_TYPE(cat_detect, cat);

struct HasCat
{
  using cat = int;
};
struct HasDog
{
  using dog = int;
};

C2H_TEST("Test CUB_DEFINE_DETECT_NESTED_TYPE", "[util][type]")
{
  STATIC_REQUIRE(cat_detect<HasCat>::value);
  STATIC_REQUIRE(!cat_detect<HasDog>::value);
}
