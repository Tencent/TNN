/******************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

#include <cub/detail/choose_offset.cuh>

#include <cuda/std/cstdint>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

#include <c2h/catch2_test_helper.cuh>

C2H_TEST("Tests choose_offset", "[util][type]")
{
  // Uses unsigned 32-bit type for signed 32-bit type
  STATIC_REQUIRE(::cuda::std::is_same<cub::detail::choose_offset_t<std::int32_t>, std::uint32_t>::value);

  // Uses unsigned 32-bit type for type smaller than 32 bits
  STATIC_REQUIRE(::cuda::std::is_same<cub::detail::choose_offset_t<std::int8_t>, std::uint32_t>::value);

  // Uses unsigned 64-bit type for signed 64-bit type
  STATIC_REQUIRE(::cuda::std::is_same<cub::detail::choose_offset_t<std::int64_t>, unsigned long long>::value);
}

C2H_TEST("Tests choose_signed_offset", "[util][type]")
{
  // Uses signed 64-bit type for unsigned signed 32-bit type
  STATIC_REQUIRE(::cuda::std::is_same<cub::detail::choose_signed_offset_t<std::uint32_t>, std::int64_t>::value);

  // Uses signed 32-bit type for signed 32-bit type
  STATIC_REQUIRE(::cuda::std::is_same<cub::detail::choose_signed_offset_t<std::int32_t>, std::int32_t>::value);

  // Uses signed 32-bit type for type smaller than 32 bits
  STATIC_REQUIRE(::cuda::std::is_same<cub::detail::choose_signed_offset_t<std::int8_t>, std::int32_t>::value);

  // Uses signed 64-bit type for signed 64-bit type
  STATIC_REQUIRE(::cuda::std::is_same<cub::detail::choose_signed_offset_t<std::int64_t>, std::int64_t>::value);

  // Offset type covers maximum number representable by a signed 32-bit integer
  REQUIRE(cudaSuccess
          == cub::detail::choose_signed_offset<std::int32_t>::is_exceeding_offset_type(
            ::cuda::std::numeric_limits<std::int32_t>::max()));

  // Offset type covers maximum number representable by a signed 64-bit integer
  REQUIRE(cudaSuccess
          == cub::detail::choose_signed_offset<std::int64_t>::is_exceeding_offset_type(
            ::cuda::std::numeric_limits<std::int64_t>::max()));

  // Offset type does not support maximum number representable by an unsigned 64-bit integer
  REQUIRE(cudaErrorInvalidValue
          == cub::detail::choose_signed_offset<std::uint64_t>::is_exceeding_offset_type(
            ::cuda::std::numeric_limits<std::uint64_t>::max()));
}

C2H_TEST("Tests promote_small_offset", "[util][type]")
{
  // Uses input type for types of at least 32 bits
  STATIC_REQUIRE(::cuda::std::is_same<typename cub::detail::promote_small_offset_t<std::int32_t>, std::int32_t>::value);

  // Uses input type for types of at least 32 bits
  STATIC_REQUIRE(
    ::cuda::std::is_same<typename cub::detail::promote_small_offset_t<std::uint32_t>, std::uint32_t>::value);

  // Uses input type for types of at least 32 bits
  STATIC_REQUIRE(
    ::cuda::std::is_same<typename cub::detail::promote_small_offset_t<std::uint64_t>, std::uint64_t>::value);

  // Uses input type for types of at least 32 bits
  STATIC_REQUIRE(::cuda::std::is_same<typename cub::detail::promote_small_offset_t<std::int64_t>, std::int64_t>::value);

  // Uses 32-bit type for type smaller than 32 bits
  STATIC_REQUIRE(::cuda::std::is_same<typename cub::detail::promote_small_offset_t<std::int8_t>, std::int32_t>::value);

  // Uses 32-bit type for type smaller than 32 bits
  STATIC_REQUIRE(::cuda::std::is_same<typename cub::detail::promote_small_offset_t<std::int16_t>, std::int32_t>::value);

  // Uses 32-bit type for type smaller than 32 bits
  STATIC_REQUIRE(
    ::cuda::std::is_same<typename cub::detail::promote_small_offset_t<std::uint16_t>, std::int32_t>::value);
}
