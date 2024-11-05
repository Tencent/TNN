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

#include <cub/util_math.cuh>

#include <cuda/std/type_traits>

#include <c2h/catch2_test_helper.cuh>

C2H_TEST("Tests safe_add_bound_to_max", "[util][math]")
{
  REQUIRE(cub::detail::safe_add_bound_to_max(0U, ::cuda::std::numeric_limits<std::uint32_t>::max())
          == ::cuda::std::numeric_limits<std::uint32_t>::max());
  REQUIRE(cub::detail::safe_add_bound_to_max(::cuda::std::numeric_limits<std::uint32_t>::max(), 0U)
          == ::cuda::std::numeric_limits<std::uint32_t>::max());

  // We do not overflow
  REQUIRE(cub::detail::safe_add_bound_to_max(std::int32_t{0}, ::cuda::std::numeric_limits<std::int32_t>::max())
          == ::cuda::std::numeric_limits<std::int32_t>::max());
  REQUIRE(cub::detail::safe_add_bound_to_max(::cuda::std::numeric_limits<std::int32_t>::max(), std::int32_t{0})
          == ::cuda::std::numeric_limits<std::int32_t>::max());
  REQUIRE(cub::detail::safe_add_bound_to_max(std::int32_t{1}, ::cuda::std::numeric_limits<std::int32_t>::max())
          == ::cuda::std::numeric_limits<std::int32_t>::max());
  REQUIRE(cub::detail::safe_add_bound_to_max(::cuda::std::numeric_limits<std::int32_t>::max(), std::int32_t{1})
          == ::cuda::std::numeric_limits<std::int32_t>::max());
  REQUIRE(cub::detail::safe_add_bound_to_max(
            ::cuda::std::numeric_limits<std::int32_t>::max(), ::cuda::std::numeric_limits<std::int32_t>::max())
          == ::cuda::std::numeric_limits<std::int32_t>::max());

  // We do not overflow
  REQUIRE(cub::detail::safe_add_bound_to_max(std::int64_t{0}, ::cuda::std::numeric_limits<std::int64_t>::max())
          == ::cuda::std::numeric_limits<std::int64_t>::max());
  REQUIRE(cub::detail::safe_add_bound_to_max(::cuda::std::numeric_limits<std::int64_t>::max(), std::int64_t{0LL})
          == ::cuda::std::numeric_limits<std::int64_t>::max());
  REQUIRE(cub::detail::safe_add_bound_to_max(std::int64_t{1LL}, ::cuda::std::numeric_limits<std::int64_t>::max())
          == ::cuda::std::numeric_limits<std::int64_t>::max());
  REQUIRE(cub::detail::safe_add_bound_to_max(::cuda::std::numeric_limits<std::int64_t>::max(), std::int64_t{1LL})
          == ::cuda::std::numeric_limits<std::int64_t>::max());
  REQUIRE(cub::detail::safe_add_bound_to_max(
            ::cuda::std::numeric_limits<std::int64_t>::max(), ::cuda::std::numeric_limits<std::int64_t>::max())
          == ::cuda::std::numeric_limits<std::int64_t>::max());

  // We do not underflow for negative rhs (not, lhs must not be negative per documentation)
  REQUIRE(cub::detail::safe_add_bound_to_max(0, -1) == -1);
  REQUIRE(cub::detail::safe_add_bound_to_max(1, -1) == 0);
}
