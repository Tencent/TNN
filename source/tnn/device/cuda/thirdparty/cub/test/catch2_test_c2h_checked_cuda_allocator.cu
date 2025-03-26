/******************************************************************************
 * Copyright (c) 2011-2024, NVIDIA CORPORATION.  All rights reserved.
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

#include <thrust/execution_policy.h>

#include <algorithm>
#include <new> // std::bad_alloc

#include "thrust/detail/execution_policy.h"
#include <c2h/catch2_test_helper.cuh>

std::size_t get_alloc_bytes()
{
  std::size_t free_bytes{};
  std::size_t total_bytes{};
  cudaError_t status = cudaMemGetInfo(&free_bytes, &total_bytes);
  REQUIRE(status == cudaSuccess);

  // Find a size that's > free but < total, preferring to return more than total if the values are
  // too close.
  constexpr std::size_t one_MiB = 1024 * 1024;
  const std::size_t alloc_bytes = ::std::max(total_bytes - one_MiB, free_bytes + one_MiB);
  CAPTURE(free_bytes, total_bytes, alloc_bytes);
  return alloc_bytes;
}

C2H_TEST("c2h::device_vector throws when requested allocations exceed free device memory",
         "[c2h][checked_cuda_allocator][device_vector]")
{
  c2h::device_vector<char> vec;

  const std::size_t alloc_bytes = get_alloc_bytes();
  REQUIRE_THROWS_AS(vec.resize(alloc_bytes), std::bad_alloc);
}

C2H_TEST("c2h::device_policy throws when requested allocations exceed free device memory",
         "[c2h][checked_cuda_allocator][device_policy]")
{
  thrust::pair<char*, std::ptrdiff_t> buffer{nullptr, 0};
  auto policy = thrust::detail::derived_cast(thrust::detail::strip_const(c2h::device_policy));

  const std::size_t alloc_bytes = get_alloc_bytes();
  REQUIRE_THROWS_AS(
    buffer = thrust::detail::get_temporary_buffer<char>(policy, static_cast<std::ptrdiff_t>(alloc_bytes)),
    std::bad_alloc);

  thrust::detail::return_temporary_buffer(policy, buffer.first, buffer.second);
}
