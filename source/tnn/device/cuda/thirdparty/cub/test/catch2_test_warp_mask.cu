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

#include <cub/util_ptx.cuh>

#include <c2h/catch2_test_helper.cuh>

template <int logical_warp_threads>
struct total_warps_t
{
private:
  static constexpr unsigned int total_warps =
    (cub::PowerOfTwo<logical_warp_threads>::VALUE) ? CUB_WARP_THREADS(0) / logical_warp_threads : 1;

public:
  static constexpr unsigned int value()
  {
    return total_warps;
  }
};

bool is_lane_involved(unsigned int member_mask, unsigned int lane)
{
  return member_mask & (1 << lane);
}

using logical_warp_threads      = c2h::iota<1, 32>;
using power_of_two_warp_threads = c2h::enum_type_list<int, 1, 2, 4, 8, 16, 32>;

C2H_TEST("Warp mask ignores lanes before current logical warp", "[mask][warp]", power_of_two_warp_threads)
{
  constexpr int logical_warp_thread  = c2h::get<0, TestType>::value;
  constexpr unsigned int total_warps = total_warps_t<logical_warp_thread>::value();

  for (unsigned int warp_id = 0; warp_id < total_warps; warp_id++)
  {
    const unsigned int warp_mask  = cub::WarpMask<logical_warp_thread>(warp_id);
    const unsigned int warp_begin = logical_warp_thread * warp_id;

    for (unsigned int prev_warp_lane = 0; prev_warp_lane < warp_begin; prev_warp_lane++)
    {
      REQUIRE_FALSE(is_lane_involved(warp_mask, prev_warp_lane));
    }
  }
}

C2H_TEST("Warp mask involves lanes of current logical warp", "[mask][warp]", logical_warp_threads)
{
  constexpr int logical_warp_thread  = c2h::get<0, TestType>::value;
  constexpr unsigned int total_warps = total_warps_t<logical_warp_thread>::value();

  for (unsigned int warp_id = 0; warp_id < total_warps; warp_id++)
  {
    const unsigned int warp_mask  = cub::WarpMask<logical_warp_thread>(warp_id);
    const unsigned int warp_begin = logical_warp_thread * warp_id;
    const unsigned int warp_end   = warp_begin + logical_warp_thread;

    for (unsigned int warp_lane = warp_begin; warp_lane < warp_end; warp_lane++)
    {
      REQUIRE(is_lane_involved(warp_mask, warp_lane));
    }
  }
}

C2H_TEST("Warp mask ignores lanes after current logical warp", "[mask][warp]", logical_warp_threads)
{
  constexpr int logical_warp_thread  = c2h::get<0, TestType>::value;
  constexpr unsigned int total_warps = total_warps_t<logical_warp_thread>::value();

  for (unsigned int warp_id = 0; warp_id < total_warps; warp_id++)
  {
    const unsigned int warp_mask  = cub::WarpMask<logical_warp_thread>(warp_id);
    const unsigned int warp_begin = logical_warp_thread * warp_id;
    const unsigned int warp_end   = warp_begin + logical_warp_thread;

    for (unsigned int post_warp_lane = warp_end; post_warp_lane < CUB_WARP_THREADS(0); post_warp_lane++)
    {
      REQUIRE_FALSE(is_lane_involved(warp_mask, post_warp_lane));
    }
  }
}
