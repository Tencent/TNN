/******************************************************************************
 * Copyright (c) 2011-2023, NVIDIA CORPORATION.  All rights reserved.
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

#if !TUNE_BASE
#  include <cub/agent/single_pass_scan_operators.cuh>

#  include <nvbench_helper.cuh>

#  if !defined(TUNE_MAGIC_NS) || !defined(TUNE_L2_WRITE_LATENCY_NS) || !defined(TUNE_DELAY_CONSTRUCTOR_ID)
#    error "TUNE_MAGIC_NS, TUNE_L2_WRITE_LATENCY_NS, and TUNE_DELAY_CONSTRUCTOR_ID must be defined"
#  endif

using delay_constructors = nvbench::type_list<
  cub::detail::no_delay_constructor_t<TUNE_L2_WRITE_LATENCY_NS>,
  cub::detail::fixed_delay_constructor_t<TUNE_MAGIC_NS, TUNE_L2_WRITE_LATENCY_NS>,
  cub::detail::exponential_backoff_constructor_t<TUNE_MAGIC_NS, TUNE_L2_WRITE_LATENCY_NS>,
  cub::detail::exponential_backoff_jitter_constructor_t<TUNE_MAGIC_NS, TUNE_L2_WRITE_LATENCY_NS>,
  cub::detail::exponential_backoff_jitter_window_constructor_t<TUNE_MAGIC_NS, TUNE_L2_WRITE_LATENCY_NS>,
  cub::detail::exponential_backon_jitter_window_constructor_t<TUNE_MAGIC_NS, TUNE_L2_WRITE_LATENCY_NS>,
  cub::detail::exponential_backon_jitter_constructor_t<TUNE_MAGIC_NS, TUNE_L2_WRITE_LATENCY_NS>,
  cub::detail::exponential_backon_constructor_t<TUNE_MAGIC_NS, TUNE_L2_WRITE_LATENCY_NS>>;

using delay_constructor_t = nvbench::tl::get<TUNE_DELAY_CONSTRUCTOR_ID, delay_constructors>;
#endif // !TUNE_BASE
