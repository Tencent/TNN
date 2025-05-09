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

// %PARAM% TEST_ERR err 0:1:2:3:4:5

#include <cub/device/device_segmented_reduce.cuh>

int main()
{
  using offset_t = float; // error
  // using offset_t = int; // ok
  float *d_in{}, *d_out{};
  offset_t* d_offsets{};
  std::size_t temp_storage_bytes{};
  std::uint8_t* d_temp_storage{};

#if TEST_ERR == 0
  // expected-error {{"Offset iterator type should be integral."}}
  cub::DeviceSegmentedReduce::Reduce(
    d_temp_storage, temp_storage_bytes, d_in, d_out, 0, d_offsets, d_offsets + 1, cub::Min(), 0);

#elif TEST_ERR == 1
  // expected-error {{"Offset iterator type should be integral."}}
  cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, 0, d_offsets, d_offsets + 1);

#elif TEST_ERR == 2
  // expected-error {{"Offset iterator type should be integral."}}
  cub::DeviceSegmentedReduce::Min(d_temp_storage, temp_storage_bytes, d_in, d_out, 0, d_offsets, d_offsets + 1);

#elif TEST_ERR == 3
  // expected-error {{"Offset iterator type should be integral."}}
  cub::DeviceSegmentedReduce::ArgMin(d_temp_storage, temp_storage_bytes, d_in, d_out, 0, d_offsets, d_offsets + 1);

#elif TEST_ERR == 4
  // expected-error {{"Offset iterator type should be integral."}}
  cub::DeviceSegmentedReduce::Max(d_temp_storage, temp_storage_bytes, d_in, d_out, 0, d_offsets, d_offsets + 1);

#elif TEST_ERR == 5
  // expected-error {{"Offset iterator type should be integral."}}
  cub::DeviceSegmentedReduce::ArgMax(d_temp_storage, temp_storage_bytes, d_in, d_out, 0, d_offsets, d_offsets + 1);
#endif
}
