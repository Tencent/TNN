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

#include <cub/device/device_for.cuh>

#include <nvbench_helper.cuh>

template <class T>
struct op_t
{
  int* d_count{};

  __device__ void operator()(T val) const
  {
    if (val == T{})
    {
      atomicAdd(d_count, 1);
    }
  }
};

template <class T, class OffsetT>
void for_each(nvbench::state& state, nvbench::type_list<T, OffsetT>)
{
  using input_it_t  = const T*;
  using output_it_t = int*;
  using offset_t    = OffsetT;

  const auto elements = static_cast<offset_t>(state.get_int64("Elements{io}"));

  thrust::device_vector<T> in(elements, T{42});

  input_it_t d_in   = thrust::raw_pointer_cast(in.data());
  output_it_t d_out = nullptr;

  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements);

  op_t<T> op{d_out};

  std::size_t temp_size{};
  cub::DeviceFor::ForEachCopyN(nullptr, temp_size, d_in, elements, op);

  thrust::device_vector<nvbench::uint8_t> temp(temp_size);
  auto* temp_storage = thrust::raw_pointer_cast(temp.data());

  state.exec(nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    cub::DeviceFor::ForEachCopyN(temp_storage, temp_size, d_in, elements, op, launch.get_stream());
  });
}

NVBENCH_BENCH_TYPES(for_each, NVBENCH_TYPE_AXES(fundamental_types, offset_types))
  .set_name("base")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4));
