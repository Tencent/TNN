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

#include <cub/device/device_radix_sort.cuh>

#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cuda/std/tuple>

#include <bitset>
#include <cstdint>
#include <functional>
#include <limits>
#include <type_traits>

#include "cub/block/radix_rank_sort_operations.cuh"

struct custom_t
{
  std::uint16_t i;
  float f;
};

struct decomposer_t
{
  __host__ __device__ //
    ::cuda::std::tuple<std::uint16_t&, float&>
    operator()(custom_t& key) const
  {
    return {key.i, key.f};
  }
};

std::bitset<64> to_binary_representation(custom_t value)
{
  std::uint64_t bits{};
  memcpy(&bits, &value, sizeof(custom_t));
  return std::bitset<64>{bits};
}

int main()
{
  std::cout << "This example illustrates use of radix sort with custom type.\n";
  std::cout << "Let's define a simple structure of the following form:\n\n";
  std::cout << "\tstruct custom_t {\n";
  std::cout << "\t  std::uint32_t i;\n";
  std::cout << "\t  float f;\n";
  std::cout << "\t};\n\n";
  std::cout << "The `i` field is already stored in the bit-lexicographical order.\n";
  std::cout << "The `f` field, however, isn't. Therefore, to feed this structure \n";
  std::cout << "into the radix sort, we have to convert `f` into bit ordered representation.\n";
  std::cout << "The `custom_t{65535, -4.2f}` has the following binary representation:\n\n";

  auto print_segment = [](std::string msg, std::size_t segment_size, char filler = '-') {
    std::string spaces((segment_size - msg.size()) / 2 - 1, filler);
    std::cout << '<' << spaces << msg << spaces << '>';
  };

  std::cout << '\t';
  print_segment(" `.f` ", 32);
  print_segment(" padding -", 16);
  print_segment(" `.s` ", 16);
  std::cout << '\n';

  std::cout << "\ts";
  print_segment(" exp. ", 8);
  print_segment(" mantissa -", 23);
  print_segment(" padding -", 16);
  print_segment(" short -", 16);
  std::cout << '\n';

  custom_t the_answer{65535, -4.2f};
  std::cout << '\t' << to_binary_representation(the_answer);
  std::cout << "\n\t";
  print_segment(" <----  higher bits  /  lower bits  ----> ", 64, ' ');
  std::cout << "\n\n";

  std::cout << "Let's say we are trying to compare l={42, -4.2f} with g={42, 4.2f}:\n";

  std::cout << "\n\t";
  print_segment(" `.f` ", 32);
  print_segment(" padding -", 16);
  print_segment(" `.s` ", 16);
  std::cout << '\n';

  custom_t l{42, -4.2f};
  custom_t g{42, 4.2f};
  std::cout << "l:\t" << to_binary_representation(l) << '\n';
  std::cout << "g:\t" << to_binary_representation(g) << "\n\n";

  std::cout << "As you can see, `l` key happened to be larger in the bit-lexicographicl order.\n";
  std::cout << "Since there's no reflection in C++, we can't inspect the type and convert \n";
  std::cout << "each field into the bit-lexicographicl order. You can tell CUB how to do that\n";
  std::cout << "by specializing cub::RadixTraits for the `custom_t`:\n\n";

  std::cout << "\tstruct decomposer_t \n";
  std::cout << "\t{\n";
  std::cout << "\t  __host__ __device__ \n";
  std::cout << "\t  ::cuda::std::tuple<std::uint16_t&, float&> operator()(custom_t &key) const \n";
  std::cout << "\t  {\n";
  std::cout << "\t    return {key.i, key.f};\n";
  std::cout << "\t  }\n";
  std::cout << "\t};\n";
  std::cout << "\n";

  std::cout << "Decomposer allows you to specify which fields are most significant and which\n";
  std::cout << "are least significant. In our case, `f` is the most significant field and\n";
  std::cout << "`i` is the least significant field. The decomposer is then used by CUB to convert\n";
  std::cout << "the `custom_t` into the bit-lexicographicl order:\n\n";

  using conversion_policy = cub::detail::radix::traits_t<custom_t>::bit_ordered_conversion_policy;
  l                       = conversion_policy::to_bit_ordered(decomposer_t{}, l);
  g                       = conversion_policy::to_bit_ordered(decomposer_t{}, g);

  std::cout << "\n\t";
  print_segment(" `.f` ", 32);
  print_segment(" padding -", 16);
  print_segment(" `.s` ", 16);
  std::cout << '\n';

  std::cout << "l:\t" << to_binary_representation(l) << '\n';
  std::cout << "g:\t" << to_binary_representation(g) << "\n\n";

  std::cout << '\n';
  std::cout << "As you can see, `g` is now actually larger than `l` in the bit-lexicographicl order.\n";
  std::cout << "After binning, CUB is able to restore the original key:\n\n";

  l = conversion_policy::from_bit_ordered(decomposer_t{}, l);
  g = conversion_policy::from_bit_ordered(decomposer_t{}, g);

  std::cout << "\n\t";
  print_segment(" `.f` ", 32);
  print_segment(" padding -", 16);
  print_segment(" `.s` ", 16);
  std::cout << '\n';

  std::cout << "l:\t" << to_binary_representation(l) << '\n';
  std::cout << "g:\t" << to_binary_representation(g) << "\n\n";

  using inversion_policy = cub::detail::radix::traits_t<custom_t>::bit_ordered_inversion_policy;
  std::cout << '\n';
  std::cout << "We are also able to inverse differentiating bits:\n";

  l = inversion_policy::inverse(decomposer_t{}, l);
  g = inversion_policy::inverse(decomposer_t{}, g);

  std::cout << "\n\t";
  print_segment(" `.f` ", 32);
  print_segment(" padding -", 16);
  print_segment(" `.s` ", 16);
  std::cout << '\n';

  std::cout << "l:\t" << to_binary_representation(l) << '\n';
  std::cout << "g:\t" << to_binary_representation(g) << "\n\n";

  std::cout << '\n';
  std::cout << "We as well can compute the minimal and minimal / maximal keys:\n";

  l = cub::detail::radix::traits_t<custom_t>::min_raw_binary_key(decomposer_t{});
  g = cub::detail::radix::traits_t<custom_t>::max_raw_binary_key(decomposer_t{});

  std::cout << "\n\t";
  print_segment(" `.f` ", 32);
  print_segment(" padding -", 16);
  print_segment(" `.s` ", 16);
  std::cout << '\n';

  std::cout << "l:\t" << to_binary_representation(l) << '\n';
  std::cout << "g:\t" << to_binary_representation(g) << "\n\n";

  std::cout << "We can even compute the number of differentiating bits:\n\n";

  std::cout << "end:\t";
  std::cout << cub::detail::radix::traits_t<custom_t>::default_end_bit(decomposer_t{});
  std::cout << '\n';
  std::cout << "size:\t";
  std::cout << sizeof(custom_t) * CHAR_BIT;
  std::cout << "\n\n";

  std::cout << "All of these operations are used behind the scenes by CUB to sort custom types:\n\n";

  constexpr int num_items            = 6;
  thrust::device_vector<custom_t> in = {{4, +2.5f}, {0, -2.5f}, {3, +1.1f}, {1, +0.0f}, {2, -0.0f}, {5, +3.7f}};

  std::cout << "in:\n";
  for (custom_t key : in)
  {
    std::cout << "\t{.i = " << key.i << ", .f = " << key.f << "},\n";
  }

  thrust::device_vector<custom_t> out(num_items);

  const custom_t* d_in = thrust::raw_pointer_cast(in.data());
  custom_t* d_out      = thrust::raw_pointer_cast(out.data());

  // 1) Get temp storage size
  std::uint8_t* d_temp_storage{};
  std::size_t temp_storage_bytes{};

  cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, decomposer_t{});

  // 2) Allocate temp storage
  thrust::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
  d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

  // 3) Sort keys
  cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, decomposer_t{});
  cudaDeviceSynchronize();

  std::cout << "\n";
  std::cout << "sort:\n";
  std::cout << "\n";

  std::cout << "\tcub::DeviceRadixSort::SortKeys(d_temp_storage,\n";
  std::cout << "\t                               temp_storage_bytes,\n";
  std::cout << "\t                               d_in,\n";
  std::cout << "\t                               d_out,\n";
  std::cout << "\t                               num_items,\n";
  std::cout << "\t                               decomposer_t{});\n\n";

  std::cout << "out:\n";
  for (custom_t key : out)
  {
    std::cout << "\t{.i = " << key.i << ", .f = " << key.f << "},\n";
  }

  std::cout << '\n';
  std::cout << "If you have any issues with radix sort support of custom types, \n";
  std::cout << "please feel free to use this example to identify the problem.\n\n";
}
