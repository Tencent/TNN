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

#undef NDEBUG
#include <algorithm>
#include <cassert>
#include <type_traits>
#include <utility>

#include "catch2_test_block_radix_sort.cuh"
#include "cub/block/radix_rank_sort_operations.cuh"

// example-begin custom-type
struct custom_t
{
  float f;
  int unused;
  long long int lli;

  custom_t() = default;
  __device__ custom_t(float f, long long int lli)
      : f(f)
      , unused(42)
      , lli(lli)
  {}
};

static __device__ bool operator==(const custom_t& lhs, const custom_t& rhs)
{
  return lhs.f == rhs.f && lhs.lli == rhs.lli;
}

struct decomposer_t
{
  __device__ ::cuda::std::tuple<float&, long long int&> //
  operator()(custom_t & key) const
  {
    return {key.f, key.lli};
  }
};
// example-end custom-type

__global__ void sort_keys()
{
  // example-begin keys
  // Specialize `cub::BlockRadixSort` for a 1D block of 2 threads owning 3 keys each
  using block_radix_sort_t = cub::BlockRadixSort<custom_t, 2, 3>;

  // Allocate shared memory for `cub::BlockRadixSort`
  __shared__ block_radix_sort_t::TempStorage temp_storage;

  // Obtain a segment of consecutive items that are blocked across threads
  custom_t thread_keys[2][3] = //
    {{
       // thread 0 keys
       {+2.5, 4}, //
       {-2.5, 0}, //
       {+1.1, 3}, //
     },
     {
       // thread 1 keys
       {+0.0, 1}, //
       {-0.0, 2}, //
       {+3.7, 5} //
     }};

  // Collectively sort the keys
  block_radix_sort_t(temp_storage).Sort(thread_keys[threadIdx.x], decomposer_t{});

  custom_t expected_output[2][3] = //
    {{
       // thread 0 expected keys
       {-2.5, 0}, //
       {+0.0, 1}, //
       {-0.0, 2} //
     },
     {
       // thread 1 expected keys
       {+1.1, 3}, //
       {+2.5, 4}, //
       {+3.7, 5} //
     }};
  // example-end keys

  assert(thread_keys[threadIdx.x][0] == expected_output[threadIdx.x][0]);
  assert(thread_keys[threadIdx.x][1] == expected_output[threadIdx.x][1]);
  assert(thread_keys[threadIdx.x][2] == expected_output[threadIdx.x][2]);
}

__global__ void sort_keys_bits()
{
  // example-begin keys-bits
  // Specialize `cub::BlockRadixSort` for a 1D block of 2 threads owning 1 key each
  using block_radix_sort_t = cub::BlockRadixSort<custom_t, 2, 1>;

  // Allocate shared memory for `cub::BlockRadixSort`
  __shared__ block_radix_sort_t::TempStorage temp_storage;

  // Obtain a segment of consecutive items that are blocked across threads
  custom_t thread_keys[2][1] = //
    {{
       {24.2, 1ll << 61} // thread 0 keys
     },
     {
       {42.4, 1ll << 60} // thread 1 keys
     }};

  constexpr int begin_bit = sizeof(long long int) * 8 - 4; // 60
  constexpr int end_bit   = sizeof(long long int) * 8 + 4; // 68

  // Decomposition orders the bits as follows:
  //
  //                    <------------- fp32 -----------> <------ int64 ------>
  // decompose(in[0]) = 01000001110000011001100110011010 00100000000000...0000
  // decompose(in[1]) = 01000010001010011001100110011010 00010000000000...0000
  //                    <-----------  higher bits  /  lower bits  ----------->
  //
  // The bit subrange `[60, 68)` specifies differentiating key bits:
  //
  //                    <------------- fp32 -----------> <------ int64 ------>
  // decompose(in[0]) = xxxxxxxxxxxxxxxxxxxxxxxxxxxx1010 0010xxxxxxxxxx...xxxx
  // decompose(in[1]) = xxxxxxxxxxxxxxxxxxxxxxxxxxxx1010 0001xxxxxxxxxx...xxxx
  //                    <-----------  higher bits  /  lower bits  ----------->

  // Collectively sort the keys
  block_radix_sort_t(temp_storage).Sort(thread_keys[threadIdx.x], decomposer_t{}, begin_bit, end_bit);

  custom_t expected_output[2][3] = //
    {{
       {42.4, 1ll << 60}, // thread 0 expected keys
     },
     {
       {24.2, 1ll << 61} // thread 1 expected keys
     }};
  // example-end keys-bits

  assert(thread_keys[threadIdx.x][0] == expected_output[threadIdx.x][0]);
}

__global__ void sort_keys_descending()
{
  // example-begin keys-descending
  // Specialize `cub::BlockRadixSort` for a 1D block of 2 threads owning 3 keys each
  using block_radix_sort_t = cub::BlockRadixSort<custom_t, 2, 3>;

  // Allocate shared memory for `cub::BlockRadixSort`
  __shared__ block_radix_sort_t::TempStorage temp_storage;

  // Obtain a segment of consecutive items that are blocked across threads
  custom_t thread_keys[2][3] = //
    {{
       // thread 0 keys
       {+1.1, 2}, //
       {+2.5, 1}, //
       {-0.0, 4}, //
     },
     {
       // thread 1 keys
       {+0.0, 3}, //
       {-2.5, 5}, //
       {+3.7, 0} //
     }};

  // Collectively sort the keys
  block_radix_sort_t(temp_storage).SortDescending(thread_keys[threadIdx.x], decomposer_t{});

  custom_t expected_output[2][3] = //
    {{
       // thread 0 expected keys
       {+3.7, 0}, //
       {+2.5, 1}, //
       {+1.1, 2}, //
     },
     {
       // thread 1 expected keys
       {-0.0, 4}, //
       {+0.0, 3}, //
       {-2.5, 5} //
     }};
  // example-end keys-descending

  assert(thread_keys[threadIdx.x][0] == expected_output[threadIdx.x][0]);
  assert(thread_keys[threadIdx.x][1] == expected_output[threadIdx.x][1]);
  assert(thread_keys[threadIdx.x][2] == expected_output[threadIdx.x][2]);
}

__global__ void sort_keys_descending_bits()
{
  // example-begin keys-descending-bits
  // Specialize `cub::BlockRadixSort` for a 1D block of 2 threads owning 1 key each
  using block_radix_sort_t = cub::BlockRadixSort<custom_t, 2, 1>;

  // Allocate shared memory for `cub::BlockRadixSort`
  __shared__ block_radix_sort_t::TempStorage temp_storage;

  // Obtain a segment of consecutive items that are blocked across threads
  custom_t thread_keys[2][1] = //
    {{
       {42.4, 1ll << 60} // thread 0 keys
     },
     {
       {24.2, 1ll << 61} // thread 1 keys
     }};

  constexpr int begin_bit = sizeof(long long int) * 8 - 4; // 60
  constexpr int end_bit   = sizeof(long long int) * 8 + 4; // 68

  // Decomposition orders the bits as follows:
  //
  //                    <------------- fp32 -----------> <------ int64 ------>
  // decompose(in[0]) = 01000010001010011001100110011010 00010000000000...0000
  // decompose(in[1]) = 01000001110000011001100110011010 00100000000000...0000
  //                    <-----------  higher bits  /  lower bits  ----------->
  //
  // The bit subrange `[60, 68)` specifies differentiating key bits:
  //
  //                    <------------- fp32 -----------> <------ int64 ------>
  // decompose(in[0]) = xxxxxxxxxxxxxxxxxxxxxxxxxxxx1010 0001xxxxxxxxxx...xxxx
  // decompose(in[1]) = xxxxxxxxxxxxxxxxxxxxxxxxxxxx1010 0010xxxxxxxxxx...xxxx
  //                    <-----------  higher bits  /  lower bits  ----------->

  // Collectively sort the keys
  block_radix_sort_t(temp_storage).SortDescending(thread_keys[threadIdx.x], decomposer_t{}, begin_bit, end_bit);

  custom_t expected_output[2][3] = //
    {{
       {24.2, 1ll << 61}, // thread 0 expected keys
     },
     {
       {42.4, 1ll << 60} // thread 1 expected keys
     }};
  // example-end keys-descending-bits

  assert(thread_keys[threadIdx.x][0] == expected_output[threadIdx.x][0]);
}

__global__ void sort_pairs()
{
  // example-begin pairs
  // Specialize `cub::BlockRadixSort` for a 1D block of 2 threads owning 3 keys and values each
  using block_radix_sort_t = cub::BlockRadixSort<custom_t, 2, 3, int>;

  // Allocate shared memory for `cub::BlockRadixSort`
  __shared__ block_radix_sort_t::TempStorage temp_storage;

  // Obtain a segment of consecutive items that are blocked across threads
  custom_t thread_keys[2][3] = //
    {{
       // thread 0 keys
       {+2.5, 4}, //
       {-2.5, 0}, //
       {+1.1, 3}, //
     },
     {
       // thread 1 keys
       {+0.0, 1}, //
       {-0.0, 2}, //
       {+3.7, 5} //
     }};

  int thread_values[2][3] = //
    {{4, 0, 3}, // thread 0 values
     {1, 2, 5}}; // thread 1 values

  // Collectively sort the keys
  block_radix_sort_t(temp_storage).Sort(thread_keys[threadIdx.x], thread_values[threadIdx.x], decomposer_t{});

  custom_t expected_keys[2][3] = //
    {{
       // thread 0 expected keys
       {-2.5, 0}, //
       {+0.0, 1}, //
       {-0.0, 2} //
     },
     {
       // thread 1 expected keys
       {+1.1, 3}, //
       {+2.5, 4}, //
       {+3.7, 5} //
     }};

  int expected_values[2][3] = //
    {{0, 1, 2}, // thread 0 expected values
     {3, 4, 5}}; // thread 1 expected values
  // example-end pairs

  assert(thread_keys[threadIdx.x][0] == expected_keys[threadIdx.x][0]);
  assert(thread_keys[threadIdx.x][1] == expected_keys[threadIdx.x][1]);
  assert(thread_keys[threadIdx.x][2] == expected_keys[threadIdx.x][2]);

  assert(thread_values[threadIdx.x][0] == expected_values[threadIdx.x][0]);
  assert(thread_values[threadIdx.x][1] == expected_values[threadIdx.x][1]);
  assert(thread_values[threadIdx.x][2] == expected_values[threadIdx.x][2]);
}

__global__ void sort_pairs_bits()
{
  // example-begin pairs-bits
  // Specialize `cub::BlockRadixSort` for a 1D block of 2 threads owning 3 keys and values each
  using block_radix_sort_t = cub::BlockRadixSort<custom_t, 2, 1, int>;

  // Allocate shared memory for `cub::BlockRadixSort`
  __shared__ block_radix_sort_t::TempStorage temp_storage;

  // Obtain a segment of consecutive items that are blocked across threads
  custom_t thread_keys[2][1] = //
    {{
       {24.2, 1ll << 61} // thread 0 keys
     },
     {
       {42.4, 1ll << 60} // thread 1 keys
     }};

  int thread_values[2][1] = //
    {{1}, // thread 0 values
     {0}}; // thread 1 values

  constexpr int begin_bit = sizeof(long long int) * 8 - 4; // 60
  constexpr int end_bit   = sizeof(long long int) * 8 + 4; // 68

  // Decomposition orders the bits as follows:
  //
  //                    <------------- fp32 -----------> <------ int64 ------>
  // decompose(in[0]) = 01000001110000011001100110011010 00100000000000...0000
  // decompose(in[1]) = 01000010001010011001100110011010 00010000000000...0000
  //                    <-----------  higher bits  /  lower bits  ----------->
  //
  // The bit subrange `[60, 68)` specifies differentiating key bits:
  //
  //                    <------------- fp32 -----------> <------ int64 ------>
  // decompose(in[0]) = xxxxxxxxxxxxxxxxxxxxxxxxxxxx1010 0010xxxxxxxxxx...xxxx
  // decompose(in[1]) = xxxxxxxxxxxxxxxxxxxxxxxxxxxx1010 0001xxxxxxxxxx...xxxx
  //                    <-----------  higher bits  /  lower bits  ----------->

  // Collectively sort the keys
  block_radix_sort_t(temp_storage)
    .Sort(thread_keys[threadIdx.x], thread_values[threadIdx.x], decomposer_t{}, begin_bit, end_bit);

  custom_t expected_keys[2][3] = //
    {{
       {42.4, 1ll << 60}, // thread 0 expected keys
     },
     {
       {24.2, 1ll << 61} // thread 1 expected keys
     }};

  int expected_values[2][1] = //
    {{0}, // thread 0 values
     {1}}; // thread 1 values
  // example-end pairs-bits

  assert(thread_keys[threadIdx.x][0] == expected_keys[threadIdx.x][0]);
  assert(thread_values[threadIdx.x][0] == expected_values[threadIdx.x][0]);
}

__global__ void sort_pairs_descending()
{
  // example-begin pairs-descending
  // Specialize `cub::BlockRadixSort` for a 1D block of 2 threads owning 3 keys and values each
  using block_radix_sort_t = cub::BlockRadixSort<custom_t, 2, 3, int>;

  // Allocate shared memory for `cub::BlockRadixSort`
  __shared__ block_radix_sort_t::TempStorage temp_storage;

  // Obtain a segment of consecutive items that are blocked across threads
  custom_t thread_keys[2][3] = //
    {{
       // thread 0 keys
       {+1.1, 2}, //
       {+2.5, 1}, //
       {-0.0, 4}, //
     },
     {
       // thread 1 keys
       {+0.0, 3}, //
       {-2.5, 5}, //
       {+3.7, 0} //
     }};

  int thread_values[2][3] = //
    {{2, 1, 4}, // thread 0 values
     {3, 5, 0}}; // thread 1 values

  // Collectively sort the keys
  block_radix_sort_t(temp_storage).SortDescending(thread_keys[threadIdx.x], thread_values[threadIdx.x], decomposer_t{});

  custom_t expected_keys[2][3] = //
    {{
       // thread 0 expected keys
       {+3.7, 0}, //
       {+2.5, 1}, //
       {+1.1, 2}, //
     },
     {
       // thread 1 expected keys
       {-0.0, 4}, //
       {+0.0, 3}, //
       {-2.5, 5} //
     }};

  int expected_values[2][3] = //
    {{0, 1, 2}, // thread 0 expected values
     {4, 3, 5}}; // thread 1 expected values
  // example-end pairs-descending

  assert(thread_keys[threadIdx.x][0] == expected_keys[threadIdx.x][0]);
  assert(thread_keys[threadIdx.x][1] == expected_keys[threadIdx.x][1]);
  assert(thread_keys[threadIdx.x][2] == expected_keys[threadIdx.x][2]);

  assert(thread_values[threadIdx.x][0] == expected_values[threadIdx.x][0]);
  assert(thread_values[threadIdx.x][1] == expected_values[threadIdx.x][1]);
  assert(thread_values[threadIdx.x][2] == expected_values[threadIdx.x][2]);
}

__global__ void sort_pairs_descending_bits()
{
  // example-begin pairs-descending-bits
  // Specialize `cub::BlockRadixSort` for a 1D block of 2 threads owning 3 keys and values each
  using block_radix_sort_t = cub::BlockRadixSort<custom_t, 2, 1, int>;

  // Allocate shared memory for `cub::BlockRadixSort`
  __shared__ block_radix_sort_t::TempStorage temp_storage;

  // Obtain a segment of consecutive items that are blocked across threads
  custom_t thread_keys[2][1] = //
    {{
       {42.4, 1ll << 60} // thread 0 keys
     },
     {
       {24.2, 1ll << 61} // thread 1 keys
     }};

  int thread_values[2][1] = //
    {{1}, // thread 0 values
     {0}}; // thread 1 values

  constexpr int begin_bit = sizeof(long long int) * 8 - 4; // 60
  constexpr int end_bit   = sizeof(long long int) * 8 + 4; // 68

  // Decomposition orders the bits as follows:
  //
  //                    <------------- fp32 -----------> <------ int64 ------>
  // decompose(in[0]) = 01000010001010011001100110011010 00010000000000...0000
  // decompose(in[1]) = 01000001110000011001100110011010 00100000000000...0000
  //                    <-----------  higher bits  /  lower bits  ----------->
  //
  // The bit subrange `[60, 68)` specifies differentiating key bits:
  //
  //                    <------------- fp32 -----------> <------ int64 ------>
  // decompose(in[0]) = xxxxxxxxxxxxxxxxxxxxxxxxxxxx1010 0001xxxxxxxxxx...xxxx
  // decompose(in[1]) = xxxxxxxxxxxxxxxxxxxxxxxxxxxx1010 0010xxxxxxxxxx...xxxx
  //                    <-----------  higher bits  /  lower bits  ----------->

  // Collectively sort the keys
  block_radix_sort_t(temp_storage)
    .SortDescending(thread_keys[threadIdx.x], thread_values[threadIdx.x], decomposer_t{}, begin_bit, end_bit);

  custom_t expected_output[2][3] = //
    {{
       {24.2, 1ll << 61}, // thread 0 expected keys
     },
     {
       {42.4, 1ll << 60} // thread 1 expected keys
     }};

  int expected_values[2][1] = //
    {{0}, // thread 0 expected values
     {1}}; // thread 1 expected values
  // example-end pairs-descending-bits

  assert(thread_keys[threadIdx.x][0] == expected_output[threadIdx.x][0]);
  assert(thread_values[threadIdx.x][0] == expected_values[threadIdx.x][0]);
}

__global__ void sort_keys_blocked_to_striped()
{
  // example-begin keys-striped
  // Specialize `cub::BlockRadixSort` for a 1D block of 2 threads owning 3 keys each
  using block_radix_sort_t = cub::BlockRadixSort<custom_t, 2, 3>;

  // Allocate shared memory for `cub::BlockRadixSort`
  __shared__ block_radix_sort_t::TempStorage temp_storage;

  // Obtain a segment of consecutive items that are blocked across threads
  custom_t thread_keys[2][3] = //
    {{
       // thread 0 keys
       {+2.5, 4}, //
       {-2.5, 0}, //
       {+1.1, 3}, //
     },
     {
       // thread 1 keys
       {+0.0, 1}, //
       {-0.0, 2}, //
       {+3.7, 5} //
     }};

  // Collectively sort the keys
  block_radix_sort_t(temp_storage).SortBlockedToStriped(thread_keys[threadIdx.x], decomposer_t{});

  custom_t expected_output[2][3] = //
    {{
       // thread 0 expected keys
       {-2.5, 0}, //
       {-0.0, 2}, //
       {+2.5, 4} //
     },
     {
       // thread 1 expected keys
       {+0.0, 1}, //
       {+1.1, 3}, //
       {+3.7, 5} //
     }};
  // example-end keys-striped

  assert(thread_keys[threadIdx.x][0] == expected_output[threadIdx.x][0]);
  assert(thread_keys[threadIdx.x][1] == expected_output[threadIdx.x][1]);
  assert(thread_keys[threadIdx.x][2] == expected_output[threadIdx.x][2]);
}

__global__ void sort_keys_blocked_to_striped_bits()
{
  // example-begin keys-striped-bits
  // Specialize `cub::BlockRadixSort` for a 1D block of 2 threads owning 2 keys each
  using block_radix_sort_t = cub::BlockRadixSort<custom_t, 2, 2>;

  // Allocate shared memory for `cub::BlockRadixSort`
  __shared__ block_radix_sort_t::TempStorage temp_storage;

  // Obtain a segment of consecutive items that are blocked across threads
  custom_t thread_keys[2][2] = //
    {{// thread 0 keys
      {24.2, 1ll << 62},
      {42.4, 1ll << 61}},
     {// thread 1 keys
      {42.4, 1ll << 60},
      {24.2, 1ll << 59}}};

  constexpr int begin_bit = sizeof(long long int) * 8 - 4; // 60
  constexpr int end_bit   = sizeof(long long int) * 8 + 4; // 68

  // Decomposition orders the bits as follows:
  //
  //                    <------------- fp32 -----------> <------ int64 ------>
  // decompose(in[0]) = 01000001110000011001100110011010 01000000000000...0000
  // decompose(in[1]) = 01000010001010011001100110011010 00100000000000...0000
  // decompose(in[2]) = 01000001110000011001100110011010 00010000000000...0000
  // decompose(in[3]) = 01000010001010011001100110011010 00001000000000...0000
  //                    <-----------  higher bits  /  lower bits  ----------->
  //
  // The bit subrange `[60, 68)` specifies differentiating key bits:
  //
  //                    <------------- fp32 -----------> <------ int64 ------>
  // decompose(in[0]) = xxxxxxxxxxxxxxxxxxxxxxxxxxxx1010 0100xxxxxxxxxx...xxxx
  // decompose(in[1]) = xxxxxxxxxxxxxxxxxxxxxxxxxxxx1010 0010xxxxxxxxxx...xxxx
  // decompose(in[2]) = xxxxxxxxxxxxxxxxxxxxxxxxxxxx1010 0001xxxxxxxxxx...xxxx
  // decompose(in[3]) = xxxxxxxxxxxxxxxxxxxxxxxxxxxx1010 0000xxxxxxxxxx...xxxx
  //                    <-----------  higher bits  /  lower bits  ----------->

  // Collectively sort the keys
  block_radix_sort_t(temp_storage).SortBlockedToStriped(thread_keys[threadIdx.x], decomposer_t{}, begin_bit, end_bit);

  custom_t expected_output[2][3] = //
    {{// thread 0 expected keys
      {24.2, 1ll << 59},
      {42.4, 1ll << 61}},
     {// thread 1 expected keys
      {42.4, 1ll << 60},
      {24.2, 1ll << 62}}};
  // example-end keys-striped-bits

  assert(thread_keys[threadIdx.x][0] == expected_output[threadIdx.x][0]);
  assert(thread_keys[threadIdx.x][1] == expected_output[threadIdx.x][1]);
}

__global__ void sort_pairs_blocked_to_striped()
{
  // example-begin pairs-striped
  // Specialize `cub::BlockRadixSort` for a 1D block of 2 threads owning 3 keys and values each
  using block_radix_sort_t = cub::BlockRadixSort<custom_t, 2, 3, int>;

  // Allocate shared memory for `cub::BlockRadixSort`
  __shared__ block_radix_sort_t::TempStorage temp_storage;

  // Obtain a segment of consecutive items that are blocked across threads
  custom_t thread_keys[2][3] = //
    {{
       // thread 0 keys
       {+2.5, 4}, //
       {-2.5, 0}, //
       {+1.1, 3}, //
     },
     {
       // thread 1 keys
       {+0.0, 1}, //
       {-0.0, 2}, //
       {+3.7, 5} //
     }};

  int thread_values[2][3] = //
    {{4, 0, 3}, // thread 0 values
     {1, 2, 5}}; // thread 1 values

  // Collectively sort the keys
  block_radix_sort_t(temp_storage)
    .SortBlockedToStriped(thread_keys[threadIdx.x], thread_values[threadIdx.x], decomposer_t{});

  custom_t expected_output[2][3] = //
    {{
       // thread 0 expected keys
       {-2.5, 0}, //
       {-0.0, 2}, //
       {+2.5, 4} //
     },
     {
       // thread 1 expected keys
       {+0.0, 1}, //
       {+1.1, 3}, //
       {+3.7, 5} //
     }};

  int expected_values[2][3] = //
    {{0, 2, 4}, // thread 0 values
     {1, 3, 5}}; // thread 1 values
  // example-end pairs-striped

  assert(thread_keys[threadIdx.x][0] == expected_output[threadIdx.x][0]);
  assert(thread_keys[threadIdx.x][1] == expected_output[threadIdx.x][1]);
  assert(thread_keys[threadIdx.x][2] == expected_output[threadIdx.x][2]);

  assert(thread_values[threadIdx.x][0] == expected_values[threadIdx.x][0]);
  assert(thread_values[threadIdx.x][1] == expected_values[threadIdx.x][1]);
  assert(thread_values[threadIdx.x][2] == expected_values[threadIdx.x][2]);
}

__global__ void sort_pairs_blocked_to_striped_bits()
{
  // example-begin pairs-striped-bits
  // Specialize `cub::BlockRadixSort` for a 1D block of 2 threads owning 2 keys and values each
  using block_radix_sort_t = cub::BlockRadixSort<custom_t, 2, 2, int>;

  // Allocate shared memory for `cub::BlockRadixSort`
  __shared__ block_radix_sort_t::TempStorage temp_storage;

  // Obtain a segment of consecutive items that are blocked across threads
  custom_t thread_keys[2][2] = //
    {{// thread 0 keys
      {24.2, 1ll << 62},
      {42.4, 1ll << 61}},
     {// thread 1 keys
      {42.4, 1ll << 60},
      {24.2, 1ll << 59}}};

  int thread_values[2][2] = //
    {{3, 2}, // thread 0 values
     {1, 0}}; // thread 1 values

  constexpr int begin_bit = sizeof(long long int) * 8 - 4; // 60
  constexpr int end_bit   = sizeof(long long int) * 8 + 4; // 68

  // Decomposition orders the bits as follows:
  //
  //                    <------------- fp32 -----------> <------ int64 ------>
  // decompose(in[0]) = 01000001110000011001100110011010 01000000000000...0000
  // decompose(in[1]) = 01000010001010011001100110011010 00100000000000...0000
  // decompose(in[2]) = 01000001110000011001100110011010 00010000000000...0000
  // decompose(in[3]) = 01000010001010011001100110011010 00001000000000...0000
  //                    <-----------  higher bits  /  lower bits  ----------->
  //
  // The bit subrange `[60, 68)` specifies differentiating key bits:
  //
  //                    <------------- fp32 -----------> <------ int64 ------>
  // decompose(in[0]) = xxxxxxxxxxxxxxxxxxxxxxxxxxxx1010 0100xxxxxxxxxx...xxxx
  // decompose(in[1]) = xxxxxxxxxxxxxxxxxxxxxxxxxxxx1010 0010xxxxxxxxxx...xxxx
  // decompose(in[2]) = xxxxxxxxxxxxxxxxxxxxxxxxxxxx1010 0001xxxxxxxxxx...xxxx
  // decompose(in[3]) = xxxxxxxxxxxxxxxxxxxxxxxxxxxx1010 0000xxxxxxxxxx...xxxx
  //                    <-----------  higher bits  /  lower bits  ----------->

  // Collectively sort the keys
  block_radix_sort_t(temp_storage)
    .SortBlockedToStriped(thread_keys[threadIdx.x], thread_values[threadIdx.x], decomposer_t{}, begin_bit, end_bit);

  custom_t expected_output[2][3] = //
    {{// thread 0 expected keys
      {24.2, 1ll << 59},
      {42.4, 1ll << 61}},
     {// thread 1 expected keys
      {42.4, 1ll << 60},
      {24.2, 1ll << 62}}};

  int expected_values[2][2] = //
    {{0, 2}, // thread 0 values
     {1, 3}}; // thread 1 values
  // example-end pairs-striped-bits

  assert(thread_keys[threadIdx.x][0] == expected_output[threadIdx.x][0]);
  assert(thread_keys[threadIdx.x][1] == expected_output[threadIdx.x][1]);

  assert(thread_values[threadIdx.x][0] == expected_values[threadIdx.x][0]);
  assert(thread_values[threadIdx.x][1] == expected_values[threadIdx.x][1]);
}

__global__ void sort_keys_descending_blocked_to_striped()
{
  // example-begin keys-striped-descending
  // Specialize `cub::BlockRadixSort` for a 1D block of 2 threads owning 3 keys each
  using block_radix_sort_t = cub::BlockRadixSort<custom_t, 2, 3>;

  // Allocate shared memory for `cub::BlockRadixSort`
  __shared__ block_radix_sort_t::TempStorage temp_storage;

  // Obtain a segment of consecutive items that are blocked across threads
  custom_t thread_keys[2][3] = //
    {{
       // thread 0 keys
       {+1.1, 2}, //
       {+2.5, 1}, //
       {-0.0, 4}, //
     },
     {
       // thread 1 keys
       {+0.0, 3}, //
       {-2.5, 5}, //
       {+3.7, 0} //
     }};

  // Collectively sort the keys
  block_radix_sort_t(temp_storage).SortDescendingBlockedToStriped(thread_keys[threadIdx.x], decomposer_t{});

  custom_t expected_output[2][3] = //
    {{
       // thread 0 expected keys
       {+3.7, 0}, //
       {+1.1, 2}, //
       {+0.0, 3} //
     },
     {
       // thread 1 expected keys
       {+2.5, 1}, //
       {-0.0, 4}, //
       {-2.5, 5} //
     }};
  // example-end keys-striped-descending

  assert(thread_keys[threadIdx.x][0] == expected_output[threadIdx.x][0]);
  assert(thread_keys[threadIdx.x][1] == expected_output[threadIdx.x][1]);
  assert(thread_keys[threadIdx.x][2] == expected_output[threadIdx.x][2]);
}

__global__ void sort_keys_descending_blocked_to_striped_bits()
{
  // example-begin keys-striped-descending-bits
  // Specialize `cub::BlockRadixSort` for a 1D block of 2 threads owning 2 keys each
  using block_radix_sort_t = cub::BlockRadixSort<custom_t, 2, 2>;

  // Allocate shared memory for `cub::BlockRadixSort`
  __shared__ block_radix_sort_t::TempStorage temp_storage;

  // Obtain a segment of consecutive items that are blocked across threads
  custom_t thread_keys[2][2] = //
    {{// thread 0 keys
      {24.2, 1ll << 62},
      {42.4, 1ll << 61}},
     {// thread 1 keys
      {42.4, 1ll << 60},
      {24.2, 1ll << 59}}};

  constexpr int begin_bit = sizeof(long long int) * 8 - 4; // 60
  constexpr int end_bit   = sizeof(long long int) * 8 + 4; // 68

  // Decomposition orders the bits as follows:
  //
  //                    <------------- fp32 -----------> <------ int64 ------>
  // decompose(in[0]) = 01000001110000011001100110011010 01000000000000...0000
  // decompose(in[1]) = 01000010001010011001100110011010 00100000000000...0000
  // decompose(in[2]) = 01000001110000011001100110011010 00010000000000...0000
  // decompose(in[3]) = 01000010001010011001100110011010 00001000000000...0000
  //                    <-----------  higher bits  /  lower bits  ----------->
  //
  // The bit subrange `[60, 68)` specifies differentiating key bits:
  //
  //                    <------------- fp32 -----------> <------ int64 ------>
  // decompose(in[0]) = xxxxxxxxxxxxxxxxxxxxxxxxxxxx1010 0100xxxxxxxxxx...xxxx
  // decompose(in[1]) = xxxxxxxxxxxxxxxxxxxxxxxxxxxx1010 0010xxxxxxxxxx...xxxx
  // decompose(in[2]) = xxxxxxxxxxxxxxxxxxxxxxxxxxxx1010 0001xxxxxxxxxx...xxxx
  // decompose(in[3]) = xxxxxxxxxxxxxxxxxxxxxxxxxxxx1010 0000xxxxxxxxxx...xxxx
  //                    <-----------  higher bits  /  lower bits  ----------->

  // Collectively sort the keys
  block_radix_sort_t(temp_storage)
    .SortDescendingBlockedToStriped(thread_keys[threadIdx.x], decomposer_t{}, begin_bit, end_bit);

  custom_t expected_output[2][2] = //
    {{
       // thread 0 expected keys
       {24.2, 1ll << 62}, //
       {42.4, 1ll << 60} //
     },
     {
       // thread 1 expected keys
       {42.4, 1ll << 61}, //
       {24.2, 1ll << 59} //
     }};
  // example-end keys-striped-descending-bits

  assert(thread_keys[threadIdx.x][0] == expected_output[threadIdx.x][0]);
  assert(thread_keys[threadIdx.x][1] == expected_output[threadIdx.x][1]);
}

__global__ void sort_pairs_descending_blocked_to_striped()
{
  // example-begin pairs-striped-descending
  // Specialize `cub::BlockRadixSort` for a 1D block of 2 threads owning 3 keys and values each
  using block_radix_sort_t = cub::BlockRadixSort<custom_t, 2, 3, int>;

  // Allocate shared memory for `cub::BlockRadixSort`
  __shared__ block_radix_sort_t::TempStorage temp_storage;

  // Obtain a segment of consecutive items that are blocked across threads
  custom_t thread_keys[2][3] = //
    {{
       // thread 0 keys
       {+1.1, 2}, //
       {+2.5, 1}, //
       {-0.0, 4}, //
     },
     {
       // thread 1 keys
       {+0.0, 3}, //
       {-2.5, 5}, //
       {+3.7, 0} //
     }};

  int thread_values[2][3] = //
    {{2, 1, 4}, // thread 0 values
     {3, 5, 0}}; // thread 1 values

  // Collectively sort the keys
  block_radix_sort_t(temp_storage)
    .SortDescendingBlockedToStriped(thread_keys[threadIdx.x], thread_values[threadIdx.x], decomposer_t{});

  custom_t expected_output[2][3] = //
    {{
       // thread 0 expected keys
       {+3.7, 0}, //
       {+1.1, 2}, //
       {+0.0, 3} //
     },
     {
       // thread 1 expected keys
       {+2.5, 1}, //
       {-0.0, 4}, //
       {-2.5, 5} //
     }};

  int expected_values[2][3] = //
    {{0, 2, 3}, // thread 0 values
     {1, 4, 5}}; // thread 1 values
  // example-end pairs-striped-descending

  assert(thread_keys[threadIdx.x][0] == expected_output[threadIdx.x][0]);
  assert(thread_keys[threadIdx.x][1] == expected_output[threadIdx.x][1]);
  assert(thread_keys[threadIdx.x][2] == expected_output[threadIdx.x][2]);

  assert(thread_values[threadIdx.x][0] == expected_values[threadIdx.x][0]);
  assert(thread_values[threadIdx.x][1] == expected_values[threadIdx.x][1]);
  assert(thread_values[threadIdx.x][2] == expected_values[threadIdx.x][2]);
}

__global__ void sort_pairs_descending_blocked_to_striped_bits()
{
  // example-begin pairs-striped-descending-bits
  // Specialize `cub::BlockRadixSort` for a 1D block of 2 threads owning 2 keys and values each
  using block_radix_sort_t = cub::BlockRadixSort<custom_t, 2, 2, int>;

  // Allocate shared memory for `cub::BlockRadixSort`
  __shared__ block_radix_sort_t::TempStorage temp_storage;

  // Obtain a segment of consecutive items that are blocked across threads
  custom_t thread_keys[2][2] = //
    {{// thread 0 keys
      {24.2, 1ll << 62},
      {42.4, 1ll << 61}},
     {// thread 1 keys
      {42.4, 1ll << 60},
      {24.2, 1ll << 59}}};

  int thread_values[2][2] = //
    {{3, 2}, // thread 0 values
     {1, 0}}; // thread 1 values

  constexpr int begin_bit = sizeof(long long int) * 8 - 4; // 60
  constexpr int end_bit   = sizeof(long long int) * 8 + 4; // 68

  // Decomposition orders the bits as follows:
  //
  //                    <------------- fp32 -----------> <------ int64 ------>
  // decompose(in[0]) = 01000001110000011001100110011010 01000000000000...0000
  // decompose(in[1]) = 01000010001010011001100110011010 00100000000000...0000
  // decompose(in[2]) = 01000001110000011001100110011010 00010000000000...0000
  // decompose(in[3]) = 01000010001010011001100110011010 00001000000000...0000
  //                    <-----------  higher bits  /  lower bits  ----------->
  //
  // The bit subrange `[60, 68)` specifies differentiating key bits:
  //
  //                    <------------- fp32 -----------> <------ int64 ------>
  // decompose(in[0]) = xxxxxxxxxxxxxxxxxxxxxxxxxxxx1010 0100xxxxxxxxxx...xxxx
  // decompose(in[1]) = xxxxxxxxxxxxxxxxxxxxxxxxxxxx1010 0010xxxxxxxxxx...xxxx
  // decompose(in[2]) = xxxxxxxxxxxxxxxxxxxxxxxxxxxx1010 0001xxxxxxxxxx...xxxx
  // decompose(in[3]) = xxxxxxxxxxxxxxxxxxxxxxxxxxxx1010 0000xxxxxxxxxx...xxxx
  //                    <-----------  higher bits  /  lower bits  ----------->

  // Collectively sort the keys
  block_radix_sort_t(temp_storage)
    .SortDescendingBlockedToStriped(
      thread_keys[threadIdx.x], thread_values[threadIdx.x], decomposer_t{}, begin_bit, end_bit);

  custom_t expected_output[2][2] = //
    {{
       // thread 0 expected keys
       {24.2, 1ll << 62}, //
       {42.4, 1ll << 60} //
     },
     {
       // thread 1 expected keys
       {42.4, 1ll << 61}, //
       {24.2, 1ll << 59} //
     }};

  int expected_values[2][2] = //
    {{3, 1}, // thread 0 values
     {2, 0}}; // thread 1 values
  // example-end pairs-striped-descending-bits

  assert(thread_keys[threadIdx.x][0] == expected_output[threadIdx.x][0]);
  assert(thread_keys[threadIdx.x][1] == expected_output[threadIdx.x][1]);

  assert(thread_values[threadIdx.x][0] == expected_values[threadIdx.x][0]);
  assert(thread_values[threadIdx.x][1] == expected_values[threadIdx.x][1]);
}

TEST_CASE("Block radix sort works in some corner cases", "[radix][sort][block]")
{
  sort_keys<<<1, 2>>>();
  REQUIRE(cudaSuccess == cudaGetLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  sort_keys_bits<<<1, 2>>>();
  REQUIRE(cudaSuccess == cudaGetLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  sort_keys_descending<<<1, 2>>>();
  REQUIRE(cudaSuccess == cudaGetLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  sort_keys_descending_bits<<<1, 2>>>();
  REQUIRE(cudaSuccess == cudaGetLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  sort_pairs<<<1, 2>>>();
  REQUIRE(cudaSuccess == cudaGetLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  sort_pairs_bits<<<1, 2>>>();
  REQUIRE(cudaSuccess == cudaGetLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  sort_pairs_descending<<<1, 2>>>();
  REQUIRE(cudaSuccess == cudaGetLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  sort_pairs_descending_bits<<<1, 2>>>();
  REQUIRE(cudaSuccess == cudaGetLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  sort_keys_blocked_to_striped<<<1, 2>>>();
  REQUIRE(cudaSuccess == cudaGetLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  sort_keys_blocked_to_striped_bits<<<1, 2>>>();
  REQUIRE(cudaSuccess == cudaGetLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  sort_pairs_blocked_to_striped<<<1, 2>>>();
  REQUIRE(cudaSuccess == cudaGetLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  sort_pairs_blocked_to_striped_bits<<<1, 2>>>();
  REQUIRE(cudaSuccess == cudaGetLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  sort_keys_descending_blocked_to_striped<<<1, 2>>>();
  REQUIRE(cudaSuccess == cudaGetLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  sort_keys_descending_blocked_to_striped_bits<<<1, 2>>>();
  REQUIRE(cudaSuccess == cudaGetLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  sort_pairs_descending_blocked_to_striped<<<1, 2>>>();
  REQUIRE(cudaSuccess == cudaGetLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  sort_pairs_descending_blocked_to_striped_bits<<<1, 2>>>();
  REQUIRE(cudaSuccess == cudaGetLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
}
