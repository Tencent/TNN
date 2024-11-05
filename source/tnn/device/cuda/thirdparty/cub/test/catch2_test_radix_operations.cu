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

#include <cub/block/radix_rank_sort_operations.cuh>

#include <thrust/detail/raw_pointer_cast.h>

#include <bitset>
#include <climits>
#include <limits>
#include <type_traits>

#include <c2h/catch2_test_helper.cuh>

template <typename KeyT>
struct fundamental_extractor_t
{
  std::uint32_t bit_start;
  std::uint32_t mask;

  __host__ __device__ fundamental_extractor_t(std::uint32_t bit_start = 0, std::uint32_t num_bits = 0)
      : bit_start(bit_start)
      , mask((1 << num_bits) - 1)
  {}

  __host__ __device__ std::uint32_t Digit(KeyT key) const
  {
    return std::uint32_t(key >> KeyT(bit_start)) & mask;
  }
};

template <class T>
c2h::host_vector<std::uint8_t> get_random_buffer()
{
  c2h::device_vector<std::uint8_t> buffer(sizeof(T));
  c2h::gen(C2H_SEED(3), buffer);
  return buffer;
}

constexpr int max_digit_bits = sizeof(std::uint32_t) * CHAR_BIT;
using digit_bits_t           = std::bitset<max_digit_bits>;

digit_bits_t buffer_to_digit_bits(const char* buffer, int current_bit, int num_bits)
{
  digit_bits_t dst; // all bits set to zero

  for (int bit = current_bit; bit < current_bit + num_bits; bit++)
  {
    const int dst_bit  = bit - current_bit;
    const int src_byte = bit / CHAR_BIT;
    const int src_bit  = bit % CHAR_BIT;

    std::bitset<CHAR_BIT> src(buffer[src_byte]);
    dst[dst_bit] = src[src_bit];
  }

  return dst;
}

using fundamental_types       = c2h::type_list<std::uint8_t, std::uint16_t, std::uint32_t, std::uint64_t>;
using a_few_fundamental_types = c2h::type_list<std::uint8_t, std::uint64_t>;

/**
 * This test checks that radix operations can extract certain bits out of unsigned integers.
 * Test runs for all possible combinations of `current_bit` and `num_bits`.
 * Example for `current_bit = 5`, and `num_bits = 4`:
 *
 *          [-------]
 *    src: 1 1 0 0 1 1 0 0 1 1
 *    bit: 9 8 7 6 5 4 3 2 1 0
 *    dst: 0 0 0 0 0 0 1 0 0 1
 *
 */
C2H_TEST("Radix operations extract digits from fundamental types", "[radix][operations]", fundamental_types)
{
  using key_t        = typename c2h::get<0, TestType>;
  using traits       = cub::detail::radix::traits_t<key_t>;
  using extractor_t  = fundamental_extractor_t<key_t>;
  using decomposer_t = cub::detail::identity_decomposer_t;

  auto decomposer            = decomposer_t{};
  constexpr int max_key_bits = sizeof(key_t) * CHAR_BIT;
  REQUIRE(traits::default_end_bit(decomposer) == max_key_bits);

  key_t val{};
  c2h::host_vector<char> output_buffer_mem(sizeof(std::uint32_t));
  const c2h::host_vector<char> input_buffer_mem = get_random_buffer<key_t>();

  char* output_buffer      = thrust::raw_pointer_cast(output_buffer_mem.data());
  const char* input_buffer = thrust::raw_pointer_cast(input_buffer_mem.data());
  std::memcpy(&val, input_buffer, sizeof(key_t));

  for (int current_bit = 0; current_bit < max_key_bits; current_bit++)
  {
    const int max_bits = std::min(max_key_bits - current_bit, max_digit_bits);

    for (int num_bits = 1; num_bits < max_bits; num_bits++)
    {
      auto extractor = traits::template digit_extractor<extractor_t>(current_bit, num_bits, decomposer);

      std::uint32_t digit = extractor.Digit(val);
      std::memcpy(output_buffer, &digit, sizeof(std::uint32_t));

      digit_bits_t result    = buffer_to_digit_bits(output_buffer, 0, num_bits);
      digit_bits_t reference = buffer_to_digit_bits(input_buffer, current_bit, num_bits);

      REQUIRE(reference == result);
    }
  }
}

template <class T>
struct tuple_decomposer_t;

template <class... Ts>
struct tuple_decomposer_t<::cuda::std::tuple<Ts...>>
{
  template <std::size_t... Is>
  __host__ __device__ ::cuda::std::tuple<Ts&...>
  extract(::cuda::std::tuple<Ts...>& key, thrust::index_sequence<Is...>) const
  {
    return ::cuda::std::tie(::cuda::std::get<Is>(key)...);
  }

  __host__ __device__ ::cuda::std::tuple<Ts&...> operator()(::cuda::std::tuple<Ts...>& key) const
  {
    return extract(key, thrust::make_index_sequence<sizeof...(Ts)>{});
  }
};

// clang-format off
template <std::size_t I, class... Ts>
typename ::cuda::std::enable_if<I == 0>::type
buffer_to_tpl_helper(const char *buffer, ::cuda::std::tuple<Ts...> &tpl)
{
  constexpr std::size_t element_size =
    sizeof(typename ::cuda::std::tuple_element<I, ::cuda::std::tuple<Ts...>>::type);
  std::memcpy(&::cuda::std::get<I>(tpl), buffer, element_size);
}

template <std::size_t I, class... Ts>
typename ::cuda::std::enable_if <I != 0>::type
buffer_to_tpl_helper(const char *buffer, ::cuda::std::tuple<Ts...> &tpl)
{
  constexpr std::size_t element_size =
    sizeof(typename ::cuda::std::tuple_element<I, ::cuda::std::tuple<Ts...>>::type);
  std::memcpy(&::cuda::std::get<I>(tpl), buffer, element_size);
  buffer_to_tpl_helper<I - 1>(buffer + element_size, tpl);
}

template <class... Ts>
void buffer_to_tpl(const char *buffer, ::cuda::std::tuple<Ts...> &tpl)
{
  buffer_to_tpl_helper<sizeof...(Ts) - 1>(buffer, tpl);
}

template <std::size_t I, class... Ts>
typename ::cuda::std::enable_if<I == 0>::type
tpl_to_buffer_helper(char *buffer, ::cuda::std::tuple<Ts...> &tpl)
{
  constexpr std::size_t element_size =
    sizeof(typename ::cuda::std::tuple_element<I, ::cuda::std::tuple<Ts...>>::type);
  std::memcpy(buffer, &::cuda::std::get<I>(tpl), element_size);
}

template <std::size_t I, class... Ts>
typename ::cuda::std::enable_if <I != 0>::type
tpl_to_buffer_helper(char *buffer, ::cuda::std::tuple<Ts...> &tpl)
{
  constexpr std::size_t element_size =
    sizeof(typename ::cuda::std::tuple_element<I, ::cuda::std::tuple<Ts...>>::type);
  std::memcpy(buffer, &::cuda::std::get<I>(tpl), element_size);
  tpl_to_buffer_helper<I - 1>(buffer + element_size, tpl);
}

template <class... Ts>
void tpl_to_buffer(char *buffer, ::cuda::std::tuple<Ts...> &tpl)
{
  tpl_to_buffer_helper<sizeof...(Ts) - 1>(buffer, tpl);
}

template <std::size_t I = 0, class... Ts>
typename ::cuda::std::enable_if<I >= sizeof...(Ts), int>::type
tpl_to_max_bits(::cuda::std::tuple<Ts...> &)
{
  return 0;
}

template <std::size_t I = 0, class... Ts>
typename ::cuda::std::enable_if <I < sizeof...(Ts), int>::type
tpl_to_max_bits(::cuda::std::tuple<Ts...> &tpl)
{
  constexpr std::size_t element_size =
    sizeof(typename ::cuda::std::tuple_element<I, ::cuda::std::tuple<Ts...>>::type);
  return element_size * CHAR_BIT + tpl_to_max_bits<I + 1>(tpl);
}

template <std::size_t I = 0, class... Ts>
typename ::cuda::std::enable_if<I >= sizeof...(Ts)>::type
tpl_to_min(::cuda::std::tuple<Ts...> &)
{}

template <std::size_t I = 0, class... Ts>
typename ::cuda::std::enable_if <I < sizeof...(Ts)>::type
tpl_to_min(::cuda::std::tuple<Ts...> &tpl)
{
  using T = typename ::cuda::std::tuple_element<I, ::cuda::std::tuple<Ts...>>::type;
  ::cuda::std::get<I>(tpl) = std::numeric_limits<T>::lowest();
  tpl_to_min<I + 1>(tpl);
}

template <std::size_t I = 0, class... Ts>
typename ::cuda::std::enable_if<I >= sizeof...(Ts)>::type
tpl_to_max(::cuda::std::tuple<Ts...> &)
{}

template <std::size_t I = 0, class... Ts>
typename ::cuda::std::enable_if <I < sizeof...(Ts)>::type
tpl_to_max(::cuda::std::tuple<Ts...> &tpl)
{
  using T = typename ::cuda::std::tuple_element<I, ::cuda::std::tuple<Ts...>>::type;
  ::cuda::std::get<I>(tpl) = std::numeric_limits<T>::max();
  tpl_to_max<I + 1>(tpl);
}
// clang-format on

/**
 * This test checks that radix operations can extract certain bits out of aggregate types.
 * Test runs for all possible combinations of `current_bit` and `num_bits` excluding padding bits.
 * For example, `struct custom_t { short s = 65535; float f = -42.2f; };` has the following binary
 * representation:
 *
 *    <------------ `.f` ------------><-- padding ---><---- `.s` ---->
 *    s< exp. ><----- mantissa ------><-- padding ---><--- short ---->
 *    1100000010000110011001100110011000000000000000001111111111111111
 *                               +---~                ~--+
 *    <           <----  higher bits  /  lower bits  ---->           >
 *
 * For `current_bit = 12`, and `num_bits = 9`:
 *    dst: 0000011011111
 *         <   fp  ><sh>
 *
 */
template <class... Ts>
void test_tuple()
{
  using tpl_t        = ::cuda::std::tuple<Ts...>;
  using traits       = cub::detail::radix::traits_t<tpl_t>;
  using decomposer_t = tuple_decomposer_t<tpl_t>;
  using extractor_t  = cub::detail::radix::custom_digit_extractor_t<decomposer_t>;

  tpl_t tpl{};
  c2h::host_vector<char> output_buffer_mem(sizeof(std::uint32_t));
  const c2h::host_vector<char> input_buffer_mem = get_random_buffer<tpl_t>();

  char* output_buffer      = thrust::raw_pointer_cast(output_buffer_mem.data());
  const char* input_buffer = thrust::raw_pointer_cast(input_buffer_mem.data());
  buffer_to_tpl(input_buffer, tpl);

  auto decomposer        = decomposer_t{};
  const int max_key_bits = tpl_to_max_bits(tpl);
  REQUIRE(traits::default_end_bit(decomposer) == max_key_bits);

  for (int current_bit = 0; current_bit < max_key_bits; current_bit++)
  {
    const int max_bits = std::min(max_key_bits - current_bit, max_digit_bits);

    for (int num_bits = 1; num_bits < max_bits; num_bits++)
    {
      auto extractor = traits::template digit_extractor<extractor_t>(current_bit, num_bits, decomposer);

      std::uint32_t digit = extractor.Digit(tpl);
      std::memcpy(output_buffer, &digit, sizeof(std::uint32_t));

      digit_bits_t result    = buffer_to_digit_bits(output_buffer, 0, num_bits);
      digit_bits_t reference = buffer_to_digit_bits(input_buffer, current_bit, num_bits);

      // Provides readable error messages:
      //  00000000000000000000000000000000
      //  ==
      //  00000000000000000000000000000001
      REQUIRE(reference == result);
    }
  }
}

C2H_TEST("Radix operations extract digits from pairs", "[radix][operations]", fundamental_types, fundamental_types)
{
  test_tuple<typename c2h::get<0, TestType>, //
             typename c2h::get<1, TestType>>();
}

C2H_TEST("Radix operations extract digits from triples",
         "[radix][operations]",
         fundamental_types,
         fundamental_types,
         fundamental_types)
{
  test_tuple<typename c2h::get<0, TestType>, //
             typename c2h::get<1, TestType>, //
             typename c2h::get<2, TestType>>();
}

C2H_TEST("Radix operations extract digits from tetrads",
         "[radix][operations]",
         a_few_fundamental_types,
         a_few_fundamental_types,
         a_few_fundamental_types,
         a_few_fundamental_types)
{
  test_tuple<typename c2h::get<0, TestType>, //
             typename c2h::get<1, TestType>, //
             typename c2h::get<2, TestType>, //
             typename c2h::get<3, TestType>>();
}

/**
 * This test checks that radix operations can invert bits (`~`) of fundamental types.
 *
 *    src: 1 1 0 0 1 1 0 0 1 1
 *    dst: 0 0 1 1 0 0 1 1 0 0
 *
 */
C2H_TEST("Radix operations inverse fundamental types", "[radix][operations]", fundamental_types)
{
  using key_t        = typename c2h::get<0, TestType>;
  using traits       = cub::detail::radix::traits_t<key_t>;
  using extractor_t  = fundamental_extractor_t<key_t>;
  using decomposer_t = cub::detail::identity_decomposer_t;

  auto decomposer = decomposer_t{};

  key_t val{};
  c2h::host_vector<char> output_buffer_mem(sizeof(key_t));
  c2h::host_vector<char> input_buffer_mem = get_random_buffer<key_t>();

  char* output_buffer = thrust::raw_pointer_cast(output_buffer_mem.data());
  char* input_buffer  = thrust::raw_pointer_cast(input_buffer_mem.data());
  std::memcpy(&val, input_buffer, sizeof(key_t));

  for (std::size_t i = 0; i < input_buffer_mem.size(); i++)
  {
    input_buffer[i] = ~input_buffer[i];
  }

  key_t inv = traits::bit_ordered_inversion_policy::inverse(decomposer, val);
  std::memcpy(output_buffer, &inv, sizeof(key_t));

  REQUIRE(input_buffer_mem == output_buffer_mem);
}

/**
 * This test checks that radix operations can invert bits (`~`) of aggregate types.
 * For example, `struct custom_t { short s = 65535; float f = -42.2f; };`:
 *
 *      <------------ `.f` ------------><-- padding ---><---- `.s` ---->
 *      s< exp. ><----- mantissa ------><-- padding ---><--- short ---->
 * src: 1100000010000110011001100110011000000000000000001111111111111111
 *      +------------------------------~                ~--------------+
 * dst: 0011111101111001100110011001100111111111111111110000000000000000
 *      <           <----  higher bits  /  lower bits  ---->           >
 *
 */
C2H_TEST("Radix operations inverse pairs", "[radix][operations]", fundamental_types, fundamental_types)
{
  using tpl_t = ::cuda::std::tuple<typename c2h::get<0, TestType>, //
                                   typename c2h::get<1, TestType>>;

  using traits       = cub::detail::radix::traits_t<tpl_t>;
  using decomposer_t = tuple_decomposer_t<tpl_t>;
  using extractor_t  = cub::detail::radix::custom_digit_extractor_t<decomposer_t>;

  auto decomposer = decomposer_t{};

  tpl_t tpl{};
  c2h::host_vector<char> input_buffer_mem = get_random_buffer<tpl_t>();

  char* input_buffer = thrust::raw_pointer_cast(input_buffer_mem.data());
  buffer_to_tpl(input_buffer, tpl);

  for (std::size_t i = 0; i < input_buffer_mem.size(); i++)
  {
    input_buffer[i] = ~input_buffer[i];
  }

  c2h::host_vector<char> output_buffer_mem = input_buffer_mem;
  char* output_buffer                      = thrust::raw_pointer_cast(output_buffer_mem.data());

  tpl_t inv = traits::bit_ordered_inversion_policy::inverse(decomposer, tpl);
  tpl_to_buffer(output_buffer, inv);

  REQUIRE(input_buffer_mem == output_buffer_mem);
}

/**
 * This tests checks that radix operations can get a value that when converted
 * to binary-comparable representation, yields smallest possible value.
 */
C2H_TEST("Radix operations infere minimal value for fundamental types", "[radix][operations]", fundamental_types)
{
  using key_t        = typename c2h::get<0, TestType>;
  using traits       = cub::detail::radix::traits_t<key_t>;
  using decomposer_t = cub::detail::identity_decomposer_t;

  c2h::host_vector<char> output_buffer_mem(sizeof(key_t));
  c2h::host_vector<char> input_buffer_mem(sizeof(key_t));

  key_t ref = std::numeric_limits<key_t>::lowest();
  key_t val = traits::min_raw_binary_key(decomposer_t{});

  REQUIRE(ref == val);
}

C2H_TEST(
  "Radix operations infere minimal value for pair types", "[radix][operations]", fundamental_types, fundamental_types)
{
  using tpl_t = ::cuda::std::tuple<typename c2h::get<0, TestType>, //
                                   typename c2h::get<1, TestType>>;

  using traits       = cub::detail::radix::traits_t<tpl_t>;
  using decomposer_t = tuple_decomposer_t<tpl_t>;

  tpl_t ref;
  tpl_to_min(ref);

  tpl_t val = traits::min_raw_binary_key(decomposer_t{});

  REQUIRE(ref == val);
}

/**
 * This tests checks that radix operations can get a value that when converted
 * to binary-comparable representation, yields largest possible value.
 */
C2H_TEST("Radix operations infere maximal value for fundamental types", "[radix][operations]", fundamental_types)
{
  using key_t        = typename c2h::get<0, TestType>;
  using traits       = cub::detail::radix::traits_t<key_t>;
  using decomposer_t = cub::detail::identity_decomposer_t;

  key_t ref = std::numeric_limits<key_t>::max();
  key_t val = traits::max_raw_binary_key(decomposer_t{});

  REQUIRE(ref == val);
}

C2H_TEST(
  "Radix operations infere maximal value for pair types", "[radix][operations]", fundamental_types, fundamental_types)
{
  using tpl_t = ::cuda::std::tuple<typename c2h::get<0, TestType>, //
                                   typename c2h::get<1, TestType>>;

  using traits       = cub::detail::radix::traits_t<tpl_t>;
  using decomposer_t = tuple_decomposer_t<tpl_t>;

  tpl_t ref;
  tpl_to_max(ref);

  tpl_t val = traits::max_raw_binary_key(decomposer_t{});

  REQUIRE(ref == val);
}

using fundamental_signed_types = c2h::type_list<std::int8_t, std::int16_t, std::int32_t, std::int64_t>;

/**
 * This tests checks that radix operations can convert a value to a binary-comparable
 * represetation. For example, `42.0f` is larger than `-42.0f`, but if we look at the
 * binary representation, it's not the case because of the sign bit:
 *
 *         s< exp. ><----- mantissa ------>
 *  42.0f: 01000010001010000000000000000000
 * -42.0f: 11000010001010000000000000000000
 *
 */
C2H_TEST("Radix operations reorder values for pair types",
         "[radix][operations]",
         fundamental_signed_types,
         fundamental_signed_types)
{
  using T1    = typename c2h::get<0, TestType>;
  using UT1   = typename std::make_unsigned<T1>::type;
  using T2    = typename c2h::get<1, TestType>;
  using UT2   = typename std::make_unsigned<T2>::type;
  using tpl_t = ::cuda::std::tuple<T1, T2>;

  using traits            = cub::detail::radix::traits_t<tpl_t>;
  using conversion_policy = typename traits::bit_ordered_conversion_policy;
  using decomposer_t      = tuple_decomposer_t<tpl_t>;

  std::bitset<sizeof(T1) * CHAR_BIT> bs_1;
  std::bitset<sizeof(T2) * CHAR_BIT> bs_2;

  // 10000(0)
  bs_1.set(sizeof(T1) * CHAR_BIT - 1);
  bs_2.set(sizeof(T2) * CHAR_BIT - 1);

  UT1 ul_1 = static_cast<UT1>(bs_1.to_ullong());
  UT2 ul_2 = static_cast<UT2>(bs_2.to_ullong());

  T1 l_1 = reinterpret_cast<T1&>(ul_1);
  T2 l_2 = reinterpret_cast<T2&>(ul_2);

  REQUIRE(l_1 == std::numeric_limits<T1>::lowest());
  REQUIRE(l_2 == std::numeric_limits<T2>::lowest());

  {
    tpl_t ref{T1{0}, T2{0}};
    const tpl_t unordered_val = tpl_t{l_1, l_2};
    const tpl_t ordered_val   = conversion_policy::to_bit_ordered(decomposer_t{}, unordered_val);

    REQUIRE(ref == ordered_val);

    const tpl_t restored_val = conversion_policy::from_bit_ordered(decomposer_t{}, ordered_val);
    REQUIRE(restored_val == unordered_val);
  }

  ul_1 = static_cast<UT1>(std::numeric_limits<T1>::max());
  ul_2 = static_cast<UT2>(std::numeric_limits<T2>::max());

  l_1 = reinterpret_cast<T1&>(ul_1);
  l_2 = reinterpret_cast<T2&>(ul_2);

  bs_1 = ul_1;
  bs_2 = ul_2;

  REQUIRE_FALSE(bs_1[sizeof(T1) * CHAR_BIT - 1]);
  REQUIRE_FALSE(bs_2[sizeof(T2) * CHAR_BIT - 1]);

  {
    const tpl_t unordered_val = tpl_t{l_1, l_2};
    const tpl_t ordered_val   = conversion_policy::to_bit_ordered(decomposer_t{}, unordered_val);

    ul_1 = reinterpret_cast<const UT1&>(::cuda::std::get<0>(ordered_val));
    ul_2 = reinterpret_cast<const UT2&>(::cuda::std::get<1>(ordered_val));

    REQUIRE(ul_1 == std::numeric_limits<UT1>::max());
    REQUIRE(ul_2 == std::numeric_limits<UT2>::max());

    const tpl_t restored_val = conversion_policy::from_bit_ordered(decomposer_t{}, ordered_val);
    REQUIRE(restored_val == unordered_val);
  }
}

struct fp_aggregate_t
{
  double fp64;
  float fp32;
};

struct fp_aggregate_decomposer_t
{
  __host__ __device__ ::cuda::std::tuple<double&, float&> operator()(fp_aggregate_t& val) const
  {
    return {val.fp64, val.fp32};
  }
};

struct flipped_fp_aggregate_decomposer_t
{
  __host__ __device__ ::cuda::std::tuple<float&, double&> operator()(fp_aggregate_t& val) const
  {
    return {val.fp32, val.fp64};
  }
};

/**
 * This tests checks radix sort guarantees to treat +0/-0 as the same value.
 */
TEST_CASE("Radix operations treat -0/+0 as being equal", "[radix][operations]")
{
  using traits            = cub::detail::radix::traits_t<fp_aggregate_t>;
  using conversion_policy = typename traits::bit_ordered_conversion_policy;
  using decomposer_t      = fp_aggregate_decomposer_t;
  using extractor_t       = cub::detail::radix::custom_digit_extractor_t<decomposer_t>;

  fp_aggregate_t negative{-0.0, -0.0f};
  fp_aggregate_t positive{+0.0, +0.0f};
  fp_aggregate_t ordered_negative = conversion_policy::to_bit_ordered(decomposer_t{}, negative);
  fp_aggregate_t ordered_positibe = conversion_policy::to_bit_ordered(decomposer_t{}, positive);

  constexpr int num_bits = CHAR_BIT;

  for (int bit = 0; bit < 8; bit += num_bits)
  {
    auto extractor = traits::digit_extractor<extractor_t>(bit, num_bits, decomposer_t{});

    const std::uint32_t digit_positive = extractor.Digit(ordered_positibe);
    const std::uint32_t digit_negative = extractor.Digit(ordered_negative);

    REQUIRE(digit_positive == digit_negative);
  }
}

/**
 * This tests checks that radix operations respect the order of fields in the
 * tuple instead of looking at the binary key representation.
 */
TEST_CASE("Radix operations allow fields permutation", "[radix][operations]")
{
  using traits            = cub::detail::radix::traits_t<fp_aggregate_t>;
  using conversion_policy = typename traits::bit_ordered_conversion_policy;
  using decomposer_t      = flipped_fp_aggregate_decomposer_t;
  using extractor_t       = cub::detail::radix::custom_digit_extractor_t<decomposer_t>;

  fp_aggregate_t lhs{4.2, 2.4f};
  fp_aggregate_t rhs{2.4, 4.2f};

  REQUIRE(::cuda::std::tie(lhs.fp64, lhs.fp32) > cuda::std::tie(rhs.fp64, rhs.fp32));

  fp_aggregate_t ordered_lhs = conversion_policy::to_bit_ordered(decomposer_t{}, lhs);
  fp_aggregate_t ordered_rhs = conversion_policy::to_bit_ordered(decomposer_t{}, lhs);

  constexpr int num_bits       = CHAR_BIT;
  constexpr int aggregate_bits = (sizeof(float) + sizeof(double)) * CHAR_BIT;

  for (int current_bit = aggregate_bits - num_bits; current_bit >= 0; current_bit -= num_bits)
  {
    auto extractor = traits::digit_extractor<extractor_t>(current_bit, num_bits, decomposer_t{});

    const std::uint32_t digit_lhs = extractor.Digit(ordered_lhs);
    const std::uint32_t digit_rhs = extractor.Digit(ordered_rhs);

    if (digit_lhs == digit_rhs)
    {
      continue;
    }

    std::bitset<32> bs_lhs(digit_lhs);
    std::bitset<32> bs_rhs(digit_rhs);

    for (int bit = 31; bit >= 0; bit--)
    {
      REQUIRE_FALSE(bs_lhs[bit]);
      if (bs_rhs[bit])
      {
        return;
      }
    }
  }
}
