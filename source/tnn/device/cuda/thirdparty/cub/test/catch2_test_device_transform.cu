// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"
// above header needs to be included first

#include <cub/device/device_for.cuh>
#include <cub/device/device_transform.cuh>

#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sequence.h>
#include <thrust/zip_function.h>

#include <sstream>

#include "catch2_test_launch_helper.h"
#include <c2h/catch2_test_helper.cuh>
#include <c2h/custom_type.cuh>
#include <c2h/test_util_vec.cuh>

// %PARAM% TEST_LAUNCH lid 0:1:2

using cub::detail::transform::Algorithm;

template <Algorithm Alg>
struct policy_hub_for_alg
{
  struct max_policy : cub::ChainedPolicy<300, max_policy, max_policy>
  {
    static constexpr int min_bif         = 64 * 1024;
    static constexpr Algorithm algorithm = Alg;
    using algo_policy =
      ::cuda::std::_If<Alg == Algorithm::prefetch,
                       cub::detail::transform::prefetch_policy_t<256>,
                       cub::detail::transform::async_copy_policy_t<256>>;
  };
};

template <Algorithm Alg,
          typename Offset,
          typename... RandomAccessIteratorsIn,
          typename RandomAccessIteratorOut,
          typename TransformOp>
CUB_RUNTIME_FUNCTION static cudaError_t transform_many_with_alg_entry_point(
  void* d_temp_storage,
  size_t& temp_storage_bytes,
  ::cuda::std::tuple<RandomAccessIteratorsIn...> inputs,
  RandomAccessIteratorOut output,
  Offset num_items,
  TransformOp transform_op,
  cudaStream_t stream = nullptr)
{
  if (d_temp_storage == nullptr)
  {
    temp_storage_bytes = 1;
    return cudaSuccess;
  }

  constexpr bool RequiresStableAddress = false;
  return cub::detail::transform::dispatch_t<RequiresStableAddress,
                                            Offset,
                                            ::cuda::std::tuple<RandomAccessIteratorsIn...>,
                                            RandomAccessIteratorOut,
                                            TransformOp,
                                            policy_hub_for_alg<Alg>>{}
    .dispatch(inputs, output, num_items, transform_op, stream);
}

DECLARE_LAUNCH_WRAPPER(cub::DeviceTransform::Transform, transform_many);
DECLARE_LAUNCH_WRAPPER(cub::DeviceTransform::TransformStableArgumentAddresses, transform_many_stable);
DECLARE_TMPL_LAUNCH_WRAPPER(transform_many_with_alg_entry_point,
                            transform_many_with_alg,
                            ESCAPE_LIST(Algorithm Alg, typename Offset),
                            ESCAPE_LIST(Alg, Offset));

using algorithms =
  c2h::enum_type_list<Algorithm,
                      Algorithm::prefetch
#ifdef _CUB_HAS_TRANSFORM_UBLKCP
                      ,
                      Algorithm::ublkcp
#endif // _CUB_HAS_TRANSFORM_UBLKCP
                      >;

using offset_types = c2h::type_list<std::int32_t, std::int64_t>;

#ifdef _CUB_HAS_TRANSFORM_UBLKCP
#  define FILTER_UBLKCP                                \
    if (alg == Algorithm::ublkcp && ptx_version < 900) \
    {                                                  \
      return;                                          \
    }
#else // _CUB_HAS_TRANSFORM_UBLKCP
#  define FILTER_UBLKCP
#endif // _CUB_HAS_TRANSFORM_UBLKCP

#define FILTER_UNSUPPORTED_ALGS                                           \
  int ptx_version = 0;                                                    \
  REQUIRE(cub::PtxVersion(ptx_version) == cudaSuccess);                   \
  _CCCL_DIAG_PUSH                                                         \
  _CCCL_DIAG_SUPPRESS_MSVC(4127) /* conditional expression is constant */ \
  FILTER_UBLKCP                                                           \
  _CCCL_DIAG_POP

C2H_TEST("DeviceTransform::Transform BabelStream add",
         "[device][device_transform]",
         c2h::type_list<std::uint8_t, std::uint16_t, std::uint32_t, std::uint64_t /*, int3, double3*/>,
         offset_types,
         algorithms)
{
  using type         = typename c2h::get<0, TestType>;
  using offset_t     = typename c2h::get<1, TestType>;
  constexpr auto alg = c2h::get<2, TestType>::value;
  FILTER_UNSUPPORTED_ALGS
  const int num_items = GENERATE(0, 1, 15, 16, 17, 127, 128, 129, 4095, 4096, 4097); // edge cases around 16 and 128
  CAPTURE(c2h::demangle(typeid(type).name()), c2h::demangle(typeid(offset_t).name()), alg, num_items);

  c2h::device_vector<type> a(num_items);
  c2h::device_vector<type> b(num_items);
  c2h::gen(C2H_SEED(1), a);
  c2h::gen(C2H_SEED(1), b);

  c2h::device_vector<type> result(num_items);
  transform_many_with_alg<alg, offset_t>(
    ::cuda::std::make_tuple(a.begin(), b.begin()), result.begin(), num_items, ::cuda::std::plus<type>{});

  // compute reference and verify
  c2h::host_vector<type> a_h = a;
  c2h::host_vector<type> b_h = b;
  c2h::host_vector<type> reference_h(num_items);
  std::transform(a_h.begin(), a_h.end(), b_h.begin(), reference_h.begin(), std::plus<type>{});
  REQUIRE(reference_h == result);
}

template <int Alignment>
struct alignas(Alignment) overaligned_addable_t
{
  int value;

  overaligned_addable_t() = default;

  _CCCL_HOST_DEVICE overaligned_addable_t(int val)
      : value{val}
  {}

  _CCCL_HOST_DEVICE static void check(const overaligned_addable_t& obj)
  {
    if (reinterpret_cast<uintptr_t>(&obj) % Alignment != 0)
    {
      printf("Error: object not aligned to %d: %p\n", Alignment, &obj);
      ::cuda::std::terminate();
    }
  }

  _CCCL_HOST_DEVICE friend auto operator==(const overaligned_addable_t& a, const overaligned_addable_t& b) -> bool
  {
    check(a);
    check(b);
    return a.value == b.value;
  }

  _CCCL_HOST_DEVICE friend auto
  operator+(const overaligned_addable_t& a, const overaligned_addable_t& b) -> overaligned_addable_t
  {
    check(a);
    check(b);
    return overaligned_addable_t{a.value + b.value};
  }

  _CCCL_HOST friend auto operator<<(std::ostream& os, const overaligned_addable_t& obj) -> std::ostream&
  {
    check(obj);
    return os << "over{" << obj.value << "}";
  }
};

using overaligned_types =
  c2h::type_list<overaligned_addable_t<32>
#ifndef _CCCL_COMPILER_MSVC // error C2719: [...] formal parameter with requested alignment of 256 won't be aligned
                 ,
                 overaligned_addable_t<256>
#endif // _CCCL_COMPILER_MSVC
                 >;

// test with types exceeding the memcpy_async and bulk copy alignments (16 and 128 bytes respectively)
C2H_TEST("DeviceTransform::Transform overaligned type", "[device][device_transform]", overaligned_types)
{
  using type = c2h::get<0, TestType>;
  CAPTURE(c2h::demangle(typeid(type).name()));

  const int num_items = GENERATE(0, 1, 100, 1000);
  c2h::device_vector<int> a(num_items, 3); // put some integers at the front, so SMEM has to handle different alignments
  c2h::device_vector<type> b(num_items, 4);

  c2h::device_vector<type> result(num_items);
  // we need raw pointers here to halfen the conversion sequence from device_reference<int> -> int -> type when calling
  // plus(...), which is too long to compile
  transform_many(::cuda::std::make_tuple(thrust::raw_pointer_cast(a.data()), thrust::raw_pointer_cast(b.data())),
                 result.begin(),
                 num_items,
                 ::cuda::std::plus<type>{});

  REQUIRE(result == c2h::device_vector<type>(num_items, 7));
}

C2H_TEST("DeviceTransform::Transform huge type", "[device][device_transform]")
{
  using huge_t = c2h::custom_type_t<c2h::equal_comparable_t, c2h::accumulateable_t, c2h::huge_data<666>::type>;
  static_assert(alignof(huge_t) == 8, "Need a large type with alignment < 16");
  CAPTURE(c2h::demangle(typeid(huge_t).name()));

  const int num_items = GENERATE(0, 1, 100, 1000);
  c2h::device_vector<huge_t> a(num_items);
  c2h::device_vector<huge_t> b(num_items);
  c2h::gen(C2H_SEED(1), a);
  c2h::gen(C2H_SEED(1), b);

  c2h::device_vector<huge_t> result(num_items);
  transform_many(::cuda::std::make_tuple(a.begin(), b.begin()), result.begin(), num_items, ::cuda::std::plus<huge_t>{});

  c2h::host_vector<huge_t> a_h = a;
  c2h::host_vector<huge_t> b_h = b;
  c2h::host_vector<huge_t> reference_h(num_items);
  std::transform(a_h.begin(), a_h.end(), b_h.begin(), reference_h.begin(), std::plus<huge_t>{});
  REQUIRE(result == reference_h);
}

struct times_seven
{
  _CCCL_HOST_DEVICE auto operator()(unsigned char v) const -> char
  {
    return static_cast<unsigned char>(v * 7);
  }
};

C2H_TEST("DeviceTransform::Transform with large input", "[device][device_transform]", algorithms)
try
{
  using type         = unsigned char;
  using offset_t     = cuda::std::int64_t;
  constexpr auto alg = c2h::get<0, TestType>::value;
  FILTER_UNSUPPORTED_ALGS
  CAPTURE(alg);

  constexpr offset_t num_items = (offset_t{1} << 32) + 123456; // a few thread blocks beyond 4GiB
  c2h::device_vector<type> input(num_items);
  c2h::gen(C2H_SEED(1), input);

  c2h::device_vector<type> result(num_items);
  transform_many_with_alg<alg, offset_t>(
    ::cuda::std::make_tuple(input.begin()), result.begin(), num_items, times_seven{});

  // compute reference and verify
  c2h::host_vector<type> input_h = input;
  c2h::host_vector<type> reference_h(num_items);
  std::transform(input_h.begin(), input_h.end(), reference_h.begin(), times_seven{});
  REQUIRE((reference_h == result));
}
catch (const std::bad_alloc&)
{
  // allocation failure is not a test failure, so we can run tests on smaller GPUs
}

template <typename T>
struct nstream_kernel
{
  static constexpr T scalar = 42;

  _CCCL_HOST_DEVICE T operator()(const T& ai, const T& bi, const T& ci) const
  {
    return ai + bi + scalar * ci;
  }
};

// overwrites one input stream
C2H_TEST("DeviceTransform::Transform BabelStream nstream",
         "[device][device_transform]",
         c2h::type_list<std::uint8_t, std::uint16_t, std::uint32_t, std::uint64_t>,
         offset_types,
         algorithms)
{
  using type         = typename c2h::get<0, TestType>;
  using offset_t     = typename c2h::get<1, TestType>;
  constexpr auto alg = c2h::get<2, TestType>::value;
  FILTER_UNSUPPORTED_ALGS
  CAPTURE(c2h::demangle(typeid(type).name()), c2h::demangle(typeid(offset_t).name()), alg);

  const int num_items = GENERATE(0, 1, 100, 1000, 10000);
  c2h::device_vector<type> a(num_items);
  c2h::device_vector<type> b(num_items);
  c2h::device_vector<type> c(num_items);
  c2h::gen(C2H_SEED(1), a, type{10}, type{100});
  c2h::gen(C2H_SEED(1), b, type{10}, type{100});
  c2h::gen(C2H_SEED(1), c, type{10}, type{100});

  // copy to host before changing
  c2h::host_vector<type> a_h = a;
  c2h::host_vector<type> b_h = b;
  c2h::host_vector<type> c_h = c;

  transform_many_with_alg<alg, offset_t>(
    ::cuda::std::make_tuple(a.begin(), b.begin(), c.begin()), a.begin(), num_items, nstream_kernel<type>{});

  // compute reference and verify
  auto z = thrust::make_zip_iterator(a_h.begin(), b_h.begin(), c_h.begin());
  std::transform(z, z + num_items, a_h.begin(), thrust::make_zip_function(nstream_kernel<type>{}));
  REQUIRE(a_h == a);
}

struct sum_five
{
  __device__ auto operator()(std::int8_t a, std::int16_t b, std::int32_t c, std::int64_t d, float e) const -> double
  {
    return a + b + c + d + e;
  }
};

C2H_TEST("DeviceTransform::Transform add five streams", "[device][device_transform]", algorithms)
{
  using offset_t     = int;
  constexpr auto alg = c2h::get<0, TestType>::value;
  FILTER_UNSUPPORTED_ALGS

  constexpr int num_items = 100;
  c2h::device_vector<std::int8_t> a(num_items, 1);
  c2h::device_vector<std::int16_t> b(num_items, 2);
  c2h::device_vector<std::int32_t> c(num_items, 3);
  c2h::device_vector<std::int64_t> d(num_items, 4);
  c2h::device_vector<float> e(num_items, 5);

  c2h::device_vector<double> result(num_items);
  transform_many_with_alg<alg, offset_t>(
    ::cuda::std::make_tuple(a.begin(), b.begin(), c.begin(), d.begin(), e.begin()),
    result.begin(),
    num_items,
    sum_five{});

  // compute reference and verify
  c2h::device_vector<double> reference(num_items, 1 + 2 + 3 + 4 + 5);
  REQUIRE(reference == result);
}

struct give_me_five
{
  __device__ auto operator()() const -> int
  {
    return 5;
  }
};

C2H_TEST("DeviceTransform::Transform no streams", "[device][device_transform]")
{
  constexpr int num_items = 100;
  c2h::device_vector<int> result(num_items);
  transform_many(::cuda::std::tuple<>{}, result.begin(), num_items, give_me_five{});

  // compute reference and verify
  c2h::device_vector<int> reference(num_items, 5);
  REQUIRE(reference == result);
}

C2H_TEST("DeviceTransform::Transform fancy input iterator types", "[device][device_transform]")
{
  using type = int;

  constexpr int num_items = 100;
  thrust::counting_iterator<type> a{0};
  thrust::counting_iterator<type> b{10};

  c2h::device_vector<type> result(num_items);
  transform_many(::cuda::std::make_tuple(a, b), result.begin(), num_items, ::cuda::std::plus<type>{});

  // compute reference and verify
  c2h::host_vector<type> reference_h(num_items);
  std::transform(a, a + num_items, b, reference_h.begin(), std::plus<type>{});
  REQUIRE(reference_h == result);
}

C2H_TEST("DeviceTransform::Transform fancy output iterator type", "[device][device_transform]", algorithms)
{
  using type         = int;
  using offset_t     = int;
  constexpr auto alg = c2h::get<0, TestType>::value;
  FILTER_UNSUPPORTED_ALGS

  constexpr int num_items = 100;
  c2h::device_vector<type> a(num_items, 13);
  c2h::device_vector<type> b(num_items, 35);
  c2h::device_vector<type> result(num_items);

  using thrust::placeholders::_1;
  auto out = thrust::make_transform_output_iterator(result.begin(), _1 + 4);
  transform_many_with_alg<alg, offset_t>(
    ::cuda::std::make_tuple(a.begin(), b.begin()), out, num_items, ::cuda::std::plus<type>{});
  REQUIRE(result == c2h::device_vector<type>(num_items, (13 + 35) + 4));
}

C2H_TEST("DeviceTransform::Transform mixed input iterator types", "[device][device_transform]")
{
  using type = int;

  constexpr int num_items = 100;
  thrust::counting_iterator<type> a{0};
  c2h::device_vector<type> b(num_items, 10);

  c2h::device_vector<type> result(num_items);
  transform_many(::cuda::std::make_tuple(a, b.begin()), result.begin(), num_items, ::cuda::std::plus<type>{});

  // compute reference and verify
  c2h::host_vector<type> b_h = b;
  c2h::host_vector<type> reference_h(num_items);
  std::transform(a, a + num_items, b_h.begin(), reference_h.begin(), std::plus<type>{});
  REQUIRE(reference_h == result);
}

struct plus_needs_stable_address
{
  int* a;
  int* b;

  _CCCL_HOST_DEVICE int operator()(const int& v) const
  {
    const auto i = &v - a;
    return v + b[i];
  }
};

C2H_TEST("DeviceTransform::Transform address stability", "[device][device_transform]")
{
  using type = int;

  constexpr int num_items = 100;
  c2h::device_vector<type> a(num_items);
  c2h::device_vector<type> b(num_items);
  thrust::sequence(a.begin(), a.end());
  thrust::sequence(b.begin(), b.end(), 42);

  c2h::device_vector<type> result(num_items);
  transform_many_stable(
    ::cuda::std::make_tuple(thrust::raw_pointer_cast(a.data())),
    result.begin(),
    num_items,
    plus_needs_stable_address{thrust::raw_pointer_cast(a.data()), thrust::raw_pointer_cast(b.data())});

  // compute reference and verify
  c2h::device_vector<type> a_h = a;
  c2h::device_vector<type> b_h = b;
  c2h::host_vector<type> reference_h(num_items);
  std::transform(a_h.begin(), a_h.end(), b_h.begin(), reference_h.begin(), std::plus<type>{});
  REQUIRE(reference_h == result);
}

// Non-trivially-copyable/relocatable type which cannot be copied using std::memcpy or cudaMemcpy
struct non_trivial
{
  int data;

  non_trivial() = default;

  _CCCL_HOST_DEVICE explicit non_trivial(int data)
      : data(data)
  {}

  _CCCL_HOST_DEVICE non_trivial(const non_trivial& nt)
      : data(nt.data)
  {}

  _CCCL_HOST_DEVICE auto operator=(const non_trivial& nt) -> non_trivial&
  {
    data = nt.data;
    return *this;
  }

  _CCCL_HOST_DEVICE auto operator-() const -> non_trivial
  {
    return non_trivial{-data};
  }

  friend _CCCL_HOST_DEVICE auto operator==(non_trivial a, non_trivial b) -> bool
  {
    return a.data == b.data;
  }
};
static_assert(!::cuda::std::is_trivially_copyable<non_trivial>::value, ""); // as required by the standard
static_assert(!thrust::is_trivially_relocatable<non_trivial>::value, ""); // CUB uses this check internally

// Note(bgruber): I gave up on writing a test that checks whether the copy ctor/assignment operator is actually called
// (e.g. by tracking/counting invocations of those), since C++ allows (but not guarantees) elision of these operations.
// Also thrust algorithms perform a lot of copies in-between, so the test needs to use only raw allocations and
// iteration for setup and checking.
C2H_TEST("DeviceTransform::Transform not trivially relocatable", "[device][device_transform]")
{
  constexpr int num_items = 100;
  c2h::device_vector<non_trivial> input(num_items, non_trivial{42});
  c2h::device_vector<non_trivial> result(num_items);
  transform_many(
    ::cuda::std::make_tuple(thrust::raw_pointer_cast(input.data())), result.begin(), num_items, ::cuda::std::negate<>{});

  const auto reference = c2h::device_vector<non_trivial>(num_items, non_trivial{-42});
  REQUIRE((reference == result));
}

C2H_TEST("DeviceTransform::Transform buffer start alignment",
         "[device][device_transform]",
         c2h::type_list<std::uint8_t, std::uint16_t, float, double>)
{
  using type = typename c2h::get<0, TestType>;

  constexpr int num_items = 1000;
  const int offset        = GENERATE(1, 2, 4, 8, 16, 32, 64, 128); // global memory is always at least 256 byte aligned
  CAPTURE(c2h::demangle(typeid(type).name()), offset);
  c2h::device_vector<type> input(num_items);
  thrust::sequence(input.begin(), input.end());
  c2h::device_vector<type> result(num_items);
  using thrust::placeholders::_1;
  transform_many(::cuda::std::make_tuple(input.begin() + offset),
                 result.begin() + offset,
                 num_items - offset,
                 _1 * 10); // FIXME(bgruber): does not work on negative

  c2h::device_vector<type> reference(num_items);
  thrust::tabulate(reference.begin() + offset, reference.end(), (_1 + offset) * 10);
  REQUIRE(reference == result);
}

namespace Catch
{
template <typename T>
struct StringMaker<cub::detail::transform::aligned_base_ptr<T>>
{
  static auto convert(cub::detail::transform::aligned_base_ptr<T> abp) -> std::string
  {
    std::stringstream ss;
    ss << "{ptr: " << abp.ptr << ", head_padding: " << abp.head_padding << "}";
    return ss.str();
  }
};
} // namespace Catch

// TODO(bgruber): rewrite this example using int3
C2H_TEST("DeviceTransform::Transform aligned_base_ptr", "[device][device_transform]")
{
  alignas(128) int arr[256];
  using namespace cub::detail::transform;
  CHECK(make_aligned_base_ptr(&arr[0], 128) == aligned_base_ptr<int>{reinterpret_cast<char*>(&arr[0]), 0});
  CHECK(make_aligned_base_ptr(&arr[1], 128) == aligned_base_ptr<int>{reinterpret_cast<char*>(&arr[0]), 4});
  CHECK(make_aligned_base_ptr(&arr[5], 128) == aligned_base_ptr<int>{reinterpret_cast<char*>(&arr[0]), 20});
  CHECK(make_aligned_base_ptr(&arr[31], 128) == aligned_base_ptr<int>{reinterpret_cast<char*>(&arr[0]), 124});
  CHECK(make_aligned_base_ptr(&arr[32], 128) == aligned_base_ptr<int>{reinterpret_cast<char*>(&arr[32]), 0});
  CHECK(make_aligned_base_ptr(&arr[33], 128) == aligned_base_ptr<int>{reinterpret_cast<char*>(&arr[32]), 4});
  CHECK(make_aligned_base_ptr(&arr[127], 128) == aligned_base_ptr<int>{reinterpret_cast<char*>(&arr[96]), 124});
  CHECK(make_aligned_base_ptr(&arr[128], 128) == aligned_base_ptr<int>{reinterpret_cast<char*>(&arr[128]), 0});
  CHECK(make_aligned_base_ptr(&arr[129], 128) == aligned_base_ptr<int>{reinterpret_cast<char*>(&arr[128]), 4});
}
