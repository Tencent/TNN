// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if defined(_CCCL_CUDA_COMPILER) && _CCCL_CUDACC_VER < 1105000
_CCCL_NV_DIAG_SUPPRESS(186)
#  include <cuda_pipeline_primitives.h>
// we cannot re-enable the warning here, because it is triggered outside the translation unit
// see also: https://godbolt.org/z/1x8b4hn3G
#endif // defined(_CCCL_CUDA_COMPILER) && _CCCL_CUDACC_VER < 1105000

#include <cub/detail/uninitialized_copy.cuh>
#include <cub/util_arch.cuh>
#include <cub/util_device.cuh>
#include <cub/util_math.cuh>
#include <cub/util_type.cuh>

#include <thrust/detail/raw_reference_cast.h>
#include <thrust/system/cuda/detail/core/triple_chevron_launch.h>
#include <thrust/type_traits/is_contiguous_iterator.h>
#include <thrust/type_traits/is_trivially_relocatable.h>

#include <cuda/cmath>
#include <cuda/ptx>
#include <cuda/std/__algorithm/clamp.h>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/array>
#include <cuda/std/bit>
#include <cuda/std/expected>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include <cassert>

// cooperative groups do not support NVHPC yet
#ifndef _CCCL_CUDA_COMPILER_NVHPC
#  include <cooperative_groups.h>
#  include <cooperative_groups/memcpy_async.h>
#endif

CUB_NAMESPACE_BEGIN

// The ublkcp kernel needs PTX features that are only available and understood by nvcc >=12.
// Also, cooperative groups do not support NVHPC yet.
#if _CCCL_CUDACC_VER_MAJOR >= 12 && !defined(_CCCL_CUDA_COMPILER_NVHPC)
#  define _CUB_HAS_TRANSFORM_UBLKCP
#endif // _CCCL_CUDACC_VER_MAJOR >= 12 && !defined(_CCCL_CUDA_COMPILER_NVHPC)

namespace detail
{
namespace transform
{
_CCCL_HOST_DEVICE constexpr int sum()
{
  return 0;
}

// TODO(bgruber): remove with C++17
template <typename... Ts>
_CCCL_HOST_DEVICE constexpr int sum(int head, Ts... tail)
{
  return head + sum(tail...);
}

#if _CCCL_STD_VER >= 2017
template <typename... Its>
_CCCL_HOST_DEVICE constexpr auto loaded_bytes_per_iteration() -> int
{
  return (int{sizeof(value_t<Its>)} + ... + 0);
}
#else // ^^^ C++17 ^^^ / vvv C++11 vvv
template <typename... Its>
_CCCL_HOST_DEVICE constexpr auto loaded_bytes_per_iteration() -> int
{
  return sum(int{sizeof(value_t<Its>)}...);
}
#endif // _CCCL_STD_VER >= 2017

enum class Algorithm
{
  // We previously had a fallback algorithm that would use cub::DeviceFor. Benchmarks showed that the prefetch algorithm
  // is always superior to that fallback, so it was removed.
  prefetch,
#ifdef _CUB_HAS_TRANSFORM_UBLKCP
  ublkcp,
#endif // _CUB_HAS_TRANSFORM_UBLKCP
};

template <typename T>
_CCCL_HOST_DEVICE _CCCL_FORCEINLINE const char* round_down_ptr(const T* ptr, unsigned alignment)
{
#if _CCCL_STD_VER > 2011
  _CCCL_ASSERT(::cuda::std::has_single_bit(alignment), "");
#endif // _CCCL_STD_VER > 2011
  return reinterpret_cast<const char*>(
    reinterpret_cast<::cuda::std::uintptr_t>(ptr) & ~::cuda::std::uintptr_t{alignment - 1});
}

template <int BlockThreads>
struct prefetch_policy_t
{
  static constexpr int block_threads = BlockThreads;
  // items per tile are determined at runtime. these (inclusive) bounds allow overriding that value via a tuning policy
  static constexpr int items_per_thread_no_input = 2; // when there are no input iterators, the kernel is just filling
  static constexpr int min_items_per_thread      = 1;
  static constexpr int max_items_per_thread      = 32;
};

// Prefetches (at least on Hopper) a 128 byte cache line. Prefetching out-of-bounds addresses has no side effects
// TODO(bgruber): there is also the cp.async.bulk.prefetch instruction available on Hopper. May improve perf a tiny bit
// as we need to create less instructions to prefetch the same amount of data.
template <typename T>
_CCCL_DEVICE _CCCL_FORCEINLINE void prefetch(const T* addr)
{
  // TODO(bgruber): prefetch to L1 may be even better
  asm volatile("prefetch.global.L2 [%0];" : : "l"(__cvta_generic_to_global(addr)) : "memory");
}

template <int BlockDim, typename T>
_CCCL_DEVICE _CCCL_FORCEINLINE void prefetch_tile(const T* addr, int tile_size)
{
  constexpr int prefetch_byte_stride = 128; // TODO(bgruber): should correspond to cache line size. Does this need to be
                                            // architecture dependent?
  const int tile_size_bytes = tile_size * sizeof(T);
  // prefetch does not stall and unrolling just generates a lot of unnecessary computations and predicate handling
#pragma unroll 1
  for (int offset = threadIdx.x * prefetch_byte_stride; offset < tile_size_bytes;
       offset += BlockDim * prefetch_byte_stride)
  {
    prefetch(reinterpret_cast<const char*>(addr) + offset);
  }
}

// TODO(miscco): we should probably constrain It to not be a contiguous iterator in C++17 (and change the overload
// above to accept any contiguous iterator)
// overload for any iterator that is not a pointer, do nothing
template <int, typename It, ::cuda::std::__enable_if_t<!::cuda::std::is_pointer<It>::value, int> = 0>
_CCCL_DEVICE _CCCL_FORCEINLINE void prefetch_tile(It, int)
{}

// This kernel guarantees that objects passed as arguments to the user-provided transformation function f reside in
// global memory. No intermediate copies are taken. If the parameter type of f is a reference, taking the address of the
// parameter yields a global memory address.
template <typename PrefetchPolicy,
          typename Offset,
          typename F,
          typename RandomAccessIteratorOut,
          typename... RandomAccessIteratorIn>
_CCCL_DEVICE void transform_kernel_impl(
  ::cuda::std::integral_constant<Algorithm, Algorithm::prefetch>,
  Offset num_items,
  int num_elem_per_thread,
  F f,
  RandomAccessIteratorOut out,
  RandomAccessIteratorIn... ins)
{
  constexpr int block_dim = PrefetchPolicy::block_threads;
  const int tile_stride   = block_dim * num_elem_per_thread;
  const Offset offset     = static_cast<Offset>(blockIdx.x) * tile_stride;
  const int tile_size     = static_cast<int>(::cuda::std::min(num_items - offset, Offset{tile_stride}));

  // move index and iterator domain to the block/thread index, to reduce arithmetic in the loops below
  {
    int dummy[] = {(ins += offset, 0)..., 0};
    (void) &dummy;
    out += offset;
  }

  {
    // TODO(bgruber): replace by fold over comma in C++17
    int dummy[] = {(prefetch_tile<block_dim>(ins, tile_size), 0)..., 0}; // extra zero to handle empty packs
    (void) &dummy; // nvcc 11.1 needs extra strong unused warning suppression
  }

#define PREFETCH_AGENT(full_tile)                                                                                  \
  /* ahendriksen: various unrolling yields less <1% gains at much higher compile-time cost */                      \
  /* bgruber: but A6000 and H100 show small gains without pragma */                                                \
  /*_Pragma("unroll 1")*/ for (int j = 0; j < num_elem_per_thread; ++j)                                            \
  {                                                                                                                \
    const int idx = j * block_dim + threadIdx.x;                                                                   \
    if (full_tile || idx < tile_size)                                                                              \
    {                                                                                                              \
      /* we have to unwrap Thrust's proxy references here for backward compatibility (try zip_iterator.cu test) */ \
      out[idx] = f(THRUST_NS_QUALIFIER::raw_reference_cast(ins[idx])...);                                          \
    }                                                                                                              \
  }

  if (tile_stride == tile_size)
  {
    PREFETCH_AGENT(true);
  }
  else
  {
    PREFETCH_AGENT(false);
  }
#undef PREFETCH_AGENT
}

template <int BlockThreads>
struct async_copy_policy_t
{
  static constexpr int block_threads = BlockThreads;
  // items per tile are determined at runtime. these (inclusive) bounds allow overriding that value via a tuning policy
  static constexpr int min_items_per_thread = 1;
  static constexpr int max_items_per_thread = 32;
};

// TODO(bgruber) cheap copy of ::cuda::std::apply, which requires C++17.
template <class F, class Tuple, std::size_t... Is>
_CCCL_DEVICE _CCCL_FORCEINLINE auto poor_apply_impl(F&& f, Tuple&& t, ::cuda::std::index_sequence<Is...>)
  -> decltype(::cuda::std::forward<F>(f)(::cuda::std::get<Is>(::cuda::std::forward<Tuple>(t))...))
{
  return ::cuda::std::forward<F>(f)(::cuda::std::get<Is>(::cuda::std::forward<Tuple>(t))...);
}

template <class F, class Tuple>
_CCCL_DEVICE _CCCL_FORCEINLINE auto poor_apply(F&& f, Tuple&& t)
  -> decltype(poor_apply_impl(
    ::cuda::std::forward<F>(f),
    ::cuda::std::forward<Tuple>(t),
    ::cuda::std::make_index_sequence<::cuda::std::tuple_size<::cuda::std::__libcpp_remove_reference_t<Tuple>>::value>{}))
{
  return poor_apply_impl(
    ::cuda::std::forward<F>(f),
    ::cuda::std::forward<Tuple>(t),
    ::cuda::std::make_index_sequence<::cuda::std::tuple_size<::cuda::std::__libcpp_remove_reference_t<Tuple>>::value>{});
}

// mult must be a power of 2
template <typename Integral>
_CCCL_HOST_DEVICE _CCCL_FORCEINLINE constexpr auto round_up_to_po2_multiple(Integral x, Integral mult) -> Integral
{
#if _CCCL_STD_VER > 2011
  _CCCL_ASSERT(::cuda::std::has_single_bit(static_cast<::cuda::std::__make_unsigned_t<Integral>>(mult)), "");
#endif // _CCCL_STD_VER > 2011
  return (x + mult - 1) & ~(mult - 1);
}

// Implementation notes on memcpy_async and UBLKCP kernels regarding copy alignment and padding
//
// For performance considerations of memcpy_async:
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#performance-guidance-for-memcpy-async
//
// We basically have to align the base pointer to 16 bytes, and copy a multiple of 16 bytes. To achieve this, when we
// copy a tile of data from an input buffer, we round down the pointer to the start of the tile to the next lower
// address that is a multiple of 16 bytes. This introduces head padding. We also round up the total number of bytes to
// copy (including head padding) to a multiple of 16 bytes, which introduces tail padding. For the bulk copy kernel, we
// have to align to 128 bytes instead of 16.
//
// However, padding memory copies like that may access the input buffer out-of-bounds. Here are some thoughts:
// * According to the CUDA programming guide
// (https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-accesses), "any address of a variable
// residing in global memory or returned by one of the memory allocation routines from the driver or runtime API is
// always aligned to at least 256 bytes."
// * Memory protection is usually done on memory page level, which is even larger than 256 bytes for CUDA and 4KiB on
// Intel x86 and 4KiB+ ARM. Front and tail padding thus never leaves the memory page of the input buffer.
// * This should count for device memory, but also for device accessible memory living on the host.
// * The base pointer alignment and size rounding also never leaves the size of a cache line.
//
// Copying larger data blocks with head and tail padding should thus be legal. Nevertheless, an out-of-bounds read is
// still technically undefined behavior in C++. Also, compute-sanitizer flags at least such reads after the end of a
// buffer. Therefore, we lean on the safer side and protect against out of bounds reads at the beginning and end.

// A note on size and alignment: The size of a type is at least as large as its alignment. We rely on this fact in some
// conditions.
// This is guaranteed by the C++ standard, and follows from the definition of arrays: the difference between neighboring
// array element addresses is sizeof element type and each array element needs to fulfill the alignment requirement of
// the element type.

// Pointer with metadata to describe readonly input memory for memcpy_async and UBLKCP kernels.
// cg::memcpy_async is most efficient when the data is 16-byte aligned and the size a multiple of 16 bytes
// UBLKCP is most efficient when the data is 128-byte aligned and the size a multiple of 16 bytes
template <typename T> // Cannot add alignment to signature, because we need a uniform kernel template instantiation
struct aligned_base_ptr
{
  using value_type = T;

  const char* ptr; // aligned pointer before the original pointer (16-byte or 128-byte). May not be aligned to
                   // alignof(T). E.g.: array of int3 starting at address 4, ptr == 0
  int head_padding; // byte offset between ptr and the original pointer. Value inside [0;15] or [0;127].

  _CCCL_HOST_DEVICE const T* ptr_to_elements() const
  {
    return reinterpret_cast<const T*>(ptr + head_padding);
  }

  _CCCL_HOST_DEVICE friend bool operator==(const aligned_base_ptr& a, const aligned_base_ptr& b)
  {
    return a.ptr == b.ptr && a.head_padding == b.head_padding;
  }
};

template <typename T>
_CCCL_HOST_DEVICE auto make_aligned_base_ptr(const T* ptr, int alignment) -> aligned_base_ptr<T>
{
  const char* base_ptr = round_down_ptr(ptr, alignment);
  return aligned_base_ptr<T>{base_ptr, static_cast<int>(reinterpret_cast<const char*>(ptr) - base_ptr)};
}

constexpr int bulk_copy_alignment     = 128;
constexpr int bulk_copy_size_multiple = 16;

#ifdef _CUB_HAS_TRANSFORM_UBLKCP
_CCCL_DEVICE _CCCL_FORCEINLINE static bool elect_one()
{
  const ::cuda::std::uint32_t membermask = ~0;
  ::cuda::std::uint32_t is_elected;
  asm volatile(
    "{\n\t .reg .pred P_OUT; \n\t"
    "elect.sync _|P_OUT, %1;\n\t"
    "selp.b32 %0, 1, 0, P_OUT; \n"
    "}"
    : "=r"(is_elected)
    : "r"(membermask)
    :);
  return threadIdx.x < 32 && static_cast<bool>(is_elected);
}

// TODO(bgruber): inline this as lambda in C++14
template <typename Offset, typename T>
_CCCL_DEVICE void bulk_copy_tile(
  ::cuda::std::uint64_t& bar,
  int tile_stride,
  char* smem,
  int& smem_offset,
  ::cuda::std::uint32_t& total_bytes_bulk_copied,
  Offset global_offset,
  const aligned_base_ptr<T>& aligned_ptr)
{
  static_assert(alignof(T) <= bulk_copy_alignment, "");

  const char* src = aligned_ptr.ptr + global_offset * sizeof(T);
  char* dst       = smem + smem_offset;
  _CCCL_ASSERT(reinterpret_cast<uintptr_t>(src) % bulk_copy_alignment == 0, "");
  _CCCL_ASSERT(reinterpret_cast<uintptr_t>(dst) % bulk_copy_alignment == 0, "");

  // TODO(bgruber): we could precompute bytes_to_copy on the host
  const int bytes_to_copy = round_up_to_po2_multiple(
    aligned_ptr.head_padding + static_cast<int>(sizeof(T)) * tile_stride, bulk_copy_size_multiple);

  ::cuda::ptx::cp_async_bulk(::cuda::ptx::space_cluster, ::cuda::ptx::space_global, dst, src, bytes_to_copy, &bar);
  total_bytes_bulk_copied += bytes_to_copy;

  // add bulk_copy_alignment to make space for the next tile's head padding
  smem_offset += static_cast<int>(sizeof(T)) * tile_stride + bulk_copy_alignment;
}

template <typename Offset, typename T>
_CCCL_DEVICE void bulk_copy_tile_fallback(
  int tile_size,
  int tile_stride,
  char* smem,
  int& smem_offset,
  Offset global_offset,
  const aligned_base_ptr<T>& aligned_ptr)
{
  const T* src = aligned_ptr.ptr_to_elements() + global_offset;
  T* dst       = reinterpret_cast<T*>(smem + smem_offset + aligned_ptr.head_padding);
  _CCCL_ASSERT(reinterpret_cast<uintptr_t>(src) % alignof(T) == 0, "");
  _CCCL_ASSERT(reinterpret_cast<uintptr_t>(dst) % alignof(T) == 0, "");

  const int bytes_to_copy = static_cast<int>(sizeof(T)) * tile_size;
  cooperative_groups::memcpy_async(cooperative_groups::this_thread_block(), dst, src, bytes_to_copy);

  // add bulk_copy_alignment to make space for the next tile's head padding
  smem_offset += static_cast<int>(sizeof(T)) * tile_stride + bulk_copy_alignment;
}

// TODO(bgruber): inline this as lambda in C++14
template <typename T>
_CCCL_DEVICE _CCCL_FORCEINLINE const T&
fetch_operand(int tile_stride, const char* smem, int& smem_offset, int smem_idx, const aligned_base_ptr<T>& aligned_ptr)
{
  const T* smem_operand_tile_base = reinterpret_cast<const T*>(smem + smem_offset + aligned_ptr.head_padding);
  smem_offset += int{sizeof(T)} * tile_stride + bulk_copy_alignment;
  return smem_operand_tile_base[smem_idx];
}

template <typename BulkCopyPolicy, typename Offset, typename F, typename RandomAccessIteratorOut, typename... InTs>
_CCCL_DEVICE void transform_kernel_ublkcp(
  Offset num_items, int num_elem_per_thread, F f, RandomAccessIteratorOut out, aligned_base_ptr<InTs>... aligned_ptrs)
{
  __shared__ uint64_t bar;
  extern __shared__ char __align__(bulk_copy_alignment) smem[];

  namespace ptx = ::cuda::ptx;

  constexpr int block_dim = BulkCopyPolicy::block_threads;
  const int tile_stride   = block_dim * num_elem_per_thread;
  const Offset offset     = static_cast<Offset>(blockIdx.x) * tile_stride;
  const int tile_size     = ::cuda::std::min(num_items - offset, Offset{tile_stride});

  const bool inner_blocks = 0 < blockIdx.x && blockIdx.x + 2 < gridDim.x;
  if (inner_blocks)
  {
    // use one thread to setup the entire bulk copy
    if (elect_one())
    {
      ptx::mbarrier_init(&bar, 1);
      ptx::fence_proxy_async(ptx::space_shared);

      int smem_offset                    = 0;
      ::cuda::std::uint32_t total_copied = 0;

      // TODO(bgruber): use a fold over comma in C++17
      // Order of evaluation is left-to-right
      int dummy[] = {(bulk_copy_tile(bar, tile_stride, smem, smem_offset, total_copied, offset, aligned_ptrs), 0)...,
                     0};
      (void) dummy;

      // TODO(ahendriksen): this could only have ptx::sem_relaxed, but this is not available yet
      ptx::mbarrier_arrive_expect_tx(ptx::sem_release, ptx::scope_cta, ptx::space_shared, &bar, total_copied);
    }

    // all threads wait for bulk copy
    __syncthreads();
    while (!ptx::mbarrier_try_wait_parity(&bar, 0))
      ;
  }
  else
  {
    // use all threads to schedule an async_memcpy
    int smem_offset = 0;

    // TODO(bgruber): use a fold over comma in C++17
    // Order of evaluation is left-to-right
    int dummy[] = {(bulk_copy_tile_fallback(tile_size, tile_stride, smem, smem_offset, offset, aligned_ptrs), 0)..., 0};
    (void) dummy;

    cooperative_groups::wait(cooperative_groups::this_thread_block());
  }

  // move the whole index and iterator to the block/thread index, to reduce arithmetic in the loops below
  out += offset;

  // note: I tried expressing the UBLKCP_AGENT as a function object but it adds a lot of code to handle the variadics
  // TODO(bgruber): use a polymorphic lambda in C++14
#  define UBLKCP_AGENT(full_tile)                                                                            \
    /* Unroll 1 tends to improve performance, especially for smaller data types (confirmed by benchmark) */  \
    _CCCL_PRAGMA(unroll 1)                                                                                   \
    for (int j = 0; j < num_elem_per_thread; ++j)                                                            \
    {                                                                                                        \
      const int idx = j * block_dim + threadIdx.x;                                                           \
      if (full_tile || idx < tile_size)                                                                      \
      {                                                                                                      \
        int smem_offset = 0;                                                                                 \
        /* need to expand into a tuple for guaranteed order of evaluation*/                                  \
        out[idx] = poor_apply(                                                                               \
          [&](const InTs&... values) {                                                                       \
            return f(values...);                                                                             \
          },                                                                                                 \
          ::cuda::std::tuple<InTs...>{fetch_operand(tile_stride, smem, smem_offset, idx, aligned_ptrs)...}); \
      }                                                                                                      \
    }
  if (tile_stride == tile_size)
  {
    UBLKCP_AGENT(true);
  }
  else
  {
    UBLKCP_AGENT(false);
  }
#  undef UBLKCP_AGENT
}

template <typename BulkCopyPolicy, typename Offset, typename F, typename RandomAccessIteratorOut, typename... InTs>
_CCCL_DEVICE void transform_kernel_impl(
  ::cuda::std::integral_constant<Algorithm, Algorithm::ublkcp>,
  Offset num_items,
  int num_elem_per_thread,
  F f,
  RandomAccessIteratorOut out,
  aligned_base_ptr<InTs>... aligned_ptrs)
{
  // only call the real kernel for sm90 and later
  NV_IF_TARGET(NV_PROVIDES_SM_90,
               (transform_kernel_ublkcp<BulkCopyPolicy>(num_items, num_elem_per_thread, f, out, aligned_ptrs...);));
}
#endif // _CUB_HAS_TRANSFORM_UBLKCP

template <typename It>
union kernel_arg
{
  aligned_base_ptr<value_t<It>> aligned_ptr;
  It iterator;

  _CCCL_HOST_DEVICE kernel_arg() {} // in case It is not default-constructible
};

template <typename It>
_CCCL_HOST_DEVICE auto make_iterator_kernel_arg(It it) -> kernel_arg<It>
{
  kernel_arg<It> arg;
  arg.iterator = it;
  return arg;
}

template <typename It>
_CCCL_HOST_DEVICE auto make_aligned_base_ptr_kernel_arg(It ptr, int alignment) -> kernel_arg<It>
{
  kernel_arg<It> arg;
  arg.aligned_ptr = make_aligned_base_ptr(ptr, alignment);
  return arg;
}

// TODO(bgruber): make a variable template in C++14
template <Algorithm Alg>
using needs_aligned_ptr_t =
  ::cuda::std::bool_constant<false
#ifdef _CUB_HAS_TRANSFORM_UBLKCP
                             || Alg == Algorithm::ublkcp
#endif // _CUB_HAS_TRANSFORM_UBLKCP
                             >;

#ifdef _CUB_HAS_TRANSFORM_UBLKCP
template <Algorithm Alg, typename It, ::cuda::std::__enable_if_t<needs_aligned_ptr_t<Alg>::value, int> = 0>
_CCCL_DEVICE _CCCL_FORCEINLINE auto select_kernel_arg(
  ::cuda::std::integral_constant<Algorithm, Alg>, kernel_arg<It>&& arg) -> aligned_base_ptr<value_t<It>>&&
{
  return ::cuda::std::move(arg.aligned_ptr);
}
#endif // _CUB_HAS_TRANSFORM_UBLKCP

template <Algorithm Alg, typename It, ::cuda::std::__enable_if_t<!needs_aligned_ptr_t<Alg>::value, int> = 0>
_CCCL_DEVICE _CCCL_FORCEINLINE auto
select_kernel_arg(::cuda::std::integral_constant<Algorithm, Alg>, kernel_arg<It>&& arg) -> It&&
{
  return ::cuda::std::move(arg.iterator);
}

// There is only one kernel for all algorithms, that dispatches based on the selected policy. It must be instantiated
// with the same arguments for each algorithm. Only the device compiler will then select the implementation. This
// saves some compile-time and binary size.
template <typename MaxPolicy,
          typename Offset,
          typename F,
          typename RandomAccessIteratorOut,
          typename... RandomAccessIteartorsIn>
__launch_bounds__(MaxPolicy::ActivePolicy::algo_policy::block_threads)
  CUB_DETAIL_KERNEL_ATTRIBUTES void transform_kernel(
    Offset num_items,
    int num_elem_per_thread,
    F f,
    RandomAccessIteratorOut out,
    kernel_arg<RandomAccessIteartorsIn>... ins)
{
  constexpr auto alg = ::cuda::std::integral_constant<Algorithm, MaxPolicy::ActivePolicy::algorithm>{};
  transform_kernel_impl<typename MaxPolicy::ActivePolicy::algo_policy>(
    alg,
    num_items,
    num_elem_per_thread,
    ::cuda::std::move(f),
    ::cuda::std::move(out),
    select_kernel_arg(alg, ::cuda::std::move(ins))...);
}

constexpr int arch_to_min_bytes_in_flight(int sm_arch)
{
  // TODO(bgruber): use if-else in C++14 for better readability
  return sm_arch >= 900 ? 48 * 1024 // 32 for H100, 48 for H200
       : sm_arch >= 800 ? 16 * 1024 // A100
                        : 12 * 1024; // V100 and below
}

template <typename... RandomAccessIteratorsIn>
_CCCL_HOST_DEVICE constexpr auto bulk_copy_smem_for_tile_size(int tile_size) -> int
{
  return round_up_to_po2_multiple(int{sizeof(int64_t)}, bulk_copy_alignment) /* bar */
       // 128 bytes of padding for each input tile (handles before + after)
       + tile_size * loaded_bytes_per_iteration<RandomAccessIteratorsIn...>()
       + sizeof...(RandomAccessIteratorsIn) * bulk_copy_alignment;
}

template <bool RequiresStableAddress, typename RandomAccessIteratorTupleIn>
struct policy_hub
{
  static_assert(sizeof(RandomAccessIteratorTupleIn) == 0, "Second parameter must be a tuple");
};

template <bool RequiresStableAddress, typename... RandomAccessIteratorsIn>
struct policy_hub<RequiresStableAddress, ::cuda::std::tuple<RandomAccessIteratorsIn...>>
{
  static constexpr bool no_input_streams = sizeof...(RandomAccessIteratorsIn) == 0;
  static constexpr bool all_contiguous =
    ::cuda::std::conjunction<THRUST_NS_QUALIFIER::is_contiguous_iterator<RandomAccessIteratorsIn>...>::value;
  static constexpr bool all_values_trivially_reloc =
    ::cuda::std::conjunction<THRUST_NS_QUALIFIER::is_trivially_relocatable<value_t<RandomAccessIteratorsIn>>...>::value;

  static constexpr bool can_memcpy = all_contiguous && all_values_trivially_reloc;

  // TODO(bgruber): consider a separate kernel for just filling

  struct policy300 : ChainedPolicy<300, policy300, policy300>
  {
    static constexpr int min_bif = arch_to_min_bytes_in_flight(300);
    // TODO(bgruber): we don't need algo, because we can just detect the type of algo_policy
    static constexpr auto algorithm = Algorithm::prefetch;
    using algo_policy               = prefetch_policy_t<256>;
  };

#ifdef _CUB_HAS_TRANSFORM_UBLKCP
  // H100 and H200
  struct policy900 : ChainedPolicy<900, policy900, policy300>
  {
    static constexpr int min_bif = arch_to_min_bytes_in_flight(900);
    using async_policy           = async_copy_policy_t<256>;
    static constexpr bool exhaust_smem =
      bulk_copy_smem_for_tile_size<RandomAccessIteratorsIn...>(
        async_policy::block_threads * async_policy::min_items_per_thread)
      > 48 * 1024;
    static constexpr bool any_type_is_overalinged =
#  if _CCCL_STD_VER >= 2017
      ((alignof(value_t<RandomAccessIteratorsIn>) > bulk_copy_alignment) || ...);
#  else
      sum((alignof(value_t<RandomAccessIteratorsIn>) > bulk_copy_alignment)...) > 0;
#  endif

    static constexpr bool use_fallback =
      RequiresStableAddress || !can_memcpy || no_input_streams || exhaust_smem || any_type_is_overalinged;
    static constexpr auto algorithm = use_fallback ? Algorithm::prefetch : Algorithm::ublkcp;
    using algo_policy               = ::cuda::std::_If<use_fallback, prefetch_policy_t<256>, async_policy>;
  };

  using max_policy = policy900;
#else // _CUB_HAS_TRANSFORM_UBLKCP
  using max_policy = policy300;
#endif // _CUB_HAS_TRANSFORM_UBLKCP
};

// TODO(bgruber): replace by ::cuda::std::expected in C++14
template <typename T>
struct PoorExpected
{
  alignas(T) char storage[sizeof(T)];
  cudaError_t error;

  _CCCL_HOST_DEVICE PoorExpected(T value)
      : error(cudaSuccess)
  {
    new (storage) T(::cuda::std::move(value));
  }

  _CCCL_HOST_DEVICE PoorExpected(cudaError_t error)
      : error(error)
  {}

  _CCCL_HOST_DEVICE explicit operator bool() const
  {
    return error == cudaSuccess;
  }

  _CCCL_HOST_DEVICE T& operator*()
  {
    _CCCL_DIAG_PUSH
    _CCCL_DIAG_SUPPRESS_GCC("-Wstrict-aliasing")
    return reinterpret_cast<T&>(storage);
    _CCCL_DIAG_POP
  }

  _CCCL_HOST_DEVICE const T& operator*() const
  {
    _CCCL_DIAG_PUSH
    _CCCL_DIAG_SUPPRESS_GCC("-Wstrict-aliasing")
    return reinterpret_cast<const T&>(storage);
    _CCCL_DIAG_POP
  }

  _CCCL_HOST_DEVICE T* operator->()
  {
    return &**this;
  }

  _CCCL_HOST_DEVICE const T* operator->() const
  {
    return &**this;
  }
};

// TODO(bgruber): this is very similar to thrust::cuda_cub::core::get_max_shared_memory_per_block. We should unify this.
_CCCL_HOST_DEVICE inline PoorExpected<int> get_max_shared_memory()
{
  //  gevtushenko promised me that I can assume that the stream passed to the CUB API entry point (where the kernels
  //  will later be launched on) belongs to the currently active device. So we can just query the active device here.
  int device = 0;
  auto error = CubDebug(cudaGetDevice(&device));
  if (error != cudaSuccess)
  {
    return error;
  }

  int max_smem = 0;
  error        = CubDebug(cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlock, device));
  if (error != cudaSuccess)
  {
    return error;
  }

  return max_smem;
}

_CCCL_HOST_DEVICE inline PoorExpected<int> get_sm_count()
{
  int device = 0;
  auto error = CubDebug(cudaGetDevice(&device));
  if (error != cudaSuccess)
  {
    return error;
  }

  int sm_count = 0;
  error        = CubDebug(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device));
  if (error != cudaSuccess)
  {
    return error;
  }

  return sm_count;
}

struct elem_counts
{
  int elem_per_thread;
  int tile_size;
  int smem_size;
};

struct prefetch_config
{
  int max_occupancy;
  int sm_count;
};

template <bool RequiresStableAddress,
          typename Offset,
          typename RandomAccessIteratorTupleIn,
          typename RandomAccessIteratorOut,
          typename TransformOp,
          typename PolicyHub = policy_hub<RequiresStableAddress, RandomAccessIteratorTupleIn>>
struct dispatch_t;

template <bool RequiresStableAddress,
          typename Offset,
          typename... RandomAccessIteratorsIn,
          typename RandomAccessIteratorOut,
          typename TransformOp,
          typename PolicyHub>
struct dispatch_t<RequiresStableAddress,
                  Offset,
                  ::cuda::std::tuple<RandomAccessIteratorsIn...>,
                  RandomAccessIteratorOut,
                  TransformOp,
                  PolicyHub>
{
  static_assert(::cuda::std::is_same<Offset, ::cuda::std::int32_t>::value
                  || ::cuda::std::is_same<Offset, ::cuda::std::int64_t>::value,
                "cub::DeviceTransform is only tested and tuned for 32-bit or 64-bit signed offset types");

  ::cuda::std::tuple<RandomAccessIteratorsIn...> in;
  RandomAccessIteratorOut out;
  Offset num_items;
  TransformOp op;
  cudaStream_t stream;

#define CUB_DETAIL_TRANSFORM_KERNEL_PTR             \
  &transform_kernel<typename PolicyHub::max_policy, \
                    Offset,                         \
                    TransformOp,                    \
                    RandomAccessIteratorOut,        \
                    THRUST_NS_QUALIFIER::try_unwrap_contiguous_iterator_t<RandomAccessIteratorsIn>...>

  static constexpr int loaded_bytes_per_iter = loaded_bytes_per_iteration<RandomAccessIteratorsIn...>();

#ifdef _CUB_HAS_TRANSFORM_UBLKCP
  // TODO(bgruber): I want to write tests for this but those are highly depending on the architecture we are running
  // on?
  template <typename ActivePolicy>
  CUB_RUNTIME_FUNCTION _CCCL_VISIBILITY_HIDDEN _CCCL_FORCEINLINE auto configure_ublkcp_kernel()
    -> PoorExpected<
      ::cuda::std::
        tuple<THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron, decltype(CUB_DETAIL_TRANSFORM_KERNEL_PTR), int>>
  {
    using policy_t          = typename ActivePolicy::algo_policy;
    constexpr int block_dim = policy_t::block_threads;
    static_assert(block_dim % bulk_copy_alignment == 0,
                  "block_threads needs to be a multiple of bulk_copy_alignment (128)"); // then tile_size is a multiple
                                                                                        // of 128-byte

    auto determine_element_counts = [&]() -> PoorExpected<elem_counts> {
      const auto max_smem = get_max_shared_memory();
      if (!max_smem)
      {
        return max_smem.error;
      }

      elem_counts last_counts{};
      // Increase the number of output elements per thread until we reach the required bytes in flight.
      static_assert(policy_t::min_items_per_thread <= policy_t::max_items_per_thread, ""); // ensures the loop below
      // runs at least once
      for (int elem_per_thread = +policy_t::min_items_per_thread; elem_per_thread < +policy_t::max_items_per_thread;
           ++elem_per_thread)
      {
        const int tile_size = block_dim * elem_per_thread;
        const int smem_size = bulk_copy_smem_for_tile_size<RandomAccessIteratorsIn...>(tile_size);
        if (smem_size > *max_smem)
        {
#  ifdef CUB_DETAIL_DEBUG_ENABLE_HOST_ASSERTIONS
          // assert should be prevented by smem check in policy
          assert(last_counts.elem_per_thread > 0 && "min_items_per_thread exceeds available shared memory");
#  endif // CUB_DETAIL_DEBUG_ENABLE_HOST_ASSERTIONS
          return last_counts;
        }

        if (tile_size >= num_items)
        {
          return elem_counts{elem_per_thread, tile_size, smem_size};
        }

        int max_occupancy = 0;
        const auto error =
          CubDebug(MaxSmOccupancy(max_occupancy, CUB_DETAIL_TRANSFORM_KERNEL_PTR, block_dim, smem_size));
        if (error != cudaSuccess)
        {
          return error;
        }

        const int bytes_in_flight_SM = max_occupancy * tile_size * loaded_bytes_per_iter;
        if (ActivePolicy::min_bif <= bytes_in_flight_SM)
        {
          return elem_counts{elem_per_thread, tile_size, smem_size};
        }

        last_counts = elem_counts{elem_per_thread, tile_size, smem_size};
      }
      return last_counts;
    };
    PoorExpected<elem_counts> config = [&]() {
      NV_IF_TARGET(NV_IS_HOST,
                   (static auto cached_config = determine_element_counts(); return cached_config;),
                   (
                     // we cannot cache the determined element count in device code
                     return determine_element_counts();));
    }();
    if (!config)
    {
      return config.error;
    }
#  ifdef CUB_DETAIL_DEBUG_ENABLE_HOST_ASSERTIONS
    assert(config->elem_per_thread > 0);
    assert(config->tile_size > 0);
    assert(config->tile_size % bulk_copy_alignment == 0);
    assert((sizeof...(RandomAccessIteratorsIn) == 0) != (config->smem_size != 0)); // logical xor
#  endif // CUB_DETAIL_DEBUG_ENABLE_HOST_ASSERTIONS

    const auto grid_dim = static_cast<unsigned int>(::cuda::ceil_div(num_items, Offset{config->tile_size}));
    return ::cuda::std::make_tuple(
      THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron(grid_dim, block_dim, config->smem_size, stream),
      CUB_DETAIL_TRANSFORM_KERNEL_PTR,
      config->elem_per_thread);
  }

  template <typename ActivePolicy, std::size_t... Is>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t
  invoke_algorithm(cuda::std::index_sequence<Is...>, ::cuda::std::integral_constant<Algorithm, Algorithm::ublkcp>)
  {
    auto ret = configure_ublkcp_kernel<ActivePolicy>();
    if (!ret)
    {
      return ret.error;
    }
    // TODO(bgruber): use a structured binding in C++17
    // auto [launcher, kernel, elem_per_thread] = *ret;

    return ::cuda::std::get<0>(*ret).doit(
      ::cuda::std::get<1>(*ret),
      num_items,
      ::cuda::std::get<2>(*ret),
      op,
      out,
      make_aligned_base_ptr_kernel_arg(
        THRUST_NS_QUALIFIER::try_unwrap_contiguous_iterator(::cuda::std::get<Is>(in)), bulk_copy_alignment)...);
  }
#endif // _CUB_HAS_TRANSFORM_UBLKCP

  template <typename ActivePolicy, std::size_t... Is>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t
  invoke_algorithm(cuda::std::index_sequence<Is...>, ::cuda::std::integral_constant<Algorithm, Algorithm::prefetch>)
  {
    using policy_t          = typename ActivePolicy::algo_policy;
    constexpr int block_dim = policy_t::block_threads;

    auto determine_config = [&]() -> PoorExpected<prefetch_config> {
      int max_occupancy = 0;
      const auto error  = CubDebug(MaxSmOccupancy(max_occupancy, CUB_DETAIL_TRANSFORM_KERNEL_PTR, block_dim, 0));
      if (error != cudaSuccess)
      {
        return error;
      }
      const auto sm_count = get_sm_count();
      if (!sm_count)
      {
        return sm_count.error;
      }
      return prefetch_config{max_occupancy, *sm_count};
    };

    PoorExpected<prefetch_config> config = [&]() {
      NV_IF_TARGET(
        NV_IS_HOST,
        (
          // this static variable exists for each template instantiation of the surrounding function and class, on which
          // the chosen element count solely depends (assuming max SMEM is constant during a program execution)
          static auto cached_config = determine_config(); return cached_config;),
        (
          // we cannot cache the determined element count in device code
          return determine_config();));
    }();
    if (!config)
    {
      return config.error;
    }

    const int items_per_thread =
      loaded_bytes_per_iter == 0
        ? +policy_t::items_per_thread_no_input
        : ::cuda::ceil_div(ActivePolicy::min_bif, config->max_occupancy * block_dim * loaded_bytes_per_iter);

    // Generate at least one block per SM. This improves tiny problem sizes (e.g. 2^16 elements).
    const int items_per_thread_evenly_spread =
      static_cast<int>(::cuda::std::min(Offset{items_per_thread}, num_items / (config->sm_count * block_dim)));

    const int items_per_thread_clamped = ::cuda::std::clamp(
      items_per_thread_evenly_spread, +policy_t::min_items_per_thread, +policy_t::max_items_per_thread);
    const int tile_size = block_dim * items_per_thread_clamped;
    const auto grid_dim = static_cast<unsigned int>(::cuda::ceil_div(num_items, Offset{tile_size}));
    return CubDebug(
      THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron(grid_dim, block_dim, 0, stream)
        .doit(
          CUB_DETAIL_TRANSFORM_KERNEL_PTR,
          num_items,
          items_per_thread_clamped,
          op,
          out,
          make_iterator_kernel_arg(THRUST_NS_QUALIFIER::try_unwrap_contiguous_iterator(::cuda::std::get<Is>(in)))...));
  }

  template <typename ActivePolicy>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t Invoke()
  {
    // // TODO(bgruber): replace the overload set by if constexpr in C++17
    return invoke_algorithm<ActivePolicy>(::cuda::std::index_sequence_for<RandomAccessIteratorsIn...>{},
                                          ::cuda::std::integral_constant<Algorithm, ActivePolicy::algorithm>{});
  }

  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t dispatch(
    ::cuda::std::tuple<RandomAccessIteratorsIn...> in,
    RandomAccessIteratorOut out,
    Offset num_items,
    TransformOp op,
    cudaStream_t stream)
  {
    if (num_items == 0)
    {
      return cudaSuccess;
    }

    int ptx_version = 0;
    auto error      = CubDebug(PtxVersion(ptx_version));
    if (cudaSuccess != error)
    {
      return error;
    }

    dispatch_t dispatch{::cuda::std::move(in), ::cuda::std::move(out), num_items, ::cuda::std::move(op), stream};
    return CubDebug(PolicyHub::max_policy::Invoke(ptx_version, dispatch));
  }

#undef CUB_DETAIL_TRANSFORM_KERNEL_PTR
};
} // namespace transform
} // namespace detail
CUB_NAMESPACE_END
