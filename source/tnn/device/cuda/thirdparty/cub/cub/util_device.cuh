/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2020, NVIDIA CORPORATION.  All rights reserved.
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

/**
 * \file
 * Properties of a given CUDA device and the corresponding PTX bundle.
 *
 * \note
 * This file contains __host__ only functions and utilities, and should not be
 * included in code paths that could be online-compiled (ex: using NVRTC).
 */

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/detail/device_synchronize.cuh> // IWYU pragma: export
#include <cub/util_debug.cuh>
#include <cub/util_type.cuh>
// for backward compatibility
#include <cub/util_temporary_storage.cuh>

#include <cuda/std/__cuda/ensure_current_device.h> // IWYU pragma: export
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include <array>
#include <atomic>
#include <cassert>

#include <nv/target>

CUB_NAMESPACE_BEGIN

#ifndef DOXYGEN_SHOULD_SKIP_THIS // Do not document

namespace detail
{
/**
 * @brief Helper class template that allows overwriting the `BLOCK_THREAD` and `ITEMS_PER_THREAD`
 * configurations of a given policy.
 */
// TODO(bgruber): this should be called something like "override_policy"
template <typename PolicyT, int BLOCK_THREADS_, int ITEMS_PER_THREAD_ = PolicyT::ITEMS_PER_THREAD>
struct policy_wrapper_t : PolicyT
{
  static constexpr int ITEMS_PER_THREAD = ITEMS_PER_THREAD_;
  static constexpr int BLOCK_THREADS    = BLOCK_THREADS_;
  static constexpr int ITEMS_PER_TILE   = BLOCK_THREADS * ITEMS_PER_THREAD;
};
} // namespace detail

/**
 * \brief Empty kernel for querying PTX manifest metadata (e.g., version) for the current device
 */
template <typename T>
CUB_DETAIL_KERNEL_ATTRIBUTES void EmptyKernel()
{}

#endif // DOXYGEN_SHOULD_SKIP_THIS

/**
 * \brief Returns the current device or -1 if an error occurred.
 */
CUB_RUNTIME_FUNCTION inline int CurrentDevice()
{
  int device = -1;
  if (CubDebug(cudaGetDevice(&device)))
  {
    return -1;
  }
  return device;
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS // Do not document

//! @brief RAII helper which saves the current device and switches to the specified device on construction and switches
//! to the saved device on destruction.
using SwitchDevice = ::cuda::__ensure_current_device;

#endif // DOXYGEN_SHOULD_SKIP_THIS

/**
 * \brief Returns the number of CUDA devices available or -1 if an error
 *        occurred.
 */
CUB_RUNTIME_FUNCTION inline int DeviceCountUncached()
{
  int count = -1;
  if (CubDebug(cudaGetDeviceCount(&count)))
  {
    // CUDA makes no guarantees about the state of the output parameter if
    // `cudaGetDeviceCount` fails; in practice, they don't, but out of
    // paranoia we'll reset `count` to `-1`.
    count = -1;
  }
  return count;
}

/**
 * \brief Cache for an arbitrary value produced by a nullary function.
 * deprecated [Since 2.6.0]
 */
template <typename T, T (*Function)()>
struct CUB_DEPRECATED ValueCache
{
  T const value;

  /**
   * \brief Call the nullary function to produce the value and construct the
   *        cache.
   */
  _CCCL_HOST inline ValueCache()
      : value(Function())
  {}
};

// Host code. This is a separate function to avoid defining a local static in a host/device function.
_CCCL_HOST inline int DeviceCountCachedValue()
{
  static int count = DeviceCountUncached();
  return count;
}

/**
 * \brief Returns the number of CUDA devices available.
 *
 * \note This function may cache the result internally.
 *
 * \note This function is thread safe.
 */
CUB_RUNTIME_FUNCTION inline int DeviceCount()
{
  int result = -1;

  NV_IF_TARGET(NV_IS_HOST, (result = DeviceCountCachedValue();), (result = DeviceCountUncached();));

  return result;
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS // Do not document
/**
 * \brief Per-device cache for a CUDA attribute value; the attribute is queried
 *        and stored for each device upon construction.
 */
struct PerDeviceAttributeCache
{
  struct DevicePayload
  {
    int attribute;
    cudaError_t error;
  };

  // Each entry starts in the `DeviceEntryEmpty` state, then proceeds to the
  // `DeviceEntryInitializing` state, and then proceeds to the
  // `DeviceEntryReady` state. These are the only state transitions allowed;
  // i.e. a linear sequence of transitions.
  enum DeviceEntryStatus
  {
    DeviceEntryEmpty = 0,
    DeviceEntryInitializing,
    DeviceEntryReady
  };

  struct DeviceEntry
  {
    std::atomic<DeviceEntryStatus> flag;
    DevicePayload payload;
  };

private:
  std::array<DeviceEntry, CUB_MAX_DEVICES> entries_;

public:
  /**
   * \brief Construct the cache.
   */
  _CCCL_HOST inline PerDeviceAttributeCache()
      : entries_()
  {
    assert(DeviceCount() <= CUB_MAX_DEVICES);
  }

  /**
   * \brief Retrieves the payload of the cached function \p f for \p device.
   *
   * \note You must pass a morally equivalent function in to every call or
   *       this function has undefined behavior.
   */
  template <typename Invocable>
  _CCCL_HOST DevicePayload operator()(Invocable&& f, int device)
  {
    if (device >= DeviceCount() || device < 0)
    {
      return DevicePayload{0, cudaErrorInvalidDevice};
    }

    auto& entry   = entries_[device];
    auto& flag    = entry.flag;
    auto& payload = entry.payload;

    DeviceEntryStatus old_status = DeviceEntryEmpty;

    // First, check for the common case of the entry being ready.
    if (flag.load(std::memory_order_acquire) != DeviceEntryReady)
    {
      // Assume the entry is empty and attempt to lock it so we can fill
      // it by trying to set the state from `DeviceEntryReady` to
      // `DeviceEntryInitializing`.
      if (flag.compare_exchange_strong(
            old_status, DeviceEntryInitializing, std::memory_order_acq_rel, std::memory_order_acquire))
      {
        // We successfully set the state to `DeviceEntryInitializing`;
        // we have the lock and it's our job to initialize this entry
        // and then release it.

        // We don't use `CubDebug` here because we let the user code
        // decide whether or not errors are hard errors.
        payload.error = ::cuda::std::forward<Invocable>(f)(payload.attribute);
        if (payload.error)
        {
          // Clear the global CUDA error state which may have been
          // set by the last call. Otherwise, errors may "leak" to
          // unrelated kernel launches.
          cudaGetLastError();
        }

        // Release the lock by setting the state to `DeviceEntryReady`.
        flag.store(DeviceEntryReady, std::memory_order_release);
      }

      // If the `compare_exchange_weak` failed, then `old_status` has
      // been updated with the value of `flag` that it observed.

      else if (old_status == DeviceEntryInitializing)
      {
        // Another execution agent is initializing this entry; we need
        // to wait for them to finish; we'll know they're done when we
        // observe the entry status as `DeviceEntryReady`.
        do
        {
          old_status = flag.load(std::memory_order_acquire);
        } while (old_status != DeviceEntryReady);
        // FIXME: Use `atomic::wait` instead when we have access to
        // host-side C++20 atomics. We could use libcu++, but it only
        // supports atomics for SM60 and up, even if you're only using
        // them in host code.
      }
    }

    // We now know that the state of our entry is `DeviceEntryReady`, so
    // just return the entry's payload.
    return entry.payload;
  }
};
#endif // DOXYGEN_SHOULD_SKIP_THIS

/**
 * \brief Retrieves the PTX version that will be used on the current device (major * 100 + minor * 10).
 */
CUB_RUNTIME_FUNCTION inline cudaError_t PtxVersionUncached(int& ptx_version)
{
  // Instantiate `EmptyKernel<void>` in both host and device code to ensure
  // it can be called.
  using EmptyKernelPtr        = void (*)();
  EmptyKernelPtr empty_kernel = EmptyKernel<void>;

  // This is necessary for unused variable warnings in host compilers. The
  // usual syntax of (void)empty_kernel; was not sufficient on MSVC2015.
  (void) reinterpret_cast<void*>(empty_kernel);

  // Define a temporary macro that expands to the current target ptx version
  // in device code.
  // <nv/target> may provide an abstraction for this eventually. For now,
  // we have to keep this usage of __CUDA_ARCH__.
#if defined(_NVHPC_CUDA)
#  define CUB_TEMP_GET_PTX __builtin_current_device_sm()
#else
#  define CUB_TEMP_GET_PTX __CUDA_ARCH__
#endif

  cudaError_t result = cudaSuccess;
  NV_IF_TARGET(
    NV_IS_HOST,
    (cudaFuncAttributes empty_kernel_attrs;

     result = CubDebug(cudaFuncGetAttributes(&empty_kernel_attrs, reinterpret_cast<void*>(empty_kernel)));

     ptx_version = empty_kernel_attrs.ptxVersion * 10;),
    // NV_IS_DEVICE
    (
      // This is necessary to ensure instantiation of EmptyKernel in device
      // code. The `reinterpret_cast` is necessary to suppress a
      // set-but-unused warnings. This is a meme now:
      // https://twitter.com/blelbach/status/1222391615576100864
      (void) reinterpret_cast<EmptyKernelPtr>(empty_kernel);

      ptx_version = CUB_TEMP_GET_PTX;));

#undef CUB_TEMP_GET_PTX

  return result;
}

/**
 * \brief Retrieves the PTX version that will be used on \p device (major * 100 + minor * 10).
 */
_CCCL_HOST inline cudaError_t PtxVersionUncached(int& ptx_version, int device)
{
  SwitchDevice sd(device);
  (void) sd;
  return PtxVersionUncached(ptx_version);
}

template <typename Tag>
_CCCL_HOST inline PerDeviceAttributeCache& GetPerDeviceAttributeCache()
{
  static PerDeviceAttributeCache cache;
  return cache;
}

struct PtxVersionCacheTag
{};
struct SmVersionCacheTag
{};

/**
 * \brief Retrieves the PTX virtual architecture that will be used on \p device (major * 100 + minor * 10). If
 * __CUDA_ARCH_LIST__ is defined, this value is one of __CUDA_ARCH_LIST__.
 *
 * \note This function may cache the result internally.
 * \note This function is thread safe.
 */
_CCCL_HOST inline cudaError_t PtxVersion(int& ptx_version, int device)
{
  // Note: the ChainedPolicy pruning (i.e., invoke_static) requites that there's an exact match between one of the
  // architectures in __CUDA_ARCH__ and the runtime queried ptx version.
  auto const payload = GetPerDeviceAttributeCache<PtxVersionCacheTag>()(
    // If this call fails, then we get the error code back in the payload, which we check with `CubDebug` below.
    [=](int& pv) {
      return PtxVersionUncached(pv, device);
    },
    device);

  if (!CubDebug(payload.error))
  {
    ptx_version = payload.attribute;
  }

  return payload.error;
}

/**
 * \brief Retrieves the PTX virtual architecture that will be used on the current device (major * 100 + minor * 10).
 *
 * \note This function may cache the result internally.
 * \note This function is thread safe.
 */
CUB_RUNTIME_FUNCTION inline cudaError_t PtxVersion(int& ptx_version)
{
  // Note: the ChainedPolicy pruning (i.e., invoke_static) requites that there's an exact match between one of the
  // architectures in __CUDA_ARCH__ and the runtime queried ptx version.
  cudaError_t result = cudaErrorUnknown;
  NV_IF_TARGET(NV_IS_HOST,
               (result = PtxVersion(ptx_version, CurrentDevice());),
               ( // NV_IS_DEVICE:
                 result = PtxVersionUncached(ptx_version);));
  return result;
}

/**
 * \brief Retrieves the SM version (i.e. compute capability) of \p device (major * 100 + minor * 10)
 */
CUB_RUNTIME_FUNCTION inline cudaError_t SmVersionUncached(int& sm_version, int device = CurrentDevice())
{
  cudaError_t error = cudaSuccess;
  do
  {
    int major = 0, minor = 0;
    error = CubDebug(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
    if (cudaSuccess != error)
    {
      break;
    }

    error = CubDebug(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
    if (cudaSuccess != error)
    {
      break;
    }
    sm_version = major * 100 + minor * 10;
  } while (0);

  return error;
}

/**
 * \brief Retrieves the SM version (i.e. compute capability) of \p device (major * 100 + minor * 10).
 *
 * \note This function may cache the result internally.
 * \note This function is thread safe.
 */
CUB_RUNTIME_FUNCTION inline cudaError_t SmVersion(int& sm_version, int device = CurrentDevice())
{
  cudaError_t result = cudaErrorUnknown;

  NV_IF_TARGET(
    NV_IS_HOST,
    (auto const payload = GetPerDeviceAttributeCache<SmVersionCacheTag>()(
       // If this call fails, then we get the error code back in the payload, which we check with `CubDebug` below.
       [=](int& pv) {
         return SmVersionUncached(pv, device);
       },
       device);

     if (!CubDebug(payload.error)) { sm_version = payload.attribute; };

     result = payload.error;),
    ( // NV_IS_DEVICE
      result = SmVersionUncached(sm_version, device);));

  return result;
}

/**
 * Synchronize the specified \p stream.
 */
CUB_RUNTIME_FUNCTION inline cudaError_t SyncStream(cudaStream_t stream)
{
  cudaError_t result = cudaErrorNotSupported;

  NV_IF_TARGET(NV_IS_HOST,
               (result = CubDebug(cudaStreamSynchronize(stream));),
               ((void) stream; result = CubDebug(cub::detail::device_synchronize());));

  return result;
}

namespace detail
{

/**
 * Same as SyncStream, but intended for use with the debug_synchronous flags
 * in device algorithms. This should not be used if synchronization is required
 * for correctness.
 *
 * If `debug_synchronous` is false, this function will immediately return
 * cudaSuccess. If true, one of the following will occur:
 *
 * If synchronization is supported by the current compilation target and
 * settings, the sync is performed and the sync result is returned.
 *
 * If syncs are not supported then no sync is performed, but a message is logged
 * via _CubLog and cudaSuccess is returned.
 */
CUB_RUNTIME_FUNCTION inline cudaError_t DebugSyncStream(cudaStream_t stream)
{
#ifndef CUB_DETAIL_DEBUG_ENABLE_SYNC

  (void) stream;
  return cudaSuccess;

#else // CUB_DETAIL_DEBUG_ENABLE_SYNC:

#  define CUB_TMP_SYNC_AVAILABLE         \
    _CubLog("%s\n", "Synchronizing..."); \
    return SyncStream(stream)

#  define CUB_TMP_DEVICE_SYNC_UNAVAILABLE                                        \
    (void) stream;                                                               \
    _CubLog("WARNING: Skipping CUB `debug_synchronous` synchronization (%s).\n", \
            "device-side sync requires <sm_90, RDC, and CDPv1");                 \
    return cudaSuccess

#  ifdef CUB_DETAIL_CDPv1

  // Can sync everywhere but SM_90+
  NV_IF_TARGET(NV_PROVIDES_SM_90, (CUB_TMP_DEVICE_SYNC_UNAVAILABLE;), (CUB_TMP_SYNC_AVAILABLE;));

#  else // CDPv2 or no CDP:

  // Can only sync on host
  NV_IF_TARGET(NV_IS_HOST, (CUB_TMP_SYNC_AVAILABLE;), (CUB_TMP_DEVICE_SYNC_UNAVAILABLE;));

#  endif // CDP version

#  undef CUB_TMP_DEVICE_SYNC_UNAVAILABLE
#  undef CUB_TMP_SYNC_AVAILABLE

#endif // CUB_DETAIL_DEBUG_ENABLE_SYNC
}

/** \brief Gets whether the current device supports unified addressing */
CUB_RUNTIME_FUNCTION inline cudaError_t HasUVA(bool& has_uva)
{
  has_uva           = false;
  int device        = -1;
  cudaError_t error = CubDebug(cudaGetDevice(&device));
  if (cudaSuccess != error)
  {
    return error;
  }

  int uva = 0;
  error   = CubDebug(cudaDeviceGetAttribute(&uva, cudaDevAttrUnifiedAddressing, device));
  if (cudaSuccess != error)
  {
    return error;
  }
  has_uva = uva == 1;
  return error;
}

} // namespace detail

/**
 * @brief Computes maximum SM occupancy in thread blocks for executing the given kernel function
 *        pointer @p kernel_ptr on the current device with @p block_threads per thread block.
 *
 * @par Snippet
 * The code snippet below illustrates the use of the MaxSmOccupancy function.
 * @par
 * @code
 * #include <cub/cub.cuh>   // or equivalently <cub/util_device.cuh>
 *
 * template <typename T>
 * __global__ void ExampleKernel()
 * {
 *     // Allocate shared memory for BlockScan
 *     __shared__ volatile T buffer[4096];
 *
 *        ...
 * }
 *
 *     ...
 *
 * // Determine SM occupancy for ExampleKernel specialized for unsigned char
 * int max_sm_occupancy;
 * MaxSmOccupancy(max_sm_occupancy, ExampleKernel<unsigned char>, 64);
 *
 * // max_sm_occupancy  <-- 4 on SM10
 * // max_sm_occupancy  <-- 8 on SM20
 * // max_sm_occupancy  <-- 12 on SM35
 *
 * @endcode
 *
 * @param[out] max_sm_occupancy
 *   maximum number of thread blocks that can reside on a single SM
 *
 * @param[in] kernel_ptr
 *   Kernel pointer for which to compute SM occupancy
 *
 * @param[in] block_threads
 *   Number of threads per thread block
 *
 * @param[in] dynamic_smem_bytes
 *   Dynamically allocated shared memory in bytes. Default is 0.
 */
template <typename KernelPtr>
_CCCL_VISIBILITY_HIDDEN CUB_RUNTIME_FUNCTION inline cudaError_t
MaxSmOccupancy(int& max_sm_occupancy, KernelPtr kernel_ptr, int block_threads, int dynamic_smem_bytes = 0)
{
  return CubDebug(
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_sm_occupancy, kernel_ptr, block_threads, dynamic_smem_bytes));
}

/******************************************************************************
 * Policy management
 ******************************************************************************/

template <typename PolicyT, typename = void>
struct PolicyWrapper : PolicyT
{
  CUB_RUNTIME_FUNCTION PolicyWrapper(PolicyT base)
      : PolicyT(base)
  {}
};

template <typename StaticPolicyT>
struct PolicyWrapper<
  StaticPolicyT,
  _CUDA_VSTD::void_t<decltype(StaticPolicyT::BLOCK_THREADS), decltype(StaticPolicyT::ITEMS_PER_THREAD)>> : StaticPolicyT
{
  CUB_RUNTIME_FUNCTION PolicyWrapper(StaticPolicyT base)
      : StaticPolicyT(base)
  {}

  CUB_RUNTIME_FUNCTION static constexpr int BlockThreads()
  {
    return StaticPolicyT::BLOCK_THREADS;
  }

  CUB_RUNTIME_FUNCTION static constexpr int ItemsPerThread()
  {
    return StaticPolicyT::ITEMS_PER_THREAD;
  }
};

template <typename PolicyT>
CUB_RUNTIME_FUNCTION PolicyWrapper<PolicyT> MakePolicyWrapper(PolicyT policy)
{
  return PolicyWrapper<PolicyT>{policy};
}

struct TripleChevronFactory;

/**
 * Kernel dispatch configuration
 */
struct KernelConfig
{
  int block_threads;
  int items_per_thread;
  int tile_size;
  int sm_occupancy;

  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE KernelConfig()
      : block_threads(0)
      , items_per_thread(0)
      , tile_size(0)
      , sm_occupancy(0)
  {}

  template <typename AgentPolicyT, typename KernelPtrT, typename LauncherFactory = TripleChevronFactory>
  CUB_RUNTIME_FUNCTION _CCCL_VISIBILITY_HIDDEN _CCCL_FORCEINLINE cudaError_t
  Init(KernelPtrT kernel_ptr, AgentPolicyT agent_policy = {}, LauncherFactory launcher_factory = {})
  {
    block_threads    = MakePolicyWrapper(agent_policy).BlockThreads();
    items_per_thread = MakePolicyWrapper(agent_policy).ItemsPerThread();
    tile_size        = block_threads * items_per_thread;
    return launcher_factory.MaxSmOccupancy(sm_occupancy, kernel_ptr, block_threads);
  }
};

/// Helper for dispatching into a policy chain
template <int PolicyPtxVersion, typename PolicyT, typename PrevPolicyT>
struct ChainedPolicy
{
  /// The policy for the active compiler pass
  using ActivePolicy = ::cuda::std::_If<(CUB_PTX_ARCH < PolicyPtxVersion), typename PrevPolicyT::ActivePolicy, PolicyT>;

  /// Specializes and dispatches op in accordance to the first policy in the chain of adequate PTX version
  template <typename FunctorT>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t Invoke(int device_ptx_version, FunctorT& op)
  {
    // __CUDA_ARCH_LIST__ is only available from CTK 11.5 onwards
#ifdef __CUDA_ARCH_LIST__
    return runtime_to_compiletime<__CUDA_ARCH_LIST__>(device_ptx_version, op);
#else
    if (device_ptx_version < PolicyPtxVersion)
    {
      return PrevPolicyT::Invoke(device_ptx_version, op);
    }
    return op.template Invoke<PolicyT>();
#endif
  }

private:
  template <int, typename, typename>
  friend struct ChainedPolicy; // let us call invoke_static of other ChainedPolicy instantiations

  template <int... CudaArches, typename FunctorT>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t runtime_to_compiletime(int device_ptx_version, FunctorT& op)
  {
    // We instantiate invoke_static for each CudaArches, but only call the one matching device_ptx_version.
    // If there's no exact match of any of the architectures in __CUDA_ARCH_LIST__ and the runtime
    // queried ptx version (i.e., the closest ptx version to the current device's architecture that the EmptyKernel was
    // compiled for), we return cudaErrorInvalidDeviceFunction. Such a scenario may arise if CUB_DISABLE_NAMESPACE_MAGIC
    // is set and different TUs are compiled for different sets of architecture.
    cudaError_t e             = cudaErrorInvalidDeviceFunction;
    const cudaError_t dummy[] = {
      (device_ptx_version == CudaArches ? (e = invoke_static<CudaArches>(op, ::cuda::std::true_type{}))
                                        : cudaSuccess)...};
    (void) dummy;
    return e;
  }

  template <int DevicePtxVersion, typename FunctorT>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t invoke_static(FunctorT& op, ::cuda::std::true_type)
  {
    // TODO(bgruber): drop diagnostic suppression in C++17
    _CCCL_DIAG_PUSH
    _CCCL_DIAG_SUPPRESS_MSVC(4127) // suppress Conditional Expression is Constant
    _CCCL_IF_CONSTEXPR (DevicePtxVersion < PolicyPtxVersion)
    {
      // TODO(bgruber): drop boolean tag dispatches in C++17, since _CCCL_IF_CONSTEXPR will discard this branch properly
      return PrevPolicyT::template invoke_static<DevicePtxVersion>(
        op, ::cuda::std::bool_constant<(DevicePtxVersion < PolicyPtxVersion)>{});
    }
    else
    {
      return do_invoke(op, ::cuda::std::bool_constant<DevicePtxVersion >= PolicyPtxVersion>{});
    }
    _CCCL_DIAG_POP
  }

  template <int, typename FunctorT>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t invoke_static(FunctorT&, ::cuda::std::false_type)
  {
    _CCCL_UNREACHABLE();
  }

  template <typename FunctorT>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t do_invoke(FunctorT& op, ::cuda::std::true_type)
  {
    return op.template Invoke<PolicyT>();
  }

  template <typename FunctorT>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t do_invoke(FunctorT&, ::cuda::std::false_type)
  {
    _CCCL_UNREACHABLE();
  }
};

/// Helper for dispatching into a policy chain (end-of-chain specialization)
template <int PolicyPtxVersion, typename PolicyT>
struct ChainedPolicy<PolicyPtxVersion, PolicyT, PolicyT>
{
  template <int, typename, typename>
  friend struct ChainedPolicy; // befriend primary template, so it can call invoke_static

  /// The policy for the active compiler pass
  using ActivePolicy = PolicyT;

  /// Specializes and dispatches op in accordance to the first policy in the chain of adequate PTX version
  template <typename FunctorT>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t Invoke(int /*ptx_version*/, FunctorT& op)
  {
    return op.template Invoke<PolicyT>();
  }

private:
  template <int, typename FunctorT>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t invoke_static(FunctorT& op, ::cuda::std::true_type)
  {
    return op.template Invoke<PolicyT>();
  }

  template <int, typename FunctorT>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t invoke_static(FunctorT&, ::cuda::std::false_type)
  {
    _CCCL_UNREACHABLE();
  }
};

CUB_NAMESPACE_END

#include <cub/launcher/cuda_runtime.cuh> // to complete the definition of TripleChevronFactory
