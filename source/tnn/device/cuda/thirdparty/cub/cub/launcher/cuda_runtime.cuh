#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/util_device.cuh>

#include <thrust/system/cuda/detail/core/triple_chevron_launch.h>

CUB_NAMESPACE_BEGIN

struct TripleChevronFactory
{
  CUB_RUNTIME_FUNCTION THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron
  operator()(dim3 grid, dim3 block, _CUDA_VSTD::size_t shared_mem, cudaStream_t stream) const
  {
    return THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron(grid, block, shared_mem, stream);
  }

  CUB_RUNTIME_FUNCTION cudaError_t PtxVersion(int& version)
  {
    return cub::PtxVersion(version);
  }

  CUB_RUNTIME_FUNCTION cudaError_t MultiProcessorCount(int& sm_count) const
  {
    int device_ordinal;
    cudaError_t error = CubDebug(cudaGetDevice(&device_ordinal));
    if (cudaSuccess != error)
    {
      return error;
    }

    // Get SM count
    return cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_ordinal);
  }

  template <typename Kernel>
  CUB_RUNTIME_FUNCTION cudaError_t
  MaxSmOccupancy(int& sm_occupancy, Kernel kernel_ptr, int block_size, int dynamic_smem_bytes = 0)
  {
    return cudaOccupancyMaxActiveBlocksPerMultiprocessor(&sm_occupancy, kernel_ptr, block_size, dynamic_smem_bytes);
  }
};

CUB_NAMESPACE_END
