#define NVTX3_CPP_REQUIRE_EXPLICIT_VERSION
#include <cub/device/device_for.cuh> // internal include of NVTX

#include <thrust/iterator/counting_iterator.h>

#include <cuda/std/functional>

#include <nvtx3/nvtx3.hpp> // user-side include of NVTX, retrieved elsewhere

int main()
{
  nvtx3::v1::scoped_range range("user-range"); // user-side use of explicit NVTX API

  thrust::counting_iterator<int> it{0};
  cub::DeviceFor::ForEach(it, it + 16, ::cuda::std::negate<int>{}); // internal use of NVTX
  cudaDeviceSynchronize();
}
