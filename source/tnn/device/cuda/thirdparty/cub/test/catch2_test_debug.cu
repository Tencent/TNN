#include <cub/util_debug.cuh>
#include <cub/util_device.cuh>

#include <c2h/catch2_test_helper.cuh>

TEST_CASE("CubDebug returns input error", "[debug][utils]")
{
  REQUIRE(CubDebug(cudaSuccess) == cudaSuccess);
  REQUIRE(CubDebug(cudaErrorInvalidConfiguration) == cudaErrorInvalidConfiguration);
}

TEST_CASE("CubDebug returns new errors", "[debug][utils]")
{
  cub::EmptyKernel<int><<<0, 0>>>();
  cudaError error = cudaPeekAtLastError();

  REQUIRE(error != cudaSuccess);
  REQUIRE(CubDebug(cudaSuccess) != cudaSuccess);
}

TEST_CASE("CubDebug prefers input errors", "[debug][utils]")
{
  cub::EmptyKernel<int><<<0, 0>>>();
  cudaError error = cudaPeekAtLastError();

  REQUIRE(error != cudaSuccess);
  REQUIRE(CubDebug(cudaErrorMemoryAllocation) != cudaSuccess);
}

TEST_CASE("CubDebug resets last error", "[debug][utils]")
{
  cub::EmptyKernel<int><<<0, 0>>>();
  cudaError error = cudaPeekAtLastError();

  REQUIRE(error != cudaSuccess);
  REQUIRE(CubDebug(cudaSuccess) != cudaSuccess);
  REQUIRE(CubDebug(cudaSuccess) == cudaSuccess);
}
