// Wrap thrust and cub in different enclosing namespaces
// (In practice, you probably want these to be the same, in which case just
// set THRUST_CUB_WRAPPED_NAMESPACE to set both).
#define THRUST_WRAPPED_NAMESPACE wrap_thrust
#define CUB_WRAPPED_NAMESPACE    wrap_cub

// Enable error checking:
#define CUB_STDERR

#include <cub/device/device_radix_sort.cuh>
#include <cub/util_debug.cuh>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>

#include <cstdint>
#include <cstdlib>

#include "test_util.h"

// Test that we can use a few common utilities and algorithms from wrapped
// Thrust/CUB namespaces at runtime. More extensive testing is performed by the
// header tests and the check_namespace.cmake test.
int main(int argc, char** argv)
{
  CommandLineArgs args(argc, argv);
  CubDebugExit(args.DeviceInit());

  constexpr std::size_t n = 2048;

  // Fill a vector with random data:
  ::wrap_thrust::thrust::host_vector<int> h_input(n);
  for (auto& val : h_input)
  {
    RandomBits(val);
  }

  // Test the qualifier macro:
  THRUST_NS_QUALIFIER::device_vector<int> d_input(h_input);
  THRUST_NS_QUALIFIER::device_vector<int> d_output(n);

  std::size_t temp_storage_bytes{};

  // Sort with DeviceRadixSort:
  auto error = ::wrap_cub::cub::DeviceRadixSort::SortKeys(
    nullptr,
    temp_storage_bytes,
    ::wrap_thrust::thrust::raw_pointer_cast(d_input.data()),
    ::wrap_thrust::thrust::raw_pointer_cast(d_output.data()),
    static_cast<std::size_t>(n));

  CubDebugExit(error);

  ::wrap_thrust::thrust::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);

  // Test the CUB qualifier macro:
  error = CUB_NS_QUALIFIER::DeviceRadixSort::SortKeys(
    ::wrap_thrust::thrust::raw_pointer_cast(temp_storage.data()),
    temp_storage_bytes,
    ::wrap_thrust::thrust::raw_pointer_cast(d_input.data()),
    ::wrap_thrust::thrust::raw_pointer_cast(d_output.data()),
    static_cast<std::size_t>(n));

  CubDebugExit(error);

  // Verify output:
  if (!::wrap_thrust::thrust::is_sorted(d_output.cbegin(), d_output.cend()))
  {
    std::cerr << "Output is not sorted!\n";
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
