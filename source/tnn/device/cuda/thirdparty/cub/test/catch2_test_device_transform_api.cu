// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/device/device_transform.cuh>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <c2h/catch2_test_helper.cuh>

// need a separate function because the ext. lambda needs to be enclosed by a function with external linkage on Windows
void test_transform_api()
{
  // example-begin transform-many
  constexpr auto num_items = 4;
  auto input1              = thrust::device_vector<int>{0, -2, 5, 3};
  auto input2              = thrust::device_vector<float>{5.2f, 3.1f, -1.1f, 3.0f};
  auto input3              = thrust::counting_iterator<int>{100};
  auto op                  = [] __device__(int a, float b, int c) {
    return (a + b) * c;
  };

  auto result = thrust::device_vector<int>(num_items);
  cub::DeviceTransform::Transform(
    ::cuda::std::make_tuple(input1.begin(), input2.begin(), input3), result.begin(), num_items, op);

  const auto expected = thrust::host_vector<float>{520, 111, 397, 618};
  // example-end transform-many
  CHECK(result == expected);
}

C2H_TEST("DeviceTransform::Transform API example", "[device][device_transform]")
{
  test_transform_api();
}

// need a separate function because the ext. lambda needs to be enclosed by a function with external linkage on Windows
void test_transform_stable_api()
{
  // example-begin transform-many-stable
  constexpr auto num_items = 4;
  auto input1              = thrust::device_vector<int>{0, -2, 5, 3};
  auto input2              = thrust::device_vector<int>{52, 31, -11, 30};

  auto* input1_ptr = thrust::raw_pointer_cast(input1.data());
  auto* input2_ptr = thrust::raw_pointer_cast(input2.data());

  auto op = [input1_ptr, input2_ptr] __device__(const int& a) -> int {
    const auto i = &a - input1_ptr; // we depend on the address of a
    return a + input2_ptr[i];
  };

  auto result = thrust::device_vector<int>(num_items);
  cub::DeviceTransform::TransformStableArgumentAddresses(
    ::cuda::std::make_tuple(input1_ptr), result.begin(), num_items, op);

  const auto expected = thrust::host_vector<float>{52, 29, -6, 33};
  // example-end transform-many-stable
  CHECK(result == expected);
}

C2H_TEST("DeviceTransform::TransformStableArgumentAddresses API example", "[device][device_transform]")
{
  test_transform_stable_api();
}
