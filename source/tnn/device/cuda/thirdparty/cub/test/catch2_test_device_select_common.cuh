// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <thrust/iterator/constant_iterator.h>

#include <cuda/std/type_traits>

#include <c2h/catch2_test_helper.cuh>

template <typename T>
struct less_than_t
{
  T compare;

  explicit __host__ less_than_t(T compare)
      : compare(compare)
  {}

  __host__ __device__ bool operator()(const T& a) const
  {
    return a < compare;
  }
};

template <typename T>
struct mod_n
{
  T mod;
  __host__ __device__ bool operator()(T x)
  {
    return (x % mod == 0) ? true : false;
  }
};

template <typename T>
struct multiply_n
{
  T multiplier;
  __host__ __device__ T operator()(T x)
  {
    return x * multiplier;
  }
};

template <typename T, typename TargetT>
struct modx_and_add_divy
{
  T mod;
  T div;

  __host__ __device__ TargetT operator()(T x)
  {
    return static_cast<TargetT>((x % mod) + (x / div));
  }
};

template <typename SelectedItT, typename RejectedItT>
struct index_to_expected_partition_op
{
  using value_t = typename ::cuda::std::iterator_traits<SelectedItT>::value_type;
  SelectedItT expected_selected_it;
  RejectedItT expected_rejected_it;
  std::int64_t expected_num_selected;

  template <typename OffsetT>
  __host__ __device__ value_t operator()(OffsetT index)
  {
    return (index < static_cast<OffsetT>(expected_num_selected))
           ? expected_selected_it[index]
           : expected_rejected_it[index - expected_num_selected];
  }
};

template <typename SelectedItT, typename RejectedItT>
static index_to_expected_partition_op<SelectedItT, RejectedItT> make_index_to_expected_partition_op(
  SelectedItT expected_selected_it, RejectedItT expected_rejected_it, std::int64_t expected_num_selected)
{
  return index_to_expected_partition_op<SelectedItT, RejectedItT>{
    expected_selected_it, expected_rejected_it, expected_num_selected};
}

template <typename ExpectedValuesItT>
struct flag_correct_writes_op
{
  ExpectedValuesItT expected_it;
  std::uint32_t* d_correctness_flags;

  static constexpr auto bits_per_element = 8 * sizeof(std::uint32_t);
  template <typename OffsetT, typename T>
  __host__ __device__ void operator()(OffsetT index, T val)
  {
    // Set bit-flag if the correct result has been written at the given index
    if (expected_it[index] == val)
    {
      OffsetT uint_index     = index / static_cast<OffsetT>(bits_per_element);
      std::uint32_t bit_flag = 0x00000001U << (index % bits_per_element);
      atomicOr(&d_correctness_flags[uint_index], bit_flag);
    }
  }
};

template <typename ExpectedValuesItT>
flag_correct_writes_op<ExpectedValuesItT> static make_checking_write_op(
  ExpectedValuesItT expected_it, std::uint32_t* d_correctness_flags)
{
  return flag_correct_writes_op<ExpectedValuesItT>{expected_it, d_correctness_flags};
}

static bool are_all_flags_set(c2h::device_vector<std::uint32_t>& flag_vector, std::size_t num_flags_to_check)
{
  static constexpr auto bits_per_element = 8 * sizeof(std::uint32_t);
  bool all_flags_set                     = thrust::equal(
    flag_vector.cbegin(),
    flag_vector.cbegin() + (num_flags_to_check / bits_per_element),
    thrust::make_constant_iterator(0xFFFFFFFFU));
  if (num_flags_to_check % bits_per_element != 0)
  {
    std::uint32_t last_element_flags = (0x00000001U << (num_flags_to_check % bits_per_element)) - 0x01U;
    all_flags_set = all_flags_set && (flag_vector[num_flags_to_check / bits_per_element] == last_element_flags);
  }
  return all_flags_set;
}
