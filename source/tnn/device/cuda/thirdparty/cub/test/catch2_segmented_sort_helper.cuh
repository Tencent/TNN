/******************************************************************************
 * Copyright (c) 2011-2024, NVIDIA CORPORATION.  All rights reserved.
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
#pragma once

// #define CCCL_TEST_ENABLE_LARGE_SEGMENTED_SORT
#include <cub/device/device_segmented_sort.cuh>

#include <thrust/device_ptr.h>
#include <thrust/for_each.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/logical.h>
#include <thrust/random.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include <cuda/std/limits>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>

#include <cstdio>

#include <c2h/catch2_test_helper.cuh>
#include <c2h/cpu_timer.cuh>
#include <c2h/extended_types.cuh>
#include <c2h/utility.cuh>
#include <catch2_test_launch_helper.h>
#include <nv/target>

#define MAKE_SEED_MOD_FUNCTION(name, xor_mask)                  \
  inline c2h::seed_t make_##name##_seed(const c2h::seed_t seed) \
  {                                                             \
    auto seed_val = seed.get();                                 \
    /* Verify assumptions: */                                   \
    static_assert(sizeof(seed_val) == 8, "");                   \
    static_assert(sizeof(xor_mask) == 8, "");                   \
    return c2h::seed_t{seed_val ^ xor_mask};                    \
  }

// Each set of params should be different to make sure that we don't reuse the same seed for different random ops:
MAKE_SEED_MOD_FUNCTION(key, 0xcccccccccccccccc)
MAKE_SEED_MOD_FUNCTION(value, 0xaaaaaaaaaaaaaaaa)
MAKE_SEED_MOD_FUNCTION(offset, 0x5555555555555555)
MAKE_SEED_MOD_FUNCTION(offset_eraser, 0x3333333333333333)

#undef MAKE_SEED_MOD_FUNCTION

template <typename T>
struct unwrap_value_t_impl
{
  using type = T;
};

#if TEST_HALF_T
template <>
struct unwrap_value_t_impl<half_t>
{
  using type = __half;
};
#endif

#if TEST_BF_T
template <>
struct unwrap_value_t_impl<bfloat16_t>
{
  using type = __nv_bfloat16;
};
#endif

template <typename T>
using unwrap_value_t = typename unwrap_value_t_impl<T>::type;

///////////////////////////////////////////////////////////////////////////////
// Derived element gen/validation

template <typename T>
_CCCL_HOST_DEVICE __forceinline__ double compute_conversion_factor(int segment_size, T)
{
  const double max_value = static_cast<double>(::cuda::std::numeric_limits<T>::max());
  return (max_value + 1) / segment_size;
}

_CCCL_HOST_DEVICE __forceinline__ double compute_conversion_factor(int segment_size, double)
{
  const double max_value = ::cuda::std::numeric_limits<double>::max();
  return max_value / segment_size;
}

_CCCL_HOST_DEVICE __forceinline__ double compute_conversion_factor(int, cub::NullType)
{
  return 1.0;
}

template <typename T>
struct segment_filler
{
  T* d_data{};
  const int* d_offsets{};
  bool descending{};

  segment_filler(T* d_data, const int* d_offsets, bool descending)
      : d_data(d_data)
      , d_offsets(d_offsets)
      , descending(descending)
  {}

  _CCCL_DEVICE void operator()(int segment_id) const
  {
    const int segment_begin = d_offsets[segment_id];
    const int segment_end   = d_offsets[segment_id + 1];
    const int segment_size  = segment_end - segment_begin;
    if (segment_size == 0)
    {
      return;
    }

    const double conversion = compute_conversion_factor(segment_size, T{});

    if (descending)
    {
      int counter = segment_size - 1;
      for (int i = segment_begin; i < segment_end; i++)
      {
        d_data[i] = static_cast<T>(conversion * counter--);
      }
    }
    else
    {
      int counter = 0;
      for (int i = segment_begin; i < segment_end; i++)
      {
        d_data[i] = static_cast<T>(conversion * counter++);
      }
    }
  }
};

template <typename KeyT, typename ValueT, bool STABLE>
struct segment_checker
{
  const KeyT* d_keys{};
  ValueT* d_values{}; // May be permuted if STABLE is false.
  const int* d_offsets{};
  bool sort_descending{};

  segment_checker(const KeyT* d_keys, ValueT* d_values, const int* d_offsets, bool sort_descending)
      : d_keys(d_keys)
      , d_values(d_values)
      , d_offsets(d_offsets)
      , sort_descending(sort_descending)
  {}

  _CCCL_DEVICE bool operator()(int segment_id)
  {
    const int segment_begin = d_offsets[segment_id];
    const int segment_end   = d_offsets[segment_id + 1];
    const int segment_size  = segment_end - segment_begin;
    if (segment_size == 0)
    {
      return true;
    }

    if (!this->check_results(segment_begin, segment_size, ValueT{}))
    {
      return false;
    }

    return true;
  }

private:
  // Keys only:
  _CCCL_DEVICE _CCCL_FORCEINLINE bool check_results(
    int segment_begin, //
    int segment_size,
    cub::NullType)
  {
    const double conversion = compute_conversion_factor(segment_size, KeyT{});

    for (int i = 0; i < segment_size; i++)
    {
      const KeyT key = this->compute_key(i, segment_size, conversion);
      if (d_keys[segment_begin + i] != key)
      {
        return false;
      }
    }

    return true;
  }

  // Pairs:
  template <typename DispatchValueT> // Same as ValueT if not cub::NullType
  _CCCL_DEVICE _CCCL_FORCEINLINE bool
  check_results(int segment_begin, //
                int segment_size,
                DispatchValueT)
  {
    // Validating values is trickier, since duplicate keys lead to different requirements for stable/unstable sorts.
    const double key_conversion   = compute_conversion_factor(segment_size, KeyT{});
    const double value_conversion = compute_conversion_factor(segment_size, ValueT{});

    // Find ranges of duplicate keys in the output:
    int key_out_dup_begin = 0;
    while (key_out_dup_begin < segment_size)
    {
      int key_out_dup_end    = key_out_dup_begin;
      const KeyT current_key = this->compute_key(key_out_dup_end, segment_size, key_conversion);

      // Find end of duplicate key range and validate all output keys as we go:
      do
      {
        if (current_key != d_keys[segment_begin + key_out_dup_end])
        {
          return false;
        }
        key_out_dup_end++;
      } while (key_out_dup_end < segment_size
               && current_key == this->compute_key(key_out_dup_end, segment_size, key_conversion));

      // Bookkeeping for validating unstable sorts:
      int unchecked_values_for_current_dup_key_begin     = segment_begin + key_out_dup_begin;
      const int unchecked_values_for_current_dup_key_end = segment_begin + key_out_dup_end;

      // NVCC claims that these variables are set-but-not-used, and the usual tricks to silence
      // those warnings don't work. This convoluted nonsense, however, does work...
      if (static_cast<bool>(unchecked_values_for_current_dup_key_begin)
          || static_cast<bool>(unchecked_values_for_current_dup_key_end))
      {
        []() {}(); // no-op lambda
      }

      // Validate all output values for the current key by determining the input key indicies and computing the matching
      // input values.
      const int num_dup_keys     = key_out_dup_end - key_out_dup_begin;
      const int key_in_dup_begin = segment_size - key_out_dup_end;

      // Validate the range of values corresponding to the current duplicate key:
      for (int dup_idx = 0; dup_idx < num_dup_keys; ++dup_idx)
      {
        const int in_seg_idx = key_in_dup_begin + dup_idx;

        // Compute the original input value corresponding to the current duplicate key.
        // NOTE: Keys and values are generated using opposing ascending/descending parameters, so the generated input
        // values are descending when generating ascending input keys for a descending sort.
        const int conv_idx         = sort_descending ? (segment_size - 1 - in_seg_idx) : in_seg_idx;
        const ValueT current_value = static_cast<ValueT>(conv_idx * value_conversion);
        _CCCL_IF_CONSTEXPR (STABLE)
        {
          // For stable sorts, the output value must appear at an exact offset:
          const int out_seg_idx = key_out_dup_begin + dup_idx;
          if (current_value != d_values[segment_begin + out_seg_idx])
          {
            return false;
          }
        }
        else
        {
          // For unstable sorts, the reference value can appear anywhere in the output values corresponding to the
          // current duplicate key.
          // For each reference value, find the corresponding value in the output and swap it out of the unchecked
          // region:
          int probe_unchecked_idx = unchecked_values_for_current_dup_key_begin;
          for (; probe_unchecked_idx < unchecked_values_for_current_dup_key_end; ++probe_unchecked_idx)
          {
            if (current_value == d_values[probe_unchecked_idx])
            {
              using thrust::swap;
              swap(d_values[probe_unchecked_idx], d_values[unchecked_values_for_current_dup_key_begin]);
              unchecked_values_for_current_dup_key_begin++;
              break;
            }
          }

          //  Check that the probe found a match:
          if (probe_unchecked_idx == unchecked_values_for_current_dup_key_end)
          {
            return false;
          }
        } // End of STABLE/UNSTABLE check
      } // End of duplicate key value validation

      // Prepare for next set of dup keys
      key_out_dup_begin = key_out_dup_end;
    }

    return true;
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE KeyT compute_key(int seg_idx, int segment_size, double conversion)
  {
    int conv_idx = sort_descending ? (segment_size - 1 - seg_idx) : seg_idx;
    return static_cast<KeyT>(conv_idx * conversion);
  }
};

// Generates segmented arrays using keys/values derived from segment indices.
// d_offsets should be populated and d_keys/d_values preallocated.
// d_values may be left empty if ValueT == cub::NullType.
// If descending_sort is true, the keys will ascend and the values will descend.
// Duplicate keys will be generated if the segment size exceeds the max KeyT.
// Sorted results may be validated with validate_sorted_derived_outputs.
template <typename KeyT, typename ValueT>
void generate_unsorted_derived_inputs(
  bool descending_sort, //
  const c2h::device_vector<int>& d_offsets,
  c2h::device_vector<KeyT>& d_keys,
  c2h::device_vector<ValueT>& d_values)
{
  C2H_TIME_SCOPE("GenerateUnsortedDerivedInputs");

  static constexpr bool sort_pairs = !::cuda::std::is_same<ValueT, cub::NullType>::value;

  const int num_segments = static_cast<int>(d_offsets.size() - 1);
  const int* offsets     = thrust::raw_pointer_cast(d_offsets.data());
  KeyT* keys             = thrust::raw_pointer_cast(d_keys.data());
  ValueT* values         = thrust::raw_pointer_cast(d_values.data());

  (void) values; // Unused for key-only sort.

  // Build keys in reversed order from how they'll eventually be sorted:
  thrust::for_each(c2h::nosync_device_policy,
                   thrust::make_counting_iterator(0),
                   thrust::make_counting_iterator(num_segments),
                   segment_filler<KeyT>{keys, offsets, !descending_sort});
  _CCCL_IF_CONSTEXPR (sort_pairs)
  {
    // Values are generated in reversed order from keys:
    thrust::for_each(c2h::nosync_device_policy,
                     thrust::make_counting_iterator(0),
                     thrust::make_counting_iterator(num_segments),
                     segment_filler<ValueT>{values, offsets, descending_sort});
  }

  // The for_each calls are using nosync policies:
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
}

// Verifies the results of sorting the segmented key/value arrays produced by generate_unsorted_derived_inputs.
// Reference values are computed on-the-fly, avoiding the need for host/device transfers and reference array sorting.
// d_values may be left empty if ValueT == cub::NullType. d_values may be permuted within duplicate key ranges if STABLE
// is false.
template <bool STABLE, typename KeyT, typename ValueT>
void validate_sorted_derived_outputs(
  bool descending_sort, //
  const c2h::device_vector<int>& d_offsets,
  const c2h::device_vector<KeyT>& d_keys,
  c2h::device_vector<ValueT>& d_values)
{
  C2H_TIME_SCOPE("validate_sorted_derived_outputs");
  const int num_segments = static_cast<int>(d_offsets.size() - 1);
  const KeyT* keys       = thrust::raw_pointer_cast(d_keys.data());
  ValueT* values         = thrust::raw_pointer_cast(d_values.data());
  const int* offsets     = thrust::raw_pointer_cast(d_offsets.data());

  REQUIRE(thrust::all_of(c2h::device_policy,
                         thrust::make_counting_iterator(0),
                         thrust::make_counting_iterator(num_segments),
                         segment_checker<KeyT, ValueT, STABLE>{keys, values, offsets, descending_sort}));
}

///////////////////////////////////////////////////////////////////////////////
// Random element gen/validation

// Generates random key/value pairs in keys/values.
// d_values may be left empty if ValueT == cub::NullType.
template <typename KeyT, typename ValueT>
void generate_random_unsorted_inputs(c2h::seed_t seed, //
                                     c2h::device_vector<KeyT>& d_keys,
                                     c2h::device_vector<ValueT>& d_values)
{
  C2H_TIME_SCOPE("generate_random_unsorted_inputs");

  (void) d_values; // Unused for key-only sort.

  c2h::gen(make_key_seed(seed), d_keys);

  _CCCL_IF_CONSTEXPR (!::cuda::std::is_same<ValueT, cub::NullType>::value)
  {
    c2h::gen(make_value_seed(seed), d_values);
  }
}

// Stable sort the segmented key/values pairs in the host arrays.
// d_values may be left empty if ValueT == cub::NullType.
template <typename KeyT, typename ValueT>
void host_sort_random_inputs(
  bool sort_descending, //
  int num_segments,
  const int* h_begin_offsets,
  const int* h_end_offsets,
  c2h::host_vector<KeyT>& h_unsorted_keys,
  c2h::host_vector<ValueT>& h_unsorted_values = {})
{
  C2H_TIME_SCOPE("host_sort_random_inputs");

  constexpr bool sort_pairs = !::cuda::std::is_same<ValueT, cub::NullType>::value;

  (void) h_unsorted_values; // Unused for key-only sort.

  for (int segment_i = 0; segment_i < num_segments; segment_i++)
  {
    const int segment_begin = h_begin_offsets[segment_i];
    const int segment_end   = h_end_offsets[segment_i];

    if (segment_end == segment_begin)
    {
      continue;
    }

    _CCCL_IF_CONSTEXPR (sort_pairs)
    {
      if (sort_descending)
      {
        thrust::stable_sort_by_key(
          h_unsorted_keys.begin() + segment_begin,
          h_unsorted_keys.begin() + segment_end,
          h_unsorted_values.begin() + segment_begin,
          thrust::greater<KeyT>{});
      }
      else
      {
        thrust::stable_sort_by_key(h_unsorted_keys.begin() + segment_begin,
                                   h_unsorted_keys.begin() + segment_end,
                                   h_unsorted_values.begin() + segment_begin);
      }
    }
    else
    {
      if (sort_descending)
      {
        thrust::stable_sort(h_unsorted_keys.begin() + segment_begin, //
                            h_unsorted_keys.begin() + segment_end,
                            thrust::greater<KeyT>{});
      }
      else
      {
        thrust::stable_sort(h_unsorted_keys.begin() + segment_begin, //
                            h_unsorted_keys.begin() + segment_end);
      }
    }
  }
}

template <typename KeyT, typename ValueT>
struct unstable_segmented_value_checker
{
  const KeyT* ref_keys{};
  const ValueT* ref_values{};
  ValueT* test_values{};
  const int* offsets_begin{};
  const int* offsets_end{};

  unstable_segmented_value_checker(
    const KeyT* ref_keys,
    const ValueT* ref_values,
    ValueT* test_values,
    const int* offsets_begin,
    const int* offsets_end)
      : ref_keys(ref_keys)
      , ref_values(ref_values)
      , test_values(test_values)
      , offsets_begin(offsets_begin)
      , offsets_end(offsets_end)
  {}

  _CCCL_DEVICE bool operator()(int segment_id) const
  {
    const int segment_begin = offsets_begin[segment_id];
    const int segment_end   = offsets_end[segment_id];

    // Identify duplicate ranges of keys in the current segment
    for (int key_offset = segment_begin; key_offset < segment_end; /*inc in loop*/)
    {
      // Per range of duplicate keys, find the corresponding range of values:
      int unchecked_values_for_current_dup_key_begin = key_offset;
      int unchecked_values_for_current_dup_key_end   = key_offset + 1;

      const KeyT current_key = ref_keys[unchecked_values_for_current_dup_key_begin];
      while (unchecked_values_for_current_dup_key_end < segment_end
             && current_key == ref_keys[unchecked_values_for_current_dup_key_end])
      {
        unchecked_values_for_current_dup_key_end++;
      }

      // Iterate through all of the ref values and verify that they each appear once-and-only-once in the test values:
      for (int value_idx = unchecked_values_for_current_dup_key_begin;
           value_idx < unchecked_values_for_current_dup_key_end;
           value_idx++)
      {
        const ValueT current_value = ref_values[value_idx];
        int probe_unchecked_idx    = unchecked_values_for_current_dup_key_begin;
        for (; probe_unchecked_idx < unchecked_values_for_current_dup_key_end; probe_unchecked_idx++)
        {
          if (current_value == test_values[probe_unchecked_idx])
          {
            // Swap the found value out of the unchecked region to reduce the search space in future iterations:
            using thrust::swap;
            swap(test_values[probe_unchecked_idx], test_values[unchecked_values_for_current_dup_key_begin]);
            unchecked_values_for_current_dup_key_begin++;
            break;
          }
        }

        // Check that the probe found a match:
        if (probe_unchecked_idx == unchecked_values_for_current_dup_key_end)
        {
          return false;
        }
      }

      key_offset = unchecked_values_for_current_dup_key_end;
    }

    return true;
  }
};

// For UNSTABLE verification, test values may be permutated within each duplicate key range.
// They will not be modified when STABLE.
template <bool STABLE, typename KeyT, typename ValueT>
void validate_sorted_random_outputs(
  int num_segments,
  const int* d_segment_begin,
  const int* d_segment_end,
  const c2h::device_vector<KeyT>& d_ref_keys,
  const c2h::device_vector<KeyT>& d_sorted_keys,
  const c2h::device_vector<ValueT>& d_ref_values,
  c2h::device_vector<ValueT>& d_sorted_values)
{
  C2H_TIME_SCOPE("validate_sorted_random_outputs");

  (void) d_ref_values;
  (void) d_sorted_values;
  (void) num_segments;
  (void) d_segment_begin;
  (void) d_segment_end;

  // Verify that the key arrays match exactly:
  REQUIRE((d_ref_keys == d_sorted_keys) == true);

  // Verify segment-by-segment that the values are appropriately sorted for an unstable key-value sort:
  _CCCL_IF_CONSTEXPR (!::cuda::std::is_same<ValueT, cub::NullType>::value)
  {
    _CCCL_IF_CONSTEXPR (STABLE)
    {
      REQUIRE((d_ref_values == d_sorted_values) == true);
    }
    else
    {
      const KeyT* ref_keys     = thrust::raw_pointer_cast(d_ref_keys.data());
      const ValueT* ref_values = thrust::raw_pointer_cast(d_ref_values.data());
      ValueT* test_values      = thrust::raw_pointer_cast(d_sorted_values.data());

      REQUIRE(thrust::all_of(
        c2h::device_policy,
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(num_segments),
        unstable_segmented_value_checker<KeyT, ValueT>{
          ref_keys, ref_values, test_values, d_segment_begin, d_segment_end}));
    }
  }
};

///////////////////////////////////////////////////////////////////////////////
// Sorting abstraction/launcher

template <typename WrappedKeyT, typename ValueT>
CUB_RUNTIME_FUNCTION cudaError_t call_cub_segmented_sort_api(
  bool descending,
  bool double_buffer,
  bool stable_sort,

  void* tmp_storage,
  std::size_t& temp_storage_bytes,

  int* keys_selector,
  int* values_selector,

  WrappedKeyT* wrapped_input_keys,
  WrappedKeyT* wrapped_output_keys,

  ValueT* input_values,
  ValueT* output_values,

  int num_items,
  int num_segments,

  const int* d_begin_offsets,
  const int* d_end_offsets,

  cudaStream_t stream = 0)
{
  using KeyT                = unwrap_value_t<WrappedKeyT>;
  constexpr bool sort_pairs = !::cuda::std::is_same<ValueT, cub::NullType>::value;

  // Unused for key-only sort.
  (void) values_selector;
  (void) input_values;
  (void) output_values;

  auto input_keys  = reinterpret_cast<KeyT*>(wrapped_input_keys);
  auto output_keys = reinterpret_cast<KeyT*>(wrapped_output_keys);

  // Use different types for the offset begin/end iterators to ensure that this is supported:
  const int* offset_begin_it                  = d_begin_offsets;
  thrust::device_ptr<const int> offset_end_it = thrust::device_pointer_cast(d_end_offsets);

  cudaError_t status = cudaErrorInvalidValue;

  if (stable_sort)
  {
    _CCCL_IF_CONSTEXPR (sort_pairs)
    {
      if (descending)
      {
        if (double_buffer)
        {
          cub::DoubleBuffer<KeyT> keys_buffer(input_keys, output_keys);
          keys_buffer.selector = *keys_selector;

          cub::DoubleBuffer<ValueT> values_buffer(input_values, output_values);
          values_buffer.selector = *values_selector;

          status = cub::DeviceSegmentedSort::StableSortPairsDescending(
            tmp_storage,
            temp_storage_bytes,
            keys_buffer,
            values_buffer,
            num_items,
            num_segments,
            offset_begin_it,
            offset_end_it,
            stream);

          *keys_selector   = keys_buffer.selector;
          *values_selector = values_buffer.selector;
        }
        else
        {
          status = cub::DeviceSegmentedSort::StableSortPairsDescending(
            tmp_storage,
            temp_storage_bytes,
            input_keys,
            output_keys,
            input_values,
            output_values,
            num_items,
            num_segments,
            offset_begin_it,
            offset_end_it,
            stream);
        }
      }
      else
      {
        if (double_buffer)
        {
          cub::DoubleBuffer<KeyT> keys_buffer(input_keys, output_keys);
          keys_buffer.selector = *keys_selector;

          cub::DoubleBuffer<ValueT> values_buffer(input_values, output_values);
          values_buffer.selector = *values_selector;

          status = cub::DeviceSegmentedSort::StableSortPairs(
            tmp_storage,
            temp_storage_bytes,
            keys_buffer,
            values_buffer,
            num_items,
            num_segments,
            offset_begin_it,
            offset_end_it,
            stream);

          *keys_selector   = keys_buffer.selector;
          *values_selector = values_buffer.selector;
        }
        else
        {
          status = cub::DeviceSegmentedSort::StableSortPairs(
            tmp_storage,
            temp_storage_bytes,
            input_keys,
            output_keys,
            input_values,
            output_values,
            num_items,
            num_segments,
            offset_begin_it,
            offset_end_it,
            stream);
        }
      }
    }
    else
    {
      if (descending)
      {
        if (double_buffer)
        {
          cub::DoubleBuffer<KeyT> keys_buffer(input_keys, output_keys);
          keys_buffer.selector = *keys_selector;

          status = cub::DeviceSegmentedSort::StableSortKeysDescending(
            tmp_storage, //
            temp_storage_bytes,
            keys_buffer,
            num_items,
            num_segments,
            offset_begin_it,
            offset_end_it,
            stream);

          *keys_selector = keys_buffer.selector;
        }
        else
        {
          status = cub::DeviceSegmentedSort::StableSortKeysDescending(
            tmp_storage, //
            temp_storage_bytes,
            input_keys,
            output_keys,
            num_items,
            num_segments,
            offset_begin_it,
            offset_end_it,
            stream);
        }
      }
      else
      {
        if (double_buffer)
        {
          cub::DoubleBuffer<KeyT> keys_buffer(input_keys, output_keys);
          keys_buffer.selector = *keys_selector;

          status = cub::DeviceSegmentedSort::StableSortKeys(
            tmp_storage, //
            temp_storage_bytes,
            keys_buffer,
            num_items,
            num_segments,
            offset_begin_it,
            offset_end_it,
            stream);

          *keys_selector = keys_buffer.selector;
        }
        else
        {
          status = cub::DeviceSegmentedSort::StableSortKeys(
            tmp_storage, //
            temp_storage_bytes,
            input_keys,
            output_keys,
            num_items,
            num_segments,
            offset_begin_it,
            offset_end_it,
            stream);
        }
      }
    }
  }
  else
  {
    _CCCL_IF_CONSTEXPR (sort_pairs)
    {
      if (descending)
      {
        if (double_buffer)
        {
          cub::DoubleBuffer<KeyT> keys_buffer(input_keys, output_keys);
          keys_buffer.selector = *keys_selector;

          cub::DoubleBuffer<ValueT> values_buffer(input_values, output_values);
          values_buffer.selector = *values_selector;

          status = cub::DeviceSegmentedSort::SortPairsDescending(
            tmp_storage,
            temp_storage_bytes,
            keys_buffer,
            values_buffer,
            num_items,
            num_segments,
            offset_begin_it,
            offset_end_it,
            stream);

          *keys_selector   = keys_buffer.selector;
          *values_selector = values_buffer.selector;
        }
        else
        {
          status = cub::DeviceSegmentedSort::SortPairsDescending(
            tmp_storage,
            temp_storage_bytes,
            input_keys,
            output_keys,
            input_values,
            output_values,
            num_items,
            num_segments,
            offset_begin_it,
            offset_end_it,
            stream);
        }
      }
      else
      {
        if (double_buffer)
        {
          cub::DoubleBuffer<KeyT> keys_buffer(input_keys, output_keys);
          keys_buffer.selector = *keys_selector;

          cub::DoubleBuffer<ValueT> values_buffer(input_values, output_values);
          values_buffer.selector = *values_selector;

          status = cub::DeviceSegmentedSort::SortPairs(
            tmp_storage,
            temp_storage_bytes,
            keys_buffer,
            values_buffer,
            num_items,
            num_segments,
            offset_begin_it,
            offset_end_it,
            stream);

          *keys_selector   = keys_buffer.selector;
          *values_selector = values_buffer.selector;
        }
        else
        {
          status = cub::DeviceSegmentedSort::SortPairs(
            tmp_storage,
            temp_storage_bytes,
            input_keys,
            output_keys,
            input_values,
            output_values,
            num_items,
            num_segments,
            offset_begin_it,
            offset_end_it,
            stream);
        }
      }
    }
    else
    {
      if (descending)
      {
        if (double_buffer)
        {
          cub::DoubleBuffer<KeyT> keys_buffer(input_keys, output_keys);
          keys_buffer.selector = *keys_selector;

          status = cub::DeviceSegmentedSort::SortKeysDescending(
            tmp_storage, //
            temp_storage_bytes,
            keys_buffer,
            num_items,
            num_segments,
            offset_begin_it,
            offset_end_it,
            stream);

          *keys_selector = keys_buffer.selector;
        }
        else
        {
          status = cub::DeviceSegmentedSort::SortKeysDescending(
            tmp_storage, //
            temp_storage_bytes,
            input_keys,
            output_keys,
            num_items,
            num_segments,
            offset_begin_it,
            offset_end_it,
            stream);
        }
      }
      else
      {
        if (double_buffer)
        {
          cub::DoubleBuffer<KeyT> keys_buffer(input_keys, output_keys);
          keys_buffer.selector = *keys_selector;

          status = cub::DeviceSegmentedSort::SortKeys(
            tmp_storage, //
            temp_storage_bytes,
            keys_buffer,
            num_items,
            num_segments,
            offset_begin_it,
            offset_end_it,
            stream);

          *keys_selector = keys_buffer.selector;
        }
        else
        {
          status = cub::DeviceSegmentedSort::SortKeys(
            tmp_storage, //
            temp_storage_bytes,
            input_keys,
            output_keys,
            num_items,
            num_segments,
            offset_begin_it,
            offset_end_it,
            stream);
        }
      }
    }
  }

  return status;
}

struct segmented_sort_launcher_t
{
private:
  bool m_is_descending;
  bool m_double_buffer;
  bool m_stable_sort;
  int* m_selectors;

public:
  explicit segmented_sort_launcher_t(bool is_descending, bool double_buffer, bool stable_sort)
      : m_is_descending(is_descending)
      , m_double_buffer(double_buffer)
      , m_stable_sort(stable_sort)
      , m_selectors(nullptr)
  {}

  void initialize()
  {
    REQUIRE(cudaSuccess == cudaMallocHost(&m_selectors, 2 * sizeof(int)));
  }

  void finalize()
  {
    REQUIRE(cudaSuccess == cudaFreeHost(m_selectors));
    m_selectors = nullptr;
  }

  void set_key_selector(int sel)
  {
    m_selectors[0] = sel;
  }

  int key_selector() const
  {
    return m_selectors[0];
  }

  void set_value_selector(int sel)
  {
    m_selectors[1] = sel;
  }

  int value_selector() const
  {
    return m_selectors[1];
  }

  template <class... As>
  CUB_RUNTIME_FUNCTION cudaError_t operator()(std::uint8_t* d_temp_storage, std::size_t& temp_storage_bytes, As... as)
  {
    const cudaError_t status = call_cub_segmented_sort_api(
      m_is_descending, //
      m_double_buffer,
      m_stable_sort,
      d_temp_storage,
      temp_storage_bytes,
      m_selectors,
      m_selectors + 1,
      as...);

    return status;
  }
};

template <typename KeyT, typename ValueT = cub::NullType>
void call_cub_segmented_sort_api(
  bool descending,
  bool double_buffer,
  bool stable_sort,

  KeyT* input_keys,
  KeyT* output_keys,

  ValueT* input_values,
  ValueT* output_values,

  int num_items,
  int num_segments,

  const int* d_begin_offsets,
  const int* d_end_offsets,

  int* keys_selector   = nullptr,
  int* values_selector = nullptr)
{
  C2H_TIME_SCOPE("cub::DeviceSegmentedSort");
  CAPTURE(descending, double_buffer, stable_sort, num_items, num_segments);

  segmented_sort_launcher_t action(descending, double_buffer, stable_sort);
  action.initialize();

  if (keys_selector)
  {
    action.set_key_selector(*keys_selector);
  }

  if (values_selector)
  {
    action.set_value_selector(*values_selector);
  }

  launch(action, //
         input_keys,
         output_keys,
         input_values,
         output_values,
         num_items,
         num_segments,
         d_begin_offsets,
         d_end_offsets);

  if (keys_selector)
  {
    *keys_selector = action.key_selector();
  }

  if (values_selector)
  {
    *values_selector = action.value_selector();
  }

  action.finalize();
}

///////////////////////////////////////////////////////////////////////////////
// Testing implementations

constexpr bool ascending  = false;
constexpr bool descending = true;

constexpr bool pointers      = false;
constexpr bool double_buffer = true;

constexpr bool unstable = false;
constexpr bool stable   = true;

// Uses analytically derived key/value pairs for generation and validation.
// Much faster that test_segments_random, as this avoids H<->D copies and host reference sorting.
// Drawback is that the unsorted keys are always reversed (though duplicate keys introduce some sorting variation
// due to stability).
template <typename KeyT, typename ValueT = cub::NullType>
void test_segments_derived(const c2h::device_vector<int>& d_offsets_vec)
{
  C2H_TIME_SECTION_INIT();
  constexpr bool sort_pairs = !::cuda::std::is_same<ValueT, cub::NullType>::value;

  const int num_items    = d_offsets_vec.back();
  const int num_segments = static_cast<int>(d_offsets_vec.size() - 1);

  C2H_TIME_SECTION("Fetch num_items D->H");

  c2h::device_vector<KeyT> keys_input(num_items);
  c2h::device_vector<KeyT> keys_output(num_items);

  c2h::device_vector<ValueT> values_input;
  c2h::device_vector<ValueT> values_output;
  _CCCL_IF_CONSTEXPR (sort_pairs)
  {
    values_input.resize(num_items);
    values_output.resize(num_items);
  }

  C2H_TIME_SECTION("Allocate device memory");

  const bool stable_sort     = GENERATE(unstable, stable);
  const bool sort_descending = GENERATE(ascending, descending);
  const bool sort_buffers    = GENERATE(pointers, double_buffer);

  CAPTURE(c2h::type_name<KeyT>(),
          c2h::type_name<ValueT>(),
          sort_pairs,
          num_items,
          num_segments,
          stable_sort,
          sort_descending,
          sort_buffers);

  generate_unsorted_derived_inputs(sort_descending, d_offsets_vec, keys_input, values_input);

  int keys_selector   = 0;
  int values_selector = 1;

  _CCCL_IF_CONSTEXPR (sort_pairs)
  {
    if (sort_buffers)
    { // Value buffer selector is initialized to read from the second buffer:
      using namespace std;
      swap(values_input, values_output);
    }
  }

  const int* d_begin_offsets = thrust::raw_pointer_cast(d_offsets_vec.data());
  const int* d_end_offsets   = thrust::raw_pointer_cast(d_offsets_vec.data() + 1);
  KeyT* d_keys_input         = thrust::raw_pointer_cast(keys_input.data());
  KeyT* d_keys_output        = thrust::raw_pointer_cast(keys_output.data());
  ValueT* d_values_input     = thrust::raw_pointer_cast(values_input.data());
  ValueT* d_values_output    = thrust::raw_pointer_cast(values_output.data());

  call_cub_segmented_sort_api(
    sort_descending,
    sort_buffers,
    stable_sort,
    d_keys_input,
    d_keys_output,
    d_values_input,
    d_values_output,
    num_items,
    num_segments,
    d_begin_offsets,
    d_end_offsets,
    &keys_selector,
    &values_selector);

  auto& keys   = (keys_selector || !sort_buffers) ? keys_output : keys_input;
  auto& values = (values_selector || !sort_buffers) ? values_output : values_input;

  if (stable_sort)
  {
    validate_sorted_derived_outputs<true>(sort_descending, d_offsets_vec, keys, values);
  }
  else
  {
    validate_sorted_derived_outputs<false>(sort_descending, d_offsets_vec, keys, values);
  }
}

template <typename KeyT, typename ValueT>
void test_segments_random(
  c2h::seed_t seed, //
  int num_items,
  int num_segments,
  const int* d_begin_offsets,
  const int* d_end_offsets)
{
  constexpr bool sort_pairs = !::cuda::std::is_same<ValueT, cub::NullType>::value;

  CAPTURE(c2h::type_name<KeyT>(), //
          c2h::type_name<ValueT>(),
          sort_pairs,
          num_items,
          num_segments);

  C2H_TIME_SECTION_INIT();

  c2h::device_vector<KeyT> keys_input(num_items);
  c2h::device_vector<KeyT> keys_output(num_items);

  c2h::device_vector<ValueT> values_input;
  c2h::device_vector<ValueT> values_output;
  _CCCL_IF_CONSTEXPR (sort_pairs)
  {
    values_input.resize(num_items);
    values_output.resize(num_items);
  }

  C2H_TIME_SECTION("Allocate device memory");

  const bool sort_descending = GENERATE(ascending, descending);

  generate_random_unsorted_inputs(seed, keys_input, values_input);

  // Initialize the output values to the inputs so unused segments will be filled with the expected random values:
  keys_output   = keys_input;
  values_output = values_input;

  C2H_TIME_SECTION_RESET();

  // Since we only have offset pointers, allocate offsets first and then copy -- using the iterator constructor for
  // cross-system copies would do a separate D->H transfer for each element:
  c2h::host_vector<int> h_begin_offsets(num_segments);
  thrust::copy(thrust::device_pointer_cast(d_begin_offsets),
               thrust::device_pointer_cast(d_begin_offsets + num_segments),
               h_begin_offsets.begin());
  c2h::host_vector<int> h_end_offsets(num_segments);
  thrust::copy(thrust::device_pointer_cast(d_end_offsets),
               thrust::device_pointer_cast(d_end_offsets + num_segments),
               h_end_offsets.begin());

  // Copying a vector D->H will do a bulk copy:
  c2h::host_vector<KeyT> h_keys_ref     = keys_input;
  c2h::host_vector<ValueT> h_values_ref = values_input;

  C2H_TIME_SECTION("D->H input arrays");

  c2h::device_vector<KeyT> keys_orig     = keys_input;
  c2h::device_vector<ValueT> values_orig = values_input;

  C2H_TIME_SECTION("Clone input arrays on device");

  host_sort_random_inputs(
    sort_descending,
    num_segments,
    thrust::raw_pointer_cast(h_begin_offsets.data()),
    thrust::raw_pointer_cast(h_end_offsets.data()),
    h_keys_ref,
    h_values_ref);

  C2H_TIME_SECTION_RESET();

  c2h::device_vector<KeyT> keys_ref     = h_keys_ref;
  c2h::device_vector<ValueT> values_ref = h_values_ref;

  C2H_TIME_SECTION("H->D reference arrays");

  bool need_reset = false;

  // Don't use GENERATE for these. We can reuse the expensive reference arrays for them:
  for (bool stable_sort : {unstable, stable})
  {
    for (bool sort_buffers : {pointers, double_buffer})
    {
      CAPTURE(stable_sort, sort_descending, sort_buffers);

      if (need_reset)
      {
        C2H_TIME_SCOPE("Reset input/output device arrays");
        keys_input   = keys_orig;
        values_input = values_orig;
        // Initialize the outputs so we have the expected random sequences in any unused segments.
        keys_output   = keys_orig;
        values_output = values_orig;
      }

      int keys_selector   = 0;
      int values_selector = 1;

      _CCCL_IF_CONSTEXPR (sort_pairs)
      {
        if (sort_buffers)
        { // Value buffer selector is initialized to read from the second buffer:
          using namespace std;
          swap(values_input, values_output);
        }
      }

      KeyT* d_keys_input      = thrust::raw_pointer_cast(keys_input.data());
      KeyT* d_keys_output     = thrust::raw_pointer_cast(keys_output.data());
      ValueT* d_values_input  = thrust::raw_pointer_cast(values_input.data());
      ValueT* d_values_output = thrust::raw_pointer_cast(values_output.data());

      call_cub_segmented_sort_api(
        sort_descending,
        sort_buffers,
        stable_sort,
        d_keys_input,
        d_keys_output,
        d_values_input,
        d_values_output,
        num_items,
        num_segments,
        d_begin_offsets,
        d_end_offsets,
        &keys_selector,
        &values_selector);

      need_reset       = true;
      const auto& keys = (keys_selector || !sort_buffers) ? keys_output : keys_input;
      auto& values     = (values_selector || !sort_buffers) ? values_output : values_input;

      if (stable_sort)
      {
        validate_sorted_random_outputs<true>(
          num_segments, d_begin_offsets, d_end_offsets, keys_ref, keys, values_ref, values);
      }
      else
      {
        validate_sorted_random_outputs<false>(
          num_segments, d_begin_offsets, d_end_offsets, keys_ref, keys, values_ref, values);
      }
    }
  }
}

template <typename KeyT, typename ValueT>
void test_segments_random(c2h::seed_t seed, const c2h::device_vector<int>& d_offsets_vec)
{
  const int num_items    = d_offsets_vec.back();
  const int num_segments = static_cast<int>(d_offsets_vec.size() - 1);

  test_segments_random<KeyT, ValueT>(
    seed, //
    num_items,
    num_segments,
    thrust::raw_pointer_cast(d_offsets_vec.data()),
    thrust::raw_pointer_cast(d_offsets_vec.data() + 1));
}

///////////////////////////////////////////////////////////////////////////////
// Offset generators

inline c2h::device_vector<int> generate_same_size_offsets(int segment_size, int num_segments)
{
  c2h::device_vector<int> offsets(num_segments + 1);
  thrust::sequence(c2h::device_policy, offsets.begin(), offsets.end(), int{}, segment_size);
  return offsets;
}

struct offset_scan_op_t
{
  int max_items;

  _CCCL_DEVICE _CCCL_FORCEINLINE int operator()(int a, int b) const
  {
    const int sum = a + b;
    return CUB_MIN(sum, max_items);
  }
};

inline c2h::device_vector<int>
generate_random_offsets(c2h::seed_t seed, int max_items, int max_segment, int num_segments)
{
  C2H_TIME_SCOPE("generate_random_offsets");
  const int expected_segment_length = ::cuda::ceil_div(max_items, num_segments);
  const int max_segment_length      = CUB_MIN(max_segment, (expected_segment_length * 2) + 1);

  c2h::device_vector<int> offsets(num_segments + 1);
  c2h::gen(make_offset_seed(seed), offsets, 0, max_segment_length);

  thrust::exclusive_scan(
    c2h::device_policy, //
    offsets.cbegin(),
    offsets.cend(),
    offsets.begin(),
    0,
    offset_scan_op_t{max_items});

  return offsets;
}

struct generate_edge_case_offsets_dispatch
{
  // Edge cases that needs to be tested
  static constexpr int empty_short_circuit_segment_size = 0;
  static constexpr int copy_short_circuit_segment_size  = 1;
  static constexpr int swap_short_circuit_segment_size  = 2;

  static constexpr int a_few      = 2;
  static constexpr int a_bunch_of = 42;
  static constexpr int a_lot_of   = 420;

  int small_segment_max_segment_size;
  int items_per_small_segment;
  int medium_segment_max_segment_size;
  int single_thread_segment_size;
  int large_cached_segment_max_segment_size;

  template <typename ActivePolicyT>
  CUB_RUNTIME_FUNCTION cudaError_t Invoke()
  {
    NV_IF_TARGET(
      NV_IS_HOST,
      (using SmallAndMediumPolicyT = typename ActivePolicyT::SmallAndMediumSegmentedSortPolicyT;
       using LargeSegmentPolicyT   = typename ActivePolicyT::LargeSegmentPolicy;

       small_segment_max_segment_size  = SmallAndMediumPolicyT::SmallPolicyT::ITEMS_PER_TILE;
       items_per_small_segment         = SmallAndMediumPolicyT::SmallPolicyT::ITEMS_PER_THREAD;
       medium_segment_max_segment_size = SmallAndMediumPolicyT::MediumPolicyT::ITEMS_PER_TILE;
       single_thread_segment_size      = items_per_small_segment;
       large_cached_segment_max_segment_size =
         LargeSegmentPolicyT::BLOCK_THREADS * LargeSegmentPolicyT::ITEMS_PER_THREAD; //
       ));

    return cudaSuccess;
  }

  c2h::device_vector<int> generate_offsets() const
  {
    c2h::host_vector<int> h_offsets;

    auto add_segments = [&h_offsets](int num_segments, int segment_size) {
      h_offsets.resize(h_offsets.size() + num_segments, segment_size);
    };

    add_segments(a_lot_of, empty_short_circuit_segment_size);
    add_segments(a_lot_of, copy_short_circuit_segment_size);
    add_segments(a_lot_of, swap_short_circuit_segment_size);
    add_segments(a_lot_of, swap_short_circuit_segment_size + 1);
    add_segments(a_lot_of, swap_short_circuit_segment_size + 1);
    add_segments(a_lot_of, single_thread_segment_size - 1);
    add_segments(a_lot_of, single_thread_segment_size);
    add_segments(a_lot_of, single_thread_segment_size + 1);
    add_segments(a_lot_of, single_thread_segment_size * 2 - 1);
    add_segments(a_lot_of, single_thread_segment_size * 2);
    add_segments(a_lot_of, single_thread_segment_size * 2 + 1);
    add_segments(a_bunch_of, small_segment_max_segment_size - 1);
    add_segments(a_bunch_of, small_segment_max_segment_size);
    add_segments(a_bunch_of, small_segment_max_segment_size + 1);
    add_segments(a_bunch_of, medium_segment_max_segment_size - 1);
    add_segments(a_bunch_of, medium_segment_max_segment_size);
    add_segments(a_bunch_of, medium_segment_max_segment_size + 1);
    add_segments(a_bunch_of, large_cached_segment_max_segment_size - 1);
    add_segments(a_bunch_of, large_cached_segment_max_segment_size);
    add_segments(a_bunch_of, large_cached_segment_max_segment_size + 1);
    add_segments(a_few, large_cached_segment_max_segment_size * 2);
    add_segments(a_few, large_cached_segment_max_segment_size * 3);
    add_segments(a_few, large_cached_segment_max_segment_size * 5);

    c2h::device_vector<int> d_offsets = h_offsets;
    thrust::exclusive_scan(c2h::device_policy, d_offsets.cbegin(), d_offsets.cend(), d_offsets.begin(), 0);
    return d_offsets;
  }
};

template <typename KeyT, typename ValueT>
c2h::device_vector<int> generate_edge_case_offsets()
{
  C2H_TIME_SCOPE("generate_edge_case_offsets");

  using MaxPolicyT = typename cub::DeviceSegmentedSortPolicy<KeyT, ValueT>::MaxPolicy;

  int ptx_version = 0;
  REQUIRE(cudaSuccess == CubDebug(cub::PtxVersion(ptx_version)));

  generate_edge_case_offsets_dispatch dispatch;
  REQUIRE(cudaSuccess == CubDebug(MaxPolicyT::Invoke(ptx_version, dispatch)));

  return dispatch.generate_offsets();
}

// Returns num_items
inline int generate_unspecified_segments_offsets(
  c2h::seed_t seed, c2h::device_vector<int>& d_begin_offsets, c2h::device_vector<int>& d_end_offsets)
{
  C2H_TIME_SCOPE("generate_unspecified_segments_offsets");

  const int max_items        = 1 << 18;
  const int max_segment_size = 1000;
  const int num_segments     = 4000;

  d_begin_offsets = generate_random_offsets(seed, max_items, max_segment_size, num_segments);
  d_end_offsets.resize(num_segments);

  // Skip the first offset -- it's always 0 at this point (exclusive_scan generates begin offsets, end offsets are
  // initialized to 0). This ensures that we test the case where the beginning of the inputs are not in any segment.
  thrust::copy(d_begin_offsets.cbegin() + 2, d_begin_offsets.cend(), d_end_offsets.begin() + 1);
  d_begin_offsets.pop_back();

  // Scatter some zeros around to erase half of the segments:
  c2h::device_vector<int> erase_indices(num_segments / 2);
  // Don't zero out the first or last segment. These are handled specially in other ways -- the first segment is always
  // zeroed explicitly in the offset arrays, and the last segment is required to be specified for the num_items
  // calculation below.
  c2h::gen(make_offset_eraser_seed(seed), erase_indices, 1, num_segments - 2);

  auto const_zero_begin = thrust::make_constant_iterator<int>(0);
  auto const_zero_end   = const_zero_begin + erase_indices.size();

  thrust::scatter(
    c2h::nosync_device_policy, const_zero_begin, const_zero_end, erase_indices.cbegin(), d_begin_offsets.begin());
  thrust::scatter(
    c2h::nosync_device_policy, const_zero_begin, const_zero_end, erase_indices.cbegin(), d_end_offsets.begin());

  REQUIRE(cudaSuccess == CubDebug(cudaDeviceSynchronize()));

  // Add more items to place another unspecified segment at the end.
  const int num_items = d_end_offsets.back() + 243;
  return num_items;
}

///////////////////////////////////////////////////////////////////////////////
// Entry points

template <typename KeyT, typename ValueT = cub::NullType>
void test_same_size_segments_derived(int segment_size, int num_segments)
{
  CAPTURE(segment_size, num_segments);
  c2h::device_vector<int> offsets = generate_same_size_offsets(segment_size, num_segments);
  test_segments_derived<KeyT, ValueT>(offsets);
}

template <typename KeyT, typename ValueT = cub::NullType>
void test_random_size_segments_derived(c2h::seed_t seed, int max_items, int max_segment, int num_segments)
{
  CAPTURE(seed.get());
  c2h::device_vector<int> offsets = generate_random_offsets(seed, max_items, max_segment, num_segments);
  test_segments_derived<KeyT, ValueT>(offsets);
}

template <typename KeyT, typename ValueT = cub::NullType>
void test_random_size_segments_random(c2h::seed_t seed, int max_items, int max_segment, int num_segments)
{
  CAPTURE(seed.get());
  c2h::device_vector<int> offsets = generate_random_offsets(seed, max_items, max_segment, num_segments);
  test_segments_random<KeyT, ValueT>(seed, offsets);
}

template <typename KeyT, typename ValueT = cub::NullType>
void test_edge_case_segments_random(c2h::seed_t seed)
{
  CAPTURE(seed.get());
  c2h::device_vector<int> offsets = generate_edge_case_offsets<KeyT, ValueT>();
  test_segments_random<KeyT, ValueT>(seed, offsets);
}

template <typename KeyT, typename ValueT = cub::NullType>
void test_unspecified_segments_random(c2h::seed_t seed)
{
  CAPTURE(seed.get());

  c2h::device_vector<int> d_begin_offsets;
  c2h::device_vector<int> d_end_offsets;
  const int num_items = generate_unspecified_segments_offsets(seed, d_begin_offsets, d_end_offsets);

  const int num_segments = static_cast<int>(d_begin_offsets.size());

  test_segments_random<KeyT, ValueT>(
    seed,
    num_items,
    num_segments,
    thrust::raw_pointer_cast(d_begin_offsets.data()),
    thrust::raw_pointer_cast(d_end_offsets.data()));
}
