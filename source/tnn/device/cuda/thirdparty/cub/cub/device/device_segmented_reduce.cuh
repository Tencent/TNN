/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2022, NVIDIA CORPORATION.  All rights reserved.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

//! @file
//! cub::DeviceSegmentedReduce provides device-wide, parallel operations for computing a batched reduction across
//! multiple sequences of data items residing within device-accessible memory.

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/detail/choose_offset.cuh>
#include <cub/detail/nvtx.cuh>
#include <cub/device/dispatch/dispatch_reduce.cuh>
#include <cub/device/dispatch/dispatch_reduce_by_key.cuh>
#include <cub/iterator/arg_index_input_iterator.cuh>
#include <cub/util_deprecated.cuh>
#include <cub/util_type.cuh>

#include <cuda/std/type_traits>

#include <iterator>

CUB_NAMESPACE_BEGIN

//! @rst
//! DeviceSegmentedReduce provides device-wide, parallel operations for
//! computing a reduction across multiple sequences of data items
//! residing within device-accessible memory.
//!
//! Overview
//! +++++++++++++++++++++++++++++++++++++++++++++
//!
//! A `reduction <http://en.wikipedia.org/wiki/Reduce_(higher-order_function)>`_
//! (or *fold*) uses a binary combining operator to compute a single aggregate
//! from a sequence of input elements.
//!
//! Usage Considerations
//! +++++++++++++++++++++++++++++++++++++++++++++
//!
//! @cdp_class{DeviceSegmentedReduce}
//!
//! @endrst
struct DeviceSegmentedReduce
{
private:
  template <typename InputIteratorT,
            typename OutputIteratorT,
            typename BeginOffsetIteratorT,
            typename EndOffsetIteratorT,
            typename OffsetT,
            typename ReductionOpT,
            typename InitT,
            typename... Ts>
  CUB_RUNTIME_FUNCTION static cudaError_t segmented_reduce(
    ::cuda::std::false_type,
    void* d_temp_storage,
    std::size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OutputIteratorT d_out,
    int num_segments,
    BeginOffsetIteratorT d_begin_offsets,
    EndOffsetIteratorT d_end_offsets,
    ReductionOpT reduction_op,
    InitT initial_value,
    cudaStream_t stream);

  template <typename InputIteratorT,
            typename OutputIteratorT,
            typename BeginOffsetIteratorT,
            typename EndOffsetIteratorT,
            typename OffsetT,
            typename ReductionOpT,
            typename InitT,
            typename... Ts>
  CUB_RUNTIME_FUNCTION static cudaError_t segmented_reduce(
    ::cuda::std::true_type,
    void* d_temp_storage,
    std::size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OutputIteratorT d_out,
    int num_segments,
    BeginOffsetIteratorT d_begin_offsets,
    EndOffsetIteratorT d_end_offsets,
    ReductionOpT reduction_op,
    InitT initial_value,
    cudaStream_t stream)
  {
    return DispatchSegmentedReduce<
      InputIteratorT,
      OutputIteratorT,
      BeginOffsetIteratorT,
      EndOffsetIteratorT,
      OffsetT,
      ReductionOpT,
      Ts...>::Dispatch(d_temp_storage,
                       temp_storage_bytes,
                       d_in,
                       d_out,
                       num_segments,
                       d_begin_offsets,
                       d_end_offsets,
                       reduction_op,
                       initial_value,
                       stream);
  }

public:
  //! @rst
  //! Computes a device-wide segmented reduction using the specified
  //! binary ``reduction_op`` functor.
  //!
  //! - Does not support binary reduction operators that are non-commutative.
  //! - Provides "run-to-run" determinism for pseudo-associative reduction
  //!   (e.g., addition of floating point types) on the same GPU device.
  //!   However, results for pseudo-associative reduction may be inconsistent
  //!   from one device to a another device of a different compute-capability
  //!   because CUB can employ different tile-sizing for different architectures.
  //! - When input a contiguous sequence of segments, a single sequence
  //!   ``segment_offsets`` (of length ``num_segments + 1``) can be aliased
  //!   for both the ``d_begin_offsets`` and ``d_end_offsets`` parameters (where
  //!   the latter is specified as ``segment_offsets + 1``).
  //! - Let ``s`` be in ``[0, num_segments)``. The range
  //!   ``[d_out + d_begin_offsets[s], d_out + d_end_offsets[s])`` shall not
  //!   overlap ``[d_in + d_begin_offsets[s], d_in + d_end_offsets[s])``,
  //!   ``[d_begin_offsets, d_begin_offsets + num_segments)`` nor
  //!   ``[d_end_offsets, d_end_offsets + num_segments)``.
  //! - @devicestorage
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates a custom min-reduction of a device vector of ``int`` data elements.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_segmented_reduce_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin segmented-reduce-reduce
  //!     :end-before: example-end segmented-reduce-reduce
  //!
  //! @endrst
  //!
  //! @tparam InputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading input items @iterator
  //!
  //! @tparam OutputIteratorT
  //!   **[inferred]** Output iterator type for recording the reduced aggregate @iterator
  //!
  //! @tparam BeginOffsetIteratorT
  //!   **[inferred]** Random-access input iterator type for reading segment beginning offsets @iterator
  //!
  //! @tparam EndOffsetIteratorT
  //!   **[inferred]** Random-access input iterator type for reading segment ending offsets @iterator
  //!
  //! @tparam ReductionOpT
  //!   **[inferred]** Binary reduction functor type having member `T operator()(const T &a, const T &b)`
  //!
  //! @tparam T
  //!   **[inferred]** Data element type that is convertible to the `value` type of `InputIteratorT`
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When `nullptr`, the
  //!   required allocation size is written to `temp_storage_bytes` and no work is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in] d_in
  //!   Pointer to the input sequence of data items
  //!
  //! @param[out] d_out
  //!   Pointer to the output aggregate
  //!
  //! @param[in] num_segments
  //!   The number of segments that comprise the sorting data
  //!
  //! @param[in] d_begin_offsets
  //!   @rst
  //!   Random-access input iterator to the sequence of beginning offsets of
  //!   length ``num_segments``, such that ``d_begin_offsets[i]`` is the first
  //!   element of the *i*\ :sup:`th` data segment in ``d_keys_*`` and ``d_values_*``
  //!   @endrst
  //!
  //! @param[in] d_end_offsets
  //!   @rst
  //!   Random-access input iterator to the sequence of ending offsets of length
  //!   ``num_segments``, such that ``d_end_offsets[i] - 1`` is the last element of
  //!   the *i*\ :sup:`th` data segment in ``d_keys_*`` and ``d_values_*``.
  //!   If ``d_end_offsets[i] - 1 <= d_begin_offsets[i]``, the *i*\ :sup:`th` is considered empty.
  //!   @endrst
  //!
  //! @param[in] reduction_op
  //!   Binary reduction functor
  //!
  //! @param[in] initial_value
  //!   Initial value of the reduction for each segment
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename InputIteratorT,
            typename OutputIteratorT,
            typename BeginOffsetIteratorT,
            typename EndOffsetIteratorT,
            typename ReductionOpT,
            typename T>
  CUB_RUNTIME_FUNCTION static cudaError_t Reduce(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OutputIteratorT d_out,
    int num_segments,
    BeginOffsetIteratorT d_begin_offsets,
    EndOffsetIteratorT d_end_offsets,
    ReductionOpT reduction_op,
    T initial_value,
    cudaStream_t stream = 0)
  {
    CUB_DETAIL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceSegmentedReduce::Reduce");

    // Integer type for global offsets
    using OffsetT               = detail::common_iterator_value_t<BeginOffsetIteratorT, EndOffsetIteratorT>;
    using integral_offset_check = ::cuda::std::is_integral<OffsetT>;

    static_assert(integral_offset_check::value, "Offset iterator value type should be integral.");

    return segmented_reduce<InputIteratorT, OutputIteratorT, BeginOffsetIteratorT, EndOffsetIteratorT, OffsetT, ReductionOpT>(
      integral_offset_check{},
      d_temp_storage,
      temp_storage_bytes,
      d_in,
      d_out,
      num_segments,
      d_begin_offsets,
      d_end_offsets,
      reduction_op,
      initial_value, // zero-initialize
      stream);
  }

#ifndef DOXYGEN_SHOULD_SKIP_THIS // Do not document
  template <typename InputIteratorT,
            typename OutputIteratorT,
            typename BeginOffsetIteratorT,
            typename EndOffsetIteratorT,
            typename ReductionOpT,
            typename T>
  CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED CUB_RUNTIME_FUNCTION static cudaError_t Reduce(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OutputIteratorT d_out,
    int num_segments,
    BeginOffsetIteratorT d_begin_offsets,
    EndOffsetIteratorT d_end_offsets,
    ReductionOpT reduction_op,
    T initial_value,
    cudaStream_t stream,
    bool debug_synchronous)
  {
    CUB_DETAIL_RUNTIME_DEBUG_SYNC_USAGE_LOG

    return Reduce<InputIteratorT, OutputIteratorT, BeginOffsetIteratorT, EndOffsetIteratorT, ReductionOpT, T>(
      d_temp_storage,
      temp_storage_bytes,
      d_in,
      d_out,
      num_segments,
      d_begin_offsets,
      d_end_offsets,
      reduction_op,
      initial_value,
      stream);
  }
#endif // DOXYGEN_SHOULD_SKIP_THIS

  //! @rst
  //! Computes a device-wide segmented sum using the addition (``+``) operator.
  //!
  //! - Uses ``0`` as the initial value of the reduction for each segment.
  //! - When input a contiguous sequence of segments, a single sequence
  //!   ``segment_offsets`` (of length ``num_segments + 1``) can be aliased
  //!   for both the ``d_begin_offsets`` and ``d_end_offsets`` parameters (where
  //!   the latter is specified as ``segment_offsets + 1``).
  //! - Does not support ``+`` operators that are non-commutative.
  //! - Let ``s`` be in ``[0, num_segments)``. The range
  //!   ``[d_out + d_begin_offsets[s], d_out + d_end_offsets[s])`` shall not
  //!   overlap ``[d_in + d_begin_offsets[s], d_in + d_end_offsets[s])``,
  //!   ``[d_begin_offsets, d_begin_offsets + num_segments)`` nor
  //!   ``[d_end_offsets, d_end_offsets + num_segments)``.
  //! - @devicestorage
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the sum reduction of a device vector of ``int`` data elements.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_segmented_reduce_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin segmented-reduce-sum
  //!     :end-before: example-end segmented-reduce-sum
  //!
  //! @endrst
  //!
  //! @tparam InputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading input items @iterator
  //!
  //! @tparam OutputIteratorT
  //!   **[inferred]** Output iterator type for recording the reduced aggregate @iterator
  //!
  //! @tparam BeginOffsetIteratorT
  //!   **[inferred]** Random-access input iterator type for reading segment beginning offsets @iterator
  //!
  //! @tparam EndOffsetIteratorT
  //!   **[inferred]** Random-access input iterator type for reading segment ending offsets @iterator
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When `nullptr`, the
  //!   required allocation size is written to `temp_storage_bytes` and no work is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in] d_in
  //!   Pointer to the input sequence of data items
  //!
  //! @param[out] d_out
  //!   Pointer to the output aggregate
  //!
  //! @param[in] num_segments
  //!   The number of segments that comprise the sorting data
  //!
  //! @param[in] d_begin_offsets
  //!   @rst
  //!   Random-access input iterator to the sequence of beginning offsets of
  //!   length ``num_segments`, such that ``d_begin_offsets[i]`` is the first
  //!   element of the *i*\ :sup:`th` data segment in ``d_keys_*`` and
  //!   ``d_values_*``
  //!   @endrst
  //!
  //! @param[in] d_end_offsets
  //!   @rst
  //!   Random-access input iterator to the sequence of ending offsets of length
  //!   ``num_segments``, such that ``d_end_offsets[i] - 1`` is the last element of
  //!   the *i*\ :sup:`th` data segment in ``d_keys_*`` and ``d_values_*``.
  //!   If ``d_end_offsets[i] - 1 <= d_begin_offsets[i]``, the *i*\ :sup:`th` is considered empty.
  //!   @endrst
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename InputIteratorT, typename OutputIteratorT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
  CUB_RUNTIME_FUNCTION static cudaError_t
  Sum(void* d_temp_storage,
      size_t& temp_storage_bytes,
      InputIteratorT d_in,
      OutputIteratorT d_out,
      int num_segments,
      BeginOffsetIteratorT d_begin_offsets,
      EndOffsetIteratorT d_end_offsets,
      cudaStream_t stream = 0)
  {
    CUB_DETAIL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceSegmentedReduce::Sum");

    // Integer type for global offsets
    using OffsetT = detail::common_iterator_value_t<BeginOffsetIteratorT, EndOffsetIteratorT>;

    // The output value type
    using OutputT               = cub::detail::non_void_value_t<OutputIteratorT, cub::detail::value_t<InputIteratorT>>;
    using integral_offset_check = ::cuda::std::is_integral<OffsetT>;

    static_assert(integral_offset_check::value, "Offset iterator value type should be integral.");

    return segmented_reduce<InputIteratorT, OutputIteratorT, BeginOffsetIteratorT, EndOffsetIteratorT, OffsetT, cub::Sum>(
      integral_offset_check{},
      d_temp_storage,
      temp_storage_bytes,
      d_in,
      d_out,
      num_segments,
      d_begin_offsets,
      d_end_offsets,
      cub::Sum(),
      OutputT(), // zero-initialize
      stream);
  }

#ifndef DOXYGEN_SHOULD_SKIP_THIS // Do not document
  template <typename InputIteratorT, typename OutputIteratorT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
  CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED CUB_RUNTIME_FUNCTION static cudaError_t
  Sum(void* d_temp_storage,
      size_t& temp_storage_bytes,
      InputIteratorT d_in,
      OutputIteratorT d_out,
      int num_segments,
      BeginOffsetIteratorT d_begin_offsets,
      EndOffsetIteratorT d_end_offsets,
      cudaStream_t stream,
      bool debug_synchronous)
  {
    CUB_DETAIL_RUNTIME_DEBUG_SYNC_USAGE_LOG

    return Sum<InputIteratorT, OutputIteratorT, BeginOffsetIteratorT, EndOffsetIteratorT>(
      d_temp_storage, temp_storage_bytes, d_in, d_out, num_segments, d_begin_offsets, d_end_offsets, stream);
  }
#endif // DOXYGEN_SHOULD_SKIP_THIS

  //! @rst
  //! Computes a device-wide segmented minimum using the less-than (``<``) operator.
  //!
  //! - Uses ``std::numeric_limits<T>::max()`` as the initial value of the reduction for each segment.
  //! - When input a contiguous sequence of segments, a single sequence
  //!   ``segment_offsets`` (of length ``num_segments + 1``) can be aliased for both
  //!   the ``d_begin_offsets`` and ``d_end_offsets`` parameters (where the latter is
  //!   specified as ``segment_offsets + 1``).
  //! - Does not support ``<`` operators that are non-commutative.
  //! - Let ``s`` be in ``[0, num_segments)``. The range
  //!   ``[d_out + d_begin_offsets[s], d_out + d_end_offsets[s])`` shall not
  //!   overlap ``[d_in + d_begin_offsets[s], d_in + d_end_offsets[s])``,
  //!   ``[d_begin_offsets, d_begin_offsets + num_segments)`` nor
  //!   ``[d_end_offsets, d_end_offsets + num_segments)``.
  //! - @devicestorage
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the min-reduction of a device vector of ``int`` data elements.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_segmented_reduce_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin segmented-reduce-custommin
  //!     :end-before: example-end segmented-reduce-custommin
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_segmented_reduce_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin segmented-reduce-min
  //!     :end-before: example-end segmented-reduce-min
  //!
  //! @endrst
  //!
  //! @tparam InputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading input items @iterator
  //!
  //! @tparam OutputIteratorT
  //!   **[inferred]** Output iterator type for recording the reduced aggregate @iterator
  //!
  //! @tparam BeginOffsetIteratorT
  //!   **[inferred]** Random-access input iterator type for reading segment beginning offsets @iterator
  //!
  //! @tparam EndOffsetIteratorT
  //!   **[inferred]** Random-access input iterator type for reading segment ending offsets @iterator
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When `nullptr`, the
  //!   required allocation size is written to `temp_storage_bytes` and no work is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in] d_in
  //!   Pointer to the input sequence of data items
  //!
  //! @param[out] d_out
  //!   Pointer to the output aggregate
  //!
  //! @param[in] num_segments
  //!   The number of segments that comprise the sorting data
  //!
  //! @param[in] d_begin_offsets
  //!   @rst
  //!   Random-access input iterator to the sequence of beginning offsets of
  //!   length ``num_segments``, such that ``d_begin_offsets[i]`` is the first
  //!   element of the *i*\ :sup:`th` data segment in ``d_keys_*`` and ``d_values_*``
  //!   @endrst
  //!
  //! @param[in] d_end_offsets
  //!   @rst
  //!   Random-access input iterator to the sequence of ending offsets of length
  //!   ``num_segments``, such that ``d_end_offsets[i] - 1`` is the last element of
  //!   the *i*\ :sup:`th` data segment in ``d_keys_*`` and ``d_values_*``.
  //!   If ``d_end_offsets[i] - 1 <= d_begin_offsets[i]``, the *i*\ :sup:`th` is considered empty.
  //!   @endrst
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename InputIteratorT, typename OutputIteratorT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
  CUB_RUNTIME_FUNCTION static cudaError_t
  Min(void* d_temp_storage,
      size_t& temp_storage_bytes,
      InputIteratorT d_in,
      OutputIteratorT d_out,
      int num_segments,
      BeginOffsetIteratorT d_begin_offsets,
      EndOffsetIteratorT d_end_offsets,
      cudaStream_t stream = 0)
  {
    CUB_DETAIL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceSegmentedReduce::Min");

    // Integer type for global offsets
    using OffsetT = detail::common_iterator_value_t<BeginOffsetIteratorT, EndOffsetIteratorT>;

    // The input value type
    using InputT                = cub::detail::value_t<InputIteratorT>;
    using integral_offset_check = ::cuda::std::is_integral<OffsetT>;

    static_assert(integral_offset_check::value, "Offset iterator value type should be integral.");

    return segmented_reduce<InputIteratorT, OutputIteratorT, BeginOffsetIteratorT, EndOffsetIteratorT, OffsetT, cub::Min>(
      integral_offset_check{},
      d_temp_storage,
      temp_storage_bytes,
      d_in,
      d_out,
      num_segments,
      d_begin_offsets,
      d_end_offsets,
      cub::Min(),
      Traits<InputT>::Max(), // replace with
                             // std::numeric_limits<T>::max()
                             // when C++11 support is
                             // more prevalent
      stream);
  }

#ifndef DOXYGEN_SHOULD_SKIP_THIS // Do not document
  template <typename InputIteratorT, typename OutputIteratorT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
  CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED CUB_RUNTIME_FUNCTION static cudaError_t
  Min(void* d_temp_storage,
      size_t& temp_storage_bytes,
      InputIteratorT d_in,
      OutputIteratorT d_out,
      int num_segments,
      BeginOffsetIteratorT d_begin_offsets,
      EndOffsetIteratorT d_end_offsets,
      cudaStream_t stream,
      bool debug_synchronous)
  {
    CUB_DETAIL_RUNTIME_DEBUG_SYNC_USAGE_LOG

    return Min<InputIteratorT, OutputIteratorT, BeginOffsetIteratorT, EndOffsetIteratorT>(
      d_temp_storage, temp_storage_bytes, d_in, d_out, num_segments, d_begin_offsets, d_end_offsets, stream);
  }
#endif // DOXYGEN_SHOULD_SKIP_THIS

  //! @rst
  //! Finds the first device-wide minimum in each segment using the
  //! less-than (``<``) operator, also returning the in-segment index of that item.
  //!
  //! - The output value type of ``d_out`` is ``cub::KeyValuePair<int, T>``
  //!   (assuming the value type of ``d_in`` is ``T``)
  //!
  //!   - The minimum of the *i*\ :sup:`th` segment is written to
  //!     ``d_out[i].value`` and its offset in that segment is written to ``d_out[i].key``.
  //!   - The ``{1, std::numeric_limits<T>::max()}`` tuple is produced for zero-length inputs
  //!
  //! - When input a contiguous sequence of segments, a single sequence
  //!   ``segment_offsets`` (of length ``num_segments + 1``) can be aliased for both
  //!   the ``d_begin_offsets`` and ``d_end_offsets`` parameters (where the latter
  //!   is specified as ``segment_offsets + 1``).
  //! - Does not support ``<`` operators that are non-commutative.
  //! - Let ``s`` be in ``[0, num_segments)``. The range
  //!   ``[d_out + d_begin_offsets[s], d_out + d_end_offsets[s])`` shall not
  //!   overlap ``[d_in + d_begin_offsets[s], d_in + d_end_offsets[s])``,
  //!   ``[d_begin_offsets, d_begin_offsets + num_segments)`` nor
  //!   ``[d_end_offsets, d_end_offsets + num_segments)``.
  //! - @devicestorage
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the argmin-reduction of a device vector of ``int`` data elements.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_segmented_reduce_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin segmented-reduce-argmin
  //!     :end-before: example-end segmented-reduce-argmin
  //!
  //! @endrst
  //!
  //! @tparam InputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading input items (of some type `T`) @iterator
  //!
  //! @tparam OutputIteratorT
  //!   **[inferred]** Output iterator type for recording the reduced aggregate
  //!   (having value type `KeyValuePair<int, T>`) @iterator
  //!
  //! @tparam BeginOffsetIteratorT
  //!   **[inferred]** Random-access input iterator type for reading segment
  //!   beginning offsets @iterator
  //!
  //! @tparam EndOffsetIteratorT
  //!   **[inferred]** Random-access input iterator type for reading segment
  //!   ending offsets @iterator
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When `nullptr`, the
  //!   required allocation size is written to `temp_storage_bytes` and no work is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in] d_in
  //!   Pointer to the input sequence of data items
  //!
  //! @param[out] d_out
  //!   Pointer to the output aggregate
  //!
  //! @param[in] num_segments
  //!   The number of segments that comprise the sorting data
  //!
  //! @param[in] d_begin_offsets
  //!   @rst
  //!   Random-access input iterator to the sequence of beginning offsets of
  //!   length ``num_segments``, such that ``d_begin_offsets[i]`` is the first
  //!   element of the *i*\ :sup:`th` data segment in ``d_keys_*`` and ``d_values_*``
  //!   @endrst
  //!
  //! @param[in] d_end_offsets
  //!   @rst
  //!   Random-access input iterator to the sequence of ending offsets of length
  //!   ``num_segments``, such that ``d_end_offsets[i] - 1`` is the last element of
  //!   the *i*\ :sup:`th` data segment in ``d_keys_*`` and ``d_values_*``.
  //!   If ``d_end_offsets[i] - 1 <= d_begin_offsets[i]``, the *i*\ :sup:`th` is considered empty.
  //!   @endrst
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename InputIteratorT, typename OutputIteratorT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
  CUB_RUNTIME_FUNCTION static cudaError_t ArgMin(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OutputIteratorT d_out,
    int num_segments,
    BeginOffsetIteratorT d_begin_offsets,
    EndOffsetIteratorT d_end_offsets,
    cudaStream_t stream = 0)
  {
    CUB_DETAIL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceSegmentedReduce::ArgMin");

    // Integer type for global offsets
    // Using common iterator value type is a breaking change, see:
    // https://github.com/NVIDIA/cccl/pull/414#discussion_r1330632615
    using OffsetT = int; // detail::common_iterator_value_t<BeginOffsetIteratorT, EndOffsetIteratorT>;

    // The input type
    using InputValueT = cub::detail::value_t<InputIteratorT>;

    // The output tuple type
    using OutputTupleT = cub::detail::non_void_value_t<OutputIteratorT, KeyValuePair<OffsetT, InputValueT>>;

    // The output value type
    using OutputValueT = typename OutputTupleT::Value;

    using AccumT = OutputTupleT;

    using InitT = detail::reduce::empty_problem_init_t<AccumT>;

    // Wrapped input iterator to produce index-value <OffsetT, InputT> tuples
    using ArgIndexInputIteratorT = ArgIndexInputIterator<InputIteratorT, OffsetT, OutputValueT>;

    ArgIndexInputIteratorT d_indexed_in(d_in);

    // Initial value
    // TODO Address https://github.com/NVIDIA/cub/issues/651
    InitT initial_value{AccumT(1, Traits<InputValueT>::Max())};

    using integral_offset_check = ::cuda::std::is_integral<OffsetT>;
    static_assert(integral_offset_check::value, "Offset iterator value type should be integral.");

    return segmented_reduce<ArgIndexInputIteratorT,
                            OutputIteratorT,
                            BeginOffsetIteratorT,
                            EndOffsetIteratorT,
                            OffsetT,
                            cub::ArgMin,
                            InitT,
                            AccumT>(
      integral_offset_check{},
      d_temp_storage,
      temp_storage_bytes,
      d_indexed_in,
      d_out,
      num_segments,
      d_begin_offsets,
      d_end_offsets,
      cub::ArgMin(),
      initial_value,
      stream);
  }

#ifndef DOXYGEN_SHOULD_SKIP_THIS // Do not document
  template <typename InputIteratorT, typename OutputIteratorT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
  CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED CUB_RUNTIME_FUNCTION static cudaError_t ArgMin(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OutputIteratorT d_out,
    int num_segments,
    BeginOffsetIteratorT d_begin_offsets,
    EndOffsetIteratorT d_end_offsets,
    cudaStream_t stream,
    bool debug_synchronous)
  {
    CUB_DETAIL_RUNTIME_DEBUG_SYNC_USAGE_LOG

    return ArgMin<InputIteratorT, OutputIteratorT, BeginOffsetIteratorT, EndOffsetIteratorT>(
      d_temp_storage, temp_storage_bytes, d_in, d_out, num_segments, d_begin_offsets, d_end_offsets, stream);
  }
#endif // DOXYGEN_SHOULD_SKIP_THIS

  //! @rst
  //! Computes a device-wide segmented maximum using the greater-than (``>``) operator.
  //!
  //! - Uses ``std::numeric_limits<T>::lowest()`` as the initial value of the reduction.
  //! - When input a contiguous sequence of segments, a single sequence
  //!   ``segment_offsets`` (of length ``num_segments + 1``) can be aliased
  //!   for both the ``d_begin_offsets`` and ``d_end_offsets`` parameters (where
  //!   the latter is specified as ``segment_offsets + 1``).
  //! - Does not support ``>`` operators that are non-commutative.
  //! - Let ``s`` be in ``[0, num_segments)``. The range
  //!   ``[d_out + d_begin_offsets[s], d_out + d_end_offsets[s])`` shall not
  //!   overlap ``[d_in + d_begin_offsets[s], d_in + d_end_offsets[s])``,
  //!   ``[d_begin_offsets, d_begin_offsets + num_segments)`` nor
  //!   ``[d_end_offsets, d_end_offsets + num_segments)``.
  //! - @devicestorage
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the max-reduction of a device vector of ``int`` data elements.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_segmented_reduce_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin segmented-reduce-max
  //!     :end-before: example-end segmented-reduce-max
  //!
  //! @endrst
  //!
  //! @tparam InputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading input items @iterator
  //!
  //! @tparam OutputIteratorT
  //!   **[inferred]** Output iterator type for recording the reduced aggregate @iterator
  //!
  //! @tparam BeginOffsetIteratorT
  //!   **[inferred]** Random-access input iterator type for reading segment beginning offsets @iterator
  //!
  //! @tparam EndOffsetIteratorT
  //!   **[inferred]** Random-access input iterator type for reading segment ending offsets @iterator
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When `nullptr`, the
  //!   required allocation size is written to `temp_storage_bytes` and no work is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in] d_in
  //!   Pointer to the input sequence of data items
  //!
  //! @param[out] d_out
  //!   Pointer to the output aggregate
  //!
  //! @param[in] num_segments
  //!   The number of segments that comprise the sorting data
  //!
  //! @param[in] d_begin_offsets
  //!   @rst
  //!   Random-access input iterator to the sequence of beginning offsets of
  //!   length ``num_segments``, such that ``d_begin_offsets[i]`` is the first
  //!   element of the *i*\ :sup:`th` data segment in ``d_keys_*`` and ``d_values_*``
  //!   @endrst
  //!
  //! @param[in] d_end_offsets
  //!   @rst
  //!   Random-access input iterator to the sequence of ending offsets of length
  //!   ``num_segments``, such that ``d_end_offsets[i] - 1`` is the last element of
  //!   the *i*\ :sup:`th` data segment in ``d_keys_*`` and ``d_values_*``.
  //!   If ``d_end_offsets[i] - 1 <= d_begin_offsets[i]``, the *i*\ :sup:`th` is considered empty.
  //!   @endrst
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename InputIteratorT, typename OutputIteratorT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
  CUB_RUNTIME_FUNCTION static cudaError_t
  Max(void* d_temp_storage,
      size_t& temp_storage_bytes,
      InputIteratorT d_in,
      OutputIteratorT d_out,
      int num_segments,
      BeginOffsetIteratorT d_begin_offsets,
      EndOffsetIteratorT d_end_offsets,
      cudaStream_t stream = 0)
  {
    CUB_DETAIL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceSegmentedReduce::Max");

    // Integer type for global offsets
    using OffsetT = detail::common_iterator_value_t<BeginOffsetIteratorT, EndOffsetIteratorT>;

    // The input value type
    using InputT = cub::detail::value_t<InputIteratorT>;

    using integral_offset_check = ::cuda::std::is_integral<OffsetT>;
    static_assert(integral_offset_check::value, "Offset iterator value type should be integral.");

    return segmented_reduce<InputIteratorT, OutputIteratorT, BeginOffsetIteratorT, EndOffsetIteratorT, OffsetT>(
      integral_offset_check{},
      d_temp_storage,
      temp_storage_bytes,
      d_in,
      d_out,
      num_segments,
      d_begin_offsets,
      d_end_offsets,
      cub::Max(),
      Traits<InputT>::Lowest(), // replace with
                                // std::numeric_limits<T>::lowest()
                                // when C++11 support is
                                // more prevalent
      stream);
  }

#ifndef DOXYGEN_SHOULD_SKIP_THIS // Do not document
  template <typename InputIteratorT, typename OutputIteratorT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
  CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED CUB_RUNTIME_FUNCTION static cudaError_t
  Max(void* d_temp_storage,
      size_t& temp_storage_bytes,
      InputIteratorT d_in,
      OutputIteratorT d_out,
      int num_segments,
      BeginOffsetIteratorT d_begin_offsets,
      EndOffsetIteratorT d_end_offsets,
      cudaStream_t stream,
      bool debug_synchronous)
  {
    CUB_DETAIL_RUNTIME_DEBUG_SYNC_USAGE_LOG

    return Max<InputIteratorT, OutputIteratorT, BeginOffsetIteratorT, EndOffsetIteratorT>(
      d_temp_storage, temp_storage_bytes, d_in, d_out, num_segments, d_begin_offsets, d_end_offsets, stream);
  }
#endif // DOXYGEN_SHOULD_SKIP_THIS

  //! @rst
  //! Finds the first device-wide maximum in each segment using the
  //! greater-than (``>``) operator, also returning the in-segment index of that item
  //!
  //! - The output value type of ``d_out`` is ``cub::KeyValuePair<int, T>``
  //!   (assuming the value type of ``d_in`` is ``T``)
  //!
  //!   - The maximum of the *i*\ :sup:`th` segment is written to
  //!     ``d_out[i].value`` and its offset in that segment is written to ``d_out[i].key``.
  //!   - The ``{1, std::numeric_limits<T>::lowest()}`` tuple is produced for zero-length inputs
  //!
  //! - When input a contiguous sequence of segments, a single sequence
  //!   ``segment_offsets`` (of length ``num_segments + 1``) can be aliased
  //!   for both the ``d_begin_offsets`` and ``d_end_offsets`` parameters (where
  //!   the latter is specified as ``segment_offsets + 1``).
  //! - Does not support ``>`` operators that are non-commutative.
  //! - Let ``s`` be in ``[0, num_segments)``. The range
  //!   ``[d_out + d_begin_offsets[s], d_out + d_end_offsets[s])`` shall not
  //!   overlap ``[d_in + d_begin_offsets[s], d_in + d_end_offsets[s])``,
  //!   ``[d_begin_offsets, d_begin_offsets + num_segments)`` nor
  //!   ``[d_end_offsets, d_end_offsets + num_segments)``.
  //! - @devicestorage
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the argmax-reduction of a device vector
  //! of `int` data elements.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_segmented_reduce_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin segmented-reduce-argmax
  //!     :end-before: example-end segmented-reduce-argmax
  //!
  //! @endrst
  //!
  //! @tparam InputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading input items
  //!   (of some type `T`) @iterator
  //!
  //! @tparam OutputIteratorT
  //!   **[inferred]** Output iterator type for recording the reduced aggregate
  //!   (having value type `KeyValuePair<int, T>`) @iterator
  //!
  //! @tparam BeginOffsetIteratorT
  //!   **[inferred]** Random-access input iterator type for reading segment
  //!   beginning offsets @iterator
  //!
  //! @tparam EndOffsetIteratorT
  //!   **[inferred]** Random-access input iterator type for reading segment
  //!   ending offsets @iterator
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When `nullptr`, the
  //!   required allocation size is written to `temp_storage_bytes` and no work
  //!   is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in] d_in
  //!   Pointer to the input sequence of data items
  //!
  //! @param[out] d_out
  //!   Pointer to the output aggregate
  //!
  //! @param[in] num_segments
  //!   The number of segments that comprise the sorting data
  //!
  //! @param[in] d_begin_offsets
  //!   @rst
  //!   Random-access input iterator to the sequence of beginning offsets of
  //!   length `num_segments`, such that ``d_begin_offsets[i]`` is the first
  //!   element of the *i*\ :sup:`th` data segment in ``d_keys_*`` and ``d_values_*``
  //!   @endrst
  //!
  //! @param[in] d_end_offsets
  //!   @rst
  //!   Random-access input iterator to the sequence of ending offsets of length
  //!   ``num_segments``, such that ``d_end_offsets[i] - 1`` is the last element of
  //!   the *i*\ :sup:`th` data segment in ``d_keys_*`` and ``d_values_*``.
  //!   If ``d_end_offsets[i] - 1 <= d_begin_offsets[i]``, the *i*\ :sup:`th` is considered empty.
  //!   @endrst
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename InputIteratorT, typename OutputIteratorT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
  CUB_RUNTIME_FUNCTION static cudaError_t ArgMax(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OutputIteratorT d_out,
    int num_segments,
    BeginOffsetIteratorT d_begin_offsets,
    EndOffsetIteratorT d_end_offsets,
    cudaStream_t stream = 0)
  {
    CUB_DETAIL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceSegmentedReduce::ArgMax");

    // Integer type for global offsets
    // Using common iterator value type is a breaking change, see:
    // https://github.com/NVIDIA/cccl/pull/414#discussion_r1330632615
    using OffsetT = int; // detail::common_iterator_value_t<BeginOffsetIteratorT, EndOffsetIteratorT>;

    // The input type
    using InputValueT = cub::detail::value_t<InputIteratorT>;

    // The output tuple type
    using OutputTupleT = cub::detail::non_void_value_t<OutputIteratorT, KeyValuePair<OffsetT, InputValueT>>;

    using AccumT = OutputTupleT;

    using InitT = detail::reduce::empty_problem_init_t<AccumT>;

    // The output value type
    using OutputValueT = typename OutputTupleT::Value;

    // Wrapped input iterator to produce index-value <OffsetT, InputT> tuples
    using ArgIndexInputIteratorT = ArgIndexInputIterator<InputIteratorT, OffsetT, OutputValueT>;

    ArgIndexInputIteratorT d_indexed_in(d_in);

    // Initial value
    // TODO Address https://github.com/NVIDIA/cub/issues/651
    InitT initial_value{AccumT(1, Traits<InputValueT>::Lowest())};

    using integral_offset_check = ::cuda::std::is_integral<OffsetT>;
    static_assert(integral_offset_check::value, "Offset iterator value type should be integral.");

    return segmented_reduce<ArgIndexInputIteratorT,
                            OutputIteratorT,
                            BeginOffsetIteratorT,
                            EndOffsetIteratorT,
                            OffsetT,
                            cub::ArgMax,
                            InitT,
                            AccumT>(
      integral_offset_check{},
      d_temp_storage,
      temp_storage_bytes,
      d_indexed_in,
      d_out,
      num_segments,
      d_begin_offsets,
      d_end_offsets,
      cub::ArgMax(),
      initial_value,
      stream);
  }

#ifndef DOXYGEN_SHOULD_SKIP_THIS // Do not document
  template <typename InputIteratorT, typename OutputIteratorT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
  CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED CUB_RUNTIME_FUNCTION static cudaError_t ArgMax(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OutputIteratorT d_out,
    int num_segments,
    BeginOffsetIteratorT d_begin_offsets,
    EndOffsetIteratorT d_end_offsets,
    cudaStream_t stream,
    bool debug_synchronous)
  {
    CUB_DETAIL_RUNTIME_DEBUG_SYNC_USAGE_LOG

    return ArgMax<InputIteratorT, OutputIteratorT, BeginOffsetIteratorT, EndOffsetIteratorT>(
      d_temp_storage, temp_storage_bytes, d_in, d_out, num_segments, d_begin_offsets, d_end_offsets, stream);
  }
#endif // DOXYGEN_SHOULD_SKIP_THIS
};

CUB_NAMESPACE_END
