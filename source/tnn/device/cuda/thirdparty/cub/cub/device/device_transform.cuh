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

#include <cub/detail/nvtx.cuh>
#include <cub/device/dispatch/dispatch_transform.cuh>
#include <cub/util_namespace.cuh>

#include <cuda/std/tuple>

CUB_NAMESPACE_BEGIN

//! DeviceTransform provides device-wide, parallel operations for transforming elements tuple-wise from multiple input
//! sequences into an output sequence.
struct DeviceTransform
{
  //! @rst
  //! Overview
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //! Transforms many input sequences into one output sequence, by applying a transformation operation on corresponding
  //! input elements and writing the result to the corresponding output element. No guarantee is given on the identity
  //! (i.e. address) of the objects passed to the call operator of the transformation operation.
  //!
  //! A Simple Example
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_transform_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin transform-many
  //!     :end-before: example-end transform-many
  //!
  //! @endrst
  //!
  //! @param inputs A tuple of iterators to the input sequences where num_items elements are read from each. The
  //! iterators' value types must be trivially relocatable.
  //! @param output An iterator to the output sequence where num_items results are written to.
  //! @param num_items The number of elements in each input sequence.
  //! @param transform_op An n-ary function object, where n is the number of input sequences. The input iterators' value
  //! types must be convertible to the parameters of the function object's call operator. The return type of the call
  //! operator must be assignable to the dereferenced output iterator.
  //! @param stream **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  template <typename... RandomAccessIteratorsIn, typename RandomAccessIteratorOut, typename TransformOp>
  CUB_RUNTIME_FUNCTION static cudaError_t Transform(
    ::cuda::std::tuple<RandomAccessIteratorsIn...> inputs,
    RandomAccessIteratorOut output,
    int num_items,
    TransformOp transform_op,
    cudaStream_t stream = nullptr)
  {
    CUB_DETAIL_NVTX_RANGE_SCOPE("cub::DeviceTransform::Transform");
    return detail::transform::
      dispatch_t<false, int, ::cuda::std::tuple<RandomAccessIteratorsIn...>, RandomAccessIteratorOut, TransformOp>::
        dispatch(
          ::cuda::std::move(inputs), ::cuda::std::move(output), num_items, ::cuda::std::move(transform_op), stream);
  }

#ifndef DOXYGEN_SHOULD_SKIP_THIS // Do not document
  // This overload has additional parameters to specify temporary storage. Provided for compatibility with other CUB
  // APIs.
  template <typename... RandomAccessIteratorsIn, typename RandomAccessIteratorOut, typename TransformOp>
  CUB_RUNTIME_FUNCTION static cudaError_t Transform(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    ::cuda::std::tuple<RandomAccessIteratorsIn...> inputs,
    RandomAccessIteratorOut output,
    int num_items,
    TransformOp transform_op,
    cudaStream_t stream = nullptr)
  {
    if (d_temp_storage == nullptr)
    {
      temp_storage_bytes = 1;
      return cudaSuccess;
    }

    return Transform(
      ::cuda::std::move(inputs), ::cuda::std::move(output), num_items, ::cuda::std::move(transform_op), stream);
  }
#endif // DOXYGEN_SHOULD_SKIP_THIS

  //! @rst
  //! Transforms one input sequence into one output sequence, by applying a transformation operation on corresponding
  //! input elements and writing the result to the corresponding output element. No guarantee is given on the identity
  //! (i.e. address) of the objects passed to the call operator of the transformation operation.
  //! @endrst
  //!
  //! @param input An iterator to the input sequence where num_items elements are read from. The iterator's value type
  //! must be trivially relocatable.
  //! @param output An iterator to the output sequence where num_items results are written to.
  //! @param num_items The number of elements in each input sequence.
  //! @param transform_op An n-ary function object, where n is the number of input sequences. The input iterators' value
  //! types must be convertible to the parameters of the function object's call operator. The return type of the call
  //! operator must be assignable to the dereferenced output iterator.
  //! @param stream **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  template <typename RandomAccessIteratorIn, typename RandomAccessIteratorOut, typename TransformOp>
  CUB_RUNTIME_FUNCTION static cudaError_t Transform(
    RandomAccessIteratorIn input,
    RandomAccessIteratorOut output,
    int num_items,
    TransformOp transform_op,
    cudaStream_t stream = nullptr)
  {
    return Transform(
      ::cuda::std::make_tuple(::cuda::std::move(input)),
      ::cuda::std::move(output),
      num_items,
      ::cuda::std::move(transform_op),
      stream);
  }

#ifndef DOXYGEN_SHOULD_SKIP_THIS // Do not document
  // This overload has additional parameters to specify temporary storage. Provided for compatibility with other CUB
  // APIs.
  template <typename RandomAccessIteratorIn, typename RandomAccessIteratorOut, typename TransformOp>
  CUB_RUNTIME_FUNCTION static cudaError_t Transform(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    RandomAccessIteratorIn input,
    RandomAccessIteratorOut output,
    int num_items,
    TransformOp transform_op,
    cudaStream_t stream = nullptr)
  {
    if (d_temp_storage == nullptr)
    {
      temp_storage_bytes = 1;
      return cudaSuccess;
    }

    return Transform(
      ::cuda::std::make_tuple(::cuda::std::move(input)),
      ::cuda::std::move(output),
      num_items,
      ::cuda::std::move(transform_op),
      stream);
  }
#endif // DOXYGEN_SHOULD_SKIP_THIS

  //! @rst
  //! Overview
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //! Transforms many input sequences into one output sequence, by applying a transformation operation on corresponding
  //! input elements and writing the result to the corresponding output element. The objects passed to the call operator
  //! of the transformation operation are guaranteed to reside in the input sequences and are never copied.
  //!
  //! A Simple Example
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_transform_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin transform-many-stable
  //!     :end-before: example-end transform-many-stable
  //!
  //! @endrst
  //!
  //! @param inputs A tuple of iterators to the input sequences where num_items elements are read from each. The
  //! iterators' value types must be trivially relocatable.
  //! @param output An iterator to the output sequence where num_items results are written to.
  //! @param num_items The number of elements in each input sequence.
  //! @param transform_op An n-ary function object, where n is the number of input sequences. The input iterators' value
  //! types must be convertible to the parameters of the function object's call operator. The return type of the call
  //! operator must be assignable to the dereferenced output iterator.
  //! @param stream **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  template <typename... RandomAccessIteratorsIn, typename RandomAccessIteratorOut, typename TransformOp>
  CUB_RUNTIME_FUNCTION static cudaError_t TransformStableArgumentAddresses(
    ::cuda::std::tuple<RandomAccessIteratorsIn...> inputs,
    RandomAccessIteratorOut output,
    int num_items,
    TransformOp transform_op,
    cudaStream_t stream = nullptr)
  {
    CUB_DETAIL_NVTX_RANGE_SCOPE("cub::DeviceTransform::TransformStableArgumentAddresses");
    return detail::transform::
      dispatch_t<true, int, ::cuda::std::tuple<RandomAccessIteratorsIn...>, RandomAccessIteratorOut, TransformOp>::
        dispatch(
          ::cuda::std::move(inputs), ::cuda::std::move(output), num_items, ::cuda::std::move(transform_op), stream);
  }

#ifndef DOXYGEN_SHOULD_SKIP_THIS // Do not document
  template <typename... RandomAccessIteratorsIn, typename RandomAccessIteratorOut, typename TransformOp>
  CUB_RUNTIME_FUNCTION static cudaError_t TransformStableArgumentAddresses(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    ::cuda::std::tuple<RandomAccessIteratorsIn...> inputs,
    RandomAccessIteratorOut output,
    int num_items,
    TransformOp transform_op,
    cudaStream_t stream = nullptr)
  {
    if (d_temp_storage == nullptr)
    {
      temp_storage_bytes = 1;
      return cudaSuccess;
    }

    return TransformStableArgumentAddresses(
      ::cuda::std::move(inputs), ::cuda::std::move(output), num_items, ::cuda::std::move(transform_op), stream);
  }
#endif // DOXYGEN_SHOULD_SKIP_THIS

  //! @rst
  //! Transforms one input sequence into one output sequence, by applying a transformation operation on corresponding
  //! input elements and writing the result to the corresponding output element. The objects passed to the call operator
  //! of the transformation operation are guaranteed to reside in the input sequences and are never copied.
  //! @endrst
  //!
  //! @param input An iterator to the input sequence where num_items elements are read from. The iterator's value type
  //! must be trivially relocatable.
  //! @param output An iterator to the output sequence where num_items results are written to.
  //! @param num_items The number of elements in each input sequence.
  //! @param transform_op An n-ary function object, where n is the number of input sequences. The input iterators' value
  //! types must be convertible to the parameters of the function object's call operator. The return type of the call
  //! operator must be assignable to the dereferenced output iterator.
  //! @param stream **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  template <typename RandomAccessIteratorIn, typename RandomAccessIteratorOut, typename TransformOp>
  CUB_RUNTIME_FUNCTION static cudaError_t TransformStableArgumentAddresses(
    RandomAccessIteratorIn input,
    RandomAccessIteratorOut output,
    int num_items,
    TransformOp transform_op,
    cudaStream_t stream = nullptr)
  {
    return TransformStableArgumentAddresses(
      ::cuda::std::make_tuple(::cuda::std::move(input)),
      ::cuda::std::move(output),
      num_items,
      ::cuda::std::move(transform_op),
      stream);
  }

#ifndef DOXYGEN_SHOULD_SKIP_THIS // Do not document
  template <typename RandomAccessIteratorIn, typename RandomAccessIteratorOut, typename TransformOp>
  CUB_RUNTIME_FUNCTION static cudaError_t TransformStableArgumentAddresses(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    RandomAccessIteratorIn input,
    RandomAccessIteratorOut output,
    int num_items,
    TransformOp transform_op,
    cudaStream_t stream = nullptr)
  {
    if (d_temp_storage == nullptr)
    {
      temp_storage_bytes = 1;
      return cudaSuccess;
    }

    return TransformStableArgumentAddresses(
      ::cuda::std::make_tuple(::cuda::std::move(input)),
      ::cuda::std::move(output),
      num_items,
      ::cuda::std::move(transform_op),
      stream);
  }
#endif // DOXYGEN_SHOULD_SKIP_THIS
};

CUB_NAMESPACE_END
