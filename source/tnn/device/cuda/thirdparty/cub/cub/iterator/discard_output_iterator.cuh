/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
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
 * @file
 * Random-access iterator types
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

#include <iostream>
#include <iterator>

#if (THRUST_VERSION >= 100700)
// This iterator is compatible with Thrust API 1.7 and newer
#  include <thrust/iterator/iterator_facade.h>
#  include <thrust/iterator/iterator_traits.h>
#endif // THRUST_VERSION

CUB_NAMESPACE_BEGIN

/**
 * @brief A discard iterator
 */
template <typename OffsetT = ptrdiff_t>
class DiscardOutputIterator
{
public:
  // Required iterator traits

  /// My own type
  using self_type = DiscardOutputIterator;

  /// Type to express the result of subtracting one iterator from another
  using difference_type = OffsetT;

  /// The type of the element the iterator can point to
  using value_type = void;

  /// The type of a pointer to an element the iterator can point to
  using pointer = void;

  /// The type of a reference to an element the iterator can point to
  using reference = void;

#if (THRUST_VERSION >= 100700)
  // Use Thrust's iterator categories so we can use these iterators in Thrust 1.7 (or newer) methods

  /// The iterator category
  using iterator_category = typename THRUST_NS_QUALIFIER::detail::iterator_facade_category<
    THRUST_NS_QUALIFIER::any_system_tag,
    THRUST_NS_QUALIFIER::random_access_traversal_tag,
    value_type,
    reference>::type;
#else
  /// The iterator category
  using iterator_category = std::random_access_iterator_tag;
#endif // THRUST_VERSION

private:
  OffsetT offset;

#if defined(_WIN32) || !defined(_WIN64)
  // Workaround for win32 parameter-passing bug (ulonglong2 argmin DeviceReduce)
  OffsetT pad[CUB_MAX(1, (16 / sizeof(OffsetT) - 1))] = {};
#endif

public:
  /**
   * @param offset
   *   Base offset
   */
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE DiscardOutputIterator(OffsetT offset = 0)
      : offset(offset)
  {}

  /// Postfix increment
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE self_type operator++(int)
  {
    self_type retval = *this;
    offset++;
    return retval;
  }

  /// Prefix increment
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE self_type operator++()
  {
    offset++;
    return *this;
  }

  /// Indirection
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE self_type& operator*()
  {
    // return self reference, which can be assigned to anything
    return *this;
  }

  /// Addition
  template <typename Distance>
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE self_type operator+(Distance n) const
  {
    self_type retval(offset + n);
    return retval;
  }

  /// Addition assignment
  template <typename Distance>
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE self_type& operator+=(Distance n)
  {
    offset += n;
    return *this;
  }

  /// Subtraction
  template <typename Distance>
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE self_type operator-(Distance n) const
  {
    self_type retval(offset - n);
    return retval;
  }

  /// Subtraction assignment
  template <typename Distance>
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE self_type& operator-=(Distance n)
  {
    offset -= n;
    return *this;
  }

  /// Distance
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE difference_type operator-(self_type other) const
  {
    return offset - other.offset;
  }

  /// Array subscript
  template <typename Distance>
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE self_type& operator[](Distance n)
  {
    // return self reference, which can be assigned to anything
    return *this;
  }

  /// Structure dereference
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE pointer operator->()
  {
    return;
  }

  /// Assignment to anything else (no-op)
  template <typename T>
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE void operator=(T const&)
  {}

  /// Cast to void* operator
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE operator void*() const
  {
    return nullptr;
  }

  /// Equal to
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE bool operator==(const self_type& rhs)
  {
    return (offset == rhs.offset);
  }

  /// Not equal to
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE bool operator!=(const self_type& rhs)
  {
    return (offset != rhs.offset);
  }

  /// ostream operator
  friend std::ostream& operator<<(std::ostream& os, const self_type& itr)
  {
    os << "[" << itr.offset << "]";
    return os;
  }
};

CUB_NAMESPACE_END
