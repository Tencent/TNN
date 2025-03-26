/******************************************************************************
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <cub/device/device_spmv.cuh>
#include <cub/util_debug.cuh>

#include <thrust/distance.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/mismatch.h>
#include <thrust/scan.h>

#include <cuda/std/type_traits>

#include <iostream>
#include <type_traits>
#include <typeinfo>

#include "test_util.h"
#include <c2h/device_policy.cuh>
#include <c2h/vector.cuh>

bool g_verbose = false;

//==============================================================================
// Casts char types to int for numeric printing
template <typename T>
T print_cast(T val)
{
  return val;
}

int print_cast(char val)
{
  return static_cast<int>(val);
}

int print_cast(signed char val)
{
  return static_cast<int>(val);
}

int print_cast(unsigned char val)
{
  return static_cast<int>(val);
}

//==============================================================================
// Print a vector to out
template <typename VectorT>
void print_vector(std::ostream& out, const VectorT& vec)
{
  bool first = true;
  for (const auto& val : vec)
  {
    if (!first)
    {
      out << ", ";
    }
    first = false;
    out << print_cast(val);
  }
}

//==============================================================================
// Simple CSR matrix implementation.
// HostStorage controls whether data is stored on the host or device.
// Use the host_csr_matrix and device_csr_matrix aliases for code clarity.
template <typename ValueT, bool HostStorage>
struct csr_matrix
{
  csr_matrix(int num_rows, int num_cols)
      : m_row_offsets(static_cast<size_t>(num_rows + 1), 0)
      , m_num_rows(num_rows)
      , m_num_columns(num_cols)
  {}

  // host/device conversion constructor
  explicit csr_matrix(const csr_matrix<ValueT, !HostStorage>& other)
      : m_values(other.m_values)
      , m_row_offsets(other.m_row_offsets)
      , m_column_indices(other.m_column_indices)
      , m_num_rows(other.m_num_rows)
      , m_num_columns(other.m_num_columns)
      , m_num_nonzeros(other.m_num_nonzeros)
  {}

  // Note that this must append to the values array. Finish filling each row
  // before adding to the next, and each row's columns must be added in order.
  // Must call `finalize` once all items are added.
  void append_value(int row, int col, ValueT value)
  {
    ++m_num_nonzeros;
    ++m_row_offsets[row];
    m_column_indices.push_back(col);
    m_values.push_back(std::move(value));
  }

  void finalize()
  {
    _CCCL_IF_CONSTEXPR (HostStorage)
    {
      thrust::exclusive_scan(thrust::host, m_row_offsets.cbegin(), m_row_offsets.cend(), m_row_offsets.begin());
    }
    else
    {
      thrust::exclusive_scan(c2h::device_policy, m_row_offsets.cbegin(), m_row_offsets.cend(), m_row_offsets.begin());
    }
    AssertEquals(m_row_offsets.back(), m_num_nonzeros);
  }

  const ValueT* get_values() const
  {
    return thrust::raw_pointer_cast(m_values.data());
  }

  const int* get_row_offsets() const
  {
    return thrust::raw_pointer_cast(m_row_offsets.data());
  }

  int get_row_offset(int row) const
  {
    return m_row_offsets[row];
  }

  int get_row_num_nonzero(int row) const
  {
    return m_row_offsets[row + 1] - m_row_offsets[row];
  }

  const int* get_column_indices() const
  {
    return thrust::raw_pointer_cast(m_column_indices.data());
  }

  int get_num_rows() const
  {
    return m_num_rows;
  }

  int get_num_columns() const
  {
    return m_num_columns;
  }

  int get_num_nonzeros() const
  {
    return m_num_nonzeros;
  }

  void print_internals(std::ostream& out) const
  {
    out << (HostStorage ? "host" : "device") << "_csr_matrix"
        << "(" << m_num_rows << ", " << m_num_columns << ")\n"
        << " - num_elems:   " << (m_num_rows * m_num_columns) << "\n"
        << " - num_nonzero: " << m_num_nonzeros << "\n"
        << " - row_offsets:\n     [";
    print_vector(out, m_row_offsets);
    out << "]\n"
        << " - column_indices:\n     [";
    print_vector(out, m_column_indices);
    out << "]\n"
        << " - values:\n     [";
    print_vector(out, m_values);
    out << "]\n";
  }

  void print_summary(std::ostream& out) const
  {
    const int num_elems = m_num_rows * m_num_columns;
    const float fill_ratio =
      num_elems == 0 ? 0.f : (static_cast<float>(m_num_nonzeros) / static_cast<float>(num_elems));

    out << m_num_rows << "x" << m_num_columns << ", " << m_num_nonzeros << "/" << num_elems << " (" << fill_ratio
        << ")\n";
  }

  friend class csr_matrix<ValueT, !HostStorage>;

private:
  template <typename VecValueT>
  using vector_t = ::cuda::std::_If<HostStorage, c2h::host_vector<VecValueT>, c2h::device_vector<VecValueT>>;

  vector_t<ValueT> m_values;
  vector_t<int> m_row_offsets;
  vector_t<int> m_column_indices;

  int m_num_rows{0};
  int m_num_columns{0};
  int m_num_nonzeros{0};
};

//==============================================================================
// Convenience aliases for host/device csr_matrix types.
template <typename ValueT>
using host_csr_matrix = csr_matrix<ValueT, true>;

template <typename ValueT>
using device_csr_matrix = csr_matrix<ValueT, false>;

//==============================================================================
// Compare two floats within a tolerance.
// This mimics the approach used by Thrust's ASSERT_ALMOST_EQUAL checks.
template <typename ValueT>
struct fp_almost_equal_functor
{
  __host__ __device__ bool operator()(ValueT v1, ValueT v2) const
  {
    constexpr double r_tol = 1e-3;
    constexpr double a_tol = 1e-2;
    const double limit     = r_tol * (std::fabs(v1) + std::fabs(v2)) + a_tol;
    return std::fabs(v1 - v2) <= limit;
  }
};

//==============================================================================
// Compare the reference and cub output vectors.
// Use fuzzy check for floating point values.
template <typename ValueT>
bool compare_results(
  std::true_type /* is_fp */, const c2h::host_vector<ValueT>& h_vec1, const c2h::device_vector<ValueT>& d_vec2)
{
  c2h::device_vector<ValueT> d_vec1(h_vec1);
  auto err = thrust::mismatch(
    c2h::device_policy, d_vec1.cbegin(), d_vec1.cend(), d_vec2.cbegin(), fp_almost_equal_functor<ValueT>{});
  if (err.first == d_vec1.cend() || err.second == d_vec2.cend())
  {
    return true;
  }
  else
  {
    c2h::host_vector<ValueT> h_vec2(d_vec2);
    const auto idx = thrust::distance(d_vec1.cbegin(), err.first);
    std::cerr << "Mismatch at position " << idx << ": " << print_cast(ValueT{h_vec1[idx]}) << " vs "
              << print_cast(ValueT{h_vec2[idx]}) << std::endl;
    return false;
  }
};

template <typename ValueT>
bool compare_results(
  std::false_type /* is_fp */, const c2h::host_vector<ValueT>& h_vec1, const c2h::device_vector<ValueT>& d_vec2)
{
  c2h::device_vector<ValueT> d_vec1(h_vec1);
  auto err = thrust::mismatch(c2h::device_policy, d_vec1.cbegin(), d_vec1.cend(), d_vec2.cbegin());
  if (err.first == d_vec1.cend() || err.second == d_vec2.cend())
  {
    return true;
  }
  else
  {
    c2h::host_vector<ValueT> h_vec2(d_vec2);
    const auto idx = thrust::distance(d_vec1.cbegin(), err.first);
    std::cerr << "Mismatch at position " << idx << ": " << print_cast(ValueT{h_vec1[idx]}) << " vs "
              << print_cast(ValueT{h_vec2[idx]}) << std::endl;
    return false;
  }
}

//==============================================================================
// Generate a random host_csr_matrix<ValueT> with the specified dimensions.
// target_fill_ratio is the target fraction of non-zero elements (may be more
// or less in the output).
template <typename ValueT>
host_csr_matrix<ValueT> make_random_csr_matrix(int num_rows, int num_cols, float target_fill_ratio)
{
  host_csr_matrix<ValueT> mat{num_rows, num_cols};

  for (int row = 0; row < num_rows; ++row)
  {
    for (int col = 0; col < num_cols; ++col)
    {
      const bool is_non_zero = RandomValue<float>(1.f) < target_fill_ratio;
      if (!is_non_zero)
      {
        continue;
      }

      if (std::is_floating_point<ValueT>::value)
      {
        // Keep fp numbers somewhat small, from -50 -> 50; otherwise we run
        // into issues with nans/infs
        ValueT value = (RandomValue(static_cast<ValueT>(100)) - static_cast<ValueT>(50));
        mat.append_value(row, col, value);
      }
      else
      {
        ValueT value{};
        InitValue(RANDOM, value);
        mat.append_value(row, col, value);
      }
    }
  }

  mat.finalize();

  const int num_elements        = num_rows * num_cols;
  const float actual_fill_ratio = static_cast<float>(mat.get_num_nonzeros()) / static_cast<float>(num_elements);

  if (g_verbose)
  {
    printf(
      "Created host_csr_matrix<%s>(%d, %d)\n"
      " - NumElements: %d\n"
      " - NumNonZero:  %d\n"
      " - Target fill: %0.2f%%\n"
      " - Actual fill: %0.2f%%\n",
      typeid(ValueT).name(),
      num_rows,
      num_cols,
      num_elements,
      mat.get_num_nonzeros(),
      target_fill_ratio,
      actual_fill_ratio);
  }

  return mat;
}

//==============================================================================
// Fill a vector with random values.
template <typename ValueT>
c2h::host_vector<ValueT> make_random_vector(int len)
{
  c2h::host_vector<ValueT> vec(len);
  for (auto& val : vec)
  {
    if (std::is_floating_point<ValueT>::value)
    { // Keep fp numbers somewhat small; otherwise we run into issues with
      // nans/infs
      val = RandomValue(static_cast<ValueT>(100)) - static_cast<ValueT>(50);
    }
    else
    {
      InitValue(RANDOM, val);
    }
  }
  return vec;
}

//==============================================================================
// Serial y = Ax computation
template <typename ValueT>
void compute_reference_solution(
  const host_csr_matrix<ValueT>& a, const c2h::host_vector<ValueT>& x, c2h::host_vector<ValueT>& y)
{
  if (a.get_num_rows() == 0 || a.get_num_columns() == 0)
  {
    return;
  }

  for (int row = 0; row < a.get_num_rows(); ++row)
  {
    const int row_offset = a.get_row_offset(row);
    const int row_length = a.get_row_num_nonzero(row);
    const int* cols      = a.get_column_indices() + row_offset;
    const int* cols_end  = cols + row_length;
    const ValueT* values = a.get_values() + row_offset;

    ValueT accum{};
    while (cols < cols_end)
    {
      accum += (*values++) * x[*cols++];
    }
    y[row] = accum;
  }
}

//==============================================================================
// cub::DeviceSpmv::CsrMV y = Ax computation
template <typename ValueT>
void compute_cub_solution(
  const device_csr_matrix<ValueT>& a, const c2h::device_vector<ValueT>& x, c2h::device_vector<ValueT>& y)
{
  c2h::device_vector<char> temp_storage;
  std::size_t temp_storage_bytes{};
  auto err = cub::DeviceSpmv::CsrMV(
    nullptr,
    temp_storage_bytes,
    a.get_values(),
    a.get_row_offsets(),
    a.get_column_indices(),
    thrust::raw_pointer_cast(x.data()),
    thrust::raw_pointer_cast(y.data()),
    a.get_num_rows(),
    a.get_num_columns(),
    a.get_num_nonzeros());
  CubDebugExit(err);

  temp_storage.resize(temp_storage_bytes);

  err = cub::DeviceSpmv::CsrMV(
    thrust::raw_pointer_cast(temp_storage.data()),
    temp_storage_bytes,
    a.get_values(),
    a.get_row_offsets(),
    a.get_column_indices(),
    thrust::raw_pointer_cast(x.data()),
    thrust::raw_pointer_cast(y.data()),
    a.get_num_rows(),
    a.get_num_columns(),
    a.get_num_nonzeros());
  CubDebugExit(err);
}

//==============================================================================
// Compute y = Ax twice, one reference and one cub::DeviceSpmv, and compare the
// results.
template <typename ValueT>
void test_spmv(const host_csr_matrix<ValueT>& h_a, const c2h::host_vector<ValueT>& h_x)
{
  if (g_verbose)
  {
    std::cout << "Testing cub::DeviceSpmv on inputs:\n";
    h_a.print_internals(std::cout);
    std::cout << "x vector:\n  [";
    print_vector(std::cout, h_x);
    std::cout << "]" << std::endl;
  }
  else
  {
    h_a.print_summary(std::cout);
  }

  const device_csr_matrix<ValueT> d_a(h_a);
  const c2h::device_vector<ValueT> d_x(h_x);

  c2h::host_vector<ValueT> h_y(h_a.get_num_rows());
  c2h::device_vector<ValueT> d_y(d_a.get_num_rows());

  compute_reference_solution(h_a, h_x, h_y);
  compute_cub_solution(d_a, d_x, d_y);

  if (g_verbose)
  {
    std::cout << "reference output:\n  [";
    print_vector(std::cout, h_y);
    std::cout << "]\n";
    c2h::host_vector<ValueT> tmp_y(d_y);
    std::cout << "cub::DeviceSpmv output:\n  [";
    print_vector(std::cout, tmp_y);
    std::cout << "]" << std::endl;
  }

  constexpr auto is_fp = std::is_floating_point<ValueT>{};
  AssertTrue(compare_results(is_fp, h_y, d_y));
}

//==============================================================================
// Test example from cub::DeviceSpmv documentation
template <typename ValueT>
void test_doc_example()
{
  std::cout << "\n\ntest_doc_example<" << typeid(ValueT).name() << ">()" << std::endl;

  host_csr_matrix<ValueT> h_a(9, 9);
  h_a.append_value(0, 1, ValueT{1});
  h_a.append_value(0, 3, ValueT{1});
  h_a.append_value(1, 0, ValueT{1});
  h_a.append_value(1, 2, ValueT{1});
  h_a.append_value(1, 4, ValueT{1});
  h_a.append_value(2, 1, ValueT{1});
  h_a.append_value(2, 5, ValueT{1});
  h_a.append_value(3, 0, ValueT{1});
  h_a.append_value(3, 4, ValueT{1});
  h_a.append_value(3, 6, ValueT{1});
  h_a.append_value(4, 1, ValueT{1});
  h_a.append_value(4, 3, ValueT{1});
  h_a.append_value(4, 5, ValueT{1});
  h_a.append_value(4, 7, ValueT{1});
  h_a.append_value(5, 2, ValueT{1});
  h_a.append_value(5, 4, ValueT{1});
  h_a.append_value(5, 8, ValueT{1});
  h_a.append_value(6, 3, ValueT{1});
  h_a.append_value(6, 7, ValueT{1});
  h_a.append_value(7, 4, ValueT{1});
  h_a.append_value(7, 6, ValueT{1});
  h_a.append_value(7, 8, ValueT{1});
  h_a.append_value(8, 5, ValueT{1});
  h_a.append_value(8, 7, ValueT{1});
  h_a.finalize();

  c2h::host_vector<ValueT> h_x(9, ValueT{1});

  test_spmv(h_a, h_x);
}

//==============================================================================
// Generate and test a random SpMV operation with the given parameters.
template <typename ValueT>
void test_random(int rows, int cols, float target_fill_ratio)
{
  std::cout << "\n\ntest_random<" << typeid(ValueT).name() << ">(" << rows << ", " << cols << ", " << target_fill_ratio
            << ")" << std::endl;

  host_csr_matrix<ValueT> h_a  = make_random_csr_matrix<ValueT>(rows, cols, target_fill_ratio);
  c2h::host_vector<ValueT> h_x = make_random_vector<ValueT>(cols);

  test_spmv(h_a, h_x);
}

//==============================================================================
// Dispatch many random SpMV tests over a variety of parameters.
template <typename ValueT>
void test_random()
{
  test_random<ValueT>(0, 0, 1.f);
  test_random<ValueT>(0, 1, 1.f);
  test_random<ValueT>(1, 0, 1.f);

  constexpr int dim_min = 1;
  constexpr int dim_max = 10000;

  constexpr int max_num_elems = 100000;

  constexpr float ratio_min  = 0.f;
  constexpr float ratio_max  = 1.1f; // a lil over to account for fp errors
  constexpr float ratio_step = 0.3334f;

  for (int rows = dim_min; rows < dim_max; rows <<= 1)
  {
    for (int cols = dim_min; cols < dim_max; cols <<= 1)
    {
      if (rows * cols >= max_num_elems)
      {
        continue;
      }

      for (float ratio = ratio_min; ratio < ratio_max; ratio += ratio_step)
      {
        test_random<ValueT>(rows, cols, ratio);
        // Test nearby non-power-of-two dims:
        test_random<ValueT>(rows + 97, cols + 83, ratio);
      }
    }
  }
}

//==============================================================================
// Dispatch many SpMV tests for a given ValueT.
template <typename ValueT>
void test_type()
{
  test_doc_example<ValueT>();
  test_random<ValueT>();
}

//==============================================================================
// Dispatch many SpMV tests over a variety of types.
void test_types()
{
  test_type<float>();
  test_type<double>();
  test_type<signed char>();
  test_type<int>();
  test_type<long long>();
}

int main(int argc, char** argv)
{
  // Initialize command line
  CommandLineArgs args(argc, argv);
  g_verbose = args.CheckCmdLineFlag("v");

  // Print usage
  if (args.CheckCmdLineFlag("help"))
  {
    printf("%s "
           "[--device=<device-id>] "
           "[--v] verbose"
           "\n",
           argv[0]);
    exit(0);
  }

  CubDebugExit(args.DeviceInit());

  test_types();
}
