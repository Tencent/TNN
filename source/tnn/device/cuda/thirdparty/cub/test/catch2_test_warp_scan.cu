/******************************************************************************
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

#include <cub/util_macro.cuh>
#include <cub/warp/warp_scan.cuh>

#include <c2h/catch2_test_helper.cuh>

template <int LOGICAL_WARP_THREADS, int TOTAL_WARPS, class T, class ActionT>
__global__ void warp_combine_scan_kernel(T* in, T* inclusive_out, T* exclusive_out, ActionT action)
{
  using warp_scan_t = cub::WarpScan<T, LOGICAL_WARP_THREADS>;
  using storage_t   = typename warp_scan_t::TempStorage;

  __shared__ storage_t storage[TOTAL_WARPS];

  const int tid = cub::RowMajorTid(blockDim.x, blockDim.y, blockDim.z);

  // Get warp index
  int warp_id = tid / LOGICAL_WARP_THREADS;

  T inc_out, exc_out;
  T thread_data = in[tid];

  warp_scan_t scan(storage[warp_id]);

  action(scan, thread_data, inc_out, exc_out);

  inclusive_out[tid] = inc_out;
  exclusive_out[tid] = exc_out;
}

template <int LOGICAL_WARP_THREADS, int TOTAL_WARPS, class T, class ActionT>
void warp_combine_scan(
  c2h::device_vector<T>& in, c2h::device_vector<T>& inclusive_out, c2h::device_vector<T>& exclusive_out, ActionT action)
{
  warp_combine_scan_kernel<LOGICAL_WARP_THREADS, TOTAL_WARPS, T, ActionT><<<1, LOGICAL_WARP_THREADS * TOTAL_WARPS>>>(
    thrust::raw_pointer_cast(in.data()),
    thrust::raw_pointer_cast(inclusive_out.data()),
    thrust::raw_pointer_cast(exclusive_out.data()),
    action);

  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
}

template <int LOGICAL_WARP_THREADS, int TOTAL_WARPS, class T, class ActionT>
__global__ void warp_scan_kernel(T* in, T* out, ActionT action)
{
  using warp_scan_t = cub::WarpScan<T, LOGICAL_WARP_THREADS>;
  using storage_t   = typename warp_scan_t::TempStorage;

  __shared__ storage_t storage[TOTAL_WARPS];

  const int tid = cub::RowMajorTid(blockDim.x, blockDim.y, blockDim.z);

  // Get warp index
  int warp_id = tid / LOGICAL_WARP_THREADS;

  T thread_data = in[tid];

  warp_scan_t scan(storage[warp_id]);

  action(scan, thread_data);

  out[tid] = thread_data;
}

template <int LOGICAL_WARP_THREADS, int TOTAL_WARPS, class T, class ActionT>
void warp_scan(c2h::device_vector<T>& in, c2h::device_vector<T>& out, ActionT action)
{
  warp_scan_kernel<LOGICAL_WARP_THREADS, TOTAL_WARPS, T, ActionT><<<1, LOGICAL_WARP_THREADS * TOTAL_WARPS>>>(
    thrust::raw_pointer_cast(in.data()), thrust::raw_pointer_cast(out.data()), action);

  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
}

enum class scan_mode
{
  exclusive,
  inclusive
};

template <scan_mode Mode>
struct sum_op_t
{
  template <class WarpScanT, class T>
  __device__ void operator()(WarpScanT& scan, T& thread_data) const
  {
    _CCCL_IF_CONSTEXPR (Mode == scan_mode::exclusive)
    {
      scan.ExclusiveSum(thread_data, thread_data);
    }
    else
    {
      scan.InclusiveSum(thread_data, thread_data);
    }
  }
};

template <class T, scan_mode Mode>
struct sum_aggregate_op_t
{
  int m_target_thread_id;
  T* m_d_warp_aggregate;

  template <int LOGICAL_WARP_THREADS>
  __device__ void operator()(cub::WarpScan<T, LOGICAL_WARP_THREADS>& scan, T& thread_data) const
  {
    T warp_aggregate{};

    _CCCL_IF_CONSTEXPR (Mode == scan_mode::exclusive)
    {
      scan.ExclusiveSum(thread_data, thread_data, warp_aggregate);
    }
    else
    {
      scan.InclusiveSum(thread_data, thread_data, warp_aggregate);
    }

    const int tid = cub::RowMajorTid(blockDim.x, blockDim.y, blockDim.z);

    if (tid % LOGICAL_WARP_THREADS == m_target_thread_id)
    {
      m_d_warp_aggregate[tid / LOGICAL_WARP_THREADS] = warp_aggregate;
    }
  }
};

template <scan_mode Mode>
struct min_op_t
{
  template <class T, class WarpScanT>
  __device__ void operator()(WarpScanT& scan, T& thread_data) const
  {
    _CCCL_IF_CONSTEXPR (Mode == scan_mode::exclusive)
    {
      scan.ExclusiveScan(thread_data, thread_data, cub::Min{});
    }
    else
    {
      scan.InclusiveScan(thread_data, thread_data, cub::Min{});
    }
  }
};

template <class T, scan_mode Mode>
struct min_aggregate_op_t
{
  int m_target_thread_id;
  T* m_d_warp_aggregate;

  template <int LOGICAL_WARP_THREADS>
  __device__ void operator()(cub::WarpScan<T, LOGICAL_WARP_THREADS>& scan, T& thread_data) const
  {
    T warp_aggregate{};

    _CCCL_IF_CONSTEXPR (Mode == scan_mode::exclusive)
    {
      scan.ExclusiveScan(thread_data, thread_data, cub::Min{}, warp_aggregate);
    }
    else
    {
      scan.InclusiveScan(thread_data, thread_data, cub::Min{}, warp_aggregate);
    }

    const int tid = cub::RowMajorTid(blockDim.x, blockDim.y, blockDim.z);

    if (tid % LOGICAL_WARP_THREADS == m_target_thread_id)
    {
      m_d_warp_aggregate[tid / LOGICAL_WARP_THREADS] = warp_aggregate;
    }
  }
};

template <class T, scan_mode Mode>
struct min_init_value_op_t
{
  T initial_value;
  template <class WarpScanT>
  __device__ void operator()(WarpScanT& scan, T& thread_data) const
  {
    _CCCL_IF_CONSTEXPR (Mode == scan_mode::exclusive)
    {
      scan.ExclusiveScan(thread_data, thread_data, initial_value, cub::Min{});
    }
    else
    {
      scan.InclusiveScan(thread_data, thread_data, initial_value, cub::Min{});
    }
  }
};

template <class T, scan_mode Mode>
struct min_init_value_aggregate_op_t
{
  int m_target_thread_id;
  T initial_value;
  T* m_d_warp_aggregate;

  template <int LOGICAL_WARP_THREADS>
  __device__ void operator()(cub::WarpScan<T, LOGICAL_WARP_THREADS>& scan, T& thread_data) const
  {
    T warp_aggregate{};

    _CCCL_IF_CONSTEXPR (Mode == scan_mode::exclusive)
    {
      scan.ExclusiveScan(thread_data, thread_data, initial_value, cub::Min{}, warp_aggregate);
    }
    else
    {
      scan.InclusiveScan(thread_data, thread_data, initial_value, cub::Min{}, warp_aggregate);
    }

    const int tid = cub::RowMajorTid(blockDim.x, blockDim.y, blockDim.z);

    if (tid % LOGICAL_WARP_THREADS == m_target_thread_id)
    {
      m_d_warp_aggregate[tid / LOGICAL_WARP_THREADS] = warp_aggregate;
    }
  }
};

struct min_scan_op_t
{
  template <class T, class WarpScanT>
  __device__ void operator()(WarpScanT& scan, T& thread_data, T& inclusive_output, T& exclusive_output) const
  {
    scan.Scan(thread_data, inclusive_output, exclusive_output, cub::Min{});
  }
};

template <class T>
struct min_init_value_scan_op_t
{
  T initial_value;
  template <class WarpScanT>
  __device__ void operator()(WarpScanT& scan, T& thread_data, T& inclusive_output, T& exclusive_output) const
  {
    scan.Scan(thread_data, inclusive_output, exclusive_output, initial_value, cub::Min{});
  }
};

template <class T, class ScanOpT>
c2h::host_vector<T> compute_host_reference(
  scan_mode mode, c2h::host_vector<T>& result, int logical_warp_threads, ScanOpT scan_op, T initial_value = T{})
{
  if (result.empty())
  {
    return c2h::host_vector<T>{};
  }
  // TODO : assert result.size() % logical_warp_threads == 0

  // The accumulator variable is used to calculate warp_aggregate without
  // taking initial_value into consideration in both exclusive and inclusive scan.
  int num_warps = CUB_QUOTIENT_CEILING(static_cast<int>(result.size()), logical_warp_threads);
  c2h::host_vector<T> warp_accumulator(num_warps);
  if (mode == scan_mode::exclusive)
  {
    for (int w = 0; w < num_warps; ++w)
    {
      T* output     = result.data() + w * logical_warp_threads;
      T accumulator = output[0];
      T current     = static_cast<T>(scan_op(initial_value, output[0]));
      output[0]     = initial_value;
      for (int i = 1; i < logical_warp_threads; i++)
      {
        accumulator = static_cast<T>(scan_op(accumulator, output[i]));
        T tmp       = output[i];
        output[i]   = current;
        current     = static_cast<T>(scan_op(current, tmp));
      }
      warp_accumulator[w] = accumulator;
    }
  }
  else
  {
    for (int w = 0; w < num_warps; ++w)
    {
      T* output     = result.data() + w * logical_warp_threads;
      T accumulator = output[0];
      T current     = static_cast<T>(scan_op(initial_value, output[0]));
      output[0]     = current;
      for (int i = 1; i < logical_warp_threads; i++)
      {
        T tmp       = output[i];
        current     = static_cast<T>(scan_op(current, tmp));
        accumulator = static_cast<T>(scan_op(accumulator, tmp));
        output[i]   = current;
      }
      warp_accumulator[w] = accumulator;
    }
  }

  return warp_accumulator;
}

using types                = c2h::type_list<std::uint8_t, std::uint16_t, std::int32_t, std::int64_t>;
using logical_warp_threads = c2h::enum_type_list<int, 32, 16, 9, 2>;
using modes                = c2h::enum_type_list<scan_mode, scan_mode::exclusive, scan_mode::inclusive>;

using vec_types = c2h::type_list<ulonglong4, uchar3, short2>;

using warp_combine_type = int;

template <int logical_warp_threads>
struct total_warps_t
{
private:
  static constexpr int max_warps      = 2;
  static constexpr bool is_arch_warp  = (logical_warp_threads == CUB_WARP_THREADS(0));
  static constexpr bool is_pow_of_two = ((logical_warp_threads & (logical_warp_threads - 1)) == 0);
  static constexpr int total_warps    = (is_arch_warp || is_pow_of_two) ? max_warps : 1;

public:
  static constexpr int value()
  {
    return total_warps;
  }
};

template <class TestType>
struct params_t
{
  using type = typename c2h::get<0, TestType>;

  static constexpr int logical_warp_threads = c2h::get<1, TestType>::value;
  static constexpr scan_mode mode           = c2h::get<2, TestType>::value;
  static constexpr int total_warps          = total_warps_t<logical_warp_threads>::value();
  static constexpr int tile_size            = total_warps * logical_warp_threads;
};

C2H_TEST("Warp scan works with sum", "[scan][warp]", types, logical_warp_threads, modes)
{
  using params = params_t<TestType>;
  using type   = typename params::type;

  c2h::device_vector<type> d_out(params::tile_size);
  c2h::device_vector<type> d_in(params::tile_size);
  c2h::gen(C2H_SEED(10), d_in);

  warp_scan<params::logical_warp_threads, params::total_warps>(d_in, d_out, sum_op_t<params::mode>{});

  c2h::host_vector<type> h_out = d_in;

  compute_host_reference(params::mode, h_out, params::logical_warp_threads, std::plus<type>{});
  REQUIRE_APPROX_EQ(h_out, d_out);
}

C2H_TEST("Warp scan works with vec_types", "[scan][warp]", vec_types, logical_warp_threads, modes)
{
  using params = params_t<TestType>;
  using type   = typename params::type;

  c2h::device_vector<type> d_out(params::tile_size);
  c2h::device_vector<type> d_in(params::tile_size);
  c2h::gen(C2H_SEED(10), d_in);

  warp_scan<params::logical_warp_threads, params::total_warps>(d_in, d_out, sum_op_t<params::mode>{});

  c2h::host_vector<type> h_out = d_in;

  compute_host_reference(params::mode, h_out, params::logical_warp_threads, std::plus<type>{});
  REQUIRE(h_out == d_out);
}

C2H_TEST("Warp scan works with custom types",
         "[scan][warp]",
         c2h::type_list<c2h::custom_type_t<c2h::accumulateable_t, c2h::equal_comparable_t>>,
         logical_warp_threads,
         modes)
{
  using params = params_t<TestType>;
  using type   = typename params::type;

  c2h::device_vector<type> d_out(params::tile_size);
  c2h::device_vector<type> d_in(params::tile_size);
  c2h::gen(C2H_SEED(10), d_in);

  warp_scan<params::logical_warp_threads, params::total_warps>(d_in, d_out, sum_op_t<params::mode>{});

  c2h::host_vector<type> h_out = d_in;

  compute_host_reference(params::mode, h_out, params::logical_warp_threads, std::plus<type>{});
  REQUIRE(h_out == d_out);
}

C2H_TEST("Warp scan returns valid warp aggregate",
         "[scan][warp]",
         c2h::type_list<c2h::custom_type_t<c2h::accumulateable_t, c2h::equal_comparable_t>>,
         logical_warp_threads,
         modes)
{
  using params = params_t<TestType>;
  using type   = typename params::type;

  c2h::device_vector<type> d_warp_aggregates(params::total_warps);
  c2h::device_vector<type> d_out(params::tile_size);
  c2h::device_vector<type> d_in(params::tile_size);
  c2h::gen(C2H_SEED(10), d_in);

  const int target_thread_id = GENERATE_COPY(take(2, random(0, params::logical_warp_threads - 1)));

  warp_scan<params::logical_warp_threads, params::total_warps>(
    d_in,
    d_out,
    sum_aggregate_op_t<type, params::mode>{target_thread_id, thrust::raw_pointer_cast(d_warp_aggregates.data())});

  c2h::host_vector<type> h_out = d_in;

  auto h_warp_aggregates = compute_host_reference(params::mode, h_out, params::logical_warp_threads, std::plus<type>{});
  REQUIRE(h_out == d_out);
  REQUIRE(h_warp_aggregates == d_warp_aggregates);
}

// TODO : Do we need all the types?
C2H_TEST("Warp scan works with custom scan op", "[scan][warp]", types, logical_warp_threads, modes)
{
  using params = params_t<TestType>;
  using type   = typename params::type;

  c2h::device_vector<type> d_out(params::tile_size);
  c2h::device_vector<type> d_in(params::tile_size);
  c2h::gen(C2H_SEED(10), d_in);

  warp_scan<params::logical_warp_threads, params::total_warps>(d_in, d_out, min_op_t<params::mode>{});

  c2h::host_vector<type> h_out = d_in;

  compute_host_reference(
    params::mode,
    h_out,
    params::logical_warp_threads,
    [](type l, type r) {
      return std::min(l, r);
    },
    std::numeric_limits<type>::max());

  // From the documentation -
  // Computes an exclusive prefix scan using the specified binary scan functor
  // across the calling warp. Because no initial value is supplied, the output
  // computed for warp-lane0 is undefined.

  // When comparing device output, the corresponding undefined data points need
  // to be fixed

  _CCCL_IF_CONSTEXPR (params::mode == scan_mode::exclusive)
  {
    for (size_t i = 0; i < h_out.size(); i += params::logical_warp_threads)
    {
      d_out[i] = h_out[i];
    }
  }
  REQUIRE_APPROX_EQ(h_out, d_out);
}

C2H_TEST("Warp custom op scan returns valid warp aggregate", "[scan][warp]", types, logical_warp_threads, modes)
{
  using params = params_t<TestType>;
  using type   = typename params::type;

  c2h::device_vector<type> d_warp_aggregates(params::total_warps);
  c2h::device_vector<type> d_out(params::tile_size);
  c2h::device_vector<type> d_in(params::tile_size);
  c2h::gen(C2H_SEED(10), d_in);

  const int target_thread_id = GENERATE_COPY(take(2, random(0, params::logical_warp_threads - 1)));

  warp_scan<params::logical_warp_threads, params::total_warps>(
    d_in,
    d_out,
    min_aggregate_op_t<type, params::mode>{target_thread_id, thrust::raw_pointer_cast(d_warp_aggregates.data())});

  c2h::host_vector<type> h_out = d_in;

  auto h_warp_aggregates = compute_host_reference(
    params::mode,
    h_out,
    params::logical_warp_threads,
    [](type l, type r) {
      return std::min(l, r);
    },
    std::numeric_limits<type>::max());

  // From the documentation -
  // Computes an exclusive prefix scan using the specified binary scan functor
  // across the calling warp. Because no initial value is supplied, the output
  // computed for warp-lane0 is undefined.

  // When comparing device output, the corresponding undefined data points need
  // to be fixed

  _CCCL_IF_CONSTEXPR (params::mode == scan_mode::exclusive)
  {
    for (size_t i = 0; i < h_out.size(); i += params::logical_warp_threads)
    {
      d_out[i] = h_out[i];
    }
  }
  REQUIRE(h_out == d_out);
  REQUIRE(h_warp_aggregates == d_warp_aggregates);
}

C2H_TEST("Warp custom op scan works with initial value", "[scan][warp]", types, logical_warp_threads, modes)
{
  using params = params_t<TestType>;
  using type   = typename params::type;

  c2h::device_vector<type> d_out(params::tile_size);
  c2h::device_vector<type> d_in(params::tile_size);
  c2h::gen(C2H_SEED(10), d_in);

  const type initial_value = static_cast<type>(GENERATE_COPY(take(2, random(0, params::tile_size))));

  warp_scan<params::logical_warp_threads, params::total_warps>(
    d_in, d_out, min_init_value_op_t<type, params::mode>{initial_value});

  c2h::host_vector<type> h_out = d_in;

  compute_host_reference(
    params::mode,
    h_out,
    params::logical_warp_threads,
    [](type l, type r) {
      return std::min(l, r);
    },
    initial_value);

  REQUIRE_APPROX_EQ(h_out, d_out);
}

C2H_TEST("Warp custom op scan with initial value returns valid warp aggregate",
         "[scan][warp]",
         types,
         logical_warp_threads,
         modes)
{
  using params = params_t<TestType>;
  using type   = typename params::type;

  c2h::device_vector<type> d_warp_aggregates(params::total_warps);
  c2h::device_vector<type> d_out(params::tile_size);
  c2h::device_vector<type> d_in(params::tile_size);
  c2h::gen(C2H_SEED(10), d_in);

  const int target_thread_id = GENERATE_COPY(take(2, random(0, params::logical_warp_threads - 1)));
  const type initial_value   = static_cast<type>(GENERATE_COPY(take(2, random(0, params::tile_size))));

  warp_scan<params::logical_warp_threads, params::total_warps>(
    d_in,
    d_out,
    min_init_value_aggregate_op_t<type, params::mode>{
      target_thread_id, initial_value, thrust::raw_pointer_cast(d_warp_aggregates.data())});

  c2h::host_vector<type> h_out = d_in;

  auto h_warp_aggregates = compute_host_reference(
    params::mode,
    h_out,
    params::logical_warp_threads,
    [](type l, type r) {
      return std::min(l, r);
    },
    initial_value);

  REQUIRE(h_out == d_out);
  REQUIRE(h_warp_aggregates == d_warp_aggregates);
}

C2H_TEST("Warp combination scan works with custom scan op", "[scan][warp]", logical_warp_threads)
{
  constexpr int logical_warp_threads = c2h::get<0, TestType>();
  constexpr int total_warps          = total_warps_t<logical_warp_threads>::value();
  using type                         = int;

  c2h::device_vector<type> d_inclusive_out(total_warps * logical_warp_threads);
  c2h::device_vector<type> d_exclusive_out(total_warps * logical_warp_threads);
  c2h::device_vector<type> d_in(total_warps * logical_warp_threads);
  c2h::gen(C2H_SEED(10), d_in);

  warp_combine_scan<logical_warp_threads, total_warps>(d_in, d_inclusive_out, d_exclusive_out, min_scan_op_t{});

  c2h::host_vector<type> h_exclusive_out = d_in;
  c2h::host_vector<type> h_inclusive_out = d_in;

  compute_host_reference(
    scan_mode::exclusive,
    h_exclusive_out,
    logical_warp_threads,
    [](type l, type r) {
      return std::min(l, r);
    },
    std::numeric_limits<type>::max());

  compute_host_reference(
    scan_mode::inclusive,
    h_inclusive_out,
    logical_warp_threads,
    [](type l, type r) {
      return std::min(l, r);
    },
    std::numeric_limits<type>::max());

  // According to WarpScan::Scan documentation -
  // Because no initial value is supplied, the exclusive_output computed for warp-lane0 is
  // undefined.

  // When comparing device output, the corresponding undefined data points need
  // to be fixed

  for (size_t i = 0; i < h_exclusive_out.size(); i += logical_warp_threads)
  {
    d_exclusive_out[i] = h_exclusive_out[i];
  }

  REQUIRE(h_inclusive_out == d_inclusive_out);
  REQUIRE(h_exclusive_out == d_exclusive_out);
}

C2H_TEST("Warp combination custom scan works with initial value", "[scan][warp]", logical_warp_threads)
{
  constexpr int logical_warp_threads = c2h::get<0, TestType>();
  constexpr int total_warps          = total_warps_t<logical_warp_threads>::value();
  using type                         = int;

  c2h::device_vector<type> d_inclusive_out(total_warps * logical_warp_threads);
  c2h::device_vector<type> d_exclusive_out(total_warps * logical_warp_threads);
  c2h::device_vector<type> d_in(total_warps * logical_warp_threads);
  c2h::gen(C2H_SEED(10), d_in);

  const type initial_value = GENERATE_COPY(take(2, random(0, total_warps * logical_warp_threads)));

  warp_combine_scan<logical_warp_threads, total_warps>(
    d_in, d_inclusive_out, d_exclusive_out, min_init_value_scan_op_t<type>{initial_value});

  c2h::host_vector<type> h_exclusive_out = d_in;
  c2h::host_vector<type> h_inclusive_out = d_in;

  compute_host_reference(
    scan_mode::exclusive,
    h_exclusive_out,
    logical_warp_threads,
    [](type l, type r) {
      return std::min(l, r);
    },
    initial_value);

  compute_host_reference(
    scan_mode::inclusive,
    h_inclusive_out,
    logical_warp_threads,
    [](type l, type r) {
      return std::min(l, r);
    },
    initial_value);

  REQUIRE(h_inclusive_out == d_inclusive_out);
  REQUIRE(h_exclusive_out == d_exclusive_out);
}
