/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
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

#include <cub/iterator/arg_index_input_iterator.cuh>
#include <cub/iterator/cache_modified_input_iterator.cuh>
#include <cub/iterator/constant_input_iterator.cuh>
#include <cub/iterator/counting_input_iterator.cuh>
#include <cub/iterator/tex_obj_input_iterator.cuh>
#include <cub/iterator/transform_input_iterator.cuh>
#include <cub/util_allocator.cuh>
#include <cub/util_type.cuh>

#include <cuda/std/__cccl/dialect.h>

#include <cstdint>

#include <c2h/catch2_test_helper.cuh>

using scalar_types = c2h::type_list<std::int8_t, std::int16_t, std::int32_t, std::int64_t, float, double>;

using types = ::cuda::std::__type_push_back<
  scalar_types,
  char2,
  short2,
  int2,
  long2,
  longlong2,
  float2,
  double2,
  char3,
  short3,
  int3,
  long3,
  longlong3,
  float3,
  double3,
  char4,
  short4,
  int4,
  long4,
  longlong4,
  float4,
  double4,
  c2h::custom_type_t<c2h::equal_comparable_t, c2h::accumulateable_t>>;

template <typename InputIteratorT, typename T>
__global__ void test_iterator_kernel(InputIteratorT d_in, T* d_out, InputIteratorT* d_itrs)
{
  d_out[0] = *d_in; // Value at offset 0
  d_out[1] = d_in[100]; // Value at offset 100
  d_out[2] = *(d_in + 1000); // Value at offset 1000
  d_out[3] = *(d_in + 10000); // Value at offset 10000

  d_in++;
  d_out[4] = d_in[0]; // Value at offset 1

  d_in += 20;
  d_out[5]  = d_in[0]; // Value at offset 21
  d_itrs[0] = d_in; // Iterator at offset 21

  d_in -= 10;
  d_out[6] = d_in[0]; // Value at offset 11;

  d_in -= 11;
  d_out[7]  = d_in[0]; // Value at offset 0
  d_itrs[1] = d_in; // Iterator at offset 0
}

template <typename InputIteratorT, typename T>
void test_iterator(InputIteratorT d_in, const c2h::host_vector<T>& h_reference)
{
  c2h::device_vector<T> d_out(h_reference.size());
  c2h::device_vector<InputIteratorT> d_itrs(2, d_in); // TODO(bgruber): using a raw allocation halves the compile time
                                                      // (nvcc 12.5), because we instantiate a lot of device_vectors

  test_iterator_kernel<<<1, 1>>>(d_in, thrust::raw_pointer_cast(d_out.data()), thrust::raw_pointer_cast(d_itrs.data()));
  CubDebugExit(cudaPeekAtLastError());
  CubDebugExit(cudaDeviceSynchronize());

  c2h::host_vector<InputIteratorT> h_itrs = d_itrs;
  CHECK(h_reference == c2h::host_vector<T>(d_out)); // comparing host_vectors compiles a lot faster than mixed vectors
  CHECK(d_in + 21 == h_itrs[0]);
  CHECK(d_in == h_itrs[1]);
}

C2H_TEST("Test constant iterator", "[iterator]", scalar_types)
{
  using T                = c2h::get<0, TestType>;
  const T base           = static_cast<T>(GENERATE(0, 99));
  const auto h_reference = c2h::host_vector<T>{base, base, base, base, base, base, base, base};
  test_iterator(cub::ConstantInputIterator<T>(base), h_reference);
}

C2H_TEST("Test counting iterator", "[iterator]", scalar_types)
{
  using T                = c2h::get<0, TestType>;
  const T base           = static_cast<T>(GENERATE(0, 99));
  const auto h_reference = c2h::host_vector<T>{
    static_cast<T>(base + 0),
    static_cast<T>(base + 100),
    static_cast<T>(base + 1000),
    static_cast<T>(base + 10000),
    static_cast<T>(base + 1),
    static_cast<T>(base + 21),
    static_cast<T>(base + 11),
    static_cast<T>(base + 0)};
  test_iterator(cub::CountingInputIterator<T>(base), h_reference);
}

using cache_modifiers =
  c2h::enum_type_list<cub::CacheLoadModifier,
                      cub::LOAD_DEFAULT,
                      cub::LOAD_CA,
                      cub::LOAD_CG,
                      cub::LOAD_CS,
                      cub::LOAD_CV,
                      cub::LOAD_LDG,
                      cub::LOAD_VOLATILE>;

C2H_TEST("Test cache modified iterator", "[iterator]", types, cache_modifiers)
{
  using T                       = c2h::get<0, TestType>;
  constexpr auto cache_modifier = c2h::get<1, TestType>::value;
  constexpr int TEST_VALUES     = 11000;

  c2h::device_vector<T> d_data(TEST_VALUES);
  c2h::gen(C2H_SEED(1), d_data);
  c2h::host_vector<T> h_data(d_data);

  const auto h_reference = c2h::host_vector<T>{
    h_data[0], h_data[100], h_data[1000], h_data[10000], h_data[1], h_data[21], h_data[11], h_data[0]};
  test_iterator(
    cub::CacheModifiedInputIterator<cache_modifier, T>(const_cast<const T*>(thrust::raw_pointer_cast(d_data.data()))),
    h_reference);
}

template <typename T>
struct transform_op_t
{
  _CCCL_HOST_DEVICE T operator()(T input) const
  {
    return input + input;
  }
};

C2H_TEST("Test transform iterator", "[iterator]", types)
{
  using T                   = c2h::get<0, TestType>;
  constexpr int TEST_VALUES = 11000;

  c2h::device_vector<T> d_data(TEST_VALUES);
  c2h::gen(C2H_SEED(1), d_data);
  c2h::host_vector<T> h_data(d_data);

  transform_op_t<T> op;
  const auto h_reference = c2h::host_vector<T>{
    op(h_data[0]),
    op(h_data[100]),
    op(h_data[1000]),
    op(h_data[10000]),
    op(h_data[1]),
    op(h_data[21]),
    op(h_data[11]),
    op(h_data[0])};
  test_iterator(cub::TransformInputIterator<T, transform_op_t<T>, const T*>(
                  const_cast<const T*>(const_cast<const T*>(thrust::raw_pointer_cast(d_data.data()))), op),
                h_reference);
}

C2H_TEST("Test tex-obj texture iterator", "[iterator]", types)
{
  using T                            = c2h::get<0, TestType>;
  constexpr unsigned int TEST_VALUES = 11000;

  c2h::device_vector<T> d_data(TEST_VALUES);
  c2h::gen(C2H_SEED(1), d_data);
  c2h::host_vector<T> h_data(d_data);

  const auto h_reference = c2h::host_vector<T>{
    h_data[0], h_data[100], h_data[1000], h_data[10000], h_data[1], h_data[21], h_data[11], h_data[0]};
  cub::TexObjInputIterator<T> d_obj_itr;
  CubDebugExit(
    d_obj_itr.BindTexture(const_cast<const T*>(thrust::raw_pointer_cast(d_data.data())), sizeof(T) * TEST_VALUES));
  test_iterator(d_obj_itr, h_reference);
}

C2H_TEST("Test texture transform iterator", "[iterator]", types)
{
  using T                   = c2h::get<0, TestType>;
  constexpr int TEST_VALUES = 11000;

  c2h::device_vector<T> d_data(TEST_VALUES);
  c2h::gen(C2H_SEED(1), d_data);
  c2h::host_vector<T> h_data(d_data.begin(), d_data.end());

  transform_op_t<T> op;
  const auto h_reference = c2h::host_vector<T>{
    op(h_data[0]),
    op(h_data[100]),
    op(h_data[1000]),
    op(h_data[10000]),
    op(h_data[1]),
    op(h_data[21]),
    op(h_data[11]),
    op(h_data[0])};

  using TextureIterator = cub::TexObjInputIterator<T>;
  TextureIterator d_tex_itr;
  CubDebugExit(
    d_tex_itr.BindTexture(const_cast<const T*>(thrust::raw_pointer_cast(d_data.data())), sizeof(T) * TEST_VALUES));
  cub::TransformInputIterator<T, transform_op_t<T>, TextureIterator> xform_itr(d_tex_itr, op);
  test_iterator(xform_itr, h_reference);
  CubDebugExit(d_tex_itr.UnbindTexture());
}
