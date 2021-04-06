// Tencent is pleased to support the open source community by making TNN available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "tnn/device/cuda/acc/cuda_layer_acc.h"
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS {

DECLARE_CUDA_ACC(GroupNorm, LAYER_GROUP_NORM);

namespace {

inline static int getThreads(int count) {
    if (count <= 0) return 0;
    if (count <= 32) return 32;
    if (count > 256) return 512;
    count -= 1;
    count |= (count >> 1);
    count |= (count >> 2);
    count |= (count >> 4);
    return count + 1;
}
inline static int getPass1Size(int a, int b, int c) {
    int prod, val;
    if (a < b && a < c) { val = a; prod = b * c; }
    else if (b < c) { val = b; prod = a * c; }
    else { val = c; prod = a * b; }
    return prod > val ? prod: val;
}

template<typename T>
struct Tuple2 {
    T v1; T v2;
    __device__ __host__ inline Tuple2<T>(const T a, const T b) : v1(a), v2(b) {}
    __device__ __host__ inline Tuple2<T>() : v1(0.), v2(0.) {}
    __device__ __host__ inline Tuple2<T>(const T& other): v1(other), v2(other) {}
    __device__ __host__ inline Tuple2<T> operator+(const Tuple2<T> &other) { return {v1 + other.v1, v2 + other.v2}; }
    __device__ __host__ inline Tuple2<T> &operator+=(const Tuple2<T> &other) { v1 += other.v1; v2 += other.v2; return *this; }
};

template<typename T> struct GNAccType {using type = T; };
template<> struct GNAccType<__half> {using type = float; };
template<> struct GNAccType<float> {using type = float; };

__device__ inline static Tuple2<float> __shfl_down_sync(unsigned mask, Tuple2<float> var, unsigned int delta, int width) {
    auto ret = ::__shfl_down_sync(mask, *(double *)&var, delta, width);
    return *(Tuple2<float>*)&ret;
}
// __device__ inline static Tuple2<__half> __shfl_down_sync(unsigned mask, Tuple2<__half> var, unsigned int delta, int width) {
//     auto ret = __shfl_down_sync(mask, *(float*)&var, delta, width);
//     return *(Tuple2<__half>*)&ret;
// }

template<typename T, int WARP_SIZE>
struct WarpReducer { __device__ inline static T reduce(T val); };
template<typename T> struct WarpReducer<T, 32> { __device__ inline static T reduce(T val) {
    val += __shfl_down_sync(0xffffffff, val, 16, 32);
    val += __shfl_down_sync(0x0000ffff, val, 8, 16);
    val += __shfl_down_sync(0x000000ff, val, 4, 8);
    val += __shfl_down_sync(0x0000000f, val, 2, 4);
    val += __shfl_down_sync(0x00000003, val, 1, 2);
    return val;
}};
template<typename T> struct WarpReducer<T, 16> { __device__ inline static T reduce(T val) {
    val += __shfl_down_sync(0x0000ffff, val, 8, 16);
    val += __shfl_down_sync(0x000000ff, val, 4, 8);
    val += __shfl_down_sync(0x0000000f, val, 2, 4);
    val += __shfl_down_sync(0x00000003, val, 1, 2);
    return val;
}};
template<typename T> struct WarpReducer<T, 8> { __device__ inline static T reduce(T val) {
    val += __shfl_down_sync(0x000000ff, val, 4, 8);
    val += __shfl_down_sync(0x0000000f, val, 2, 4);
    val += __shfl_down_sync(0x00000003, val, 1, 2);
    return val;
}};
template<typename T> struct WarpReducer<T, 4> { __device__ inline static T reduce(T val) {
    val += __shfl_down_sync(0x0000000f, val, 2, 4);
    val += __shfl_down_sync(0x00000003, val, 1, 2);
    return val;
}};
template<typename T> struct WarpReducer<T, 2> { __device__ inline static T reduce(T val) {
    val += __shfl_down_sync(0x00000003, val, 1, 2);
    return val;
}};
template<typename T> struct WarpReducer<T, 1> { __device__ inline static T reduce(T val) { return val; }};

template<typename T> using UFunc = T(*)(T);
template<typename T> __device__ __host__ inline T idn(T val) { return val; }
template<typename T> __device__ __host__ inline T sqr(T val) { return val * val; }
template<typename T> __device__ __host__ inline Tuple2<T> idn(Tuple2<T> val) { return val; }
template<typename T> __device__ __host__ inline Tuple2<T> idn_sqr(Tuple2<T> val) { return {val.v1, val.v2 * val.v2}; }
}

template<int THREAD_PER_BLOCK, typename T, typename AccType, UFunc<AccType> ufunc>
__device__ static void reduce(const T* input, AccType* output, const int count, const int in_elem_step = 1) {

    static_assert(THREAD_PER_BLOCK % 32 == 0 && THREAD_PER_BLOCK >= 32, "");
    __shared__ char _sm_static[(THREAD_PER_BLOCK / 32) * sizeof(AccType)];
    AccType *ssum = reinterpret_cast<AccType*>(_sm_static);
    AccType sum = AccType(0.);

    const T* ptr = input + threadIdx.x * in_elem_step;
    const auto actual_step = THREAD_PER_BLOCK * in_elem_step;
    for (int i = threadIdx.x; i < count; i += THREAD_PER_BLOCK, ptr += actual_step) {
        auto value = static_cast<AccType>(*ptr);
        sum += ufunc(value);
    }
    sum = WarpReducer<AccType, 32>::reduce(sum);
    if (threadIdx.x % 32 == 0) { ssum[threadIdx.x / 32] = sum; }
    __syncthreads();

    sum = threadIdx.x < THREAD_PER_BLOCK / 32 ? ssum[threadIdx.x] : AccType(0.);
    sum = WarpReducer<AccType, THREAD_PER_BLOCK / 32>::reduce(sum);
    if (threadIdx.x == 0) { *output = sum; }
    __syncthreads();
}

template<int THREAD_PER_BLOCK, typename T, typename AccType, UFunc<AccType> ufunc = idn<AccType> >
__global__ static void reduce_unalign(const T* input, AccType* output, const int count,
                                      const int in_block_step, const int in_elem_step = 1,
                                      const int out_step_T = (sizeof(AccType) + sizeof(T) - 1) / sizeof(T)) {
    reduce<THREAD_PER_BLOCK, T, AccType, ufunc>(
        input + in_block_step * blockIdx.x, 
        reinterpret_cast<AccType*>(reinterpret_cast<T*>(output) + blockIdx.x * out_step_T),
        count, in_elem_step);
}

template<typename T>
__device__ void fuse_param_and_affine(const T *input, T *output, const float *gamma, const float *beta,
                                      const int c_per_g, const int hw, const float eps,
                                      typename GNAccType<T>::type sum1, typename GNAccType<T>::type sum2) {
    using AccType = typename GNAccType<T>::type;
    extern __shared__ char _sm[];
    AccType* scale = reinterpret_cast<AccType*>(_sm);
    AccType* bias = scale + c_per_g;
    const int c_off = c_per_g * blockIdx.x;
    for (int i = threadIdx.x; i < c_per_g; i += blockDim.x) {
        AccType mean = sum1 / (c_per_g * hw) ;
        AccType var = sum2 / (c_per_g * hw) - mean * mean;
        AccType k = rsqrt(var + eps) * gamma[c_off + i];
        scale[i] = k;
        bias[i] = - mean * k + beta[c_off + i];
    }
    __syncthreads();

    const auto count = c_per_g * hw;
    const auto offset = count * blockIdx.x;
    const T* in_ptr = input + offset;
    T* out_ptr = output + offset;
    for (int i = threadIdx.x; i < count; i += blockDim.x) {
        auto c_idx = i / hw;
        out_ptr[i] = static_cast<AccType>(in_ptr[i]) * scale[c_idx] + bias[c_idx];
    }
}

template<int THREAD_PER_BLOCK, typename T>
__global__ void group_norm_1pass(const T *input, T *output, const float *gamma, const float *beta,
                                 const int c_per_g, const int hw, const float eps) {
    // 1 group per block, used when c_per_g * hw <= 4096
    // assert (c == g * c_per_g)
    using AccType = typename GNAccType<T>::type;

    __shared__ char _sums[sizeof(Tuple2<AccType>)];
    Tuple2<AccType> *sums = reinterpret_cast<Tuple2<AccType>*>(_sums);
    reduce<THREAD_PER_BLOCK, T, Tuple2<AccType>, idn_sqr<AccType> >(
        input + blockIdx.x * hw * c_per_g, sums, c_per_g * hw);

    fuse_param_and_affine<T>(input, output, gamma, beta, c_per_g, hw, eps, sums[0].v1, sums[0].v2);
}

template<int THREAD_PER_BLOCK, typename T>
__global__ void group_norm_2pass(const T *input, T *output, const float *gamma, const float *beta,
                                 const int c_per_g, const int hw, const int part_per_g, const int part_count,
                                 const float eps) {
    // 1 group per block, used when c_per_g * hw >= 4096
    // assert (part_per_g * 2 * sizeof(AccType) <= c_per_g * hw * sizeof(T))
    // assert (c_per_g * hw == part_per_g * part_count)
    using AccType = typename GNAccType<T>::type;

    extern __shared__ char _part_sums[];
    __shared__ char _sums[sizeof(Tuple2<AccType>)];
    Tuple2<AccType> *sums = reinterpret_cast<Tuple2<AccType>*>(_sums);

    if (part_count % (sizeof(Tuple2<AccType>) / sizeof(T)) == 0) {
        Tuple2<AccType> *part_sums = reinterpret_cast<Tuple2<AccType>*>(_part_sums);
        for (int i = threadIdx.x; i < part_per_g; i += blockDim.x)
            part_sums[i] = *reinterpret_cast<Tuple2<AccType>*>(output + blockIdx.x * c_per_g * hw + i * part_count);
        __syncthreads();
        reduce<THREAD_PER_BLOCK, Tuple2<AccType>, Tuple2<AccType>, idn<AccType> >(part_sums, sums, part_per_g);
    } else {
        reduce<THREAD_PER_BLOCK, Tuple2<AccType>, Tuple2<AccType>, idn<AccType> >(
            reinterpret_cast<Tuple2<AccType>*>(output + blockIdx.x * c_per_g * hw), sums,
            part_per_g, static_cast<int>(part_count * sizeof(T) / sizeof(Tuple2<AccType>)));
    }
    fuse_param_and_affine<T>(input, output, gamma, beta, c_per_g, hw, eps, sums[0].v1, sums[0].v2);
}

template<typename T>
static Status group_norm_v2(const T *input, T* output, const float *gamma, const float *beta,
                            const int n, const int c, const int g, const int c_per_g, const int h, const int w,
                            const float eps, cudaStream_t s) {
    using AccType = typename GNAccType<T>::type;
    static std::map<int, void(*)(
        const T*, T*, const float *, const float *,
        const int, const int, const float)> group_norm_1pass_funcs = {
        {32,  group_norm_1pass<32, T>},
        {64,  group_norm_1pass<64, T>},
        {128, group_norm_1pass<128, T>},
        {256, group_norm_1pass<256, T>},
        {512, group_norm_1pass<512, T>},
    };
    static std::map<int, void(*)(
        const T*, T*, const float *, const float *,
        const int, const int, const int, const int, const float)> group_norm_2pass_funcs = {
        {32,  group_norm_2pass<32, T>},
        {64,  group_norm_2pass<64, T>},
        {128, group_norm_2pass<128, T>},
        {256, group_norm_2pass<256, T>},
        {512, group_norm_2pass<512, T>},
    };
    static std::map<int, void(*)(const T*, Tuple2<AccType>*, const int,
                                 const int, const int, const int)> reduce_funcs = {
        {32,  reduce_unalign<32, T, Tuple2<AccType>, idn_sqr<AccType> >},
        {64,  reduce_unalign<64, T, Tuple2<AccType>, idn_sqr<AccType> >},
        {128, reduce_unalign<128, T, Tuple2<AccType>, idn_sqr<AccType> >},
        {256, reduce_unalign<256, T, Tuple2<AccType>, idn_sqr<AccType> >},
        {512, reduce_unalign<512, T, Tuple2<AccType>, idn_sqr<AccType> >},
    };
    const int BLOCK_MAX = 512;
    const int hw = h * w;
    auto block = getThreads(c_per_g * hw);
    auto grid = n * g;
    if (c_per_g * hw <= 512 * BLOCK_MAX) {
        group_norm_1pass_funcs[block]<<<grid, block, 2 * c_per_g * sizeof(AccType), s>>>(
            input, output, gamma, beta, c_per_g, hw, eps);
        auto err = cudaGetLastError();
        if (err != cudaSuccess)
            return Status(TNNERR_CUDA_TENSORRT_ERROR, "GN Plugin 1pass failed: " + std::to_string(err));
    } else {
        // assert (part_per_g * 2 * sizeof(AccType) <= c_per_g * hw * sizeof(T))
        auto count_pass1 = getPass1Size(h, w, c_per_g);
        auto part_per_g = hw * c_per_g / count_pass1;
        auto block_pass1 = getThreads(count_pass1);
        auto grid_pass1 = n * g * part_per_g;
        if (part_per_g * 2 * sizeof(AccType) > c_per_g * hw * sizeof(T)) {
            group_norm_1pass_funcs[block]<<<grid, block, 2 * c_per_g * sizeof(AccType), s>>>(
            input, output, gamma, beta, c_per_g, hw, eps);
            auto err = cudaGetLastError();
            if (err != cudaSuccess)
                return Status(TNNERR_CUDA_TENSORRT_ERROR, "GN Plugin 1pass failed: " + std::to_string(err));
        } else {
            reduce_funcs[block_pass1]<<<grid_pass1, block_pass1, 0, s>>>(
                input, reinterpret_cast<Tuple2<AccType>*>(output), count_pass1, count_pass1, 1,
                count_pass1);
            auto err = cudaGetLastError();
            if (err != cudaSuccess)
                return Status(TNNERR_CUDA_TENSORRT_ERROR, "GN Plugin 2pass_1 failed: " + std::to_string(err));

            auto pass2_shm1 = 2 * c_per_g * sizeof(AccType);
            auto pass2_shm2 = part_per_g * sizeof(Tuple2<AccType>);
            group_norm_2pass_funcs[block]<<<grid, block, pass2_shm1 > pass2_shm2 ? pass2_shm1 : pass2_shm2, s>>>(
                input, output, gamma, beta, c_per_g, hw, part_per_g, count_pass1, eps);
            err = cudaGetLastError();
            if (err != cudaSuccess)
                return Status(TNNERR_CUDA_TENSORRT_ERROR, "GN Plugin 2pass_2 failed: " + std::to_string(err));
        }
    }
    return TNN_OK;
}

template<int THREAD_PER_BLOCK, typename T>
__global__ void group_norm_kernel(const T* input, T* output, const float * gamma,
        const float * beta, const int size, const int batch_size, const int channels_per_group,
        const int group, const int channels, const float eps) {
    using AccType = typename GNAccType<T>::type;
    __shared__ AccType ssum1[THREAD_PER_BLOCK/32];
    __shared__ AccType ssum2[THREAD_PER_BLOCK/32];
    __shared__ AccType k;
    __shared__ AccType b;
    extern __shared__ char _sm[];
    T *sm = reinterpret_cast<T*>(_sm);

    const int block_offset = (blockIdx.x * channels + blockIdx.y * channels_per_group) * size;
    const T * ptr = input + block_offset;
    T * dst = output + block_offset;

    AccType thread_sum1 = AccType(0.);
    AccType thread_sum2 = AccType(0.);

    for (int i = threadIdx.x; i < channels_per_group * size; i+=THREAD_PER_BLOCK) {
        AccType value = static_cast<AccType>(ptr[i]);
        thread_sum1 += value;
        thread_sum2 += value * value;
    }

    thread_sum1 += __shfl_down_sync(0xffffffff, thread_sum1, 16, 32);
    thread_sum1 += __shfl_down_sync(0x0000ffff, thread_sum1, 8, 16);
    thread_sum1 += __shfl_down_sync(0x000000ff, thread_sum1, 4, 8);
    thread_sum1 += __shfl_down_sync(0x0000000f, thread_sum1, 2, 4);
    thread_sum1 += __shfl_down_sync(0x00000003, thread_sum1, 1, 2);

    thread_sum2 += __shfl_down_sync(0xffffffff, thread_sum2, 16, 32);
    thread_sum2 += __shfl_down_sync(0x0000ffff, thread_sum2, 8, 16);
    thread_sum2 += __shfl_down_sync(0x000000ff, thread_sum2, 4, 8);
    thread_sum2 += __shfl_down_sync(0x0000000f, thread_sum2, 2, 4);
    thread_sum2 += __shfl_down_sync(0x00000003, thread_sum2, 1, 2);

    if (threadIdx.x % 32 == 0) {
        ssum1[threadIdx.x / 32] = thread_sum1;
        ssum2[threadIdx.x / 32] = thread_sum2;
    }
    __syncthreads();

    if (threadIdx.x < blockDim.x / 32) {
        thread_sum1 = ssum1[threadIdx.x];
        thread_sum2 = ssum2[threadIdx.x];
    } else {
        thread_sum1 = 0;
        thread_sum2 = 0;
    }
    thread_sum1 += __shfl_down_sync(0x0000000f, thread_sum1, 2, 4);
    thread_sum1 += __shfl_down_sync(0x00000003, thread_sum1, 1, 2);

    thread_sum2 += __shfl_down_sync(0x0000000f, thread_sum2, 2, 4);
    thread_sum2 += __shfl_down_sync(0x00000003, thread_sum2, 1, 2);

    if (threadIdx.x == 0) {
        AccType mean = thread_sum1 / (size * channels_per_group) ;
        AccType var = thread_sum2 / (size * channels_per_group) - mean * mean;

        k = 1.f / sqrt(var + eps);
        b = - mean * k;;
    }

    __syncthreads();
    for (int c = threadIdx.x; c < channels_per_group; c+=THREAD_PER_BLOCK) {
        float scale = gamma[blockIdx.y * channels_per_group + c];
        float bias = beta == nullptr ? 0.f : beta[blockIdx.y * channels_per_group + c];
        sm[c] = static_cast<T>(k * scale);
        sm[channels_per_group+c] = static_cast<T>(bias + b * scale);
    }
    __syncthreads();
    for (int c = 0; c < channels_per_group; c++) {
        T scale = sm[c];
        T bias = sm[channels_per_group + c];
        for (int i = threadIdx.x; i < size; i += THREAD_PER_BLOCK) {
             dst[c*size+i] = ptr[c*size+i] * scale + bias;
        }
    }
}

Status CudaGroupNormLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
        const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return CudaLayerAcc::Init(context, param, resource, inputs, outputs);
}

Status CudaGroupNormLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CudaGroupNormLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto params = dynamic_cast<GroupNormLayerParam*>(param_);
    auto dtype = inputs[0]->GetBlobDesc().data_type;

    Blob *input_blob = inputs[0];
    Blob *scale_blob = inputs[1];
    Blob *bias_blob  = inputs[2];
    Blob *output_blob = outputs[0];
    auto input_dims = inputs[0]->GetBlobDesc().dims;
    if (dtype == DATA_TYPE_FLOAT) {
        float* input_data = static_cast<float*>(input_blob->GetHandle().base);
        float* scale_data = static_cast<float*>(scale_blob->GetHandle().base);
        float* bias_data  = static_cast<float*>(bias_blob->GetHandle().base);
        float* output_data = static_cast<float*>(output_blob->GetHandle().base);
        int channels_per_group = input_dims[1] / params->group;

        // dim3 grid(input_dims[0], params->group);
        // const int THREAD_PER_BLOCK = 128;
        // int sm_size = channels_per_group * 2 * sizeof(float);
        // group_norm_kernel<THREAD_PER_BLOCK, float><<<grid, THREAD_PER_BLOCK, sm_size, context_->GetStream()>>>(input_data,
        //     output_data, scale_data, bias_data, input_dims[2]*input_dims[3], input_dims[0], channels_per_group, params->group,
        //     input_dims[1], params->eps);
        return group_norm_v2<float>(input_data, output_data, scale_data, bias_data,
                                    input_dims[0], input_dims[1], params->group, channels_per_group,
                                    input_dims[2], input_dims[3], params->eps, context_->GetStream());
    } else if (dtype == DATA_TYPE_HALF) {
        __half* input_data = static_cast<__half*>(input_blob->GetHandle().base);
        float* scale_data = static_cast<float*>(scale_blob->GetHandle().base);
        float* bias_data  = static_cast<float*>(bias_blob->GetHandle().base);
        __half* output_data = static_cast<__half*>(output_blob->GetHandle().base);
        int channels_per_group = input_dims[1] / params->group;

        // dim3 grid(input_dims[0], params->group);
        // const int THREAD_PER_BLOCK = 128;
        // int sm_size = channels_per_group * 2 * sizeof(__half);
        // group_norm_kernel<THREAD_PER_BLOCK, __half><<<grid, THREAD_PER_BLOCK, sm_size, context_->GetStream()>>>(input_data,
        //     output_data, scale_data, bias_data, input_dims[2]*input_dims[3], input_dims[0], channels_per_group, params->group,
        //     input_dims[1], params->eps);
        return group_norm_v2<__half>(input_data, output_data, scale_data, bias_data,
                                    input_dims[0], input_dims[1], params->group, channels_per_group,
                                    input_dims[2], input_dims[3], params->eps, context_->GetStream());
    } else {
        return Status(TNNERR_CUDA_TENSORRT_ERROR, "Unexpected data type " + std::to_string(dtype));
    }

    return TNN_OK;
}

REGISTER_CUDA_ACC(GroupNorm, LAYER_GROUP_NORM);

}  // namespace TNN_NS

