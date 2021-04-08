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

#include <cub/cub.cuh>
#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/block_radix_sort.cuh>

#include "tnn/device/cuda/acc/cuda_layer_acc.h"
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

DECLARE_CUDA_ACC(ArgMaxOrMin, LAYER_ARG_MAX_OR_MIN);

template <typename K, typename V>
using KeyValuePair = cub::KeyValuePair<K, V>;

template <typename K, typename V>
using BlockReduce =
    cub::BlockReduce<KeyValuePair<K, V>, TNN_CUDA_NUM_THREADS>;

template <typename T, typename ReductionOpT>
__global__ void argmaxmin_kernel(
    const T* input,
    const int outer_size,
    const int inner_size,
    const int stride,
    const ReductionOpT reducer,
    const T init,
    int* output) {
  __shared__ typename BlockReduce<int, T>::TempStorage temp_storage;
  for (int idx = blockIdx.x; idx < outer_size; idx += gridDim.x) {
    int i = idx / stride;
    int j = idx % stride;
    KeyValuePair<int, T> kv = {-1, init};
    for (int k = threadIdx.x; k < inner_size; k += blockDim.x) {
        kv = reducer({k, input[i * inner_size * stride + k * stride + j]}, kv);
    }
    kv = BlockReduce<int, T>(temp_storage).Reduce(kv, reducer);
    if (threadIdx.x == 0) {
      output[idx] = static_cast<int>(kv.key);
    }
    __syncthreads();
  }
}

Status CudaArgMaxOrMinLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
        const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return CudaLayerAcc::Init(context, param, resource, inputs, outputs);
}

Status CudaArgMaxOrMinLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CudaArgMaxOrMinLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto params = dynamic_cast<ArgMaxOrMinLayerParam *>(param_);
    CHECK_PARAM_NULL(params);

    Blob *input_blob  = inputs[0];
    Blob *output_blob = outputs[0];

    auto input_dims  = input_blob->GetBlobDesc().dims;
    int axis         = params->axis;
    int num          = DimsVectorUtils::Count(input_dims, 0, axis);
    int channels     = input_dims[axis];
    int stride       = DimsVectorUtils::Count(input_dims, axis + 1);
    if (stride == 0) {
        stride = 1;
    }

    if (input_blob->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
        float *input_data  = static_cast<float *>(input_blob->GetHandle().base);
        int *output_data = static_cast<int *>(output_blob->GetHandle().base);
        if (params->mode == 0) {
            argmaxmin_kernel<<<num * stride, TNN_CUDA_NUM_THREADS, 0, context_->GetStream()>>>(
                input_data, num * stride, channels, stride, cub::ArgMin(), FLT_MAX, output_data);
        } else {
            argmaxmin_kernel<<<num * stride, TNN_CUDA_NUM_THREADS, 0, context_->GetStream()>>>(
                input_data, num * stride, channels, stride, cub::ArgMax(), -FLT_MAX, output_data);
        }
    } else if (output_blob->GetBlobDesc().data_type == DATA_TYPE_INT8) {
        LOGE("Error: layer acc dont support datatype: %d\n", output_blob->GetBlobDesc().data_type);
        return Status(TNNERR_MODEL_ERR, "Error: layer acc dont support datatype");
    } else {
        LOGE("Error: layer acc dont support datatype: %d\n", output_blob->GetBlobDesc().data_type);
        return Status(TNNERR_MODEL_ERR, "Error: layer acc dont support datatype");
    }

    return TNN_OK;
}

REGISTER_CUDA_ACC(ArgMaxOrMin, LAYER_ARG_MAX_OR_MIN);

}  // namespace TNN_NS

