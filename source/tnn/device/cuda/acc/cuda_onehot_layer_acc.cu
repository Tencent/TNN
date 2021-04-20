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
#include "tnn/device/cuda/fastdiv.h"
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

DECLARE_CUDA_ACC(OneHot, LAYER_ONEHOT);

__global__ void onehot_kernel(
    const int* indices_data,
    float* output_data,
    int count,
    const fastdiv depth_suffix,
    const fastdiv suffix,
    const int depth,
    const float value_on,
    const float value_off) {
    CUDA_KERNEL_LOOP(index, count) {
        int prefix_index = index / depth_suffix; 
        int prefix_offset = index - prefix_index * depth_suffix;
        int depth_index = prefix_offset / suffix;
        int suffix_index = depth_index - depth_index * suffix;
        int indices_index = prefix_index * suffix + suffix_index;
        bool is_valid_range = indices_data[indices_index] >= -depth && indices_data[indices_index] < depth;
        int adjusted_indice = (indices_data[indices_index] + depth) % depth;
        output_data[index] = (is_valid_range && adjusted_indice == depth_index) ? value_on : value_off;
    }
}

Status CudaOneHotLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
        const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return CudaLayerAcc::Init(context, param, resource, inputs, outputs);;
}

Status CudaOneHotLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CudaOneHotLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto layer_param = dynamic_cast<OneHotLayerParam*>(param_);
    CHECK_PARAM_NULL(layer_param);

    auto input = inputs[0];
    auto output = outputs[0];

    auto output_dims = output->GetBlobDesc().dims;
    int axis = layer_param->axis;
    if(axis < 0) {
        axis += output_dims.size();
    }
    
    auto input_data = (int*)(input->GetHandle().base);
    auto output_data = (float *)(output->GetHandle().base);
 
    int depth = output_dims[axis];
    fastdiv depth_suffix, suffix;
    depth_suffix.init(DimsVectorUtils::Count(output_dims, axis));
    suffix.init(DimsVectorUtils::Count(output_dims, axis+1));

    const int count = DimsVectorUtils::Count(output_dims);
    onehot_kernel<<<TNN_CUDA_GET_BLOCKS(count), TNN_CUDA_NUM_THREADS, 0, context_->GetStream()>>>(
            input_data, output_data, count, depth_suffix, suffix, depth, 
            layer_param->value_on, layer_param->value_off);

    return TNN_OK;
}

REGISTER_CUDA_ACC(OneHot, LAYER_ONEHOT);

}  // namespace TNN_NS
