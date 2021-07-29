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
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

DECLARE_CUDA_ACC(ReduceL2, LAYER_REDUCE_L2);

__global__ void reduce_l2_kernel(const int num, const int channels,
        const int spatial_dim, const float* input, float* output) {
    CUDA_KERNEL_LOOP(index, num * spatial_dim) {
        int n = index / spatial_dim;
        int s = index % spatial_dim;
        float tmp = 0;
        for (int c = 0; c < channels; ++c) {
            float value = input[(n * channels + c) * spatial_dim + s];
            tmp += value * value;
        }
        output[n * spatial_dim + s] = sqrt(tmp);
    }
}

Status CudaReduceL2LayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
        const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return CudaLayerAcc::Init(context, param, resource, inputs, outputs);
}

Status CudaReduceL2LayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CudaReduceL2LayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto params = dynamic_cast<ReduceLayerParam *>(param_);
    if (!params) {
        LOGE("Error: layer param is null\n");
        return Status(TNNERR_MODEL_ERR, "Error: layer param is null");
    }

    Blob *input_blob  = inputs[0];
    Blob *output_blob = outputs[0];

    int channels = 1;
    int first_axis = 4;
    int last_axis = 0;
    // remove duplicate axes
    auto axis = params->axis;
    std::sort(axis.begin(), axis.end());
    axis.erase(std::unique(axis.begin(), axis.end() ), axis.end());
    for (int i = 0; i < axis.size(); i++) {
        channels *= input_blob->GetBlobDesc().dims[axis[i]];
        first_axis = std::min(axis[i], first_axis);
        last_axis = std::max(axis[i], last_axis);
    }

    for(int i=first_axis; i<=last_axis; ++i) {
        if (std::find(axis.begin(), axis.end(), i) == axis.end()) {
            LOGE("Error: discontinuous reduce axes!");
            return Status(TNNERR_PARAM_ERR, "Error: discontinuous reduce axes!"); 
        }
    }

    int outer_dim = DimsVectorUtils::Count(input_blob->GetBlobDesc().dims, 0, first_axis);
    int inner_dim = DimsVectorUtils::Count(input_blob->GetBlobDesc().dims, last_axis+1);
    int count = DimsVectorUtils::Count(output_blob->GetBlobDesc().dims);
    float* input_data = static_cast<float*>(input_blob->GetHandle().base);
    float* output_data = static_cast<float*>(output_blob->GetHandle().base);
    reduce_l2_kernel<<<TNN_CUDA_GET_BLOCKS(count), TNN_CUDA_NUM_THREADS, 0, context_->GetStream()>>>(
        outer_dim, channels, inner_dim, input_data, output_data);
    return TNN_OK;
}

REGISTER_CUDA_ACC(ReduceL2, LAYER_REDUCE_L2);

}  // namespace TNN_NS
