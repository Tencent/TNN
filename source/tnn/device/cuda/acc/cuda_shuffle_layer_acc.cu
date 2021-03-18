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

DECLARE_CUDA_ACC(Shuffle, LAYER_SHUFFLE_CHANNEL);

__global__ void shuffle_kernel(const int count, const int feature_map_size, const float *input, 
        float *output, int group_row, int group_column, int len) {
    CUDA_KERNEL_LOOP(index, count) {
        const int n = index / group_row / group_column / len;
        const int i = (index / group_column / len) % group_row;
        const int j = index / len % group_column;
        const int k = index - (n * feature_map_size + (i * group_column + j) * len);
        float* p_o = output + n * feature_map_size + (j * group_row + i) * len;
        p_o[k] = input[index];
    }
}

Status CudaShuffleLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
        const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return CudaLayerAcc::Init(context, param, resource, inputs, outputs);
}

Status CudaShuffleLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CudaShuffleLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto param = dynamic_cast<ShuffleLayerParam *>(param_);
    if (!param) {
        LOGE("Error: ShuffleLayerParam is nil\n");
        return Status(TNNERR_MODEL_ERR, "Error: ShuffleLayerParam is nil");
    }

    Blob *input_blob  = inputs[0];
    Blob *output_blob = outputs[0];

    auto dims   = input_blob->GetBlobDesc().dims;
    const int num              = dims[0];
    const int feature_map_size = DimsVectorUtils::Count(dims, 1);
    const int sp_sz            = DimsVectorUtils::Count(dims, 2);
    const int chs              = dims[1];

    int group_row    = param->group;
    int group_column = int(chs / group_row);
    assert(chs == (group_column * group_row));
    int count = DimsVectorUtils::Count(dims);

    float* input_data = static_cast<float*>(input_blob->GetHandle().base);
    float* output_data = static_cast<float*>(output_blob->GetHandle().base);
    shuffle_kernel<<<TNN_CUDA_GET_BLOCKS(count), TNN_CUDA_NUM_THREADS, 0, context_->GetStream()>>>(
        count, feature_map_size, input_data, output_data, group_row, group_column, sp_sz);
    return TNN_OK;
}

REGISTER_CUDA_ACC(Shuffle, LAYER_SHUFFLE_CHANNEL);

}  // namespace TNN_NS
