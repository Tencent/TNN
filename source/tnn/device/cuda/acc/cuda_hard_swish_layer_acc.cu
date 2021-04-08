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

DECLARE_CUDA_ACC(HardSwish, LAYER_HARDSWISH);

__global__ void hard_swish_kernel(int count, const float* in1, const float* in2, float* out, int in_n1,
        int in_c1, int in_h1, int in_w1, int in_n2, int in_c2, int in_h2, int in_w2, int out_c, int out_h,
        int out_w, const float alpha, const float beta) {
    CUDA_KERNEL_LOOP(index, count) {
        int b = index / (out_c * out_h * out_w);
        int c = index / (out_h * out_w) % out_c;
        int h = index / out_w % out_h;
        int w = index % out_w;
        int input_index_b_1 = min(b, in_n1-1) * in_c1 * in_h1 * in_w1;
        int input_index_b_2 = min(b, in_n2-1) * in_c2 * in_h2 * in_w2;
        int input_index_c_1 = min(c, in_c1-1) * in_h1 * in_w1 + input_index_b_1;
        int input_index_c_2 = min(c, in_c2-1) * in_h2 * in_w2 + input_index_b_2;
        int input_index_h_1 = min(h, in_h1-1) * in_w1 + input_index_c_1;
        int input_index_h_2 = min(h, in_h2-1) * in_w1 + input_index_c_2;
        int input_index_w_1 = min(w, in_w1-1) + input_index_h_1;
        int input_index_w_2 = min(w, in_w2-1) + input_index_h_2;
        out[index] = in1[input_index_w_1] * max(min(in2[input_index_w_2] * alpha + beta, 1.f), 0.f);
    }
}

Status CudaHardSwishLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
        const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return CudaLayerAcc::Init(context, param, resource, inputs, outputs);
}

Status CudaHardSwishLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CudaHardSwishLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto params = dynamic_cast<HardSwishLayerParam *>(param_);
    if (!params) {
        LOGE("Error: HardSwishLayerParam is nil\n");
        return Status(TNNERR_MODEL_ERR, "Error: HardSwishLayerParam is nil");
    }

    int count = DimsVectorUtils::Count(outputs[0]->GetBlobDesc().dims);

    Blob* input_blob1 = inputs[0];
    Blob* input_blob2 = inputs[0];
    Blob* output_blob = outputs[0];
    if (inputs.size() != 1) {
        input_blob2 = inputs[1];
    }
    float* input_data1 = static_cast<float*>(input_blob1->GetHandle().base);
    float* input_data2 = static_cast<float*>(input_blob2->GetHandle().base);
    float* output_data = static_cast<float*>(output_blob->GetHandle().base);

    auto input_dims1 = input_blob1->GetBlobDesc().dims;
    auto input_dims2 = input_blob2->GetBlobDesc().dims;
    auto output_dims = output_blob->GetBlobDesc().dims;

    int in_n1 = DimsFunctionUtils::GetDim(input_dims1, 0);
    int in_c1 = DimsFunctionUtils::GetDim(input_dims1, 1);
    int in_h1 = DimsFunctionUtils::GetDim(input_dims1, 2);
    int in_w1 = DimsFunctionUtils::GetDim(input_dims1, 3);

    int in_n2 = DimsFunctionUtils::GetDim(input_dims2, 0);
    int in_c2 = DimsFunctionUtils::GetDim(input_dims2, 1);
    int in_h2 = DimsFunctionUtils::GetDim(input_dims2, 2);
    int in_w2 = DimsFunctionUtils::GetDim(input_dims2, 3);

    int out_c = DimsFunctionUtils::GetDim(output_dims, 1);
    int out_h = DimsFunctionUtils::GetDim(output_dims, 2);
    int out_w = DimsFunctionUtils::GetDim(output_dims, 3);

    hard_swish_kernel<<<TNN_CUDA_GET_BLOCKS(count), TNN_CUDA_NUM_THREADS, 0, context_->GetStream()>>>(
        count, input_data1, input_data2, output_data, in_n1, in_c1, in_h1, in_w1, in_n2, in_c2, in_h2,
        in_w2, out_c, out_h, out_w, params->alpha, params->beta);
    
    return TNN_OK;
}

REGISTER_CUDA_ACC(HardSwish, LAYER_HARDSWISH);

}  // namespace TNN_NS
