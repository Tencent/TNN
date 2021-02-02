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
#include "cuda_strided_slice_layer_acc_kernel.cuh"
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS {

DECLARE_CUDA_ACC(StrideSliceV2, LAYER_STRIDED_SLICE_V2);

Status CudaStrideSliceV2LayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
        const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    Status ret = CudaLayerAcc::Init(context, param, resource, inputs, outputs);
    if (ret != TNN_OK) {
        return ret;
    }    
    CreateTempBuf(5 * sizeof(int));
    CreateTempBuf(5 * sizeof(int));
    auto params = dynamic_cast<StrideSliceV2LayerParam *>(param);
    if (!params) {
        LOGE("Error: ShuffleLayerParam is nil\n");
        return Status(TNNERR_MODEL_ERR, "Error: ShuffleLayerParam is nil");
    }

    auto input_dims = inputs[0]->GetBlobDesc().dims;

    auto param_begins = params->begins;
    auto param_strides = params->strides;
    auto axes = params->axes;
    std::vector<int> begins(5, 0), strides(5, 1);
    for(int i = 0; i < axes.size(); ++i) {
        int axis = axes[i];
        int begin = param_begins[i];
        begins[axis] = begin >= 0? begin : begin + input_dims[axis];
        strides[axis] = param_strides[i];
    }
    std::reverse(begins.begin(), begins.end());
    std::reverse(strides.begin(), strides.end());

    cudaMemcpy(tempbufs_[0].ptr, &(begins[0]), 5 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(tempbufs_[1].ptr, &(strides[0]), 5 * sizeof(int), cudaMemcpyHostToDevice);

    return TNN_OK;
}

Status CudaStrideSliceV2LayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CudaStrideSliceV2LayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    Blob *input_blob  = inputs[0];
    Blob *output_blob = outputs[0];

    auto input_dims   = input_blob->GetBlobDesc().dims;
    auto output_dims = output_blob->GetBlobDesc().dims;
    int input_n = input_dims[0];
    int input_c = input_dims[1];
    int output_c = output_dims[1];
    int input_d = 1, output_d = 1;
    if(input_dims.size() > 2) {
        input_d = input_dims[2];
        output_d = output_dims[2];
    }
    int input_h = 1, output_h = 1;
    if(input_dims.size() > 3) {
        input_h = input_dims[3];
        output_h = output_dims[3];
    }
    int input_w = 1, output_w = 1;
    if(input_dims.size() > 4) {
        input_w = input_dims[4];
        output_w = output_dims[4];
    }

    int div_d = output_w * output_h;
    int div_c = output_w * output_h * output_d;
    int div_n = output_w * output_h * output_d * output_c;
    int count = DimsVectorUtils::Count(output_dims);

    float* input_data = static_cast<float*>(input_blob->GetHandle().base);
    float* output_data = static_cast<float*>(output_blob->GetHandle().base);

    return RunStrideSlice(count, input_data, input_c, input_d, input_h, input_w, (const int*)tempbufs_[0].ptr,
                (const int*)tempbufs_[1].ptr, output_data, output_c, output_d, output_h, output_w, div_d, div_c, div_n, context_->GetStream());
}

REGISTER_CUDA_ACC(StrideSliceV2, LAYER_STRIDED_SLICE_V2);

}  // namespace TNN_NS
