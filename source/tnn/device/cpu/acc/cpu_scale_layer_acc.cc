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

#include "tnn/device/cpu/acc/cpu_layer_acc.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS {

DECLARE_CPU_ACC_WITH_FP32_RESOURCE(Scale, LAYER_SCALE);

Status CpuScaleLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CpuScaleLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto resource = dynamic_cast<BatchNormLayerResource *>(resource_);
    if (!resource) {
        return Status(TNNERR_MODEL_ERR, "Error: BatchNormLayerResource is nil");
    }

    Blob *input_blob       = inputs[0];
    Blob *output_blob      = outputs[0];
    float *input_data      = static_cast<float *>(input_blob->GetHandle().base);
    float *output_data     = static_cast<float *>(output_blob->GetHandle().base);
    int channel            = input_blob->GetBlobDesc().dims[1];
    int hw                 = input_blob->GetBlobDesc().dims[2] * input_blob->GetBlobDesc().dims[3];
    int count              = DimsVectorUtils::Count(input_blob->GetBlobDesc().dims);
    RawBuffer scale_handle = resource->scale_handle;
    float *k_data          = resource->scale_handle.force_to<float *>();
    bool share_channel     = scale_handle.GetBytesSize() == DataTypeUtils::GetBytesSize(scale_handle.GetDataType());
    float *b_data          = resource->bias_handle.force_to<float *>();

    for (int index = 0; index < count; ++index) {
        float result = 0.0f;
        int c        = share_channel ? 0 : (index / hw) % channel;
        result       = input_data[index] * k_data[c];
        if (b_data != nullptr) {
            result += b_data[c];
        }
        output_data[index] = result;
    }
    return TNN_OK;
}

REGISTER_CPU_ACC(Scale, LAYER_SCALE);

}  // namespace TNN_NS
