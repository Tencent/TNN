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

#include "tnn/device/x86/acc/x86_layer_acc.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_vector_utils.h"
#include "tnn/device/x86/acc/compute/x86_compute.h"

namespace TNN_NS {

DECLARE_X86_ACC(GroupNorm, LAYER_GROUP_NORM);

Status X86GroupNormLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto param = dynamic_cast<GroupNormLayerParam *>(param_);

    Blob *input_blob  = inputs[0];
    Blob *scale_blob  = inputs[1];
    Blob *bias_blob  = inputs[2];
    Blob *output_blob = outputs[0];

    const int group = param->group;
    const int batch_time_group = output_blob->GetBlobDesc().dims[0] * group;
    const int channels_per_group = output_blob->GetBlobDesc().dims[1] / group;
    const int channel_area = DimsVectorUtils::Count(output_blob->GetBlobDesc().dims, 2);
    const int group_area = channel_area * channels_per_group;

    if (0 == group_area || 0 == channels_per_group) {
        LOGE("Error: blob count is zero\n");
        return Status(TNNERR_COMMON_ERROR, "Error: blob count is zero");
    }
    
    float *k_data = (float *)((char*)scale_blob->GetHandle().base + scale_blob->GetHandle().bytes_offset);
    float *b_data = (float *)((char*)bias_blob->GetHandle().base + bias_blob->GetHandle().bytes_offset);

    const float epsilon = param->eps;
    auto x86_groupnorm_func = X86_GroupNorm_FMA<Float4, 4>;
    if (arch_ == avx2) {
        x86_groupnorm_func = X86_GroupNorm_FMA<Float8, 8>;
    }

    if (output_blob->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
        float *input_data  = (float *)((char *)input_blob->GetHandle().base+ input_blob->GetHandle().bytes_offset);
        float *output_data = (float *)((char *)output_blob->GetHandle().base + output_blob->GetHandle().bytes_offset);
        x86_groupnorm_func(input_data, output_data, k_data, b_data, group, epsilon, batch_time_group, channels_per_group, channel_area, group_area);
    } else {
        LOGE("Error: X86GroupNormLayerAcc layer acc dont support datatype: %d\n", output_blob->GetBlobDesc().data_type);
        return Status(TNNERR_MODEL_ERR, "Error: X86GroupNormLayerAcc layer acc dont support datatype");
    }

    return TNN_OK;
}

REGISTER_X86_ACC(GroupNorm, LAYER_GROUP_NORM);

}