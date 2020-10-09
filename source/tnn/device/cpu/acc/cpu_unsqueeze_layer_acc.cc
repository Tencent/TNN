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

#include "cpu_layer_acc.h"
#include "tnn/utils/dims_vector_utils.h"
namespace TNN_NS {

DECLARE_CPU_ACC(Unsqueeze, LAYER_UNSQUEEZE);

Status CpuUnsqueezeLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CpuUnsqueezeLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    const auto& param = dynamic_cast<UnsqueezeLayerParam*>(param_);
    if (param->data_in_resource) {
        auto output_data = outputs[0]->GetHandle().base;
        const auto& resource = dynamic_cast<UnsqueezeLayerResource*>(resource_);
        const auto& data_dims = resource->data_dims;
        auto data_size = resource->data.GetDataCount();
        if (resource->data.GetDataType() == DATA_TYPE_INT32) {
            memcpy(output_data, resource->data.force_to<int32_t*>(), data_size*sizeof(int32_t));
        } else {
            LOGE("Unqueeze acc: do not support data type\n");
            return TNNERR_UNSUPPORT_NET;
        }
    } else {
        const auto input_blob = inputs[0];
        const auto &input_dims  = input_blob->GetBlobDesc().dims;
        const auto &output_blob = outputs[0];
        if (input_blob->GetBlobDesc().data_type == DATA_TYPE_INT32) {
            auto input_data  = static_cast<int *>(input_blob->GetHandle().base);
            auto output_data = static_cast<int *>(output_blob->GetHandle().base);
            auto data_size = DimsVectorUtils::Count(input_dims);
            memcpy(output_data, input_data, data_size * sizeof(int32_t));
        } else {
            LOGE("Unqueeze acc: do not support data type\n");
            return TNNERR_UNSUPPORT_NET;
        }
    }
    return TNN_OK;
}

REGISTER_CPU_ACC(Unsqueeze, LAYER_UNSQUEEZE);
}  // namespace TNN_NS