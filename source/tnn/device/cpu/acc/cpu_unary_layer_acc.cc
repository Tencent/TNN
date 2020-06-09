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

#include "tnn/device/cpu/acc/cpu_unary_layer_acc.h"
#include "tnn/device/cpu/cpu_device.h"

#include "tnn/utils/naive_compute.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS {

Status CpuUnaryLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                              const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto ret = CpuLayerAcc::Init(context, param, resource, inputs, outputs);
    if (ret != TNN_OK) {
        return ret;
    }
    if (NULL == op_) {
        LOGE("Error: Unary layer init got null op\n");
        return Status(TNNERR_LAYER_ERR, "Unary layer init got null op");
    }
    return op_->Init(param);
}

Status CpuUnaryLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CpuUnaryLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    if (inputs.size() < 1) {
        LOGE("Error: invalid inputs count\n");
        return Status(TNNERR_LAYER_ERR, "layer's inputs size must >= 1");
    }

    if (NULL == op_) {
        LOGE("Error: Unary layer got null op\n");
        return Status(TNNERR_LAYER_ERR, "Unary layer got undefined op");
    }

    Blob *input_blob  = inputs[0];
    Blob *output_blob = outputs[0];
    int count         = DimsVectorUtils::Count(output_blob->GetBlobDesc().dims);
    if (output_blob->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
        float *input_data  = static_cast<float *>(input_blob->GetHandle().base);
        float *output_data = static_cast<float *>(output_blob->GetHandle().base);
        for (int index = 0; index < count; ++index) {
            output_data[index] = (*op_)(input_data[index]);
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

}  // namespace TNN_NS
