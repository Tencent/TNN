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
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

DECLARE_CPU_ACC_WITH_FUNC(Squeeze, LAYER_SQUEEZE,
                          virtual Status InferRuntimeOutputShape(const std::vector<Blob *> &inputs,
                                                                 const std::vector<Blob *> &outputs););

Status CpuSqueezeLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CpuSqueezeLayerAcc::InferRuntimeOutputShape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto *layer_param = dynamic_cast<SqueezeLayerParam *>(param_);
    CHECK_PARAM_NULL(layer_param);

    DimsVector input_dims  = inputs[0]->GetBlobDesc().dims;
    RETURN_VALUE_ON_NEQ(input_dims.size() > 0, true, Status(TNNERR_PARAM_ERR, "SqueezeLayer has invalid inpu    t size"));

    std::vector<int> axes  = layer_param->axes;

    if (!axes.empty()) {
        DimsVector output_dims = input_dims;
        for (auto iter = axes.rbegin(); iter != axes.rend(); iter++) {
            int axis = *iter;
            axis =  axis < 0 ? axis + (int)output_dims.size() : axis;
            if (axis < 0 || axis >= output_dims.size() || output_dims[axis] != 1) {
                return Status(TNNERR_PARAM_ERR, "SqueezeLayer has invalid input axes");
            }
            output_dims.erase(output_dims.begin() + axis);
        }
        outputs[0]->GetBlobDesc().dims = output_dims;
        return TNN_OK;
    } else {
        // axes is empty, this may occur in pytorch
        // https://pytorch.org/docs/stable/generated/torch.squeeze.html?highlight=squeeze#torch.squeeze
        // This Squeeze may be dangerous, pytorch has the following warning:
        // If the tensor has a batch dimension of size 1, then squeeze(input) will also remove the batch dimension, which can lead to unexpected errors.
        DimsVector output_dims = {};
        for (int i=0; i<input_dims.size(); i++) {
            if (input_dims[i] == 1) {
                axes.push_back(i);
            } else {
                output_dims.push_back(input_dims[i]);
            }
        }
        if (output_dims.empty()) {
            output_dims.push_back(0);
        }
        layer_param->axes = axes;
        outputs[0]->GetBlobDesc().dims = output_dims;
        return TNN_OK;
    }

    return TNN_OK;
}

Status CpuSqueezeLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    void *input_data  = inputs[0]->GetHandle().base;
    void *output_data = outputs[0]->GetHandle().base;
    auto input_dims   = outputs[0]->GetBlobDesc().dims;
    auto count        = DimsVectorUtils::Count(input_dims);
    auto ele_size     = DataTypeUtils::GetBytesSize(outputs[0]->GetBlobDesc().data_type);

    if (input_data != output_data) {
        memcpy(output_data, input_data, count * ele_size);
    }

    return TNN_OK;
}

REGISTER_CPU_ACC(Squeeze, LAYER_SQUEEZE);
}  // namespace TNN_NS
