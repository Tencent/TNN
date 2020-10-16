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

#include "tnn/device/x86/acc/x86_binary_op_layer_acc.h"

#include "tnn/core/blob_int8.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/device/x86/acc/compute/x86_compute.h"
#include <iostream>

namespace TNN_NS {
X86BinaryOpLayerAcc::~X86BinaryOpLayerAcc() {}

Status X86BinaryOpLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                              const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    RETURN_ON_NEQ(X86LayerAcc::Init(context, param, resource, inputs, outputs), TNN_OK);
    return op_->Init(param);
}

Status X86BinaryOpLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status X86BinaryOpLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto layer_param = dynamic_cast<MultidirBroadcastLayerParam *>(param_);
    if (!layer_param) {
        LOGE("Error: layer param is nil\n");
        return Status(TNNERR_PARAM_ERR, "Error: layer param is nil");
    }
    auto layer_res = dynamic_cast<EltwiseLayerResource *>(resource_);
    if (!((inputs.size() == 1 && layer_res) || inputs.size() >= 2)) {
        LOGE("Error: X86BinaryLayerAcc invalid inputs count\n");
        return Status(TNNERR_LAYER_ERR, "X86BinaryLayerAcc invalid inputs count");
    }

    auto output = outputs[0];
    auto dims   = output->GetBlobDesc().dims;
    std::vector<void *> input_ptrs;
    std::vector<DimsVector> input_shapes;

    if (inputs.size() >= 2) {
        for (size_t inid = 0; inid < inputs.size(); inid++) {
            input_ptrs.push_back(inputs[inid]->GetHandle().base);
            input_shapes.push_back(inputs[inid]->GetBlobDesc().dims);
        }
    } else {
        DimsVector input_shape0 = inputs[0]->GetBlobDesc().dims;
        if (layer_param->weight_input_index == 0) {
            input_ptrs.push_back(layer_res->element_handle.force_to<void *>());
            input_shapes.push_back(layer_res->element_shape);

            input_ptrs.push_back(inputs[0]->GetHandle().base);
            input_shapes.push_back(input_shape0);
        } else {
            input_ptrs.push_back(inputs[0]->GetHandle().base);
            input_shapes.push_back(input_shape0);

            input_ptrs.push_back(layer_res->element_handle.force_to<void *>());
            input_shapes.push_back(layer_res->element_shape);
        }
    }
    Status status = Calculate(inputs, input_ptrs, input_shapes, output);
    return status;
}

Status X86BinaryOpLayerAcc::Calculate(const std::vector<Blob *> &input_blobs, const std::vector<void *> &input_ptrs,
                                      const std::vector<DimsVector> &input_shapes, Blob *output){
    if (output->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
        return X86_BINARY_CALCULATE(input_ptrs, input_shapes, output, op_);
    } else {
        LOGE("Error: X86AddLayerAcc don't support data type: %d\n", output->GetBlobDesc().data_type);
        return Status(TNNERR_MODEL_ERR, "Error: X86AddLayerAcc don't support data type");
    }
}

}  // namespace TNN_NS