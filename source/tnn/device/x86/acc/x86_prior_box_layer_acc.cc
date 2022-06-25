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
// specific language governing permissions and limitations under the License./

#include "tnn/device/x86/acc/x86_prior_box_layer_acc.h"

#include "tnn/utils/pribox_generator_utils.h"

namespace TNN_NS {

X86PriorBoxLayerAcc::~X86PriorBoxLayerAcc(){};

Status X86PriorBoxLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto param = dynamic_cast<PriorBoxLayerParam *>(param_);
    if (!param) {
        return Status(TNNERR_MODEL_ERR, "Error: PriorBoxLayerParam is empyt");
    }

    Blob *output_blob  = outputs[0];
    void *output_data  = handle_ptr<void*>(output_blob->GetHandle());
    DataType data_type = output_blob->GetBlobDesc().data_type;

    // compute
    if (data_type == DATA_TYPE_FLOAT) {
        auto prior_box = GeneratePriorBox(inputs, outputs, param);
        memcpy(output_data, prior_box.data(), prior_box.size() * sizeof(float));
    } else {
        return Status(TNNERR_LAYER_ERR, "datatype not support");
    }
    return TNN_OK;
}

REGISTER_X86_ACC(PriorBox, LAYER_PRIOR_BOX)

}  // namespace TNN_NS
