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

#include "tnn/device/x86/acc/x86_unary_layer_acc.h"
#include "tnn/device/x86/x86_context.h"

namespace TNN_NS {

X86UnaryLayerAcc::~X86UnaryLayerAcc() {}

Status X86UnaryLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                              const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    RETURN_ON_NEQ(X86LayerAcc::Init(context, param, resource, inputs, outputs), TNN_OK);
    return op_->Init(param);
}

Status X86UnaryLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto input  = inputs[0];
    auto output = outputs[0];

    auto dims = output->GetBlobDesc().dims;

    // int count = dims[0] * ROUND_UP(dims[1], 4) * dims[2] * dims[3];
    int count = dims[0] * dims[1] * dims[2] * dims[3];
    auto input_data  = static_cast<float*>(input->GetHandle().base);
    auto output_data = static_cast<float*>(output->GetHandle().base);

    for (int n = 0; n < count; n++) {
        output_data[n] = (*op_)(input_data[n]);
    }

    return TNN_OK;
}

Status X86UnaryLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

}   // namespace TNN_NS