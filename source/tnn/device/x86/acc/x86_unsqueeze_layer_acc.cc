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
#include "tnn/utils/dims_utils.h"
#include "tnn/utils/data_type_utils.h"

namespace TNN_NS {

DECLARE_X86_ACC(Unsqueeze, LAYER_UNSQUEEZE);

Status X86UnsqueezeLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    void *input_data  = handle_ptr<void*>(inputs[0]->GetHandle());
    void *output_data = handle_ptr<void*>(outputs[0]->GetHandle());
    auto dims         = outputs[0]->GetBlobDesc().dims;
    auto count        = DimsVectorUtils::Count(dims);
    auto ele_size     = DataTypeUtils::GetBytesSize(outputs[0]->GetBlobDesc().data_type);

    if (input_data != output_data) {
        memcpy(output_data, input_data, count * ele_size);
    }

    return TNN_OK;
}

REGISTER_X86_ACC(Unsqueeze, LAYER_UNSQUEEZE);
}  // namespace TNN_NS
