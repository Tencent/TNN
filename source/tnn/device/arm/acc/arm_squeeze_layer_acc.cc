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

#include "tnn/device/arm/acc/arm_layer_acc.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS {

DECLARE_ARM_ACC(Squeeze, LAYER_SQUEEZE);

Status ArmSqueezeLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto input       = inputs[0];
    auto output      = outputs[0];
    auto input_data  = GetBlobHandlePtr(input->GetHandle());
    auto output_data = GetBlobHandlePtr(output->GetHandle());
    auto count       = DimsVectorUtils::Count(output->GetBlobDesc().dims);
    auto ele_size    = DataTypeUtils::GetBytesSize(output->GetBlobDesc().data_type);

    if (input_data != output_data) {
        memcpy(output_data, input_data, count * ele_size);
    }

    return TNN_OK;
}

REGISTER_ARM_ACC(Squeeze, LAYER_SQUEEZE);
REGISTER_ARM_PRECISION_FP16(LAYER_SQUEEZE);
REGISTER_ARM_LAYOUT(LAYER_SQUEEZE, DATA_FORMAT_NCHW)

}  // namespace TNN_NS
