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
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

DECLARE_ARM_ACC(Cast, LAYER_CAST);

Status ArmCastLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    const auto param = dynamic_cast<CastLayerParam*>(param_);
    void *input_data = inputs[0]->GetHandle().base;
    auto input_data_type = inputs[0]->GetBlobDesc().data_type;
    void *output_data = outputs[0]->GetHandle().base;
    auto output_data_type = outputs[0]->GetBlobDesc().data_type;
    
    const int ele_size = DataTypeUtils::GetBytesSize(outputs[0]->GetBlobDesc().data_type);
    
    int count = DimsVectorUtils::Count(outputs[0]->GetBlobDesc().dims);
    if (outputs[0]->GetBlobDesc().data_format == DATA_FORMAT_NC4HW4 ||
        outputs[0]->GetBlobDesc().data_format == DATA_FORMAT_NC8HW8) {
        DimsVector output_dims = outputs[0]->GetBlobDesc().dims;
        int channel = 1;
        if (output_dims.size() > 1) {
            channel = output_dims[2];
        }
        if (output_data_type == DATA_TYPE_FLOAT ||
            output_data_type == DATA_TYPE_BFP16) {
            count = count / channel;
            count = count * ROUND_UP(channel, 4);
        } else if (output_data_type == DATA_TYPE_HALF) {
            count = count / channel;
            count = count * ROUND_UP(channel, 8);
        }
    }

    if (input_data_type == output_data_type) {
        if (output_data != input_data) {
            memcpy(output_data, input_data, count * ele_size);
        }
    } else {
        LOGE("unsupport data type to cast\n");
    }
    return TNN_OK;
}

REGISTER_ARM_ACC(Cast, LAYER_CAST);
REGISTER_ARM_PRECISION_FP16(LAYER_CAST)
REGISTER_ARM_LAYOUT(LAYER_CAST, DATA_FORMAT_NC4HW4)
REGISTER_ARM_LAYOUT(LAYER_CAST, DATA_FORMAT_NCHW)

}  // namespace TNN_NS
