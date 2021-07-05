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

#include "tnn/device/arm/acc/arm_nchw_layer_acc.h"
#include "tnn/utils/naive_compute.h"

namespace TNN_NS {

DECLARE_ARM_NCHW_ACC(DetectionOutput, LAYER_DETECTION_OUTPUT);

Status ArmDetectionOutputLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto param = dynamic_cast<DetectionOutputLayerParam *>(param_);
    CHECK_PARAM_NULL(param);

    AllocConvertBuffer(inputs, outputs);

    // call cpu naive prior box
    if (outputs[0]->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
        UnPackInputs<float>(inputs);
        NaiveDetectionOutput(GetNchwBlobVector(nchw_blob_in), GetNchwBlobVector(nchw_blob_out), param);
        PackOutputs<float>(outputs);
    } else {
        return Status(TNNERR_LAYER_ERR, "NO IMPLEMENT data type");
    }

    return TNN_OK;
}

REGISTER_ARM_ACC(DetectionOutput, LAYER_DETECTION_OUTPUT)
REGISTER_ARM_LAYOUT(LAYER_DETECTION_OUTPUT, DATA_FORMAT_NC4HW4)

}  // namespace TNN_NS
