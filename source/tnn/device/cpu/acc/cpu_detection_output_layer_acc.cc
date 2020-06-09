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

#include "tnn/device/cpu/acc/cpu_detection_output_layer_acc.h"

#include <algorithm>
#include <cmath>

#include "tnn/device/cpu/acc/compute/normalized_bbox.h"
#include "tnn/utils/bbox_util.h"

namespace TNN_NS {

CpuDetectionOuputLayerAcc::~CpuDetectionOuputLayerAcc(){};

Status CpuDetectionOuputLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CpuDetectionOuputLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    DetectionOutputLayerParam *param = dynamic_cast<DetectionOutputLayerParam *>(param_);
    CHECK_PARAM_NULL(param);
    NaiveDetectionOutput(inputs, outputs, param);
    return TNN_OK;
}

CpuTypeLayerAccRegister<TypeLayerAccCreator<CpuDetectionOuputLayerAcc>> g_cpu_detection_output_layer_acc_register(
    LAYER_DETECTION_OUTPUT);

}  // namespace TNN_NS
