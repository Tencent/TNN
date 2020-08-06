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

#include "cpu_detection_post_process_layer_acc.h"

#include "tnn/utils/detection_post_process_utils.h"

namespace TNN_NS {

CpuDetectionPostProcessLayerAcc::~CpuDetectionPostProcessLayerAcc(){};

Status CpuDetectionPostProcessLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CpuDetectionPostProcessLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto param    = dynamic_cast<DetectionPostProcessLayerParam *>(param_);
    auto resource = dynamic_cast<DetectionPostProcessLayerResource *>(resource_);
    if (!param || !resource) {
        return Status(TNNERR_MODEL_ERR, "Error: ConvLayerParam or ConvLayerResource is empty");
    }
    CenterSizeEncoding scale_values;
    scale_values.y = param->center_size_encoding[0];
    scale_values.x = param->center_size_encoding[1];
    scale_values.h = param->center_size_encoding[2];
    scale_values.w = param->center_size_encoding[3];
    BlobDesc decode_boxes_desc;
    decode_boxes_desc.dims = {inputs[0]->GetBlobDesc().dims[2], 4, 1, 1};
    Blob decode_boxes_blob = Blob(decode_boxes_desc, true);
    DecodeBoxes(param, resource, inputs[0], scale_values, &decode_boxes_blob);

    if (param->use_regular_nms) {
        return TNNERR_UNSUPPORT_NET;
    } else {
        NonMaxSuppressionMultiClassFastImpl(param, resource, &decode_boxes_blob, inputs[1], outputs[0], outputs[1],
                                            outputs[2], outputs[3]);
    }
    return TNN_OK;
}

CpuTypeLayerAccRegister<TypeLayerAccCreator<CpuDetectionPostProcessLayerAcc>>
    g_cpu_detection_post_process_layer_acc_register(LAYER_DETECTION_POST_PROCESS);
}  // namespace TNN_NS
