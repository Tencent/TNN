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

#include "tnn/utils/data_format_converter.h"
#include "tnn/utils/detection_post_process_utils.h"
#include "tnn/utils/dims_utils.h"

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
    if (param->use_regular_nms) {
        return TNNERR_UNSUPPORT_NET;
    }
    Blob* nhwc_input0 = new Blob(inputs[0]->GetBlobDesc(), true);
    DataFormatConverter::ConvertFromNCHWToNHWC<float>(inputs[0], nhwc_input0);
    nhwc_input0->GetBlobDesc().dims = DimsVectorUtils::NCHW2NHWC(nhwc_input0->GetBlobDesc().dims);
    Blob* nhwc_input1 = new Blob(inputs[1]->GetBlobDesc(), true);
    DataFormatConverter::ConvertFromNCHWToNHWC<float>(inputs[1], nhwc_input1);
    nhwc_input1->GetBlobDesc().dims = DimsVectorUtils::NCHW2NHWC(nhwc_input1->GetBlobDesc().dims);
    CenterSizeEncoding scale_values;
    scale_values.y = param->center_size_encoding[0];
    scale_values.x = param->center_size_encoding[1];
    scale_values.h = param->center_size_encoding[2];
    scale_values.w = param->center_size_encoding[3];
    BlobDesc decode_boxes_desc;
    decode_boxes_desc.dims = {nhwc_input0->GetBlobDesc().dims[1], 4, 1, 1};
    Blob decode_boxes_blob = Blob(decode_boxes_desc, true);
    DecodeBoxes(param, resource, nhwc_input0, scale_values, &decode_boxes_blob);

    NonMaxSuppressionMultiClassFastImpl(param, resource, &decode_boxes_blob, nhwc_input1, outputs[0], outputs[1],
                                        outputs[2], outputs[3]);

    delete nhwc_input0;
    delete nhwc_input1;
    return TNN_OK;
}

CpuTypeLayerAccRegister<TypeLayerAccCreator<CpuDetectionPostProcessLayerAcc>>
    g_cpu_detection_post_process_layer_acc_register(LAYER_DETECTION_POST_PROCESS);
}  // namespace TNN_NS
