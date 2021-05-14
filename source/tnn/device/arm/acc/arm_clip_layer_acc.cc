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

#include <cmath>

#include "tnn/device/arm/acc/Float4.h"
#include "tnn/device/arm/acc/arm_layer_acc.h"
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

DECLARE_ARM_ACC(Clip, LAYER_CLIP);

Status ArmClipLayerAcc::DoForward(const std::vector<Blob *> &input_blobs, const std::vector<Blob *> &output_blobs) {
    auto layer_param = dynamic_cast<ClipLayerParam *>(param_);
    if (!layer_param) {
        LOGE("Error: layer param is nil\n");
        return Status(TNNERR_MODEL_ERR, "Error: layer param is nil");
    }

    auto input_blob  = input_blobs[0];
    auto output_blob = output_blobs[0];
    auto dims        = output_blob->GetBlobDesc().dims;
    int count        = dims[0] * ROUND_UP(dims[1], 4) * DimsVectorUtils::Count(dims, 2);
    int count_quad   = UP_DIV(count, 4);

    if (output_blob->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
        auto input_data  = reinterpret_cast<float *>(GetBlobHandlePtr(input_blob->GetHandle()));
        auto output_data = reinterpret_cast<float *>(GetBlobHandlePtr(output_blob->GetHandle()));

        for (int n = 0; n < count_quad; n++) {
            Float4::save(output_data + n * 4,
                         Float4::min(Float4(layer_param->max),
                                     Float4::max(Float4(layer_param->min), Float4::load(input_data + n * 4))));
        }
    } else if (output_blob->GetBlobDesc().data_type == DATA_TYPE_INT8) {
        LOGE("Error: layer acc dont support datatype: %d\n", output_blob->GetBlobDesc().data_type);
        return Status(TNNERR_MODEL_ERR, "Error: layer acc dont support datatype");
    } else {
        LOGE("Error: ArmClipLayerAcc dont support datatype: %d\n", output_blob->GetBlobDesc().data_type);
        return Status(TNNERR_MODEL_ERR, "Error: ArmClipLayerAcc dont support datatype");
    }
    return TNN_OK;
}

REGISTER_ARM_ACC(Clip, LAYER_CLIP);
REGISTER_ARM_LAYOUT(LAYER_CLIP, DATA_FORMAT_NC4HW4)

}  // namespace TNN_NS
