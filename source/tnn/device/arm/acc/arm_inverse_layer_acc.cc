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
#include "tnn/utils/omp_utils.h"

namespace TNN_NS {

DECLARE_ARM_ACC(Inverse, LAYER_INVERSE);

Status ArmInverseLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    const auto input_blob = inputs[0];
    auto input_dims       = input_blob->GetBlobDesc().dims;
    if (input_dims.size() < 2) {
        return Status(TNNERR_PARAM_ERR, "CpuInverseLayerAcc has invalid input dims");
    }
    if ((input_dims[input_dims.size() - 1] != 2) || (input_dims[input_dims.size() - 2] != 2)) {
        LOGE("ArmInverseLayerAcc now only support inverse of matrix batchx2x2\n");
        return Status(TNNERR_UNSUPPORT_NET, "ArmInverseLayerAcc now only support inverse of matrix batchx2x2\n");
    }
    if (input_blob->GetBlobDesc().data_type != DATA_TYPE_FLOAT) {
        LOGE("ArmInverseLayerAcc now only support float data type\n");
        return Status(TNNERR_UNSUPPORT_NET,"ArmInverseLayerAcc now only support float data type\n");
    }
    float *input_data  = reinterpret_cast<float *>(GetBlobHandlePtr(inputs[0]->GetHandle()));
    float *output_data = reinterpret_cast<float *>(GetBlobHandlePtr(outputs[0]->GetHandle()));

    const int batch = DimsVectorUtils::Count(input_dims, 0, (int)input_dims.size() - 2);
    for (int b = 0; b < batch; b++) {
        float det         = input_data[0] * input_data[3] - input_data[1] * input_data[2];
        float det_inverse = 1.0f / det;

        output_data[0] = input_data[3] * det_inverse;
        output_data[1] = -input_data[1] * det_inverse;
        output_data[2] = -input_data[2] * det_inverse;
        output_data[3] = input_data[0] * det_inverse;

        input_data += 4;
        output_data += 4;
    }
    return TNN_OK;
}

REGISTER_ARM_ACC(Inverse, LAYER_INVERSE);
REGISTER_ARM_LAYOUT(LAYER_INVERSE, DATA_FORMAT_NCHW);
}  // namespace TNN_NS