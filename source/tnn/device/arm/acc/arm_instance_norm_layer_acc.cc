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
#include "tnn/device/arm/arm_device.h"
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS {

DECLARE_ARM_ACC(InstanceNorm, LAYER_INST_BATCH_NORM);

Status ArmInstanceNormLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto layer_res = dynamic_cast<InstanceNormLayerResource *>(resource_);
    if (!layer_res) {
        return Status(TNNERR_MODEL_ERR, "Error: layer resource is nil");
    }

    auto desc    = outputs[0]->GetBlobDesc();
    int batch    = desc.dims[0];
    int channels = desc.dims[1];
    int c_r4     = ROUND_UP(channels, 4);
    int area     = DimsVectorUtils::Count(desc.dims, 2);

    float *k_data = layer_res->scale_handle.force_to<float *>();
    float *b_data = layer_res->bias_handle.force_to<float *>();

    if (desc.data_type == DATA_TYPE_FLOAT) {
        if (channels != c_r4) {
            float *tmp = new float[c_r4];
            memcpy(tmp, k_data, channels * sizeof(float));
            memset(tmp + channels, 0, (c_r4 - channels) * sizeof(float));
            k_data = tmp;
            if (b_data) {
                tmp = new float[c_r4];
                memcpy(tmp, b_data, channels * sizeof(float));
                memset(tmp + channels, 0, (c_r4 - channels) * sizeof(float));
                b_data = tmp;
            }
        }
        float *input_data  = reinterpret_cast<float *>(GetBlobHandlePtr(inputs[0]->GetHandle()));
        float *output_data = reinterpret_cast<float *>(GetBlobHandlePtr(outputs[0]->GetHandle()));

        for (int b = 0; b < batch; b++) {
            for (int c = 0; c < c_r4; c += 4) {
                Float4 sum(0.f);
                auto input_c  = input_data + b * c_r4 * area + c * area;
                auto output_c = output_data + b * c_r4 * area + c * area;
                for (int hw = 0; hw < area; ++hw) {
                    auto v = Float4::load(input_c + hw * 4);
                    sum    = sum + v;
                }
                Float4 mean = Float4::div(sum, area);
                Float4 sum2(0.f);
                for (int hw = 0; hw < area; ++hw) {
                    auto v = Float4::load(input_c + hw * 4) - mean;
                    sum2   = sum2 + v * v;
                }
                auto variance = Float4::div(sum2, area);
                Float4 k      = Float4::load(k_data + c);
                variance      = Float4::div(1.0f, Float4::sqrt(variance + Float4(0.00001f)));
                variance      = variance * k;

                Float4 b = b_data ? Float4::load(b_data + c) : Float4(0.f);
                b        = b - mean * variance;

                for (int hw = 0; hw < area; ++hw) {
                    Float4::save(output_c + hw * 4, Float4::load(input_c + hw * 4) * variance + b);
                }
            }
        }
        if (channels != c_r4) {
            delete[] k_data;
            if (b_data)
                delete[] b_data;
        }
    } else {
        LOGE("Error: layer acc dont support datatype: %d\n", desc.data_type);
        return Status(TNNERR_MODEL_ERR, "Error: layer acc dont support datatype");
    }

    return TNN_OK;
}

REGISTER_ARM_ACC(InstanceNorm, LAYER_INST_BATCH_NORM);

}  // namespace TNN_NS
