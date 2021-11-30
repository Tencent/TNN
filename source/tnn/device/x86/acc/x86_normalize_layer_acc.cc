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
#include "tnn/device/x86/x86_device.h"

#include <limits.h>
#include "tnn/utils/naive_compute.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

DECLARE_X86_ACC(Normalize, X86_NORMALIZE_OP);

bool X86CheckNormalizeLayerParam(const int p, const int axis, const int across_spatial) {
    return (p != 1 && p != 2 && p != INT_MAX && p != INT_MIN) || axis != 1 || across_spatial != 0;
}

Status X86NormalizeLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    if (inputs.size() < 1) {
        LOGE("Error: invalid inputs count\n");
        return Status(TNNERR_LAYER_ERR, "layer's inputs size must >= 2");
    }
    auto layer_param = dynamic_cast<NormalizeLayerParam *>(param_);
    if (!layer_param) {
        LOGE("Error: layer param is nil\n");
        return Status(TNNERR_MODEL_ERR, "Error: layer param is nil");
    }

    float epsilon = layer_param->epsilon;
    int axis      = layer_param->axis;
    int p         = layer_param->p;

    int across_spatial = layer_param->across_spatial;
    int channel_shared = layer_param->channel_shared;

    // old tnn support scale the result of normalize and only norm2
    if (X86CheckNormalizeLayerParam(p, axis, across_spatial)) {
        LOGE("Error: layer param is not supported now\n");
        return Status(TNNERR_INST_ERR, "Error: layer param is not supported now");
    }

    Blob *input_blob  = inputs[0];
    Blob *output_blob = outputs[0];
    auto output_dims  = output_blob->GetBlobDesc().dims;
    int batch         = output_dims[0];
    int channel       = output_dims[1];
    int channel_size  = DimsVectorUtils::Count(output_blob->GetBlobDesc().dims, 2);
    if (output_blob->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
        float *input_data  = static_cast<float *>(input_blob->GetHandle().base);
        float *output_data = static_cast<float *>(output_blob->GetHandle().base);

        float *denominator = reinterpret_cast<float *>(context_->GetSharedWorkSpace(channel_size * sizeof(float)));

        for (int b = 0; b < batch; b++) {
            memset(denominator, 0, channel_size * sizeof(float));
            float *input_data_b  = input_data + b * channel * channel_size;
            float *output_data_b = output_data + b * channel * channel_size;
            int start_channel    = 0;
            if (layer_param->p == INT_MAX || layer_param->p == INT_MIN) {
                memcpy(denominator, input_data_b, channel_size * sizeof(float));
                start_channel = 1;
            }

            for (int c = start_channel; c < channel; c++) {
                float *input_data_c = input_data_b + c * channel_size;
                for (int index = 0; index < channel_size; index++) {
                    if (layer_param->p == 1) {
                        // sum - abs(x)
                        denominator[index] += fabs(input_data_c[index]);
                    } else if (layer_param->p == 2) {
                        // sum - x*x
                        denominator[index] += input_data_c[index] * input_data_c[index];
                    } else if (layer_param->p == INT_MAX) {
                        denominator[index] = std::max(denominator[index], input_data_c[index]);
                    } else if (layer_param->p == INT_MIN) {
                        denominator[index] = std::min(denominator[index], input_data_c[index]);
                    }
                }
            }

            if (layer_param->p == 2) {
                // max - sqrt
                for (int index = 0; index < channel_size; index++) {
                    denominator[index] = std::max((float)sqrt(denominator[index]), epsilon);
                }
            }

            // div
            for (int c = 0; c < channel; c++) {
                float *input_data_c  = input_data_b + c * channel_size;
                float *output_data_c = output_data_b + c * channel_size;
                for (int index = 0; index < channel_size; index++) {
                    output_data_c[index] = input_data_c[index] / denominator[index];
                }
            }
        }
    } else {
        LOGE("Error: layer acc dont support datatype: %d\n", output_blob->GetBlobDesc().data_type);
        return Status(TNNERR_MODEL_ERR, "Error: layer acc dont support datatype");
    }

    return TNN_OK;
}

REGISTER_X86_ACC(Normalize, LAYER_NORMALIZE);

}  // namespace TNN_NS
