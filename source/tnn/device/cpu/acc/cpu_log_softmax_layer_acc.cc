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

#include <algorithm>
#include <cmath>

#include "tnn/device/cpu/acc/cpu_layer_acc.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

DECLARE_CPU_ACC(LogSoftMax, LAYER_LOGSOFTMAX);

Status CpuLogSoftMaxLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CpuLogSoftMaxLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto params = dynamic_cast<LogSoftmaxLayerParam *>(param_);

    if (!params) {
        LOGE("Error: LogSoftmaxLayerParam is unsupported\n");
        return Status(TNNERR_MODEL_ERR, "Error: LogSoftmaxLayerParam is unsupported");
    }

    Blob *input_blob   = inputs[0];
    Blob *output_blob  = outputs[0];
    float *input_data  = static_cast<float *>(input_blob->GetHandle().base);
    float *output_data = static_cast<float *>(output_blob->GetHandle().base);
    auto dims          = input_blob->GetBlobDesc().dims;
    int axis           = params->axis;
    axis               = static_cast<int>((axis + dims.size()) % dims.size());
    int batch          = DimsVectorUtils::Count(dims, 0, axis);
    int channel        = dims[axis];
    int count          = DimsVectorUtils::Count(dims, axis + 1);

    float *const temp = new float[count];

    for (int n = 0; n < batch; n++) {
        float *const input_batch  = input_data + n * channel * count;
        float *const output_batch = output_data + n * channel * count;
        // max
        memcpy(temp, input_batch, count * sizeof(float));
        for (int c = 1; c < channel; c++) {
            float *input_channel = input_batch + c * count;
            for (int ele = 0; ele < count; ele++) {
                temp[ele] = std::max(temp[ele], input_channel[ele]);
            }
        }

        // exp
        for (int c = 0; c < channel; c++) {
            float *input_channel  = input_batch + c * count;
            float *output_channel = output_batch + c * count;

            for (int ele = 0; ele < count; ele++) {
                output_channel[ele] = expf(input_channel[ele] - temp[ele]);
            }
        }

        // sum
        memcpy(temp, output_batch, count * sizeof(float));
        for (int c = 1; c < channel; c++) {
            float *output_channel = output_batch + c * count;
            for (int ele = 0; ele < count; ele++) {
                temp[ele] += output_channel[ele];
            }
        }

        // division
        for (int ele = 0; ele < count; ele++) {
            temp[ele] = 1.0f / temp[ele];
        }
        for (int c = 0; c < channel; c++) {
            float *output_channel = output_batch + c * count;
            for (int ele = 0; ele < count; ele++) {
                output_channel[ele] *= temp[ele];
            }
        }

        // log
        for (int c = 0; c < channel; c++) {
            float *output_channel = output_batch + c * count;
            for (int ele = 0; ele < count; ele++) {
                output_channel[ele] = log(output_channel[ele]);
            }
        }
    }

    delete[] temp;
    return TNN_OK;
}

REGISTER_CPU_ACC(LogSoftMax, LAYER_LOGSOFTMAX);

}  // namespace TNN_NS
