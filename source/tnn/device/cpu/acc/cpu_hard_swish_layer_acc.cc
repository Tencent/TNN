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
#include "tnn/utils/naive_compute.h"
#include "tnn/device/cpu/acc/cpu_layer_acc.h"
#include "tnn/device/cpu/cpu_device.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

DECLARE_CPU_ACC(HardSwish, LAYER_HARDSWISH);

Status CpuHardSwishLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CpuHardSwishLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto layer_param = dynamic_cast<HardSwishLayerParam *>(param_);
    if (!layer_param) {
        LOGE("Error: HardSwishLayerParam is nil\n");
        return Status(TNNERR_MODEL_ERR, "Error: HardSwishLayerParam is nil");
    }
    const float alpha = layer_param->alpha;
    const float beta  = layer_param->beta;
    const float minV  = -beta / alpha;
    const float maxV  = (1.0f - beta) / alpha;
    LOGD("alpha: %.6f  beta: %.6f\n", alpha, beta);
    LOGD("minV: %.6f  maxV: %.6f\n", minV, maxV);

    Blob *input_blob_0 = inputs[0];
    Blob *input_blob_1 = inputs.size() > 1 ? inputs[1] : input_blob_0;
    Blob *output_blob  = outputs[0];
    // TO-DO : support input size is 1 and resource is not null

    auto shape_input_0     = input_blob_0->GetBlobDesc().dims;
    auto shape_input_1     = input_blob_1->GetBlobDesc().dims;
    auto shape_output      = output_blob->GetBlobDesc().dims;
    // TODO: refactor later
    while(shape_input_0.size() < 4)
        shape_input_0.push_back(1);
    while(shape_input_1.size() < 4)
        shape_input_1.push_back(1);
    while(shape_output.size() < 4)
        shape_output.push_back(1);

    const int batch        = shape_output[0];
    const int channel      = shape_output[1];
    const int height       = DimsFunctionUtils::GetDim(shape_output, 2);
    const int width        = DimsFunctionUtils::GetDim(shape_output, 3);
    const int channel_size = height * width;

    const int input0_dim2 = DimsFunctionUtils::GetDim(shape_input_0, 2);
    const int input0_dim3 = DimsFunctionUtils::GetDim(shape_input_0, 3);
    const int input1_dim2 = DimsFunctionUtils::GetDim(shape_input_1, 2);
    const int input1_dim3 = DimsFunctionUtils::GetDim(shape_input_1, 3);

    // y =  x0 * clip(x1*alpha + beta, 0, 1)
    if (output_blob->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
        float *input_data_0 = static_cast<float *>(input_blob_0->GetHandle().base);
        float *input_data_1 = static_cast<float *>(input_blob_1->GetHandle().base);
        float *output_data  = static_cast<float *>(output_blob->GetHandle().base);

        for (int b = 0; b < batch; b++) {
            int output_index_b = b * channel * channel_size;

            int input_index_b_0 =
                std::min(b, shape_input_0[0] - 1) * shape_input_0[1] * input0_dim2 * input0_dim3;
            int input_index_b_1 =
                std::min(b, shape_input_1[0] - 1) * shape_input_1[1] * input1_dim2 * input1_dim3;
            //            OMP_PARALLEL_FOR_
            for (int c = 0; c < channel; c++) {
                int output_index_c = c * channel_size + output_index_b;

                int input_index_c_0 =
                    std::min(c, shape_input_0[1] - 1) * input0_dim2 * input0_dim3 + input_index_b_0;
                int input_index_c_1 =
                    std::min(c, shape_input_1[1] - 1) * input1_dim2 * input1_dim3 + input_index_b_1;

                for (int h = 0; h < height; h++) {
                    int output_index_h = h * width + output_index_c;

                    int input_index_h_0 = std::min(h, input0_dim2 - 1) * input0_dim3 + input_index_c_0;
                    int input_index_h_1 = std::min(h, input1_dim2 - 1) * input1_dim3 + input_index_c_1;
                    for (int w = 0; w < width; w++) {
                        int input_index_w_0 = std::min(w, input0_dim3 - 1) + input_index_h_0;
                        int input_index_w_1 = std::min(w, input1_dim3 - 1) + input_index_h_1;

                        float temp0                     = input_data_0[input_index_w_0];
                        float temp1                     = input_data_1[input_index_w_1] * alpha + beta;
                        output_data[output_index_h + w] = temp0 * std::max(std::min(temp1, 1.0f), 0.0f);
                    }
                }
            }
        }
    } else {
        return Status(TNNERR_PARAM_ERR, "Error: CpuHardSwishLayerAcc datatype not support ");
    }
    return TNN_OK;
}

REGISTER_CPU_ACC(HardSwish, LAYER_HARDSWISH);

}  // namespace TNN_NS
