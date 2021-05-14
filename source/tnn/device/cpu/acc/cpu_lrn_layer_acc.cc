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
// specific language governing permissions and limitations under the License./

#include "tnn/utils/naive_compute.h"
#include "tnn/device/cpu/acc/cpu_layer_acc.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

DECLARE_CPU_ACC(LRN, LAYER_LRN);

Status CpuLRNLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CpuLRNLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto param = dynamic_cast<LRNLayerParam *>(param_);
    if (!param) {
        return Status(TNNERR_MODEL_ERR, "Error: LRNLayerParam is empyt");
    }
    float alpha = param->alpha;
    float beta  = param->beta;
    float bias  = param->bias;
    int size    = param->size;

    Blob *input_blob   = inputs[0];
    Blob *output_blob  = outputs[0];
    float *input_data  = static_cast<float *>(input_blob->GetHandle().base);
    float *output_data = static_cast<float *>(output_blob->GetHandle().base);

    int batch   = output_blob->GetBlobDesc().dims[0];
    int channel = output_blob->GetBlobDesc().dims[1];
    int height  = output_blob->GetBlobDesc().dims[2];
    int width   = output_blob->GetBlobDesc().dims[3];
    int count   = DimsVectorUtils::Count(output_blob->GetBlobDesc().dims, 0);

    auto square_data = (float *)calloc(count, sizeof(float));
    if (square_data == nullptr) {
        LOGE("Error: calloc square data failed\n");
        return Status(TNNERR_MODEL_ERR, "Error: calloc square data failed!");
    }
    for (int n = 0; n < batch; ++n) {
        for (int c = 0; c < channel; ++c) {
            // max(0, c - int(math.floor((nsize - 1) / 2)))
            // :
            // min(C-1, c + int(math.ceil((nsize - 1) / 2)) + 1),
            int begin = std::max(0, c - int(std::floor((size - 1) / 2)));
            int end   = std::min(channel, c + int(std::ceil((size - 1) / 2)) + 1);
            for (int i = begin; i < end; ++i) {
                int count = height * width;
                for (int j = 0; j < count; ++j) {
                    int input_index = n * channel * count + i * count + j;
                    int index       = n * channel * count + c * count + j;
                    // square
                    square_data[index] += std::pow(input_data[input_index], 2);
                }
            }
        }
    }
    // y = x / ((bias + (alpha / nsize) * square_sum) ** beta)
    for (int n = 0; n < batch; ++n) {
        for (int c = 0; c < channel; ++c) {
            int count = height * width;
            for (int i = 0; i < count; ++i) {
                int index        = n * channel * count + c * count + i;
                int output_index = index;
                output_data[output_index] =
                    input_data[index] / (std::pow(bias + (alpha / float(size)) * square_data[index], beta));
            }
        }
    }
    free(square_data);

    return TNN_OK;
}  // namespace TNN_NS

REGISTER_CPU_ACC(LRN, LAYER_LRN);

}  // namespace TNN_NS
