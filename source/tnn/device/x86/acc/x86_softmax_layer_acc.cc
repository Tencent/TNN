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

#include "tnn/device/x86/acc/Float4.h"
#include "tnn/device/x86/acc/Float8.h"
#include "tnn/device/x86/acc/x86_layer_acc.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

DECLARE_X86_ACC(SoftMax, LAYER_SOFTMAX);

template <typename VEC, int pack>
static void softmax_channel_func(float *input_ptr, float *output_ptr, int channel, int count, void *workspace) {
    float *temp = reinterpret_cast<float *>(workspace);

    temp[0] = input_ptr[0];
    // max
    int ele    = 0;
    auto v_max = VEC(temp[0]);
    float vec_buf[pack];
    for (; ele + pack - 1 < channel; ele += pack) {
        v_max = VEC::max(v_max, VEC::loadu(input_ptr + ele));
    }
    for (; ele < channel; ele++) {
        temp[0] = std::max(temp[0], input_ptr[ele]);
    }
    VEC::saveu(vec_buf, v_max);
    for (int i = 0; i < pack; i++) {
        temp[0] = std::max(temp[0], vec_buf[i]);
    }

    // exp
    ele = 0;
    for (; ele + pack - 1 < channel; ele += pack) {
        VEC::saveu(output_ptr + ele, VEC::exp(VEC::loadu(input_ptr + ele) - VEC(temp[0])));
    }
    for (; ele < channel; ele++) {
        output_ptr[ele] = expf(input_ptr[ele] - temp[0]);
    }

    // sum
    temp[0]    = 0.f;
    auto v_sum = VEC(0.f);
    ele        = 0;
    for (; ele + pack - 1 < channel; ele += pack) {
        v_sum = v_sum + VEC::loadu(output_ptr + ele);
    }
    for (; ele < channel; ele++) {
        temp[0] += output_ptr[ele];
    }
    VEC::saveu(vec_buf, v_sum);
    for (int i = 0; i < pack; i++) {
        temp[0] += vec_buf[i];
    }

    // division
    temp[0] = 1.f / temp[0];

    //
    ele = 0;
    for (; ele + pack - 1 < channel; ele += pack) {
        VEC::saveu(output_ptr + ele, VEC::loadu(output_ptr + ele) * temp[0]);
    }
    for (; ele < channel; ele++) {
        output_ptr[ele] *= temp[0];
    }
}

template <typename VEC, int pack>
static void softmax_func(float *input_ptr, float *output_ptr, int channel, int count, void *workspace) {
    if (count == 1) {
        return softmax_channel_func<VEC, pack>(input_ptr, output_ptr, channel, count, workspace);
    }

    float *temp = reinterpret_cast<float *>(workspace);
    // max
    memcpy(temp, input_ptr, count * sizeof(float));
    for (int c = 1; c < channel; c++) {
        float *input_channel = input_ptr + c * count;
        int ele              = 0;
        for (; ele + pack - 1 < count; ele += pack) {
            VEC::saveu(temp + ele, VEC::max(VEC::loadu(temp + ele), VEC::loadu(input_channel + ele)));
        }
        for (; ele < count; ele++) {
            temp[ele] = std::max(temp[ele], input_channel[ele]);
        }
    }

    // exp
    for (int c = 0; c < channel; c++) {
        float *input_channel  = input_ptr + c * count;
        float *output_channel = output_ptr + c * count;

        int ele = 0;
        for (; ele + pack - 1 < count; ele += pack) {
            VEC::saveu(output_channel + ele, VEC::exp(VEC::loadu(input_channel + ele) - VEC::loadu(temp + ele)));
        }
        for (; ele < count; ele++) {
            output_channel[ele] = expf(input_channel[ele] - temp[ele]);
        }
    }

    // sum
    memcpy(temp, output_ptr, count * sizeof(float));
    for (int c = 1; c < channel; c++) {
        float *output_channel = output_ptr + c * count;
        int ele               = 0;
        for (; ele + pack - 1 < count; ele += pack) {
            VEC::saveu(temp + ele, VEC::loadu(temp + ele) + VEC::loadu(output_channel + ele));
        }
        for (; ele < count; ele++) {
            temp[ele] += output_channel[ele];
        }
    }

    // division
    int ele = 0;
    for (; ele + pack - 1 < count; ele += pack) {
        VEC::saveu(temp + ele, VEC::div(VEC(1.f), VEC::loadu(temp + ele)));
    }
    for (; ele < count; ele++) {
        temp[ele] = 1.0f / temp[ele];
    }

    for (int c = 0; c < channel; c++) {
        float *output_channel = output_ptr + c * count;
        int ele               = 0;
        for (; ele + pack - 1 < count; ele += pack) {
            VEC::saveu(output_channel + ele, VEC::loadu(temp + ele) * VEC::loadu(output_channel + ele));
        }
        for (; ele < count; ele++) {
            output_channel[ele] *= temp[ele];
        }
    }
}

Status X86SoftMaxLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto params = dynamic_cast<SoftmaxLayerParam *>(param_);

    if (!params) {
        LOGE("Error: SoftmaxLayerParam is unsupported\n");
        return Status(TNNERR_MODEL_ERR, "Error: SoftmaxLayerParam is unsupported");
    }

    Blob *input_blob   = inputs[0];
    Blob *output_blob  = outputs[0];
    float *input_data  = handle_ptr<float *>(input_blob->GetHandle());
    float *output_data = handle_ptr<float *>(output_blob->GetHandle());
    auto dims          = input_blob->GetBlobDesc().dims;
    int axis           = params->axis;
    axis               = static_cast<int>((axis + dims.size()) % dims.size());
    int batch          = DimsVectorUtils::Count(dims, 0, axis);
    int channel        = dims[axis];
    int count          = DimsVectorUtils::Count(dims, axis + 1);

    auto workspace = context_->GetSharedWorkSpace(count * sizeof(float));

    auto func = softmax_func<Float8, 8>;
    if (arch_ == sse42) {
        func = softmax_func<Float4, 4>;
    }

    for (int n = 0; n < batch; n++) {
        float *const input_batch  = input_data + n * channel * count;
        float *const output_batch = output_data + n * channel * count;

        func(input_batch, output_batch, channel, count, workspace);
    }

    return TNN_OK;
}

REGISTER_X86_ACC(SoftMax, LAYER_SOFTMAX);

}  // namespace TNN_NS
