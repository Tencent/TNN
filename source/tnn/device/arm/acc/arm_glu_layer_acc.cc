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

#include "tnn/device/arm/acc/arm_glu_layer_acc.h"

#include "tnn/device/arm/acc/Float4.h"
#include "tnn/device/arm/acc/arm_layer_acc.h"
#include "tnn/device/arm/arm_common.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

Status ArmGLULayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                            const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return ArmLayerAcc::Init(context, param, resource, inputs, outputs);
}

template <class T>
Status ArmGLULayerAcc::Exec(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto &input_blob       = inputs[0];
    auto &output_blob      = outputs[0];
    const auto &input_dims = input_blob->GetBlobDesc().dims;
    const auto &param      = dynamic_cast<GLULayerParam *>(param_);
    const int axis         = param->axis;
    auto *input_ptr        = reinterpret_cast<float *>(GetBlobHandlePtr(input_blob->GetHandle()));
    auto *output_ptr       = reinterpret_cast<float *>(GetBlobHandlePtr(output_blob->GetHandle()));

    bool is_split_channel = false;
    if (axis == 1) {
        is_split_channel = true;
        if (k_param_->ic_r4 % 8 != 0) {
            LOGE("ArmGLULayerAcc does not support for now\n");
            return {TNNERR_UNSUPPORT_NET, "ArmGLULayerAcc does not support for now\n"};
        }
    }
    if (is_split_channel) {
        const int batch = input_dims[0];
        const int ic_r4 = k_param_->ic_r4;
        const int oc_r4 = k_param_->oc_r4;
        const int count = DimsVectorUtils::Count(input_dims, axis + 1);
        Float4 one_v    = Float4(1.0f);
        for (int b = 0; b < batch; ++b) {
            auto *input_batch  = input_ptr + b * ic_r4 * count;
            auto *output_batch = output_ptr + b * oc_r4 * count;
            for (int c = 0; c < oc_r4; c += 4) {
                auto *first_slice_ptr  = input_batch + c * count;
                auto *second_slice_ptr = input_batch + (c + oc_r4) * count;
                auto *output_slice_ptr = output_batch + c * count;
                for (int i = 0; i < count; ++i) {
                    Float4 b_v0 = Float4::load(second_slice_ptr + i * 4);
                    b_v0        = Float4::neg(b_v0);
                    b_v0        = Float4::exp(b_v0);
                    Float4 a_v0 = Float4::load(first_slice_ptr + i * 4);
                    b_v0        = b_v0 + one_v;
                    Float4 o_v0 = Float4::div(a_v0, b_v0);
                    Float4::save(output_slice_ptr + i * 4, o_v0);
                }
            }
        }
    } else {
        // split in hw
        Float4 one_v         = Float4(1.0f);
        const int batch      = input_dims[0] * k_param_->ic_r4 * DimsVectorUtils::Count(input_dims, 2, axis);
        const int split_dim  = input_dims[axis];
        const int count      = DimsVectorUtils::Count(input_dims, axis + 1);
        const int output_dim = split_dim / 2;
        for (int b = 0; b < batch; b += 4) {
            auto *input_batch  = input_ptr + b * split_dim * count;
            auto *output_batch = output_ptr + b * output_dim * count;
            for (int c = 0; c < output_dim; c += 1) {
                auto *first_slice_ptr  = input_batch + c * count * 4;
                auto *second_slice_ptr = input_batch + (c + output_dim) * count * 4;
                auto *output_slice_ptr = output_batch + c * count * 4;
                for (int i = 0; i < count; ++i) {
                    Float4 a_v0 = Float4::load(first_slice_ptr + i * 4);
                    Float4 b_v0 = Float4::load(second_slice_ptr + i * 4);
                    b_v0        = Float4::neg(b_v0);
                    b_v0        = Float4::exp(b_v0);
                    b_v0        = b_v0 + one_v;
                    Float4 o_v0 = Float4::div(a_v0, b_v0);
                    Float4::save(output_slice_ptr + i * 4, o_v0);
                }
            }
        }
    }
    return TNN_OK;
}

Status ArmGLULayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto input_blob = inputs[0];
    auto data_type  = input_blob->GetBlobDesc().data_type;
    if (data_type == DATA_TYPE_FLOAT) {
        Exec<float>(inputs, outputs);
    }
#if TNN_ARM82
    else if (data_type == DATA_TYPE_HALF) {
        ExecFp16(inputs, outputs);
    }
#endif
    else if (data_type == DATA_TYPE_INT8) {
        LOGE("Error: layer acc don't support datatype: %d\n", input_blob->GetBlobDesc().data_type);
        return Status{TNNERR_MODEL_ERR, "Error: layer acc dont support datatype"};
    } else {
        LOGE("Error: layer acc dont support datatype: %d\n", input_blob->GetBlobDesc().data_type);
        return Status{TNNERR_MODEL_ERR, "Error: layer acc dont support datatype"};
    }
    return TNN_OK;
}

REGISTER_ARM_ACC(GLU, LAYER_GLU);
REGISTER_ARM_LAYOUT(LAYER_GLU, DATA_FORMAT_NC4HW4)
REGISTER_ARM_PRECISION_FP16(LAYER_GLU);

}  // namespace TNN_NS
