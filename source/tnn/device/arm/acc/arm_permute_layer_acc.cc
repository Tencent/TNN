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

#include "tnn/device/arm/acc/arm_nchw_layer_acc.h"
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

DECLARE_ARM_NCHW_ACC(Permute, LAYER_PERMUTE);

template <typename T>
void ArmPermute(const int count, DimsVector dims, T *bottom_data, const std::vector<int> &permute_order,
                const std::vector<int> &old_steps, const std::vector<int> &new_steps, const int num_axes, T *top_data) {
    if (num_axes == 5) {
        for (int n = 0; n < dims[0]; ++n) {
            int idx     = n * new_steps[0];
            int old_idx = n * old_steps[permute_order[0]];
            for (int c = 0; c < dims[1]; ++c) {
                int idx_c     = idx + c * new_steps[1];
                int old_idx_c = old_idx + c * old_steps[permute_order[1]];
                for (int h = 0; h < dims[2]; ++h) {
                    int idx_h     = idx_c + h * new_steps[2];
                    int old_idx_h = old_idx_c + h * old_steps[permute_order[2]];
                    for (int w = 0; w < dims[3]; ++w) {
                        int idx_w     = idx_h + w * new_steps[3];
                        int old_idx_w = old_idx_h + w * old_steps[permute_order[3]];
                        for (int x = 0; x < dims[4]; ++x) {
                            int idx_x       = idx_w + x * new_steps[4];
                            int old_idx_x   = old_idx_w + x * old_steps[permute_order[4]];
                            top_data[idx_x] = bottom_data[old_idx_x];
                        }
                    }
                }
            }
        }
    } else if (num_axes == 4) {
        for (int n = 0; n < dims[0]; ++n) {
            int idx     = n * new_steps[0];
            int old_idx = n * old_steps[permute_order[0]];
            for (int c = 0; c < dims[1]; ++c) {
                int idx_c     = idx + c * new_steps[1];
                int old_idx_c = old_idx + c * old_steps[permute_order[1]];
                for (int h = 0; h < dims[2]; ++h) {
                    int idx_h     = idx_c + h * new_steps[2];
                    int old_idx_h = old_idx_c + h * old_steps[permute_order[2]];
                    for (int w = 0; w < dims[3]; ++w) {
                        int idx_w       = idx_h + w * new_steps[3];
                        int old_idx_w   = old_idx_h + w * old_steps[permute_order[3]];
                        top_data[idx_w] = bottom_data[old_idx_w];
                    }
                }
            }
        }
    } else if (num_axes == 3) {
        for (int n = 0; n < dims[0]; ++n) {
            int idx     = n * new_steps[0];
            int old_idx = n * old_steps[permute_order[0]];
            for (int c = 0; c < dims[1]; ++c) {
                int idx_c     = idx + c * new_steps[1];
                int old_idx_c = old_idx + c * old_steps[permute_order[1]];
                for (int h = 0; h < dims[2]; ++h) {
                    int idx_h       = idx_c + h * new_steps[2];
                    int old_idx_h   = old_idx_c + h * old_steps[permute_order[2]];
                    top_data[idx_h] = bottom_data[old_idx_h];
                }
            }
        }
    } else if (num_axes == 2) {
        for (int n = 0; n < dims[0]; ++n) {
            int idx     = n * new_steps[0];
            int old_idx = n * old_steps[permute_order[0]];
            for (int c = 0; c < dims[1]; ++c) {
                int idx_c       = idx + c * new_steps[1];
                int old_idx_c   = old_idx + c * old_steps[permute_order[1]];
                top_data[idx_c] = bottom_data[old_idx_c];
            }
        }
    } else if (num_axes == 1) {
        for (int n = 0; n < dims[0]; ++n) {
            int idx       = n * new_steps[0];
            int old_idx   = n * old_steps[permute_order[0]];
            top_data[idx] = bottom_data[old_idx];
        }
    } else {
        for (int i = 0; i < count; ++i) {
            int old_idx = 0;
            int idx     = i;
            for (int j = num_axes - 1; j >= 0; --j) {
                int order = permute_order[j];
                old_idx += (idx % dims[j]) * old_steps[order];
                idx /= dims[j];
            }
            top_data[i] = bottom_data[old_idx];
        }
    }
};

Status ArmPermuteLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto permute_param = dynamic_cast<PermuteLayerParam *>(param_);
    CHECK_PARAM_NULL(permute_param);

    auto packed = inputs[0]->GetBlobDesc().data_format != DATA_FORMAT_NCHW;

    Blob *input_blob;
    Blob *output_blob;
    if (packed) {
        AllocConvertBuffer(inputs, outputs);
        input_blob  = nchw_blob_in[0].get();
        output_blob = nchw_blob_out[0].get();
    } else {
        input_blob  = inputs[0];
        output_blob = outputs[0];
    }

    auto input_dims  = input_blob->GetBlobDesc().dims;
    auto output_dims = output_blob->GetBlobDesc().dims;

    std::vector<int> input_step;
    std::vector<int> output_step;
    int num_dims     = int(input_dims.size());
    int output_count = DimsVectorUtils::Count(output_dims);
    for (int i = 0; i < input_dims.size(); ++i) {
        input_step.push_back(DimsVectorUtils::Count(input_dims, i + 1));
        output_step.push_back(DimsVectorUtils::Count(output_dims, i + 1));
    }

    if (outputs[0]->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
        if (packed) {
            UnPackInputs<float>(inputs);
        }
        float *input_data  = reinterpret_cast<float *>(GetBlobHandlePtr(input_blob->GetHandle()));
        float *output_data = reinterpret_cast<float *>(GetBlobHandlePtr(output_blob->GetHandle()));
        ArmPermute<float>(output_count, output_dims, input_data, permute_param->orders, input_step, output_step,
                          num_dims, output_data);
        if (packed) {
            PackOutputs<float>(outputs);
        }
    } else if (outputs[0]->GetBlobDesc().data_type == DATA_TYPE_INT32) {
        if (packed) {
            LOGE("ArmPermuteLayerAcc::DoForward only support nchw format");
            return Status(TNNERR_PARAM_ERR, "Arm permute layer got wrong format");
        }
        int *input_data  = reinterpret_cast<int *>(GetBlobHandlePtr(input_blob->GetHandle()));
        int *output_data = reinterpret_cast<int *>(GetBlobHandlePtr(output_blob->GetHandle()));
        ArmPermute<int>(output_count, output_dims, input_data, permute_param->orders, input_step, output_step, num_dims,
                        output_data);
    }
#if TNN_ARM82
    else if (outputs[0]->GetBlobDesc().data_type == DATA_TYPE_HALF) {
        if (packed) {
            UnPackInputs<fp16_t>(inputs);
        }
        fp16_t *input_data  = reinterpret_cast<fp16_t *>(GetBlobHandlePtr(input_blob->GetHandle()));
        fp16_t *output_data = reinterpret_cast<fp16_t *>(GetBlobHandlePtr(output_blob->GetHandle()));
        ArmPermute<fp16_t>(output_count, output_dims, input_data, permute_param->orders, input_step, output_step,
                           num_dims, output_data);
        if (packed) {
            PackOutputs<fp16_t>(outputs);
        }
    }
#endif
    else {
        LOGE("ArmPermuteLayerAcc::DoForward not supported dtype: %d\n", outputs[0]->GetBlobDesc().data_type);
        return Status(TNNERR_PARAM_ERR, "Arm permute layer got wrong dtype");
    }

    return TNN_OK;
}

REGISTER_ARM_ACC(Permute, LAYER_PERMUTE);
REGISTER_ARM_PRECISION_FP16(LAYER_PERMUTE)
REGISTER_ARM_LAYOUT(LAYER_PERMUTE, DATA_FORMAT_NC4HW4)
REGISTER_ARM_LAYOUT(LAYER_PERMUTE, DATA_FORMAT_NCHW)

}  // namespace TNN_NS
