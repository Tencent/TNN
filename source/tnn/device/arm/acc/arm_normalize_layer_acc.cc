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

#include <limits.h>
#include <cmath>

#include "tnn/device/arm/acc/Float4.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

DECLARE_ARM_ACC(Normalize, LAYER_NORMALIZE);

static inline void _sum_abs(float *dst, float *src, int channel, int plane_num) {
    for (int c = 0; c < UP_DIV(channel, 4); c++) {
        float *input_data_c = src + c * 4 * plane_num;
        for (int i = 0; i < plane_num; i++) {
            Float4::save(dst + i * 4, Float4::load(dst + i * 4) + Float4::abs(Float4::load(input_data_c + i * 4)));
        }
    }
    for (int i = 0; i < plane_num; i++) {
        dst[i] = dst[i * 4 + 0] + dst[i * 4 + 1] + dst[i * 4 + 2] + dst[i * 4 + 3];
    }
}

static inline void _sum_valsq(float *dst, float *src, int channel, int plane_num) {
    // sum  x*x
    for (int c = 0; c < UP_DIV(channel, 4); c++) {
        float *input_data_c = src + c * 4 * plane_num;
        for (int i = 0; i < plane_num; i++) {
            Float4::save(dst + i * 4, Float4::load(dst + i * 4) +
                                        Float4::load(input_data_c + i * 4) * Float4::load(input_data_c + i * 4));
        }
    }
    for (int i = 0; i < plane_num; i++) {
        dst[i] = dst[i * 4 + 0] + dst[i * 4 + 1] + dst[i * 4 + 2] + dst[i * 4 + 3];
    }
}

static inline void _max(float *dst, float *src, int channel, int plane_num) {
    for (int c = 0; c < UP_DIV(channel, 4); c++) {
        float *input_data_c = src + c * 4 * plane_num;
        for (int i = 0; i < plane_num; i++) {
            Float4::save(dst + i * 4, Float4::max(Float4::load(dst + i * 4), Float4::load(input_data_c + i * 4)));
        }
    }
    for (int i = 0; i < plane_num; i++) {
        dst[i] = std::max(std::max(std::max(dst[i * 4 + 0], dst[i * 4 + 1]), dst[i * 4 + 2]), dst[i * 4 + 3]);
    }
}

static inline void _min(float *dst, float *src, int channel, int plane_num) {
    for (int c = 0; c < UP_DIV(channel, 4); c++) {
        float *input_data_c = src + c * 4 * plane_num;
        for (int i = 0; i < plane_num; i++) {
            Float4::save(dst + i * 4, Float4::min(Float4::load(dst + i * 4), Float4::load(input_data_c + i * 4)));
        }
    }
    for (int i = 0; i < plane_num; i++) {
        dst[i] = std::min(std::min(std::min(dst[i * 4 + 0], dst[i * 4 + 1]), dst[i * 4 + 2]), dst[i * 4 + 3]);
    }
}

Status ArmNormalizeLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    if (inputs.size() < 1) {
        LOGE("Error: invalid inputs count\n");
        return Status(TNNERR_LAYER_ERR, "layer's inputs size must >= 2");
    }
    auto layer_param = dynamic_cast<NormalizeLayerParam *>(param_);
    CHECK_PARAM_NULL(layer_param);

    int axis           = layer_param->axis;
    int p              = layer_param->p;
    int across_spatial = layer_param->across_spatial;

    // old tnn support scale the result of normalize and only norm2
    if ((p != 1 && p != 2 && p != INT_MAX && p != INT_MIN) || axis != 1 || across_spatial != 0) {
        LOGE("Error: layer param is not supported now\n");
        return Status(TNNERR_INST_ERR, "Error: layer param is not supported now");
    }

    float epsilon      = layer_param->epsilon;
    int channel_shared = layer_param->channel_shared;

    Blob *input_blob  = inputs[0];
    Blob *output_blob = outputs[0];
    auto output_dims  = output_blob->GetBlobDesc().dims;
    int batch         = output_dims[0];
    int channel       = output_dims[1];
    int plane_num     = DimsVectorUtils::Count(output_blob->GetBlobDesc().dims, 2);
    if (output_blob->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
        float *input_data  = reinterpret_cast<float *>(GetBlobHandlePtr(input_blob->GetHandle()));
        float *output_data = reinterpret_cast<float *>(GetBlobHandlePtr(output_blob->GetHandle()));

        RawBuffer temp(4 * plane_num * sizeof(float));

        for (int b = 0; b < batch; b++) {
            float *input_data_b  = input_data + b * channel * plane_num;
            float *output_data_b = output_data + b * channel * plane_num;

            if (layer_param->p == INT_MAX) {
                float fl_min = FLT_MIN;
                memset(temp.force_to<void *>(), *(reinterpret_cast<int *>(&fl_min)), temp.GetBytesSize());
            } else if (layer_param->p == INT_MIN) {
                float fl_max = FLT_MAX;
                memset(temp.force_to<void *>(), *(reinterpret_cast<int *>(&fl_max)), temp.GetBytesSize());
            } else {
                memset(temp.force_to<void *>(), 0, temp.GetBytesSize());
            }

            if (layer_param->p == 1) {
                auto workspace = temp.force_to<float *>();
                // sum - abs(x)
                _sum_abs(workspace, input_data_b, channel, plane_num);
            } else if (layer_param->p == 2) {
                auto workspace = temp.force_to<float *>();
                // sum  x*x
                _sum_valsq(workspace, input_data_b, channel, plane_num);
                // max - sqrt
                for (int i = 0; i < plane_num; i++) {
                    workspace[i] = std::max(sqrtf(workspace[i]), epsilon);
                }
            } else if (layer_param->p == INT_MAX) {
                auto workspace = temp.force_to<float *>();
                _max(workspace, input_data_b, channel, plane_num);
            } else if (layer_param->p == INT_MIN) {
                auto workspace = temp.force_to<float *>();
                _min(workspace, input_data_b, channel, plane_num);
            }

            // div
            auto workspace = temp.force_to<float *>();
            for (int c = 0; c < UP_DIV(channel, 4); c++) {
                float *input_data_c  = input_data_b + c * 4 * plane_num;
                float *output_data_c = output_data_b + c * 4 * plane_num;
                for (int i = 0; i < plane_num; i++) {
                    Float4::save(output_data_c + i * 4,
                                 Float4::div(Float4::load(input_data_c + i * 4), Float4(workspace[i])));
                }
            }
        }
    } else if (output_blob->GetBlobDesc().data_type == DATA_TYPE_INT8) {
        LOGE("Error: layer acc dont support datatype: %d\n", output_blob->GetBlobDesc().data_type);
        return Status(TNNERR_MODEL_ERR, "Error: layer acc dont support datatype");
    } else {
        LOGE("Error: layer acc dont support datatype: %d\n", output_blob->GetBlobDesc().data_type);
        return Status(TNNERR_MODEL_ERR, "Error: layer acc dont support datatype");
    }

    return TNN_OK;
}

REGISTER_ARM_ACC(Normalize, LAYER_NORMALIZE);
REGISTER_ARM_LAYOUT(LAYER_NORMALIZE, DATA_FORMAT_NC4HW4)

}  // namespace TNN_NS
