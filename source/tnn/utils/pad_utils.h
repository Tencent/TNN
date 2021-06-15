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
#ifndef TNN_SOURCE_TNN_UTILS_PAD_UTILS_H
#define TNN_SOURCE_TNN_UTILS_PAD_UTILS_H

#include "tnn/core/common.h"
#include "tnn/core/macro.h"
#include "tnn/core/status.h"

namespace TNN_NS {
class PadUtils {
public:
    struct PadContext {
        // input context
        int32_t input_batch      = 1;
        int32_t input_channel    = 1;
        int32_t input_channel_r4 = 4;
        int32_t input_depth      = 1;
        int32_t input_height     = 1;
        int32_t input_width      = 1;
        // output context
        int32_t output_batch      = 1;
        int32_t output_channel    = 1;
        int32_t output_channel_r4 = 4;
        int32_t output_depth      = 1;
        int32_t output_height     = 1;
        int32_t output_width      = 1;
        // param context
        int32_t pad_b_b = 0;
        int32_t pad_b_e = 0;
        int32_t pad_c_b = 0;
        int32_t pad_c_e = 0;
        int32_t pad_d_b = 0;
        int32_t pad_d_e = 0;
        int32_t pad_t   = 0;
        int32_t pad_b   = 0;
        int32_t pad_l   = 0;
        int32_t pad_r   = 0;
        int32_t type    = 0;
        float value     = 0.0f;
    };
    static Status ConstPadV2(float *input_data, float *output_data, DimsVector input_dims, DimsVector output_dims,
                             PadContext);

    static Status ReflectPadV2(float *input_data, float *output_data, DimsVector input_dims, DimsVector output_dims,
                               PadContext);
};
}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_UTILS_PAD_UTILS_H
