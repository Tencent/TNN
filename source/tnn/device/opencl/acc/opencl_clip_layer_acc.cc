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

#include "tnn/device/opencl/acc/opencl_unary_layer_acc.h"
#include "tnn/utils/string_utils_inner.h"
#include <math.h>

namespace TNN_NS {

DECLARE_OPENCL_UNARY_ACC(Clip);

Status OpenCLClipLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                                const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Init Clip Acc\n");
    Status ret = OpenCLUnaryLayerAcc::Init(context, param, resource, inputs, outputs);
    CHECK_TNN_OK(ret)

    op_name_ = "Clip";

    return TNN_OK;
}

float ConvertInfNum(float num) {
    bool is_inf = isinf(num);
    if (is_inf) {
        if (num > 0) return FLT_MAX;
        if (num < 0) return -FLT_MAX;
    } else {
        return num;
    }
}

std::set<std::string> OpenCLClipLayerAcc::CreateBuildOptions() {
    std::set<std::string> build_options;
    ClipLayerParam *clip_param = dynamic_cast<ClipLayerParam *>(param_);
    if (clip_param == nullptr) {
        LOGE("clip param is nil");
        return build_options;
    }

    // declare to float type
    std::string min_clip_str = ToString(ConvertInfNum(clip_param->min)) + "f";
    std::string max_clip_str = ToString(ConvertInfNum(clip_param->max)) + "f";

    std::string compute = "clamp(in,(FLOAT4)(" + min_clip_str + "),(FLOAT4)(" + max_clip_str + "))";
    build_options.emplace(" -DOPERATOR=" + compute);
    return build_options;
}

OpenCLClipLayerAcc::~OpenCLClipLayerAcc() {}

REGISTER_OPENCL_ACC(Clip, LAYER_CLIP)
REGISTER_OPENCL_LAYOUT(LAYER_CLIP, DATA_FORMAT_NHC4W4);

}  // namespace TNN_NS
