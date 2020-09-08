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
#if (defined __ANDROID_API__) && (__ANDROID_API__ >= 21)
#include <sys/system_properties.h>
#endif
namespace TNN_NS {

DECLARE_OPENCL_UNARY_ACC(Elu);

Status OpenCLEluLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                               const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Init Elu Acc\n");
    Status ret = OpenCLUnaryLayerAcc::Init(context, param, resource, inputs, outputs);
    CHECK_TNN_OK(ret)

    op_name_ = "Elu";

    return TNN_OK;
}

std::set<std::string> OpenCLEluLayerAcc::CreateBuildOptions() {
    std::set<std::string> build_options;
    EluLayerParam *elu_param = dynamic_cast<EluLayerParam *>(param_);
    if (elu_param == nullptr) {
        LOGE("elu param is nil");
        return build_options;
    }
    std::string compute = "select(in,(FLOAT)(" + ToString(elu_param->alpha) + "f)*(exp(in)-(FLOAT)(1.0f)),in<0)";
    build_options.emplace(" -DOPERATOR=" + compute);

    bool force_fp32 = false;
#if (defined __ANDROID_API__) && (__ANDROID_API__ >= 21)
    char sdk[128] = "0";
    __system_property_get("ro.build.version.sdk", sdk);

    int sdk_version = atoi(sdk);

    // Android 7.1之前版本 fp16 exp 部分机型上的速度有问题，改用fp32版本的kernel
    force_fp32 = (sdk_version <= 25);
#elif (defined __ANDROID_API__) && (__ANDROID_API__ < 21)
    force_fp32 = true;
#endif

    if (force_fp32) {
        build_options.emplace("-DFORCE_FP32");
    }

    return build_options;
}

OpenCLEluLayerAcc::~OpenCLEluLayerAcc() {}

REGISTER_OPENCL_ACC(Elu, LAYER_ELU)

}  // namespace TNN_NS
