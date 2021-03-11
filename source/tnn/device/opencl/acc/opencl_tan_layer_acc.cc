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

namespace TNN_NS {

DECLARE_OPENCL_UNARY_ACC(Tan);

Status OpenCLTanLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                               const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Init Tan Acc\n");
    Status ret = OpenCLUnaryLayerAcc::Init(context, param, resource, inputs, outputs);
    CHECK_TNN_OK(ret)

    op_name_ = "Tan";

    return TNN_OK;
}

std::set<std::string> OpenCLTanLayerAcc::CreateBuildOptions() {
    std::set<std::string> build_options;
    // 使用sin和cos计算Tan，而不是tan计算的原因是OpenCL tan在部分机型上（如LON-AL00）会出现错误
    std::string compute = "sin(in)/cos(in)";
    build_options.emplace(" -DOPERATOR=" + compute);
    return build_options;
}

OpenCLTanLayerAcc::~OpenCLTanLayerAcc() {}

REGISTER_OPENCL_ACC(Tan, LAYER_TAN)
REGISTER_OPENCL_LAYOUT(LAYER_TAN, DATA_FORMAT_NHC4W4);

}  // namespace TNN_NS
