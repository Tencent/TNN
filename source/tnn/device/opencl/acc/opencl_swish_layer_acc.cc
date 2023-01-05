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

DECLARE_OPENCL_UNARY_ACC(Swish);

Status OpenCLSwishLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                                 const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Init Swish Acc\n");
    Status ret = OpenCLUnaryLayerAcc::Init(context, param, resource, inputs, outputs);
    CHECK_TNN_OK(ret)

    op_name_ = "Swish";

    return TNN_OK;
}

std::set<std::string> OpenCLSwishLayerAcc::CreateBuildOptions() {
    std::set<std::string> build_options;
    std::string compute = "in*(FLOAT)(1.0f)/((FLOAT)(1.0f)+exp(-in))";
    build_options.emplace(" -DOPERATOR=" + compute);
    return build_options;
}

OpenCLSwishLayerAcc::~OpenCLSwishLayerAcc() {}

REGISTER_OPENCL_ACC(Swish, LAYER_SWISH)
REGISTER_OPENCL_LAYOUT(LAYER_SWISH, DATA_FORMAT_NHC4W4);

}  // namespace TNN_NS
