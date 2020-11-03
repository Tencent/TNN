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

#include "tnn/device/opencl/acc/opencl_reduce_layer_acc.h"

namespace TNN_NS {

DECLARE_OPENCL_REDUCE_ACC(ReduceL2);

Status OpenCLReduceL2LayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                                    const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Init ReduceL2 Acc\n");
    Status ret = OpenCLReduceLayerAcc::Init(context, param, resource, inputs, outputs);
    CHECK_TNN_OK(ret)

    op_name_ = "ReduceL2";

    return TNN_OK;
}

std::set<std::string> OpenCLReduceL2LayerAcc::CreateBuildOptions() {
    std::set<std::string> build_options;
    std::string init    = " -DDATAINIT=0 ";
    std::string compute = " -DOPERATOR(r,t)=r=(r+pow(t,2)); ";
    std::string reduce  = " -DREDUCEOPERATOR(r,t)=r=(r+t); ";
    std::string inner   = " -DINNEROPERATOR(r)=r.x+r.y+r.z+r.w ";
    std::string post    = " -DPOSTOPERATOR(r)=sqrt(r) ";
    build_options.emplace(init + compute + reduce + inner + post);
    return build_options;
}

OpenCLReduceL2LayerAcc::~OpenCLReduceL2LayerAcc() {}

REGISTER_OPENCL_ACC(ReduceL2, LAYER_REDUCE_L2)

}  // namespace TNN_NS
