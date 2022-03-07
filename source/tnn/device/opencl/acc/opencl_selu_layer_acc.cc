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

#include <sstream>
#include "tnn/device/opencl/acc/opencl_unary_layer_acc.h"

namespace TNN_NS {

DECLARE_OPENCL_ACC(Selu);

Status OpenCLSeluLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                                const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Init Selu Acc\n");
    Status ret = OpenCLLayerAcc::Init(context, param, resource, inputs, outputs);
    CHECK_TNN_OK(ret)

    run_3d_ndrange_ = true;
    op_name_        = "Selu";

    std::set<std::string> build_options;
    AdjustBuildOptionForFp32(build_options);
    build_options.insert(build_options_.begin(), build_options_.end());

    // create kernel
    std::string kernel_name = "Selu";
    ret                     = CreateExecuteUnit(execute_units_[0], "selu", kernel_name, build_options);
    if (ret != TNN_OK) {
        LOGE("create execute unit failed!\n");
        return ret;
    }

    return TNN_OK;
}

Status OpenCLSeluLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Selu Acc Reshape\n");
    Status ret = OpenCLLayerAcc::Reshape(inputs, outputs);
    CHECK_TNN_OK(ret)

    SeluLayerParam *selu_param = dynamic_cast<SeluLayerParam *>(param_);
    if (!selu_param) {
        LOGE("Error: layer param is null\n");
        return Status(TNNERR_MODEL_ERR, "Error: layer param is null");
    }
    float factor1 = selu_param->alpha * selu_param->gamma;
    float factor2 = selu_param->gamma;
    uint32_t idx = SetExecuteUnit3DSizeInfoDefault(execute_units_[0], outputs[0]->GetBlobDesc().dims);
    execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)inputs[0]->GetHandle().base));
    execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)outputs[0]->GetHandle().base));
    execute_units_[0].ocl_kernel.setArg(idx++, factor1);
    execute_units_[0].ocl_kernel.setArg(idx++, factor2);
    return TNN_OK;
}

REGISTER_OPENCL_ACC(Selu, LAYER_SELU)
REGISTER_OPENCL_LAYOUT(LAYER_SELU, DATA_FORMAT_NHC4W4);

}  // namespace TNN_NS
