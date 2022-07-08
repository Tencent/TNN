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

#include "tnn/device/opencl/acc/opencl_layer_acc.h"
#include "tnn/device/opencl/imagebuffer_convertor.h"

namespace TNN_NS {

DECLARE_OPENCL_ACC(Pow);

Status OpenCLPowLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                               const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Init Pow Acc\n");
    Status ret = OpenCLLayerAcc::Init(context, param, resource, inputs, outputs);
    CHECK_TNN_OK(ret)

    run_3d_ndrange_ = true;
    op_name_        = "Pow";

    // create kernel
    std::string kernel_name = "Power";
    ret                     = CreateExecuteUnit(execute_units_[0], "pow", kernel_name, build_options_);
    if (ret != TNN_OK) {
        LOGE("create execute unit failed!\n");
        return ret;
    }

    return TNN_OK;
}

Status OpenCLPowLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Pow Acc Reshape\n");
    Status ret = OpenCLLayerAcc::Reshape(inputs, outputs);
    CHECK_TNN_OK(ret)

    PowLayerParam *pow_param = dynamic_cast<PowLayerParam *>(param_);
    if (!pow_param) {
        LOGE("Error: layer param is null\n");
        return Status(TNNERR_MODEL_ERR, "Error: layer param is null");
    }
    auto output_dims = outputs[0]->GetBlobDesc().dims;
    uint32_t idx = SetExecuteUnit3DSizeInfoDefault(execute_units_[0], output_dims);
    execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)inputs[0]->GetHandle().base));
    execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)outputs[0]->GetHandle().base));
    execute_units_[0].ocl_kernel.setArg(idx++, pow_param->scale);
    execute_units_[0].ocl_kernel.setArg(idx++, pow_param->shift);
    execute_units_[0].ocl_kernel.setArg(idx++, pow_param->exponent);
    return TNN_OK;
}

REGISTER_OPENCL_ACC(Pow, LAYER_POWER)
REGISTER_OPENCL_LAYOUT(LAYER_POWER, DATA_FORMAT_NHC4W4);

}  // namespace TNN_NS
