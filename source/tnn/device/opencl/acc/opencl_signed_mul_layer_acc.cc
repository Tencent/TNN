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

namespace TNN_NS {

DECLARE_OPENCL_ACC(SignedMul);

Status OpenCLSignedMulLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                                   const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Init SignedMul Acc\n");
    Status ret = OpenCLLayerAcc::Init(context, param, resource, inputs, outputs);
    CHECK_TNN_OK(ret)

    run_3d_ndrange_ = true;
    op_name_        = "SignedMul";

    SignedMulLayerParam *signed_mul_param = dynamic_cast<SignedMulLayerParam *>(param);
    if (!signed_mul_param) {
        LOGE("Error: singed mul layer param is null\n");
        return Status(TNNERR_MODEL_ERR, "Error: signed mul layer param is null");
    }

    // create kernel
    std::string kernel_name = "SignedMul";
    ret = CreateExecuteUnit(execute_units_[0], "signed_mul", kernel_name, build_options_);
    if (ret != TNN_OK) {
        LOGE("create execute unit failed!\n");
        return ret;
    }

    return TNN_OK;
}

Status OpenCLSignedMulLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("SignedMul Layer Reshape\n");
    Status ret = OpenCLLayerAcc::Reshape(inputs, outputs);
    CHECK_TNN_OK(ret)

    SignedMulLayerParam *signed_mul_param = dynamic_cast<SignedMulLayerParam *>(param_);
    if (!signed_mul_param) {
        LOGE("Error: layer param is null\n");
        return Status(TNNERR_MODEL_ERR, "Error: layer param is null");
    }

    auto output_dims = outputs[0]->GetBlobDesc().dims;

    const int batch    = DimsFunctionUtils::GetDim(output_dims, 0);
    const int channels = DimsFunctionUtils::GetDim(output_dims, 1);
    const int height   = DimsFunctionUtils::GetDim(output_dims, 2);
    const int width    = DimsFunctionUtils::GetDim(output_dims, 3);

    uint32_t idx = 0;
    execute_units_[0].global_work_size = {static_cast<uint32_t>(width), static_cast<uint32_t>(UP_DIV(channels, 4)),
                                                static_cast<uint32_t>(height * batch)};
    execute_units_[0].ocl_kernel.setArg(idx++, execute_units_[0].global_work_size[0]);
    execute_units_[0].ocl_kernel.setArg(idx++, execute_units_[0].global_work_size[1]);
    execute_units_[0].ocl_kernel.setArg(idx++, execute_units_[0].global_work_size[2]);
    execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)inputs[0]->GetHandle().base));
    execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)outputs[0]->GetHandle().base));
    execute_units_[0].ocl_kernel.setArg(idx++, signed_mul_param->alpha);
    execute_units_[0].ocl_kernel.setArg(idx++, signed_mul_param->beta);
    execute_units_[0].ocl_kernel.setArg(idx++, 1.0f / signed_mul_param->gamma);
    execute_units_[0].local_work_size = LocalWS3DDefault(execute_units_[0]);
    return TNN_OK;
}

REGISTER_OPENCL_ACC(SignedMul, LAYER_SIGNED_MUL)
REGISTER_OPENCL_LAYOUT(LAYER_SIGNED_MUL, DATA_FORMAT_NHC4W4);

}  // namespace TNN_NS
