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

#include "tnn/device/opencl/acc/deconvolution/opencl_deconv_layer_depthwise_acc.h"
#include "tnn/device/opencl/imagebuffer_convertor.h"

namespace TNN_NS {

bool OpenCLDeconvLayerDepthwiseAcc::IsPrefered(const ConvLayerParam *param, const std::vector<Blob *> &inputs,
                                               const std::vector<Blob *> &outputs) {
    if (!param) {
        return false;
    }

    return param->group == DimsFunctionUtils::GetDim(inputs[0]->GetBlobDesc().dims, 1) &&
           param->group == DimsFunctionUtils::GetDim(outputs[0]->GetBlobDesc().dims, 1);
}

Status OpenCLDeconvLayerDepthwiseAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                                           const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Init Deconv Depthwise Acc\n");

    op_name_     = "Deconv_Depthwise";
    deconv_type_ = CT_DECONV_DEPTHWISE;

    Status ret = OpenCLDeconvLayerAccImpl::Init(context, param, resource, inputs, outputs);
    CHECK_TNN_OK(ret)

    // create kernel
    std::set<std::string> build_options;
    std::string kernel_name = "DepthwiseDeconv2D";

    ret                     = CreateExecuteUnit(execute_units_[0], "deconvolution", kernel_name, build_options);
    if (ret != TNN_OK) {
        LOGE("create execute unit failed!\n");
        return ret;
    }

    return TNN_OK;
}

OpenCLDeconvLayerDepthwiseAcc::~OpenCLDeconvLayerDepthwiseAcc() {}

void OpenCLDeconvLayerDepthwiseAcc::SetExtraKernelParameters(uint32_t idx, const std::vector<Blob *> &inputs,
                                                             const std::vector<Blob *> &outputs) {
    auto output_dims                = outputs[0]->GetBlobDesc().dims;
    const int output_channel_blocks = UP_DIV(DimsFunctionUtils::GetDim(output_dims, 1), 4);
    execute_units_[0].ocl_kernel.setArg(idx++, static_cast<int32_t>(output_channel_blocks));
    execute_units_[0].ocl_kernel.setArg(idx++, (int)deconv_params_.activation_type);
}

}  // namespace TNN_NS
