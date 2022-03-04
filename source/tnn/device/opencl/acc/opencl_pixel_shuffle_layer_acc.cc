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

DECLARE_OPENCL_ACC(PixelShuffle);

Status OpenCLPixelShuffleLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                                       const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Init PixelShuffle Acc\n");
    Status ret = OpenCLLayerAcc::Init(context, param, resource, inputs, outputs);
    CHECK_TNN_OK(ret)

    run_3d_ndrange_ = false;
    op_name_        = "PixelShuffle";

    PixelShuffleLayerParam *pixel_shuffle_param = dynamic_cast<PixelShuffleLayerParam *>(param);
    if (!pixel_shuffle_param) {
        LOGE("Error: layer param is null\n");
        return Status(TNNERR_MODEL_ERR, "Error: layer param is null");
    }

    // create kernel
    std::string kernel_name = "PixelShuffle";

    ret = CreateExecuteUnit(execute_units_[0], "pixel_shuffle", kernel_name, build_options_);
    if (ret != TNN_OK) {
        LOGE("create execute unit failed!\n");
        return ret;
    }

    return TNN_OK;
}

Status OpenCLPixelShuffleLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("PixelShuffle Acc Reshape\n");
    Status ret = OpenCLLayerAcc::Reshape(inputs, outputs);
    CHECK_TNN_OK(ret)

    PixelShuffleLayerParam *pixel_shuffle_param = dynamic_cast<PixelShuffleLayerParam *>(param_);
    if (!pixel_shuffle_param) {
        LOGE("Error: layer param is null\n");
        return Status(TNNERR_MODEL_ERR, "Error: layer param is null");
    }
    auto output_dims    = outputs[0]->GetBlobDesc().dims;
    auto input_dims     = inputs[0]->GetBlobDesc().dims;
    int upscale_factor  = pixel_shuffle_param->upscale_factor;

    uint32_t idx = SetExecuteUnit2DSizeInfoDefault(execute_units_[0], output_dims);
    execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)inputs[0]->GetHandle().base));
    execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)outputs[0]->GetHandle().base));
    execute_units_[0].ocl_kernel.setArg(idx++, DimsFunctionUtils::GetDim(output_dims, 2));
    execute_units_[0].ocl_kernel.setArg(idx++, DimsFunctionUtils::GetDim(output_dims, 3));
    execute_units_[0].ocl_kernel.setArg(idx++, DimsFunctionUtils::GetDim(input_dims, 2));
    execute_units_[0].ocl_kernel.setArg(idx++, DimsFunctionUtils::GetDim(input_dims, 3));
    execute_units_[0].ocl_kernel.setArg(idx++, static_cast<int32_t>(upscale_factor));
    execute_units_[0].ocl_kernel.setArg(idx++, static_cast<int32_t>(upscale_factor * upscale_factor));

    return TNN_OK;
}

REGISTER_OPENCL_ACC(PixelShuffle, LAYER_PIXEL_SHUFFLE)
REGISTER_OPENCL_LAYOUT(LAYER_PIXEL_SHUFFLE, DATA_FORMAT_NHC4W4);

}  // namespace TNN_NS