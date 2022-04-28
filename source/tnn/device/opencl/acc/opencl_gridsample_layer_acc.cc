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

class OpenCLGridsampleLayerAcc : public OpenCLLayerAcc {
public:
    virtual Status Init(Context *context, LayerParam *param, LayerResource *resource, const std::vector<Blob *> &inputs,
                        const std::vector<Blob *> &outputs) override;

    virtual Status Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) override;

    virtual Status ReloadConstantBlobs(const std::vector<Blob *> &inputs,
                                       bool only_reload_shape_differ_blob = false) override {
        return TNN_OK;
    }
};

Status OpenCLGridsampleLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                                      const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Init Upsample Acc\n");
    Status ret = OpenCLLayerAcc::Init(context, param, resource, inputs, outputs);
    CHECK_TNN_OK(ret)

    op_name_ = "Gridsample";

    GridSampleLayerParam *gridsample_param = dynamic_cast<GridSampleLayerParam *>(param);
    if (!gridsample_param) {
        LOGE("Error: layer param is null\n");
        return Status(TNNERR_MODEL_ERR, "Error: layer param is null");
    }

    // create kernel
    std::string kernel_name;
    if (gridsample_param->mode == 2) {  // bilinear
        kernel_name = "BilinearGridSample";
    } else {
        LOGE("Not support Gridsample type: %d\n", gridsample_param->mode);
        return Status(TNNERR_OPENCL_ACC_INIT_ERROR, "invalid upsample mode");
    }

    ret = CreateExecuteUnit(execute_units_[0], "gridsample", kernel_name, build_options_);
    if (ret != TNN_OK) {
        LOGE("create execute unit failed!\n");
        return ret;
    }

    return TNN_OK;
}

Status OpenCLGridsampleLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Girdsample Acc Reshape\n");
    Status ret = OpenCLLayerAcc::Reshape(inputs, outputs);
    CHECK_TNN_OK(ret)

    GridSampleLayerParam *gridsample_param = dynamic_cast<GridSampleLayerParam *>(param_);
    if (!gridsample_param) {
        LOGE("Error: layer param is null\n");
        return Status(TNNERR_MODEL_ERR, "Error: layer param is null");
    }
    if (gridsample_param->mode != 2 || gridsample_param->pad_type != 0 || gridsample_param->align_corners != 0) {
        return Status(TNNERR_PARAM_ERR,
                      "OpenclGridSampleLayerAcc dont support some mode or pade type or align_corners");
    }

    auto input  = inputs[0];
    auto grid   = inputs[1];
    auto output = outputs[0];

    auto input_dims  = input->GetBlobDesc().dims;
    auto output_dims = output->GetBlobDesc().dims;
    if (input_dims.size() != 4 || output_dims.size() != 4) {
        LOGE("GridSample Layer (OpenCL) only support 4-dim by now\n");
        return Status(TNNERR_INVALID_INPUT, "GridSample Layer (OpenCL) only support 4-dim by now\n");
    }
    const int batch        = DimsFunctionUtils::GetDim(input_dims, 0);
    const int channels     = DimsFunctionUtils::GetDim(input_dims, 1);
    const int input_height = DimsFunctionUtils::GetDim(input_dims, 2);
    const int input_width  = DimsFunctionUtils::GetDim(input_dims, 3);

    const int output_height = DimsFunctionUtils::GetDim(output_dims, 2);
    const int output_width  = DimsFunctionUtils::GetDim(output_dims, 3);

    const int channel_blocks = UP_DIV(channels, 4);

    uint32_t idx         = 0;
    auto output_upheight = output_dims;
    output_upheight[2]   = UP_DIV(output_upheight[2], 4);
    idx                  = SetExecuteUnit2DSizeInfoDefault(execute_units_[0], output_upheight);

    execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)input->GetHandle().base));
    execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)grid->GetHandle().base));
    execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)output->GetHandle().base));

    execute_units_[0].ocl_kernel.setArg(idx++, static_cast<int32_t>(input_height));
    execute_units_[0].ocl_kernel.setArg(idx++, static_cast<int32_t>(input_width));
    execute_units_[0].ocl_kernel.setArg(idx++, static_cast<int32_t>(output_height));
    execute_units_[0].ocl_kernel.setArg(idx++, static_cast<int32_t>(output_width));

    return TNN_OK;
}

REGISTER_OPENCL_ACC(Gridsample, LAYER_GRIDSAMPLE)
REGISTER_OPENCL_LAYOUT(LAYER_GRIDSAMPLE, DATA_FORMAT_NHC4W4);

}  // namespace TNN_NS
