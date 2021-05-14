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

#include "tnn/device/opencl/acc/convolution/opencl_conv_layer_depthwise_acc.h"
#include "tnn/device/opencl/imagebuffer_convertor.h"

namespace TNN_NS {

bool OpenCLConvLayerDepthwiseAcc::IsPrefered(const ConvLayerParam *param, const std::vector<Blob *> &inputs,
                                             const std::vector<Blob *> &outputs) {
    if (!param) {
        return false;
    }

    return param->group == DimsFunctionUtils::GetDim(inputs[0]->GetBlobDesc().dims, 1) &&
           param->group == DimsFunctionUtils::GetDim(outputs[0]->GetBlobDesc().dims, 1);
}

Status OpenCLConvLayerDepthwiseAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                                         const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Init Conv Depthwise Acc\n");

    conv_type_ = CT_CONV_DEPTHWISE;
    op_name_   = "Conv_Depthwise";

    Status ret = OpenCLConvLayerAccImpl::Init(context, param, resource, inputs, outputs);
    CHECK_TNN_OK(ret)

    ret = AllocateWeightsBias(resource);
    CHECK_TNN_OK(ret)

    std::string program_name = "convolution_depthwise";
    std::string kernel_name = "DepthwiseConv2D";
    if (conv_params_.stride_x == 1 && conv_params_.stride_y == 1 && conv_params_.dilation_x == 1 &&
        conv_params_.dilation_y == 1) {
        kernel_name = "DepthwiseConv2DS1";
    }
    ret = CreateExecuteUnit(execute_units_[0], program_name, kernel_name, build_options_);
    if (ret != TNN_OK) {
        LOGE("create execute unit failed!\n");
        return ret;
    }

    return TNN_OK;
}

OpenCLConvLayerDepthwiseAcc::~OpenCLConvLayerDepthwiseAcc() {}

Status OpenCLConvLayerDepthwiseAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Conv Depthwise Acc Reshape\n");
    auto input_dims  = inputs[0]->GetBlobDesc().dims;
    auto output_dims = outputs[0]->GetBlobDesc().dims;

    const int output_height = DimsFunctionUtils::GetDim(output_dims, 2);
    const int output_width  = DimsFunctionUtils::GetDim(output_dims, 3);

    const int input_height   = DimsFunctionUtils::GetDim(input_dims, 2);
    const int input_width    = DimsFunctionUtils::GetDim(input_dims, 3);
    const int input_channels = DimsFunctionUtils::GetDim(input_dims, 1);

    execute_units_[0].global_work_size = {
        static_cast<uint32_t>(UP_DIV(DimsFunctionUtils::GetDim(output_dims, 1), 4) *
                              UP_DIV(DimsFunctionUtils::GetDim(output_dims, 3), 4)),
        static_cast<uint32_t>(DimsFunctionUtils::GetDim(output_dims, 0) *
                              DimsFunctionUtils::GetDim(output_dims, 2))};

    int kernel_shape[2]      = {conv_params_.kernel_x, conv_params_.kernel_y};
    int stride_shape[2]      = {conv_params_.stride_x, conv_params_.stride_y};
    int padding_shape[2]     = {conv_params_.pad_x, conv_params_.pad_y};
    int dilation_shape[2]    = {conv_params_.dilation_x, conv_params_.dilation_y};

    int input_imageshape[2]  = {input_width, input_height};
    int output_imageshape[2] = {output_width, output_height};

    uint32_t idx = 0;
    execute_units_[0].ocl_kernel.setArg(idx++, execute_units_[0].global_work_size[0]);
    execute_units_[0].ocl_kernel.setArg(idx++, execute_units_[0].global_work_size[1]);
    execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)inputs[0]->GetHandle().base));
    execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)ocl_weights_->GetData()));
    execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)ocl_bias_->GetData()));
    execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)outputs[0]->GetHandle().base));
    execute_units_[0].ocl_kernel.setArg(idx++, sizeof(input_imageshape), input_imageshape);
    execute_units_[0].ocl_kernel.setArg(idx++, sizeof(output_imageshape), output_imageshape);
    execute_units_[0].ocl_kernel.setArg(idx++, sizeof(kernel_shape), kernel_shape);
    execute_units_[0].ocl_kernel.setArg(idx++, sizeof(padding_shape), padding_shape);
    
    if (conv_params_.stride_x != 1 || conv_params_.stride_y != 1 || conv_params_.dilation_x != 1 ||
        conv_params_.dilation_y != 1) {
        execute_units_[0].ocl_kernel.setArg(idx++, sizeof(dilation_shape), dilation_shape);
        execute_units_[0].ocl_kernel.setArg(idx++, sizeof(stride_shape), stride_shape);
    }
    execute_units_[0].ocl_kernel.setArg(idx++, (int)conv_params_.activation_type);

    execute_units_[0].local_work_size = Conv2dCommonLocalWS2D(
            execute_units_[0].global_work_size, execute_units_[0].workgroupsize_max, execute_units_[0].sub_group_size);

    if (ocl_context_->GetEnableTuneKernel()) {
        execute_units_[0].local_work_size = LocalTune(execute_units_[0], ocl_context_, GenerateTuneKernelKey(execute_units_[0]));
    }

    return TNN_OK;
}

}  // namespace TNN_NS
