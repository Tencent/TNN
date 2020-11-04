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

#include "tnn/device/opencl/acc/convolution/opencl_conv_layer_common_acc.h"
#include "tnn/device/opencl/imagebuffer_convertor.h"
#include "tnn/utils/string_utils_inner.h"

namespace TNN_NS {

bool OpenCLConvLayerCommonAcc::IsPrefered(const ConvLayerParam *param, const std::vector<Blob *> &,
                                          const std::vector<Blob *> &) {
    if (!param) {
        return false;
    }

    return true;
}

Status OpenCLConvLayerCommonAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                                      const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Init Conv Common Acc\n");

    Status ret = OpenCLConvLayerAccImpl::Init(context, param, resource, inputs, outputs);
    CHECK_TNN_OK(ret)

    conv_type_ = CT_CONV_COMMON;
    op_name_   = "Conv_" + ToString(conv_params_.kernel_x) + "x" + ToString(conv_params_.kernel_y);

    if(conv_params_.kernel_x != conv_params_.kernel_y) {
        run_3d_ndrange_ = false;
    }

    ret = AllocateWeightsBias(resource);
    CHECK_TNN_OK(ret)

    // create kernel
    std::set<std::string> build_options;
    if (conv_params_.activation_type == ActivationType_ReLU) {
        build_options.emplace("-DRELU");
    } else if (conv_params_.activation_type == ActivationType_ReLU6) {
        build_options.emplace("-DRELU6");
    }
    std::string kernel_name = "Conv2D";
    if (run_3d_ndrange_)
        kernel_name = "Conv2DGS3D";
    ret = CreateExecuteUnit(execute_units_[0], "convolution", kernel_name, build_options);
    if (ret != TNN_OK) {
        LOGE("create execute unit failed!\n");
        return ret;
    }

    return TNN_OK;
}

OpenCLConvLayerCommonAcc::~OpenCLConvLayerCommonAcc() {}

Status OpenCLConvLayerCommonAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Conv Common Acc Reshape\n");
    auto input_dims  = inputs[0]->GetBlobDesc().dims;
    auto output_dims = outputs[0]->GetBlobDesc().dims;

    const int output_height = output_dims[2];
    const int output_width  = output_dims[3];

    const int input_height   = input_dims[2];
    const int input_width    = input_dims[3];

    int input_imageshape[2]  = {input_width, input_height};
    int output_imageshape[2] = {output_width, output_height};
    int kernel_shape[2]      = {conv_params_.kernel_x, conv_params_.kernel_y};
    int stride_shape[2]      = {conv_params_.stride_x, conv_params_.stride_y};
    int padding_shape[2]     = {conv_params_.pad_x, conv_params_.pad_y};
    int dilation_shape[2]    = {conv_params_.dilation_x, conv_params_.dilation_y};

    if (run_3d_ndrange_) {
        execute_units_[0].global_work_size = {static_cast<uint32_t>(UP_DIV(output_dims[1], 4)),
                                            static_cast<uint32_t>(UP_DIV(output_dims[3], 4)),
                                            static_cast<uint32_t>(output_dims[0] * output_dims[2])};
        if(kernel_shape[0] == 3 && kernel_shape[1] == 3) {
            execute_units_[0].local_work_size  = Conv2dCommonLocalWS3DKernel3x3(
                execute_units_[0].global_work_size, kernel_shape[0] * kernel_shape[1], execute_units_[0].workgroupsize_max);
        } else {
            execute_units_[0].local_work_size  = Conv2dCommonLocalWS3DGeneral(
                execute_units_[0].global_work_size, kernel_shape[0] * kernel_shape[1], execute_units_[0].workgroupsize_max);
        }
    } else {
        execute_units_[0].global_work_size = {
            static_cast<uint32_t>(UP_DIV(output_dims[1], 4) * UP_DIV(output_dims[3], 4)),
            static_cast<uint32_t>(output_dims[0] * output_dims[2])};
        execute_units_[0].local_work_size = LocalWS2DDefault(execute_units_[0]);
    }


    const int input_channels = input_dims[1];
    const int input_channel_blocks = UP_DIV(input_channels, 4);

    uint32_t idx = 0;
    for (auto gws : execute_units_[0].global_work_size) {
        execute_units_[0].ocl_kernel.setArg(idx++, gws);
    }

    execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)inputs[0]->GetHandle().base));
    execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)ocl_weights_->GetData()));
    execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)ocl_bias_->GetData()));
    execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)outputs[0]->GetHandle().base));
    execute_units_[0].ocl_kernel.setArg(idx++, sizeof(input_imageshape), input_imageshape);
    
    execute_units_[0].ocl_kernel.setArg(idx++, input_channel_blocks);
    execute_units_[0].ocl_kernel.setArg(idx++, sizeof(output_imageshape), output_imageshape);
    execute_units_[0].ocl_kernel.setArg(idx++, sizeof(kernel_shape), kernel_shape);
    execute_units_[0].ocl_kernel.setArg(idx++, sizeof(stride_shape), stride_shape);
    execute_units_[0].ocl_kernel.setArg(idx++, sizeof(padding_shape), padding_shape);
    execute_units_[0].ocl_kernel.setArg(idx++, sizeof(dilation_shape), dilation_shape);
    execute_units_[0].ocl_kernel.setArg(idx++, UP_DIV(output_width, 4));

    return TNN_OK;
}

}  // namespace TNN_NS
