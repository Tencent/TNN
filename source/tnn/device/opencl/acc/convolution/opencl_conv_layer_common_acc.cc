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

    if (!run_3d_ndrange_) {
        if (MALI_T == gpu_info_.type || MALI_G == gpu_info_.type) {
            use_buffer_ = true;
        }
    }

    ret = AllocateWeightsBias(resource);
    CHECK_TNN_OK(ret)

    auto output_dims = outputs[0]->GetBlobDesc().dims;
    const int output_batch      = DimsFunctionUtils::GetDim(output_dims, 0);
    const int output_channel    = DimsFunctionUtils::GetDim(output_dims, 1);
    const int output_height     = DimsFunctionUtils::GetDim(output_dims, 2);
    const int output_width      = DimsFunctionUtils::GetDim(output_dims, 3);

    std::string program_name = "convolution";
    std::string kernel_name = "Conv2D";
    if (run_3d_ndrange_) {
        program_name = "convolution_gws_3d";
        kernel_name = "Conv2DGS3D";
        if (output_channel > 4) {
            is_channel_blocking_ = true;
            kernel_name += "_CB2";
        }
    } else {
        if (use_buffer_) {
            program_name = "convolution_mix";
            kernel_name += "_MIX";
        }
        int task_size = output_batch * UP_DIV(output_channel, 4) * output_height * output_width;
        if (task_size > 4096 && output_channel > 4) {
            is_channel_blocking_ = true;
            kernel_name += "_CB2";
        }
    }

    {
        // When the GPU used is PowerVR Rogue GE8320, the calculation result of Conv2D_CB2 is incorrect,
        // so use Conv2D for calculation.
        //
        // The problem is that in Conv2D_CB2, use the following code to read the weights,
        // weights_c0_s0 = RI_F(weights, SAMPLER, (int2)(weights_x_idx.x, weights_y_idx.x));
        // weights_c0_s1 = RI_F(weights, SAMPLER, (int2)(weights_x_idx.x, weights_y_idx.y));
        // The expected behavior is that weights_c0_s0 and weights_c0_s1 read different values using different indices.
        // However, on the PowerVR Rogue GE8320, the value of weights_c0_s0 and weights_c0_s1 read using different
        // indexes are the same, which is not as expected, so the calculation result is incorrect.
        // (both weights_c0_s0 and weights_c0_s1 indexes have correct values)
        //
        // Use Conv2D for calculations on the PowerVR Rogue GE8320 to avoid this problem.
        if (kernel_name == "Conv2D_CB2") {
            std::vector<cl::Device> devices;
            std::vector<cl::Platform> platforms;
            cl::Platform::get(&platforms);
            if (platforms.size() == 1) {
                platforms.begin()->getDevices(CL_DEVICE_TYPE_GPU, &devices);
                std::string device_name = devices.begin()->getInfo<CL_DEVICE_NAME>();
                if (device_name == "PowerVR Rogue GE8320") {
                    kernel_name          = "Conv2D";
                    is_channel_blocking_ = false;
                }
            }
        }
    }

    ret = CreateExecuteUnit(execute_units_[0], program_name, kernel_name, build_options_);
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

    const int output_height = DimsFunctionUtils::GetDim(output_dims, 2);
    const int output_width  = DimsFunctionUtils::GetDim(output_dims, 3);

    const int input_height   = DimsFunctionUtils::GetDim(input_dims, 2);
    const int input_width    = DimsFunctionUtils::GetDim(input_dims, 3);

    int input_imageshape[2]  = {input_width, input_height};
    int output_imageshape[2] = {output_width, output_height};
    int kernel_shape[2]      = {conv_params_.kernel_x, conv_params_.kernel_y};
    int stride_shape[2]      = {conv_params_.stride_x, conv_params_.stride_y};
    int padding_shape[2]     = {conv_params_.pad_x, conv_params_.pad_y};
    int dilation_shape[2]    = {conv_params_.dilation_x, conv_params_.dilation_y};

    if (run_3d_ndrange_) {
        if (is_channel_blocking_) {
            execute_units_[0].global_work_size = {
                static_cast<uint32_t>(UP_DIV(DimsFunctionUtils::GetDim(output_dims, 1), 8)),
                static_cast<uint32_t>(UP_DIV(DimsFunctionUtils::GetDim(output_dims, 3), 4)),
                static_cast<uint32_t>(DimsFunctionUtils::GetDim(output_dims, 0) *
                                      DimsFunctionUtils::GetDim(output_dims, 2))};
        } else {
            execute_units_[0].global_work_size = {
                static_cast<uint32_t>(UP_DIV(DimsFunctionUtils::GetDim(output_dims, 1), 4)),
                static_cast<uint32_t>(UP_DIV(DimsFunctionUtils::GetDim(output_dims, 3), 4)),
                static_cast<uint32_t>(DimsFunctionUtils::GetDim(output_dims, 0) *
                                      DimsFunctionUtils::GetDim(output_dims, 2))};
        }

        if (kernel_shape[0] == 3 && kernel_shape[1] == 3) {
            execute_units_[0].local_work_size =
                Conv2dCommonLocalWS3DKernel3x3(execute_units_[0].global_work_size, kernel_shape[0] * kernel_shape[1],
                                               execute_units_[0].workgroupsize_max);
        } else {
            execute_units_[0].local_work_size =
                Conv2dCommonLocalWS3DGeneral(execute_units_[0].global_work_size, kernel_shape[0] * kernel_shape[1],
                                             execute_units_[0].workgroupsize_max);
        }
    } else {
        if (is_channel_blocking_) {
            execute_units_[0].global_work_size = {
                static_cast<uint32_t>(UP_DIV(DimsFunctionUtils::GetDim(output_dims, 1), 8) *
                                      UP_DIV(DimsFunctionUtils::GetDim(output_dims, 3), 4)),
                static_cast<uint32_t>(DimsFunctionUtils::GetDim(output_dims, 0) *
                                      DimsFunctionUtils::GetDim(output_dims, 2))};
        } else {
            execute_units_[0].global_work_size = {
                static_cast<uint32_t>(UP_DIV(DimsFunctionUtils::GetDim(output_dims, 1), 4) *
                                      UP_DIV(DimsFunctionUtils::GetDim(output_dims, 3), 4)),
                static_cast<uint32_t>(DimsFunctionUtils::GetDim(output_dims, 0) *
                                      DimsFunctionUtils::GetDim(output_dims, 2))};
        }
        execute_units_[0].local_work_size = LocalWS2DDefault(execute_units_[0]);
    }

    const int input_channels = DimsFunctionUtils::GetDim(input_dims, 1);
    const int input_channel_blocks = UP_DIV(input_channels, 4);

    const int output_channels = DimsFunctionUtils::GetDim(output_dims, 1);
    const int output_channel_blocks = UP_DIV(output_channels, 4);

    uint32_t idx = 0;
    for (auto gws : execute_units_[0].global_work_size) {
        execute_units_[0].ocl_kernel.setArg(idx++, gws);
    }

    execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)inputs[0]->GetHandle().base));
    if (use_buffer_) {
        execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Buffer *)ocl_weights_->GetData()));
        execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Buffer *)ocl_bias_->GetData()));
    } else {
        execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)ocl_weights_->GetData()));
        execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)ocl_bias_->GetData()));
    }
    execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)outputs[0]->GetHandle().base));
    execute_units_[0].ocl_kernel.setArg(idx++, sizeof(input_imageshape), input_imageshape);
    
    execute_units_[0].ocl_kernel.setArg(idx++, input_channel_blocks);
    if (is_channel_blocking_) {
        execute_units_[0].ocl_kernel.setArg(idx++, output_channel_blocks);
    }
    execute_units_[0].ocl_kernel.setArg(idx++, sizeof(output_imageshape), output_imageshape);
    execute_units_[0].ocl_kernel.setArg(idx++, sizeof(kernel_shape), kernel_shape);
    execute_units_[0].ocl_kernel.setArg(idx++, sizeof(stride_shape), stride_shape);
    execute_units_[0].ocl_kernel.setArg(idx++, sizeof(padding_shape), padding_shape);
    execute_units_[0].ocl_kernel.setArg(idx++, sizeof(dilation_shape), dilation_shape);
    if (is_channel_blocking_) {
        execute_units_[0].ocl_kernel.setArg(idx++, kernel_shape[0] * kernel_shape[1]);
    }
    execute_units_[0].ocl_kernel.setArg(idx++, UP_DIV(output_width, 4));
    execute_units_[0].ocl_kernel.setArg(idx++, (int)conv_params_.activation_type);

    if (ocl_context_->GetEnableTuneKernel()) {
            execute_units_[0].local_work_size = LocalTune(execute_units_[0], ocl_context_, GenerateTuneKernelKey(execute_units_[0]));
    }

    return TNN_OK;
}

}  // namespace TNN_NS
