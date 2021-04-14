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

#include "tnn/device/opencl/acc/convolution/opencl_conv_layer_1x1_acc.h"
#include "tnn/device/opencl/imagebuffer_convertor.h"

namespace TNN_NS {

#define LowOpHWThre 9
#define HighOpIntensityThre 128

// magic number
static const uint32_t lws_limit         = 128;

bool OpenCLConvLayer1x1Acc::IsPrefered(const ConvLayerParam *param, const std::vector<Blob *> &,
                                       const std::vector<Blob *> &) {
    if (!param) {
        return false;
    }
    return param->group == 1 && param->kernels[0] == 1 && param->kernels[1] == 1 && param->dialations[0] == 1 && 
            param->dialations[1] == 1 && param->pads[0] == 0 && param->pads[1] == 0;
}

Status OpenCLConvLayer1x1Acc::Init(Context *context, LayerParam *param, LayerResource *resource,
                                   const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Init Conv 1x1 Acc\n");

    conv_type_ = CT_CONV_1x1;
    op_name_   = "Conv_1x1";

    // AccImpl init first
    Status ret = OpenCLConvLayerAccImpl::Init(context, param, resource, inputs, outputs);
    CHECK_TNN_OK(ret)

    if (1 == conv_params_.stride_x && 1 == conv_params_.stride_y) {
        stride_is_1_ = true;
    }

    if (stride_is_1_ && !run_3d_ndrange_) {
        if (MALI_T == gpu_info_.type || MALI_G == gpu_info_.type) {
            use_buffer_ = true;
        }
    }

    ret = AllocateWeightsBias(resource);
    CHECK_TNN_OK(ret)

    auto input_dims = inputs[0]->GetBlobDesc().dims;
    auto output_dims = outputs[0]->GetBlobDesc().dims;

    const int input_channel     = input_dims[1];
    const int output_batch      = output_dims[0];
    const int output_channel    = output_dims[1];
    const int output_height     = output_dims[2];
    const int output_width      = output_dims[3];

    std::string kernel_name;
    if (run_3d_ndrange_) {
        kernel_name = "Conv2D1x1GS3D";
    } else {
        kernel_name = "Conv2D1x1";
    }
    if (stride_is_1_) {
        kernel_name += "_S1";
    }
    if (use_buffer_) {
        kernel_name += "_MIX";
    }

    uint32_t compute_units = OpenCLRuntime::GetInstance()->DeviceComputeUnits();
    int task_size = output_batch * UP_DIV(output_channel, 4) * output_height * output_width;
    int task_size_per_cu = task_size / static_cast<int>(compute_units);
    if (!run_3d_ndrange_ && stride_is_1_ && use_buffer_ && task_size_per_cu < 256) {
        width_blocking_is_1_ = true;
        kernel_name += "_WB1";
        run_local_work_ = (UP_DIV(input_channel, 4) >= HighOpIntensityThre) &&
                          (output_height * output_width <= LowOpHWThre);
        if (run_local_work_) {
            kernel_name += "_Local";
        }
    } else if (output_channel > 4 && run_3d_ndrange_ && !use_buffer_) {
        is_channel_blocking_ = true;
        kernel_name += "_CB2";
    }

    ret = CreateExecuteUnit(execute_units_[0], "convolution", kernel_name, build_options_);
    if (ret != TNN_OK) {
        LOGE("create execute unit failed!\n");
        return ret;
    }

    LOGD("conv1x1: use buffer: %s  run_3d: %s\n", use_buffer_ ? "Yes" : "No", run_3d_ndrange_ ? "Yes" : "No");

    return TNN_OK;
}

OpenCLConvLayer1x1Acc::~OpenCLConvLayer1x1Acc() {}

Status OpenCLConvLayer1x1Acc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Conv 1x1 Acc Reshape\n");

    auto input_dims  = inputs[0]->GetBlobDesc().dims;
    auto output_dims = outputs[0]->GetBlobDesc().dims;

    const int input_channel_blocks = UP_DIV(input_dims[1], 4);

    const int output_channels = output_dims[1];
    const int output_channel_blocks = UP_DIV(output_channels, 4);

    int type_size = sizeof(float);
    if (OpenCLRuntime::GetInstance()->GetPrecision() != PRECISION_HIGH) {
        type_size = 2;
    }
    auto &unit              = execute_units_[0];
    uint32_t workgroup_size = 0;

    if (run_3d_ndrange_) {
        if (is_channel_blocking_) {
            execute_units_[0].global_work_size = {static_cast<uint32_t>(UP_DIV(output_dims[1], 8)),
                                                  static_cast<uint32_t>(UP_DIV(output_dims[3], 4)),
                                                  static_cast<uint32_t>(output_dims[0] * output_dims[2])};
        } else {
            execute_units_[0].global_work_size = {static_cast<uint32_t>(UP_DIV(output_dims[1], 4)),
                                                  static_cast<uint32_t>(UP_DIV(output_dims[3], 4)),
                                                  static_cast<uint32_t>(output_dims[0] * output_dims[2])};
        }

        execute_units_[0].local_work_size =
            Conv2d1x1LocalWS3D(execute_units_[0].global_work_size, execute_units_[0].workgroupsize_max);

    } else {
         if (width_blocking_is_1_) {
            if (run_local_work_) {
                auto &unit              = execute_units_[0];
                workgroup_size = std::min(static_cast<uint32_t>(unit.local_mem_size / (4 * type_size)),
                                          unit.workgroupsize_max);
                workgroup_size = std::min(static_cast<uint32_t>(input_channel_blocks), workgroup_size);
                int temp_size = 1;
                while ((temp_size <<= 1) <= workgroup_size);
                workgroup_size = temp_size >> 1;

                execute_units_[0].global_work_size = {static_cast<uint32_t>(UP_DIV(output_dims[1], 4) * output_dims[3] * workgroup_size),
                                                      static_cast<uint32_t>(output_dims[0] * output_dims[2])};
            } else {
                execute_units_[0].global_work_size = {static_cast<uint32_t>(UP_DIV(output_dims[1], 4) * output_dims[3]),
                                                      static_cast<uint32_t>(output_dims[0] * output_dims[2])};
            }
        } else if (is_channel_blocking_) {
            execute_units_[0].global_work_size = {
                static_cast<uint32_t>(UP_DIV(output_dims[1], 8) * UP_DIV(output_dims[3], 4)),
                static_cast<uint32_t>(output_dims[0] * output_dims[2])};
        } else {
            execute_units_[0].global_work_size = {
                static_cast<uint32_t>(UP_DIV(output_dims[1], 4) * UP_DIV(output_dims[3], 4)),
                static_cast<uint32_t>(output_dims[0] * output_dims[2])};
        }

        if (!run_local_work_) {
            execute_units_[0].local_work_size = Conv2dCommonLocalWS2D(
                execute_units_[0].global_work_size, execute_units_[0].workgroupsize_max, execute_units_[0].sub_group_size);
        } else {
            execute_units_[0].local_work_size = {workgroup_size, 1};
        }
    }
    //input width, input height
    int input_imageshape[2]  = {input_dims[3], input_dims[2]};
    //output width, output height
    int output_imageshape[2] = {output_dims[3], output_dims[2]};
    int stride_shape[2]      = {conv_params_.stride_x, conv_params_.stride_y};
    uint32_t idx             = 0;
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
    execute_units_[0].ocl_kernel.setArg(idx++, static_cast<int>(input_channel_blocks));
    if (is_channel_blocking_) {
        execute_units_[0].ocl_kernel.setArg(idx++, static_cast<int>(output_channel_blocks));
    }
    if (!stride_is_1_) {
        execute_units_[0].ocl_kernel.setArg(idx++, sizeof(output_imageshape), output_imageshape);
        execute_units_[0].ocl_kernel.setArg(idx++, sizeof(stride_shape), stride_shape);
    }
    if (!width_blocking_is_1_) {
        // set value (output width / 4)
        execute_units_[0].ocl_kernel.setArg(idx++, UP_DIV(output_dims[3], 4));
    }

    if (run_local_work_) {
        execute_units_[0].ocl_kernel.setArg(idx++, UP_DIV(input_channel_blocks, workgroup_size));
        execute_units_[0].ocl_kernel.setArg(idx++, workgroup_size * 4 * type_size, nullptr);
    }

    if (!run_local_work_ && ocl_context_->GetEnableTuneKernel()) {
        execute_units_[0].local_work_size = LocalTune(execute_units_[0], ocl_context_, GenerateTuneKernelKey(execute_units_[0]));
    }

    return TNN_OK;
}

std::vector<uint32_t> OpenCLConvLayer1x1Acc::Conv2d1x1LocalWS3D(std::vector<uint32_t> &gws,
                                                                const uint32_t max_workgroup_size) {
    uint32_t compute_units = OpenCLRuntime::GetInstance()->DeviceComputeUnits();
    uint64_t cache_size    = OpenCLRuntime::GetInstance()->DeviceGlobalMemeryCacheSize();
    LOGD("cache_size: %d\n", (int)cache_size);
    const uint32_t base = std::max<uint32_t>(cache_size / g_base_gpu_mem_cachesize, 1);
    // (inputs + weights + outputs) * array_size * sizeof(float)
    uint32_t kernel_cache_size = is_channel_blocking_ ? (4 + 8 + 8) * 4 * 4 : (4 + 4 + 4) * 4 * 4;

    std::vector<uint32_t> lws(3, 1);
    if(max_workgroup_size > 0) {
        lws[1]              = std::min<uint32_t>(gws[1], max_workgroup_size);
        if (lws[1] >= base) {
            lws[0] = std::min<uint32_t>(gws[0], base);
        } else if ((1 < lws[1] && lws[1] < base) && gws[0] >= lws_limit) {
            lws[0] = std::min<uint32_t>(gws[0], base);
        } else {
            lws[0] = gws[0] / 8;
            if (lws[0] < base) {
                lws[0] = std::max<uint32_t>(gws[0] / 4, base);
            }
        }
        lws[0]                  = std::min<uint32_t>(lws[0], max_workgroup_size / lws[1]);
        const uint32_t lws_size = lws[0] * lws[1];
        lws[2] = std::min<uint32_t>((cache_size / kernel_cache_size / lws_size / compute_units) * 2, gws[2]);
        if (lws[2] == 0) {
            lws[2] = std::min<uint32_t>(gws[2], base);
        }
        lws[2] = std::max<uint32_t>(std::min<uint32_t>(lws[2], max_workgroup_size / lws_size), 1);
    }
    LOGD("compute_units : %d , max_workgroup_size : %d\n", compute_units, max_workgroup_size);
    LOGD("layer: %s conv1x1 [%d, %d, %d] -- [%d, %d, %d] \n", layer_name_.c_str(), gws[0], gws[1], gws[2], lws[0],
         lws[1], lws[2]);

    return lws;
}

}  // namespace TNN_NS
