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

DECLARE_OPENCL_ACC(Softmax);

#define HighOpIntensityThre 128

Status OpenCLSoftmaxLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                                   const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Init SoftMax Acc\n");
    Status ret = OpenCLLayerAcc::Init(context, param, resource, inputs, outputs);
    CHECK_TNN_OK(ret)

    run_3d_ndrange_ = true;
    op_name_        = "SoftMax";

    SoftmaxLayerParam *softmax_param = dynamic_cast<SoftmaxLayerParam *>(param);
    if (!softmax_param) {
        LOGE("Error: layer param is null\n");
        return Status(TNNERR_MODEL_ERR, "Error: layer param is null");
    }

    // create kernel
    std::string kernel_name;
    if (softmax_param->axis == 1) {
        kernel_name = "SoftmaxChannel";
    } else if (softmax_param->axis == 2) {
        kernel_name = "SoftmaxHeight";
    } else {
        LOGE("not support axis = %d in softmax yet!\n", softmax_param->axis);
        return Status(TNNERR_OPENCL_ACC_INIT_ERROR, "invalid softmax axis");
    }

    auto output_dims    = outputs[0]->GetBlobDesc().dims;
    const int batch     = output_dims[0];
    int cw              = output_dims[3] * UP_DIV(output_dims[1], 4);

    auto input_dims     = inputs[0]->GetBlobDesc().dims;
    int axis_n          = input_dims[softmax_param->axis];

    // only support fine-grained parallelism in softmax height
    bool run_local_work = softmax_param->axis == 2 && axis_n >= HighOpIntensityThre;
    if (run_local_work) {
        kernel_name += "Local";
    }

    std::set<std::string> build_options;
    AdjustBuildOptionForFp32(build_options);

    ret = CreateExecuteUnit(execute_units_[0], "softmax", kernel_name, build_options);
    if (ret != TNN_OK) {
        LOGE("create execute unit failed!\n");
        return ret;
    }

    return TNN_OK;
}

Status OpenCLSoftmaxLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("SoftMax Layer Reshape\n");
    SoftmaxLayerParam *softmax_param = dynamic_cast<SoftmaxLayerParam *>(param_);
    if (!softmax_param) {
        LOGE("Error: layer param is null\n");
        return Status(TNNERR_MODEL_ERR, "Error: layer param is null");
    }

    ASSERT(inputs.size() == 1);

    auto input_dims  = inputs[0]->GetBlobDesc().dims;
    auto output_dims = outputs[0]->GetBlobDesc().dims;

    const int batch     = output_dims[0];
    const int channels  = output_dims[1];
    const int height    = output_dims[2];
    const int width     = output_dims[3];
    int c4_n            = input_dims[1] / 4;

    const int channelBlocks  = UP_DIV(channels, 4);
    const int remainChannels = channelBlocks * 4 - channels;

    int cw      = output_dims[3] * channelBlocks;
    int axis_n  = input_dims[softmax_param->axis];

    uint32_t idx = 0;

    OpenCLRuntime *opencl_runtime = OpenCLRuntime::GetInstance();
    int type_size = sizeof(float);
    if (opencl_runtime->GetPrecision() != PRECISION_HIGH) {
        type_size = 2;
    }

    if (1 == softmax_param->axis) {
        execute_units_[0].global_work_size = {static_cast<uint32_t>(channelBlocks), static_cast<uint32_t>(width),
                                                static_cast<uint32_t>(height * batch)};
        uint32_t idx                       = 0;
        execute_units_[0].ocl_kernel.setArg(idx++, execute_units_[0].global_work_size[0]);
        execute_units_[0].ocl_kernel.setArg(idx++, execute_units_[0].global_work_size[1]);
        execute_units_[0].ocl_kernel.setArg(idx++, execute_units_[0].global_work_size[2]);

        execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)inputs[0]->GetHandle().base));
        execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)outputs[0]->GetHandle().base));
        execute_units_[0].ocl_kernel.setArg(idx++, static_cast<int>(channels));
        execute_units_[0].ocl_kernel.setArg(idx++, remainChannels);
        execute_units_[0].local_work_size = LocalWS3DDefault(execute_units_[0]);
    } else if (2 == softmax_param->axis) {
        bool run_local_work     = axis_n >= HighOpIntensityThre;
        uint32_t workgroup_size = 0;
        auto &unit              = execute_units_[0];
        if (run_local_work) {
            workgroup_size = std::min(static_cast<uint32_t>(unit.local_mem_size / (4 * type_size)),
                                      unit.workgroupsize_max);
            workgroup_size = std::min(static_cast<uint32_t>(axis_n), workgroup_size);
            int temp_size = 1;
            while ((temp_size <<= 1) <= workgroup_size);
            workgroup_size = temp_size >> 1;

            unit.global_work_size = {static_cast<uint32_t>(cw * workgroup_size), static_cast<uint32_t>(batch)};
            unit.local_work_size  = {workgroup_size, 1};
        } else {
            if (execute_units_[0].workgroupsize_max > 256) {
                execute_units_[0].local_work_size = {16, 16, 1};
            } else {
                execute_units_[0].local_work_size = {8, 8, 1};
            }
            execute_units_[0].global_work_size = {(uint32_t)channelBlocks * width, (uint32_t)batch, 1};
        }
        int shape[]                        = {batch, channelBlocks, height, width};

        execute_units_[0].ocl_kernel.setArg(idx++, execute_units_[0].global_work_size[0]);
        execute_units_[0].ocl_kernel.setArg(idx++, execute_units_[0].global_work_size[1]);
        execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)inputs[0]->GetHandle().base));
        execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)outputs[0]->GetHandle().base));
        execute_units_[0].ocl_kernel.setArg(idx++, shape);
        if (run_local_work) {
            execute_units_[0].ocl_kernel.setArg(idx++, UP_DIV(axis_n, workgroup_size));
            execute_units_[0].ocl_kernel.setArg(idx++, workgroup_size * 4 * type_size, nullptr);
        }
    } else {
        LOGE("not support axis = %d in softmax yet!\n", softmax_param->axis);
        return Status(TNNERR_OPENCL_ACC_RESHAPE_ERROR, "invalid softmax axis");
    }

    return TNN_OK;
}

REGISTER_OPENCL_ACC(Softmax, LAYER_SOFTMAX)

}  // namespace TNN_NS
