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

#include "tnn/device/opencl/acc/opencl_reduce_layer_acc.h"
#include "tnn/device/opencl/imagebuffer_convertor.h"

namespace TNN_NS {

#define LowOpParallelismThre 256
#define HighOpIntensityThre 128

Status OpenCLReduceLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                                  const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Init Reduce Acc\n");
    Status ret = OpenCLLayerAcc::Init(context, param, resource, inputs, outputs);
    CHECK_TNN_OK(ret)

    auto reduce_param = dynamic_cast<ReduceLayerParam *>(param);
    if (!reduce_param) {
        LOGE("Error: layer param is null\n");
        return Status(TNNERR_MODEL_ERR, "Error: layer param is null");
    }

    auto output_dims = outputs[0]->GetBlobDesc().dims;

    int hb   = output_dims[0] * output_dims[2];
    int cw   = output_dims[3] * UP_DIV(output_dims[1], 4);

    auto input_dims  = inputs[0]->GetBlobDesc().dims;
    int axis = reduce_param->axis[0];
    axis     = axis >= 0 ? axis : axis + (int)input_dims.size();

    int axis_n = input_dims[axis];

    run_local_work_ = cw * hb < LowOpParallelismThre && axis_n >= HighOpIntensityThre;

    run_3d_ndrange_         = false;
    std::string kernel_name;
    if (axis == 0) {
        kernel_name = "ReduceC0";
    } else if (axis == 1) {
        kernel_name = "ReduceC1";
    } else if (axis == 2) {
        kernel_name = "ReduceC2";
    } else {
        kernel_name = "ReduceC3";
    }

    if (run_local_work_) {
        kernel_name += "Local";
    }

    std::set<std::string> build_options = CreateBuildOptions();

    ret = CreateExecuteUnit(execute_units_[0], "reduce", kernel_name, build_options);
    if (ret != TNN_OK) {
        LOGE("create execute unit failed!\n");
        return ret;
    }

    return TNN_OK;
}

OpenCLReduceLayerAcc::~OpenCLReduceLayerAcc() {}

Status OpenCLReduceLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Reduce Layer Reshape\n");
    auto reduce_param = dynamic_cast<ReduceLayerParam *>(param_);
    if (!reduce_param) {
        LOGE("Error: layer param is null\n");
        return Status(TNNERR_MODEL_ERR, "Error: layer param is null");
    }

    ASSERT(inputs.size() == 1);

    auto input_dims  = inputs[0]->GetBlobDesc().dims;
    auto output_dims = outputs[0]->GetBlobDesc().dims;

    int hb   = output_dims[0] * output_dims[2];
    int cw   = output_dims[3] * UP_DIV(output_dims[1], 4);
    int c4_n = input_dims[1] / 4;
    int c4_r = input_dims[1] % 4;
    int cw4  = input_dims[3] * c4_n;

    int axis = reduce_param->axis[0];
    axis     = axis >= 0 ? axis : axis + (int)input_dims.size();

    int axis_n = input_dims[axis];

    auto &unit            = execute_units_[0];
    uint32_t workgroup_size = 0;

    OpenCLRuntime *opencl_runtime = OpenCLRuntime::GetInstance();
    int type_size = sizeof(float);
    if (opencl_runtime->GetFp16Enable()) {
        type_size = 2;
    }

    if (run_local_work_) {
        workgroup_size = std::min(static_cast<uint32_t>(unit.local_mem_size / (4 * type_size)),
                                  unit.workgroupsize_max);
        workgroup_size = std::min(static_cast<uint32_t>(axis == 1 ? c4_n : axis_n), workgroup_size);
        int temp_size = 1;
        while ((temp_size <<= 1) <= workgroup_size);
        workgroup_size = temp_size >> 1;

        unit.global_work_size = {static_cast<uint32_t>(cw * workgroup_size), static_cast<uint32_t>(hb)};
        unit.local_work_size  = {workgroup_size, 1};
    } else {
        unit.global_work_size = {static_cast<uint32_t>(cw), static_cast<uint32_t>(hb)};
        unit.local_work_size  = LocalWS2DDefault(unit);
    }

    uint32_t idx = 0;
    execute_units_[0].ocl_kernel.setArg(idx++, execute_units_[0].global_work_size[0]);
    execute_units_[0].ocl_kernel.setArg(idx++, execute_units_[0].global_work_size[1]);

    execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)inputs[0]->GetHandle().base));
    execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)outputs[0]->GetHandle().base));
    execute_units_[0].ocl_kernel.setArg(idx++, input_dims[0]);
    execute_units_[0].ocl_kernel.setArg(idx++, input_dims[1]);
    execute_units_[0].ocl_kernel.setArg(idx++, input_dims[2]);
    execute_units_[0].ocl_kernel.setArg(idx++, input_dims[3]);
    execute_units_[0].ocl_kernel.setArg(idx++, c4_n);
    execute_units_[0].ocl_kernel.setArg(idx++, c4_r);
    execute_units_[0].ocl_kernel.setArg(idx++, cw4);
    execute_units_[0].ocl_kernel.setArg(idx++, axis_n);

    if (run_local_work_) {
        if (axis == 1) {
            execute_units_[0].ocl_kernel.setArg(idx++, UP_DIV(c4_n, workgroup_size));
        } else {
            execute_units_[0].ocl_kernel.setArg(idx++, UP_DIV(axis_n, workgroup_size));
        }
        execute_units_[0].ocl_kernel.setArg(idx++, workgroup_size * 4 * type_size, nullptr);
    }

    return TNN_OK;
}

}  // namespace TNN_NS
