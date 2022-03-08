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
#include "tnn/utils/string_utils_inner.h"

namespace TNN_NS {

DECLARE_OPENCL_ACC(Pooling);

#define LowOpParallelismThre 256
#define HighOpIntensityThre 128

Status OpenCLPoolingLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                                   const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Init Pooling Acc\n");
    Status ret = OpenCLLayerAcc::Init(context, param, resource, inputs, outputs);
    CHECK_TNN_OK(ret)

    run_3d_ndrange_ = true;
    op_name_        = "Pooling";

    PoolingLayerParam *pooling_param = dynamic_cast<PoolingLayerParam *>(param);
    if (!pooling_param) {
        LOGE("Error: layer param is null\n");
        return Status(TNNERR_MODEL_ERR, "Error: layer param is null");
    }

    if (pooling_param->pad_type == 1) {  // VALID Type
        pooling_param->pads[0] = 0;
        pooling_param->pads[2] = 0;
    }

    // create kernel
    std::set<std::string> build_options;
    std::string kernel_name = "Pooling";

    auto input  = inputs[0];
    auto output = outputs[0];

    auto input_dims  = input->GetBlobDesc().dims;
    auto output_dims = output->GetBlobDesc().dims;

    const int batch         = DimsFunctionUtils::GetDim(output_dims, 0);
    const int output_height = DimsFunctionUtils::GetDim(output_dims, 2);
    const int output_width  = DimsFunctionUtils::GetDim(output_dims, 3);
    const int channels      = DimsFunctionUtils::GetDim(output_dims, 1);

    const int kernel_height = pooling_param->kernels[1];
    const int kernel_width  = pooling_param->kernels[0];

    const int channel_blocks = UP_DIV(channels, 4);

    bool run_local_work = batch * output_height * output_width * channel_blocks < LowOpParallelismThre &&
                          kernel_width * kernel_height >= HighOpIntensityThre;
    if (run_local_work) {
        kernel_name += "Local";
    }

    if (pooling_param->pool_type != 0) {  // 0:max_pooling  other:average pooling
        build_options.emplace("-DPOOL_AVG");
    }
    build_options.insert(build_options_.begin(), build_options_.end());
    ret = CreateExecuteUnit(execute_units_[0], "pooling", kernel_name, build_options);
    if (ret != TNN_OK) {
        LOGE("create execute unit failed!\n");
        return ret;
    }

    return TNN_OK;
}

Status OpenCLPoolingLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Pooling Acc Reshape\n");
    Status ret = OpenCLLayerAcc::Reshape(inputs, outputs);
    CHECK_TNN_OK(ret)

    PoolingLayerParam *pooling_param = dynamic_cast<PoolingLayerParam *>(param_);
    if (!pooling_param) {
        LOGE("Error: layer param is null\n");
        return Status(TNNERR_MODEL_ERR, "Error: layer param is null");
    }

    auto input  = inputs[0];
    auto output = outputs[0];
    auto &unit  = execute_units_[0];

    auto input_dims  = input->GetBlobDesc().dims;
    auto output_dims = output->GetBlobDesc().dims;

    const int batch         = DimsFunctionUtils::GetDim(output_dims, 0);
    const int output_height = DimsFunctionUtils::GetDim(output_dims, 2);
    const int output_width  = DimsFunctionUtils::GetDim(output_dims, 3);
    const int channels      = DimsFunctionUtils::GetDim(output_dims, 1);

    const int input_height = DimsFunctionUtils::GetDim(input_dims, 2);
    const int input_width  = DimsFunctionUtils::GetDim(input_dims, 3);

    const int channel_blocks    = UP_DIV(channels, 4);
    uint32_t workgroup_size     = 0;

    OpenCLRuntime * opencl_runtime = OpenCLRuntime::GetInstance();
    int type_size = sizeof(float);
    if (opencl_runtime->GetPrecision() != PRECISION_HIGH && pooling_param->pool_type == 0) {
        type_size = 2;
    }

    int compute_intensity = pooling_param->kernels[0] * pooling_param->kernels[1];
    bool run_local_work = batch * output_height * output_width * channel_blocks < LowOpParallelismThre &&
                          compute_intensity >= HighOpIntensityThre;

    if (run_local_work) {
        workgroup_size = std::min(static_cast<uint32_t>(unit.local_mem_size / (4 * type_size)),
                                  unit.workgroupsize_max);
        workgroup_size = std::min(static_cast<uint32_t>(compute_intensity), workgroup_size);
        uint32_t temp_size = 1;
        while ((temp_size <<= 1) <= workgroup_size);
        workgroup_size = temp_size >> 1;

        int workgroup_w_size = 1, workgroup_h_size;
        while ((workgroup_w_size <<= 1) <= pooling_param->kernels[0] && workgroup_w_size <= workgroup_size);
        workgroup_w_size >>= 1;
        workgroup_h_size = workgroup_size / workgroup_w_size;

        execute_units_[0].global_work_size = {
            static_cast<uint32_t>(channel_blocks * workgroup_size),
            static_cast<uint32_t>(output_width),
            static_cast<uint32_t>(batch * output_height),
        };

        execute_units_[0].local_work_size = {workgroup_size, 1, 1};

        int input_image_shape[2]        = {input_width, input_height};
        int padding_shape[2]            = {pooling_param->pads[0], pooling_param->pads[2]};
        int stride_shape[2]             = {pooling_param->strides[0], pooling_param->strides[1]};
        int kernel_shape[2]             = {pooling_param->kernels[0], pooling_param->kernels[1]};
        int local_block_size_shape[2]   = {workgroup_w_size, workgroup_h_size};
        int local_block_count_shape[2]  = {UP_DIV(pooling_param->kernels[0], workgroup_w_size),
                                           UP_DIV(pooling_param->kernels[1], workgroup_h_size)};

        uint32_t idx                      = 0;
        execute_units_[0].ocl_kernel.setArg(idx++, execute_units_[0].global_work_size[0]);
        execute_units_[0].ocl_kernel.setArg(idx++, execute_units_[0].global_work_size[1]);
        execute_units_[0].ocl_kernel.setArg(idx++, execute_units_[0].global_work_size[2]);
        execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)input->GetHandle().base));
        execute_units_[0].ocl_kernel.setArg(idx++, sizeof(input_image_shape), input_image_shape);
        execute_units_[0].ocl_kernel.setArg(idx++, static_cast<int32_t>(output_height));
        execute_units_[0].ocl_kernel.setArg(idx++, sizeof(padding_shape), padding_shape);
        execute_units_[0].ocl_kernel.setArg(idx++, sizeof(stride_shape), stride_shape);
        execute_units_[0].ocl_kernel.setArg(idx++, sizeof(kernel_shape), kernel_shape);
        execute_units_[0].ocl_kernel.setArg(idx++, static_cast<int32_t>(workgroup_size));
        execute_units_[0].ocl_kernel.setArg(idx++, sizeof(local_block_size_shape), local_block_size_shape);
        execute_units_[0].ocl_kernel.setArg(idx++, sizeof(local_block_count_shape), local_block_count_shape);
        execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)output->GetHandle().base));
        execute_units_[0].ocl_kernel.setArg(idx++, workgroup_size * 4 * type_size, nullptr);
    } else {
        execute_units_[0].global_work_size = {
            static_cast<uint32_t>(channel_blocks),
            static_cast<uint32_t>(output_width),
            static_cast<uint32_t>(batch * output_height),
        };

        int input_image_shape[2] = {input_width, input_height};
        int padding_shape[2]     = {pooling_param->pads[0], pooling_param->pads[2]};
        int stride_shape[2]      = {pooling_param->strides[0], pooling_param->strides[1]};
        int kernel_shape[2]      = {pooling_param->kernels[0], pooling_param->kernels[1]};

        execute_units_[0].local_work_size = LocalWS3DDefault(execute_units_[0]);
        uint32_t idx                      = 0;
        execute_units_[0].ocl_kernel.setArg(idx++, execute_units_[0].global_work_size[0]);
        execute_units_[0].ocl_kernel.setArg(idx++, execute_units_[0].global_work_size[1]);
        execute_units_[0].ocl_kernel.setArg(idx++, execute_units_[0].global_work_size[2]);
        execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)input->GetHandle().base));
        execute_units_[0].ocl_kernel.setArg(idx++, sizeof(input_image_shape), input_image_shape);
        execute_units_[0].ocl_kernel.setArg(idx++, static_cast<int32_t>(output_height));
        execute_units_[0].ocl_kernel.setArg(idx++, sizeof(padding_shape), padding_shape);
        execute_units_[0].ocl_kernel.setArg(idx++, sizeof(stride_shape), stride_shape);
        execute_units_[0].ocl_kernel.setArg(idx++, sizeof(kernel_shape), kernel_shape);
        execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)output->GetHandle().base));
    }

    if (ocl_context_->GetEnableTuneKernel()) {
        std::string tune_key = unit.program_name + "_" + unit.kernel_name + "_" + "param[" +
                               "kernel_" + ToString(pooling_param->kernels[0]) + "_" + ToString(pooling_param->kernels[1]) + "_" +
                               "pad_" + ToString(pooling_param->pads[0]) + "_" + ToString(pooling_param->pads[1]) + "_" +
                               "stride_" + ToString(pooling_param->strides[0]) + "_" + ToString(pooling_param->strides[1]) + "_" +
                               "pool_type_" + ToString(pooling_param->pool_type) + "_" + 
                               "ceil_mode_" + ToString(pooling_param->ceil_mode) + "_" + 
                               "pad_type_" + ToString(pooling_param->pad_type) + "]_global";
        for (auto size : unit.global_work_size) {
            tune_key += "_" + ToString(size);
        }
        execute_units_[0].local_work_size = LocalTune(execute_units_[0], ocl_context_, tune_key);
    }

    return TNN_OK;
}

REGISTER_OPENCL_ACC(Pooling, LAYER_POOLING)
REGISTER_OPENCL_LAYOUT(LAYER_POOLING, DATA_FORMAT_NHC4W4);

}  // namespace TNN_NS
