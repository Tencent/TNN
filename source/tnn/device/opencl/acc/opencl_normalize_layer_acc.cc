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

#include "tnn/utils/data_type_utils.h"

namespace TNN_NS {

DECLARE_OPENCL_ACC(Normalize);

Status OpenCLNormalizeLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                                     const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Init Normalize Acc\n");
    Status ret = OpenCLLayerAcc::Init(context, param, resource, inputs, outputs);
    CHECK_TNN_OK(ret)

    run_3d_ndrange_ = false;
    op_name_        = "Normalize";

    return TNN_OK;
}

Status OpenCLNormalizeLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Normalize Layer Reshape\n");
    Status ret = OpenCLLayerAcc::Reshape(inputs, outputs);
    CHECK_TNN_OK(ret)

    auto norm_param = dynamic_cast<NormalizeLayerParam *>(param_);
    if (!norm_param) {
        LOGE("Error: layer param is null\n");
        return Status(TNNERR_MODEL_ERR, "Error: layer param is null");
    }
    if (norm_param->p != 1 && norm_param->p != 2) {
        LOGE("the param p=%d is not support yet\n", norm_param->p);
        return Status(TNNERR_MODEL_ERR, "invalid param p");
    }

    ASSERT(inputs.size() == 1);

    auto input_dims  = inputs[0]->GetBlobDesc().dims;
    auto output_dims = outputs[0]->GetBlobDesc().dims;

    // create kernel
    std::set<std::string> build_options;
    if (2 == norm_param->p) {
        build_options.emplace("-DNORMALIZE_P2");
    }
    std::string kernel_name;
    if (DimsFunctionUtils::GetDim(input_dims, 1) % 4 == 0) {
        kernel_name = "NormalizeCommon0";
    } else {
        kernel_name = "NormalizeCommon";
    }

    build_options.insert(build_options_.begin(), build_options_.end());
    ret = CreateExecuteUnit(execute_units_[0], "normalize", kernel_name, build_options);
    if (ret != TNN_OK) {
        LOGE("create execute unit failed!\n");
        return ret;
    }

    const int batch    = DimsFunctionUtils::GetDim(input_dims, 0);
    const int height   = DimsFunctionUtils::GetDim(input_dims, 2);
    const int width    = DimsFunctionUtils::GetDim(input_dims, 3);
    const int channels = DimsFunctionUtils::GetDim(input_dims, 1);

    const int channel_blocks = UP_DIV(channels, 4);
    const int channel_remain = channels % 4;

    execute_units_[0].global_work_size = {static_cast<uint32_t>(width), static_cast<uint32_t>(batch * height)};
    execute_units_[0].local_work_size  = LocalWS2DDefault(execute_units_[0]);

    uint32_t idx = 0;
    execute_units_[0].ocl_kernel.setArg(idx++, execute_units_[0].global_work_size[0]);
    execute_units_[0].ocl_kernel.setArg(idx++, execute_units_[0].global_work_size[1]);
    execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)inputs[0]->GetHandle().base));
    execute_units_[0].ocl_kernel.setArg(idx++, channel_blocks);
    if (channel_remain != 0) {
        execute_units_[0].ocl_kernel.setArg(idx++, channel_remain);
    }
    execute_units_[0].ocl_kernel.setArg(idx++, width);
    execute_units_[0].ocl_kernel.setArg(idx++, norm_param->epsilon);
    execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)outputs[0]->GetHandle().base));

    return TNN_OK;
}

REGISTER_OPENCL_ACC(Normalize, LAYER_NORMALIZE)
REGISTER_OPENCL_LAYOUT(LAYER_NORMALIZE, DATA_FORMAT_NHC4W4);

}  // namespace TNN_NS
