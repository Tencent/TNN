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
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

DECLARE_OPENCL_ACC(PadV2);

Status OpenCLPadV2LayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                                 const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Init PadV2 Acc\n");
    Status ret = OpenCLLayerAcc::Init(context, param, resource, inputs, outputs);
    CHECK_TNN_OK(ret)

    auto output_dims = outputs[0]->GetBlobDesc().dims;
    auto input_dims  = inputs[0]->GetBlobDesc().dims;

    if (input_dims.size() != 4 || output_dims.size() != 4) {
        LOGE("PadV2 Layer (OpenCL) only support 4-dim by now\n");
        return Status(TNNERR_INVALID_INPUT, "PadV2 Layer in OpenCL only support 4-dim by now\n");
    }

    run_3d_ndrange_ = true;
    op_name_        = "PadV2";

    PadLayerParam *pad_param = dynamic_cast<PadLayerParam *>(param);
    if (!pad_param) {
        LOGE("Error: layer param is null\n");
        return Status(TNNERR_MODEL_ERR, "Error: layer param is null");
    }

    auto dim_size = pad_param->pads.size() / 2;
    dim_size      = dim_size <= output_dims.size() ? dim_size : output_dims.size();
    for (int i = 0; i < dim_size; i++) {
        if (pad_param->pads[i] != pad_param->pads[i + dim_size]) {
            LOGE("PadV2 Layer (OpenCL) pad param is invalid (begin must be equal to end by now)\n");
            return Status(TNNERR_PARAM_ERR,
                          "PadV2 Layer (OpenCL) pad param is invalid (begin must be equal to end by now)\n");
        }
    }

    if (0 == pad_param->type) {
        ret = CreateExecuteUnit(execute_units_[0], "pad", "PadConst", build_options_);
    } else if (1 == pad_param->type) {
        ret = CreateExecuteUnit(execute_units_[0], "pad", "PadReflect", build_options_);
    } else {
        return Status(TNNERR_PARAM_ERR, "this pad type is not support yet!");
    }
    if (ret != TNN_OK) {
        LOGE("create execute unit failed!\n");
        return ret;
    }

    return TNN_OK;
}

Status OpenCLPadV2LayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("PadV2 Acc Reshape\n");
    Status ret = OpenCLLayerAcc::Reshape(inputs, outputs);
    CHECK_TNN_OK(ret)

    PadLayerParam *pad_param = dynamic_cast<PadLayerParam *>(param_);
    if (!pad_param) {
        LOGE("Error: layer param is null\n");
        return Status(TNNERR_MODEL_ERR, "Error: layer param is null");
    }
    auto output_dims = outputs[0]->GetBlobDesc().dims;
    auto input_dims  = inputs[0]->GetBlobDesc().dims;
    uint32_t idx     = SetExecuteUnit3DSizeInfoDefault(execute_units_[0], output_dims);
    execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)inputs[0]->GetHandle().base));
    execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)outputs[0]->GetHandle().base));
    execute_units_[0].ocl_kernel.setArg(idx++, DimsFunctionUtils::GetDim(output_dims, 2));
    execute_units_[0].ocl_kernel.setArg(idx++, DimsFunctionUtils::GetDim(input_dims, 1));
    execute_units_[0].ocl_kernel.setArg(idx++, DimsFunctionUtils::GetDim(input_dims, 2));
    execute_units_[0].ocl_kernel.setArg(idx++, DimsFunctionUtils::GetDim(input_dims, 3));
    execute_units_[0].ocl_kernel.setArg(idx++, pad_param->pads[3]);
    execute_units_[0].ocl_kernel.setArg(idx++, pad_param->pads[2]);
    execute_units_[0].ocl_kernel.setArg(idx++, pad_param->pads[1]);
    if (0 == pad_param->type) {
        execute_units_[0].ocl_kernel.setArg(idx++, pad_param->value);
    }

    return TNN_OK;
}

REGISTER_OPENCL_ACC(PadV2, LAYER_PADV2)
REGISTER_OPENCL_LAYOUT(LAYER_PADV2, DATA_FORMAT_NHC4W4);

}  // namespace TNN_NS
