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

DECLARE_OPENCL_ACC(Split);

Status OpenCLSplitLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                                 const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Init Split Acc\n");
    Status ret = OpenCLLayerAcc::Init(context, param, resource, inputs, outputs);
    CHECK_TNN_OK(ret)

    run_3d_ndrange_ = false;
    op_name_        = "Split";

    // create kernel
    execute_units_.resize(outputs.size());
    for (size_t i = 0; i < execute_units_.size(); i++) {
        ret = CreateExecuteUnit(execute_units_[i], "copy", "CopyImage", build_options_);
        if (ret != TNN_OK) {
            LOGE("create execute unit failed!\n");
            return ret;
        }
    }

    return TNN_OK;
}

Status OpenCLSplitLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Split Acc Reshape\n");
    Status ret = OpenCLLayerAcc::Reshape(inputs, outputs);
    CHECK_TNN_OK(ret)

    auto input  = inputs[0];
    auto output = outputs[0];

    auto input_dims  = input->GetBlobDesc().dims;
    auto output_dims = output->GetBlobDesc().dims;

    const int batch         = DimsFunctionUtils::GetDim(output_dims, 0);
    const int channels      = DimsFunctionUtils::GetDim(output_dims, 1);
    const int output_height = DimsFunctionUtils::GetDim(output_dims, 2);
    const int output_width  = DimsFunctionUtils::GetDim(output_dims, 3);

    int inputWH[]      = {DimsFunctionUtils::GetDim(input_dims, 3), DimsFunctionUtils::GetDim(input_dims, 2)};
    int inputOffset[]  = {0, 0, 0, 0};
    int outputOffset[] = {0, 0, 0, 0};
    for (int i = 0; i < execute_units_.size(); ++i) {
        auto output    = outputs[i];
        int outputWH[] = {output_width, output_height};

        auto &unit = execute_units_[i];
        int idx    = SetExecuteUnit2DSizeInfoDefault(unit, output_dims);
        unit.ocl_kernel.setArg(idx++, *((cl::Image *)inputs[0]->GetHandle().base));
        unit.ocl_kernel.setArg(idx++, *((cl::Image *)output->GetHandle().base));
        unit.ocl_kernel.setArg(idx++, inputOffset);
        unit.ocl_kernel.setArg(idx++, outputOffset);
        unit.ocl_kernel.setArg(idx++, inputWH);
        unit.ocl_kernel.setArg(idx++, outputWH);
        unit.ocl_kernel.setArg(idx++, outputWH);
    }

    return TNN_OK;
}

REGISTER_OPENCL_ACC(Split, LAYER_SPLITING)
REGISTER_OPENCL_LAYOUT(LAYER_SPLITING, DATA_FORMAT_NHC4W4);

}  // namespace TNN_NS
