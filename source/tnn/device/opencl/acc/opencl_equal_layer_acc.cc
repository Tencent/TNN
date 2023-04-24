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

class OpenCLEqualLayerAcc : public OpenCLLayerAcc {
public:
    virtual Status Init(Context *context, LayerParam *param, LayerResource *resource, const std::vector<Blob *> &inputs,
                        const std::vector<Blob *> &outputs) override;

    virtual Status Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) override;

private:
    virtual std::vector<DataType> SupportDataType(int dims_size, BlobType blob_type) override;
};

Status OpenCLEqualLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                                 const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Init Equal Acc\n");
    Status ret = OpenCLLayerAcc::Init(context, param, resource, inputs, outputs);
    CHECK_TNN_OK(ret)

    execute_units_.resize(1);

    ret = CreateExecuteUnit(execute_units_[0], "equal", "EqualGeneral");
    if (ret != TNN_OK) {
        LOGE("create equal execute unit failed!\n");
        return ret;
    }

    return TNN_OK;
}

Status OpenCLEqualLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Equal Acc Reshape\n");
    Status ret = OpenCLLayerAcc::Reshape(inputs, outputs);
    CHECK_TNN_OK(ret)

    auto input0  = inputs[0];
    auto input1  = inputs[1];
    auto output = outputs[0];

    auto input0_dims  = input0->GetBlobDesc().dims;
    auto input1_dims  = input1->GetBlobDesc().dims;
    auto output_dims = output->GetBlobDesc().dims;

    if (input0_dims.size() > 4 || input1_dims.size() > 4 || output_dims.size() > 4) {
        return Status(TNNERR_OPENCL_ACC_RESHAPE_ERROR, "opencl equal layer inputs not support dims > 4");
    }

    std::vector<int> output_shape(4), input0_shape(4), input1_shape(4);

    for (int i = 0; i < 4; ++i) {
        input0_shape[i] = DimsFunctionUtils::GetDim(input0_dims, i);
        input1_shape[i] = DimsFunctionUtils::GetDim(input1_dims, i);
        output_shape[i] = DimsFunctionUtils::GetDim(output_dims, i);
    }

    auto &unit = execute_units_[0];
    int idx    = SetExecuteUnit2DSizeInfoDefault(unit, output_dims);
    unit.ocl_kernel.setArg(idx++, *((cl::Image *)inputs[0]->GetHandle().base));
    unit.ocl_kernel.setArg(idx++, *((cl::Image *)inputs[1]->GetHandle().base));
    unit.ocl_kernel.setArg(idx++, 4 * sizeof(int), output_shape.data());
    unit.ocl_kernel.setArg(idx++, 4 * sizeof(int), input0_shape.data());
    unit.ocl_kernel.setArg(idx++, 4 * sizeof(int), input1_shape.data());
    unit.ocl_kernel.setArg(idx++, UP_DIV(input0_shape[1], 4));
    unit.ocl_kernel.setArg(idx++, UP_DIV(input1_shape[1], 4));
    unit.ocl_kernel.setArg(idx++, *((cl::Image *)output->GetHandle().base));

    return TNN_OK;
}

std::vector<DataType> OpenCLEqualLayerAcc::SupportDataType(int dims_size, BlobType blob_type) {
    return {DATA_TYPE_INT32, DATA_TYPE_INT8};
}


REGISTER_OPENCL_ACC(Equal, LAYER_EQUAL)
REGISTER_OPENCL_LAYOUT(LAYER_EQUAL, DATA_FORMAT_NHC4W4);

}  // namespace TNN_NS
