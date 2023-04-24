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

class OpenCLWhereLayerAcc : public OpenCLLayerAcc {
public:
    virtual Status Init(Context *context, LayerParam *param, LayerResource *resource, const std::vector<Blob *> &inputs,
                        const std::vector<Blob *> &outputs) override;

    virtual Status Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) override;

private:
    virtual std::vector<DataType> SupportDataType(int dims_size, BlobType blob_type) override;
};

Status OpenCLWhereLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                                 const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Init Where Acc\n");
    Status ret = OpenCLLayerAcc::Init(context, param, resource, inputs, outputs);
    CHECK_TNN_OK(ret)

    auto output_dims = outputs[0]->GetBlobDesc().dims;

    execute_units_.resize(1);

    if (output_dims.size() == 5) {
        ret = CreateExecuteUnit(execute_units_[0], "where", "WhereGeneral5D");
        if (ret != TNN_OK) {
            LOGE("create where execute unit failed!\n");
            return ret;
        }
    } else {
        ret = CreateExecuteUnit(execute_units_[0], "where", "WhereGeneral");
        if (ret != TNN_OK) {
            LOGE("create where execute unit failed!\n");
            return ret;
        }
    }

    return TNN_OK;
}

Status OpenCLWhereLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Where Acc Reshape\n");
    Status ret = OpenCLLayerAcc::Reshape(inputs, outputs);
    CHECK_TNN_OK(ret)

    auto input0  = inputs[0];
    auto input1  = inputs[1];
    auto input2  = inputs[2];
    auto output = outputs[0];

    auto input0_dims  = input0->GetBlobDesc().dims;
    auto input1_dims  = input1->GetBlobDesc().dims;
    auto input2_dims  = input2->GetBlobDesc().dims;
    auto output_dims = output->GetBlobDesc().dims;

    if (input0_dims.size() > 5 || input1_dims.size() > 5 || input2_dims.size() > 5 || output_dims.size() > 5) {
        LOGE("input0_dims.size(): %d, input1_dims.size(): %d, input2_dims.size(): %d, output_dims.size(): %d\n",
             input0_dims.size(), input1_dims.size(), input2_dims.size(), output_dims.size());
        return Status(TNNERR_OPENCL_ACC_RESHAPE_ERROR, "opencl where layer inputs not support dims > 5");
    }

    if (output_dims.size() == 5) {
        const int n_dims = 5;
        std::vector<int> output_shape(n_dims), input0_shape(n_dims), input1_shape(n_dims), input2_shape(n_dims);

        for (int i = 0; i < n_dims; ++i) {
            input0_shape[i] = DimsFunctionUtils::GetDim(input0_dims, i);
            input1_shape[i] = DimsFunctionUtils::GetDim(input1_dims, i);
            input2_shape[i] = DimsFunctionUtils::GetDim(input2_dims, i);
            output_shape[i] = DimsFunctionUtils::GetDim(output_dims, i);
        }

        auto &unit = execute_units_[0];
        int idx    = SetExecuteUnit2DSizeInfoDefault(unit, output_dims);
        unit.ocl_kernel.setArg(idx++, *((cl::Image *)inputs[0]->GetHandle().base));
        unit.ocl_kernel.setArg(idx++, *((cl::Image *)inputs[1]->GetHandle().base));
        unit.ocl_kernel.setArg(idx++, *((cl::Image *)inputs[2]->GetHandle().base));
        unit.ocl_kernel.setArg(idx++, n_dims * sizeof(int), output_shape.data());
        unit.ocl_kernel.setArg(idx++, n_dims * sizeof(int), input0_shape.data());
        unit.ocl_kernel.setArg(idx++, n_dims * sizeof(int), input1_shape.data());
        unit.ocl_kernel.setArg(idx++, n_dims * sizeof(int), input2_shape.data());
        unit.ocl_kernel.setArg(idx++, UP_DIV(input0_shape[1], 4));
        unit.ocl_kernel.setArg(idx++, UP_DIV(input1_shape[1], 4));
        unit.ocl_kernel.setArg(idx++, UP_DIV(input2_shape[1], 4));
        unit.ocl_kernel.setArg(idx++, *((cl::Image *)output->GetHandle().base));
    } else {
        std::vector<int> output_shape(4), input0_shape(4), input1_shape(4), input2_shape(4);

        for (int i = 0; i < 4; ++i) {
            input0_shape[i] = DimsFunctionUtils::GetDim(input0_dims, i);
            input1_shape[i] = DimsFunctionUtils::GetDim(input1_dims, i);
            input2_shape[i] = DimsFunctionUtils::GetDim(input2_dims, i);
            output_shape[i] = DimsFunctionUtils::GetDim(output_dims, i);
        }

        auto &unit = execute_units_[0];
        int idx    = SetExecuteUnit2DSizeInfoDefault(unit, output_dims);
        unit.ocl_kernel.setArg(idx++, *((cl::Image *)inputs[0]->GetHandle().base));
        unit.ocl_kernel.setArg(idx++, *((cl::Image *)inputs[1]->GetHandle().base));
        unit.ocl_kernel.setArg(idx++, *((cl::Image *)inputs[2]->GetHandle().base));
        unit.ocl_kernel.setArg(idx++, 4 * sizeof(int), output_shape.data());
        unit.ocl_kernel.setArg(idx++, 4 * sizeof(int), input0_shape.data());
        unit.ocl_kernel.setArg(idx++, 4 * sizeof(int), input1_shape.data());
        unit.ocl_kernel.setArg(idx++, 4 * sizeof(int), input2_shape.data());
        unit.ocl_kernel.setArg(idx++, UP_DIV(input0_shape[1], 4));
        unit.ocl_kernel.setArg(idx++, UP_DIV(input1_shape[1], 4));
        unit.ocl_kernel.setArg(idx++, UP_DIV(input2_shape[1], 4));
        unit.ocl_kernel.setArg(idx++, *((cl::Image *)output->GetHandle().base));
    }

    return TNN_OK;
}

std::vector<DataType> OpenCLWhereLayerAcc::SupportDataType(int dims_size, BlobType blob_type) {
    return {DATA_TYPE_FLOAT, DATA_TYPE_HALF, DATA_TYPE_INT32, DATA_TYPE_INT8};
}


REGISTER_OPENCL_ACC(Where, LAYER_WHERE)
REGISTER_OPENCL_LAYOUT(LAYER_WHERE, DATA_FORMAT_NHC4W4);

}  // namespace TNN_NS
