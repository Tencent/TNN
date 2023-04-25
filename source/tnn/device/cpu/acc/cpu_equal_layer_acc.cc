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

#include "cpu_binary_op_layer_acc.h"
#include "tnn/core/blob_int8.h"
#include "tnn/utils/naive_compute.h"

namespace TNN_NS {

DECLARE_CPU_ACC(Equal, LAYER_EQUAL);

Status CpuEqualLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CpuEqualLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    void *output_data       = outputs[0]->GetHandle().base;
    const auto &output_dims = outputs[0]->GetBlobDesc().dims;
    auto layer_param        = dynamic_cast<MultidirBroadcastLayerParam *>(param_);
    auto layer_res          = dynamic_cast<EltwiseLayerResource *>(resource_);

    const float FLOAT_EQUAL_EPSILON = 1e-6;

    DataType in0_dtype, in1_dtype;
    std::vector<void *> input_ptrs;
    std::vector<DimsVector> input_shapes;
    if (inputs.size() >= 2) {
        for (size_t inid = 0; inid < inputs.size(); inid++) {
            input_ptrs.push_back(inputs[inid]->GetHandle().base);
            input_shapes.push_back(inputs[inid]->GetBlobDesc().dims);
        }
        in0_dtype = inputs[0]->GetBlobDesc().data_type;
        in1_dtype = inputs[1]->GetBlobDesc().data_type;
    } else {
        DimsVector input_shape0 = inputs[0]->GetBlobDesc().dims;
        if (layer_param->weight_input_index == 0) {
            input_ptrs.push_back(layer_res->element_handle.force_to<void *>());
            input_shapes.push_back(layer_res->element_shape);
            in0_dtype = layer_res->element_handle.GetDataType();

            input_ptrs.push_back(inputs[0]->GetHandle().base);
            input_shapes.push_back(input_shape0);
            in1_dtype = inputs[0]->GetBlobDesc().data_type;
        } else {
            input_ptrs.push_back(inputs[0]->GetHandle().base);
            input_shapes.push_back(input_shape0);
            in0_dtype = inputs[0]->GetBlobDesc().data_type;

            input_ptrs.push_back(layer_res->element_handle.force_to<void *>());
            input_shapes.push_back(layer_res->element_shape);
            in1_dtype = layer_res->element_handle.GetDataType();
        }
    }

    if (inputs.size()<=2 && in0_dtype != in1_dtype) {
        if (in0_dtype==DATA_TYPE_FLOAT && in1_dtype==DATA_TYPE_HALF) {
            CPU_ELEMENT_WISE_BINARY_TYPECAST<float, fp16_t, int8_t>(input_ptrs, input_shapes, output_data, output_dims,
                                    [FLOAT_EQUAL_EPSILON](float a, fp16_t b) -> int8_t { return std::abs(a-float(b)) < FLOAT_EQUAL_EPSILON ? 1 : 0; });
        } else if (in0_dtype==DATA_TYPE_HALF && in1_dtype==DATA_TYPE_FLOAT) {
            CPU_ELEMENT_WISE_BINARY_TYPECAST<fp16_t, float, int8_t>(input_ptrs, input_shapes, output_data, output_dims,
                                    [FLOAT_EQUAL_EPSILON](fp16_t a, float b) -> int8_t { return std::abs(float(a)-b) < FLOAT_EQUAL_EPSILON ? 1 : 0; });
        } else if (in0_dtype==DATA_TYPE_FLOAT && in1_dtype==DATA_TYPE_INT32) {
            CPU_ELEMENT_WISE_BINARY_TYPECAST<float, int, int8_t>(input_ptrs, input_shapes, output_data, output_dims,
                                    [FLOAT_EQUAL_EPSILON](float a, int b) -> int8_t { return std::abs(a-float(b)) < FLOAT_EQUAL_EPSILON ? 1 : 0; });
        } else if (in0_dtype==DATA_TYPE_INT32 && in1_dtype==DATA_TYPE_FLOAT) {
            CPU_ELEMENT_WISE_BINARY_TYPECAST<int, float, int8_t>(input_ptrs, input_shapes, output_data, output_dims,
                                    [FLOAT_EQUAL_EPSILON](int a, float b) -> int8_t { return std::abs(float(a)-b) < FLOAT_EQUAL_EPSILON ? 1 : 0; });
        } else if (in0_dtype==DATA_TYPE_HALF && in1_dtype==DATA_TYPE_INT32) {
            CPU_ELEMENT_WISE_BINARY_TYPECAST<fp16_t, int, int8_t>(input_ptrs, input_shapes, output_data, output_dims,
                                    [FLOAT_EQUAL_EPSILON](fp16_t a, int b) -> int8_t { return std::abs(float(a)-float(b)) < FLOAT_EQUAL_EPSILON ? 1 : 0; });
        } else if (in0_dtype==DATA_TYPE_INT32 && in1_dtype==DATA_TYPE_HALF) {
            CPU_ELEMENT_WISE_BINARY_TYPECAST<int, fp16_t, int8_t>(input_ptrs, input_shapes, output_data, output_dims,
                                    [FLOAT_EQUAL_EPSILON](int a, fp16_t b) -> int8_t { return std::abs(float(a)-float(b)) < FLOAT_EQUAL_EPSILON ? 1 : 0; });
        } else {
            LOGE("Error: CpuEqualLayerAcc don't support in0.type: %d and in1.type: %d\n", in0_dtype, in1_dtype);
            return Status(TNNERR_MODEL_ERR, "CpuEqualLayerAcc don't support in0, in1 data type combination");
        }
    } else {
        if (in0_dtype == DATA_TYPE_FLOAT) {
            CPU_ELEMENT_WISE_COMPARE<float, int8_t>(input_ptrs, input_shapes, output_data, output_dims,
                [FLOAT_EQUAL_EPSILON](float a, float b) -> int8_t { return std::abs(a-b) < FLOAT_EQUAL_EPSILON ? 1 : 0; });
        } else if (in0_dtype == DATA_TYPE_HALF) {
            CPU_ELEMENT_WISE_COMPARE<fp16_t, int8_t>(input_ptrs, input_shapes, output_data, output_dims,
                [FLOAT_EQUAL_EPSILON](fp16_t a, fp16_t b) -> int8_t { return std::abs(float(a)-float(b)) < FLOAT_EQUAL_EPSILON ? 1 : 0; });
        } else if (in0_dtype == DATA_TYPE_BFP16) {
            CPU_ELEMENT_WISE_COMPARE<fp16_t, int8_t>(input_ptrs, input_shapes, output_data, output_dims,
                [FLOAT_EQUAL_EPSILON](fp16_t a, fp16_t b) -> int8_t { return std::abs(float(a)-float(b)) < FLOAT_EQUAL_EPSILON ? 1 : 0; });
        } else if (in0_dtype == DATA_TYPE_INT32) {
            CPU_ELEMENT_WISE_COMPARE<int, int8_t>(input_ptrs, input_shapes, output_data, output_dims,
                [](int a, int b) -> int8_t { return a == b; });
        } else if (in0_dtype == DATA_TYPE_INT8) {
            CPU_ELEMENT_WISE_COMPARE<int8_t, int8_t>(input_ptrs, input_shapes, output_data, output_dims,
                [](int8_t a, int8_t b) -> int8_t { return a == b; });
        } else {
            LOGE("Error: CpuEqualLayerAcc don't support data type: %d\n", inputs[0]->GetBlobDesc().data_type);
            return Status(TNNERR_MODEL_ERR, "Error: CpuEqualLayerAcc don't support data type");
        }
    }

    return TNN_OK;
}

REGISTER_CPU_ACC(Equal, LAYER_EQUAL);

}  // namespace TNN_NS
