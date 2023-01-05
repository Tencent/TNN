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

#include "tnn/device/arm/acc/arm_layer_acc.h"
#include "tnn/device/arm/acc/arm_expand_layer_acc.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

ArmExpandLayerAcc::~ArmExpandLayerAcc() {}

Status ArmExpandLayerAcc::InferRuntimeOutputShape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto expand_param = dynamic_cast<ExpandLayerParam*>(param_);
    CHECK_PARAM_NULL(expand_param);
    
    if (inputs.size() == 2) {
        auto data_dims = inputs[0]->GetBlobDesc().dims;
        DimsVector shape_dims;
        auto shape_data = reinterpret_cast<int *>(GetBlobHandlePtr(inputs[1]->GetHandle()));
        auto shape_data_count = DimsVectorUtils::Count(inputs[1]->GetBlobDesc().dims);
        for (int i=0; i<shape_data_count; i++) {
            shape_dims.push_back(shape_data[i]);
        }
        
        expand_param->shape = shape_dims;
        
        auto output_dims = DimsFunctionUtils::Expand(data_dims, shape_dims, nullptr);
        outputs[0]->GetBlobDesc().dims = output_dims;
    }
    
    return AbstractLayerAcc::InferRuntimeOutputShape(inputs, outputs);
}

static void ExpandComputeOffset(DimsVector &offset, const DimsVector dims_in, const DimsVector dims_out) {
    DimsVector dims_pad_in;
    int pad_size = dims_out.size() - dims_in.size();
    int i = 0;
    for (; i < pad_size; i++) {
        dims_pad_in.push_back(1);
    }
    for (; i < dims_out.size(); i++) {
        dims_pad_in.push_back(dims_in[i - pad_size]);
    }

    offset.resize(dims_out.size());
    int s = 1;
    for (i = dims_out.size() - 1; i >= 0; i--) {
        offset[i] = (dims_pad_in[i] == dims_out[i]) ? s : 0;
        s *= dims_pad_in[i];
    }
}

template <typename T>
static void ArmExpand(DimsVector output_shape, DimsVector input_shape,
                      T *output_ptr, const T *input_ptr) {
    DimsVector output_offset;
    ExpandComputeOffset(output_offset, output_shape, output_shape);

    DimsVector input_offset;
    ExpandComputeOffset(input_offset, input_shape, output_shape);

    DimsVector out_shape;
    DimsVector in_offset;
    DimsVector ou_offset;
    // support maximum 6 dimension, may be extended in furture
    out_shape.resize(6);
    in_offset.resize(6);
    ou_offset.resize(6);
    // if dim < 6, pad to 6
    int pad_size = 6 - output_shape.size();
    for (int i = 0; i < pad_size; i++) {
        out_shape[i] = 1;
        in_offset[i] = 0;
        ou_offset[i] = 0;
    }
    for (int i = pad_size; i < 6; i++) {
        out_shape[i] = output_shape[i - pad_size];
        in_offset[i] = input_offset[i - pad_size];
        ou_offset[i] = output_offset[i - pad_size];
    }

    for (int i0 = 0; i0 < out_shape[0]; i0++) {
        auto in_i0 = input_ptr + i0 * in_offset[0];
        auto ou_i0 = output_ptr + i0 * ou_offset[0];
        for (int i1 = 0; i1 < out_shape[1]; i1++) {
            auto in_i1 = in_i0 + i1 * in_offset[1];
            auto ou_i1 = ou_i0 + i1 * ou_offset[1];
            for (int i2 = 0; i2 < out_shape[2]; i2++) {
                auto in_i2 = in_i1 + i2 * in_offset[2];
                auto ou_i2 = ou_i1 + i2 * ou_offset[2];
                for (int i3 = 0; i3 < out_shape[3]; i3++) {
                    auto in_i3 = in_i2 + i3 * in_offset[3];
                    auto ou_i3 = ou_i2 + i3 * ou_offset[3];
                    for (int i4 = 0; i4 < out_shape[4]; i4++) {
                        auto in_i4 = in_i3 + i4 * in_offset[4];
                        auto ou_i4 = ou_i3 + i4 * ou_offset[4];
                        for (int i5 = 0; i5 < out_shape[5]; i5++) {
                            auto in_i5 = in_i4 + i5 * in_offset[5];
                            auto ou_i5 = ou_i4 + i5 * ou_offset[5];
                            ou_i5[0] = in_i5[0];
                        }
                    }
                }
            }
        }
    }
}

Status ArmExpandLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto input_blob  = inputs[0];
    auto output_blob = outputs[0];
    auto output_dims = output_blob->GetBlobDesc().dims;
    auto input_dims = input_blob->GetBlobDesc().dims;

    if (output_dims.size() > 6) {
        return Status(TNNERR_MODEL_ERR, "arm expand only support dims <= 6");
    }

    if (output_blob->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
        const float *input_data = reinterpret_cast<const float *>(GetBlobHandlePtr(input_blob->GetHandle()));
        float *output_data = reinterpret_cast<float *>(GetBlobHandlePtr(output_blob->GetHandle()));
        ArmExpand<float>(output_dims, input_dims, output_data, input_data);
    } else if (output_blob->GetBlobDesc().data_type == DATA_TYPE_BFP16) {
        const bfp16_t *input_data = reinterpret_cast<const bfp16_t *>(GetBlobHandlePtr(input_blob->GetHandle()));
        bfp16_t *output_data = reinterpret_cast<bfp16_t *>(GetBlobHandlePtr(output_blob->GetHandle()));
        ArmExpand<bfp16_t>(output_dims, input_dims, output_data, input_data);
    } else if (output_blob->GetBlobDesc().data_type == DATA_TYPE_INT32) {
        const int32_t *input_data = reinterpret_cast<const int32_t *>(GetBlobHandlePtr(input_blob->GetHandle()));
        int32_t *output_data      = reinterpret_cast<int32_t *>(GetBlobHandlePtr(output_blob->GetHandle()));
        ArmExpand<int32_t>(output_dims, input_dims, output_data, input_data);
    }
#ifdef TNN_ARM82
    else if (output_blob->GetBlobDesc().data_type == DATA_TYPE_HALF) {
        const fp16_t *input_data = reinterpret_cast<const fp16_t *>(GetBlobHandlePtr(input_blob->GetHandle()));
        fp16_t *output_data = reinterpret_cast<fp16_t *>(GetBlobHandlePtr(output_blob->GetHandle()));
        ArmExpand<fp16_t>(output_dims, input_dims, output_data, input_data);
    }
#endif
    else {
        return Status(TNNERR_MODEL_ERR, "blob type is unsupported");
    }

    return TNN_OK;
}

REGISTER_ARM_ACC(Expand, LAYER_EXPAND);
REGISTER_ARM_PRECISION_FP16(LAYER_EXPAND)
REGISTER_ARM_LAYOUT(LAYER_EXPAND, DATA_FORMAT_NCHW)

}  // namespace TNN_NS
