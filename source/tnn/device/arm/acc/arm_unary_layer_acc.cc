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

#include "tnn/device/arm/acc/arm_unary_layer_acc.h"
#include "tnn/device/arm/arm_common.h"
#include "tnn/device/arm/arm_context.h"
#include "tnn/utils/omp_utils.h"

namespace TNN_NS {

ArmUnaryLayerAcc::~ArmUnaryLayerAcc() {}

Status ArmUnaryLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                              const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    RETURN_ON_NEQ(ArmLayerAcc::Init(context, param, resource, inputs, outputs), TNN_OK);
    return op_->Init(param);
}

// SUPPORTED DATATYPES
bool ArmUnaryLayerAcc::DataTypeSupported(DataType data_type) {
    if (data_type == DATA_TYPE_FLOAT || data_type == DATA_TYPE_BFP16 || data_type == DATA_TYPE_HALF)
        return true;
    else
        return false;
}

template <typename T>
Status ArmUnaryLayerAcc::Exec(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto input  = inputs[0];
    auto output = outputs[0];

    auto dims = output->GetBlobDesc().dims;

    int count      = dims[0] * ROUND_UP(dims[1], 4) * dims[2] * dims[3];
    int count_quad = UP_DIV(count, 4);

    auto input_ptr  = reinterpret_cast<T *>(GetBlobHandlePtr(input->GetHandle()));
    auto output_ptr = reinterpret_cast<T *>(GetBlobHandlePtr(output->GetHandle()));

    if (context_->GetPrecision() == PRECISION_HIGH) {
        OMP_PARALLEL_FOR_
        for (int n = 0; n < count_quad; n++) {
            Float4::save(output_ptr + n * 4, (*op_)(Float4::load(input_ptr + n * 4)));
        }
    } else {
        OMP_PARALLEL_FOR_
        for (int n = 0; n < count_quad; n++) {
            Float4::save(output_ptr + n * 4, op_->fast_op(Float4::load(input_ptr + n * 4)));
        }
    }

    return TNN_OK;
}

#if TNN_ARM82
template <>
Status ArmUnaryLayerAcc::Exec<fp16_t>(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto input  = inputs[0];
    auto output = outputs[0];

    auto dims = output->GetBlobDesc().dims;

    int count      = dims[0] * ROUND_UP(dims[1], 8) * dims[2] * dims[3];
    int count_div8 = UP_DIV(count, 8);

    auto input_ptr  = reinterpret_cast<fp16_t *>(GetBlobHandlePtr(input->GetHandle()));
    auto output_ptr = reinterpret_cast<fp16_t *>(GetBlobHandlePtr(output->GetHandle()));

    OMP_PARALLEL_FOR_
    for (int n = 0; n < count_div8; n++) {
        Half8::save(output_ptr + n * 8, (*op_)(Half8::load(input_ptr + n * 8)));
    }

    return TNN_OK;
}
#endif

Status ArmUnaryLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    if (inputs[0]->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
        return Exec<float>(inputs, outputs);
    } else if (inputs[0]->GetBlobDesc().data_type == DATA_TYPE_BFP16) {
        return Exec<bfp16_t>(inputs, outputs);
    }
#if TNN_ARM82
    else if (inputs[0]->GetBlobDesc().data_type == DATA_TYPE_HALF) {
        return Exec<fp16_t>(inputs, outputs);
    }
#endif
    return TNNERR_LAYER_ERR;
}

}  // namespace TNN_NS
