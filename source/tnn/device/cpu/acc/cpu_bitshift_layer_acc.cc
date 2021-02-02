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

#include "cpu_layer_acc.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS {

DECLARE_CPU_ACC(BitShift, LAYER_BITSHIFT);

Status CpuBitShiftLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CpuBitShiftLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto layer_param = dynamic_cast<BitShiftLayerParam*>(param_);
    CHECK_PARAM_NULL(layer_param);
    
    auto input_data_type  = inputs[0]->GetBlobDesc().data_type;
    auto input_data     = (unsigned int *)((char *)inputs[0]->GetHandle().base + inputs[0]->GetHandle().bytes_offset);
    auto output_data     = (unsigned int *)((char *)outputs[0]->GetHandle().base + outputs[0]->GetHandle().bytes_offset);
    
    const int count = DimsVectorUtils::Count(inputs[0]->GetBlobDesc().dims);
    
    if (input_data_type == DATA_TYPE_INT32 || input_data_type == DATA_TYPE_UINT32) {
        if (layer_param->direction == 0) {
            for (int index = 0; index < count; ++index) {
                output_data[index] = input_data[index] >> layer_param->bits;
            }
        } else {
            for (int index = 0; index < count; ++index) {
                output_data[index] = input_data[index] << layer_param->bits;
            }
        }

    } else {
        LOGE("unsupport data type to Histogram\n");
    }
    return TNN_OK;
}

REGISTER_CPU_ACC(BitShift, LAYER_BITSHIFT);
}  // namespace TNN_NS
