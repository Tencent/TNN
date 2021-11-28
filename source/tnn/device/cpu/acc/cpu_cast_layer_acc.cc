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
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

DECLARE_CPU_ACC(Cast, LAYER_CAST);

Status CpuCastLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CpuCastLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    const auto param      = dynamic_cast<CastLayerParam *>(param_);
    void *input_data      = inputs[0]->GetHandle().base;
    auto input_data_type  = inputs[0]->GetBlobDesc().data_type;
    void *output_data     = outputs[0]->GetHandle().base;
    auto output_data_type = outputs[0]->GetBlobDesc().data_type;

    const int ele_size = DataTypeUtils::GetBytesSize(outputs[0]->GetBlobDesc().data_type);

    const int count = DimsVectorUtils::Count(outputs[0]->GetBlobDesc().dims);
    if (input_data_type == output_data_type) {
        if (output_data != input_data) {
            memcpy(output_data, input_data, count * ele_size);
        }
    } else if (input_data_type == DATA_TYPE_FLOAT && output_data_type == DATA_TYPE_INT32) {
        auto *input_data_ptr  = (float *)input_data;
        auto *output_data_ptr = (int *)output_data;
        for (int i = 0; i < count; ++i) {
            output_data_ptr[i] = static_cast<int>(input_data_ptr[i]);
        }
    } else if (input_data_type == DATA_TYPE_INT32 && output_data_type == DATA_TYPE_FLOAT) {
        auto *input_data_ptr  = (int *)input_data;
        auto *output_data_ptr = (float *)output_data;
        for (int i = 0; i < count; ++i) {
            output_data_ptr[i] = static_cast<float>(input_data_ptr[i]);
        }
    } else if (input_data_type == DATA_TYPE_FLOAT && output_data_type == DATA_TYPE_INT8) {
        auto *input_data_ptr  = (float *)input_data;
        auto *output_data_ptr = (int8_t *)output_data;
        for (int i = 0; i < count; ++i) {
            output_data_ptr[i] = static_cast<int8_t>(input_data_ptr[i]);
        }
    } else if (input_data_type == DATA_TYPE_INT8 && output_data_type == DATA_TYPE_FLOAT) {
        auto *input_data_ptr  = (int8_t *)input_data;
        auto *output_data_ptr = (float *)output_data;
        for (int i = 0; i < count; ++i) {
            output_data_ptr[i] = static_cast<float>(input_data_ptr[i]);
        }
    } else if (input_data_type == DATA_TYPE_INT8 && output_data_type == DATA_TYPE_INT32) {
        auto *input_data_ptr  = (int8_t *)input_data;
        auto *output_data_ptr = (int *)output_data;
        for (int i = 0; i < count; ++i) {
            output_data_ptr[i] = static_cast<int>(input_data_ptr[i]);
        }
    } else if (input_data_type == DATA_TYPE_INT32 && output_data_type == DATA_TYPE_INT8) {
        auto *input_data_ptr  = (int *)input_data;
        auto *output_data_ptr = (int8_t *)output_data;
        for (int i = 0; i < count; ++i) {
            output_data_ptr[i] = static_cast<int8_t>(input_data_ptr[i]);
        }
    } else if (input_data_type == DATA_TYPE_INT32 && output_data_type == DATA_TYPE_UINT32) {
        auto *input_data_ptr  = (int *)input_data;
        auto *output_data_ptr = (unsigned int *)output_data;
        for (int i = 0; i < count; ++i) {
            output_data_ptr[i] = static_cast<unsigned int>(input_data_ptr[i]);
        }
    } else if (input_data_type == DATA_TYPE_UINT32 && output_data_type == DATA_TYPE_INT32) {
        auto *input_data_ptr  = (unsigned int *)input_data;
        auto *output_data_ptr = (int *)output_data;
        for (int i = 0; i < count; ++i) {
            output_data_ptr[i] = static_cast<int>(input_data_ptr[i]);
        }
    } else {
        LOGE("unsupport data type to cast\n");
    }
    return TNN_OK;
}

REGISTER_CPU_ACC(Cast, LAYER_CAST);
}  // namespace TNN_NS
