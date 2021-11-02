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

DECLARE_CPU_ACC_WITH_FUNC(Ragne, LAYER_RANGE,
                          virtual Status InferRuntimeOutputShape(const std::vector<Blob *> &inputs,
                                                                 const std::vector<Blob *> &outputs););

Status CpuRagneLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CpuRagneLayerAcc::InferRuntimeOutputShape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto *layer_param = dynamic_cast<RangeLayerParam *>(param_);
    CHECK_PARAM_NULL(layer_param);
    
    if (inputs.size() >= 3) {
        //start
        {
            layer_param->data_type = inputs[0]->GetBlobDesc().data_type;
            
            auto start_data = (void *)((char *)inputs[0]->GetHandle().base + inputs[0]->GetHandle().bytes_offset);
            auto start = layer_param->start;
            if (inputs[0]->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
                start.f = *((float *)start_data);
            } else if (inputs[0]->GetBlobDesc().data_type == DATA_TYPE_INT32) {
                start.i = *((int *)start_data);
            } else {
                return Status(TNNERR_PARAM_ERR, "RangeLayer has invalid limit data type");
            }
            layer_param->start = start;
        }

        //limit
        {
            auto limit_data = (void *)((char *)inputs[1]->GetHandle().base + inputs[1]->GetHandle().bytes_offset);
            auto limit = layer_param->limit;
            if (inputs[1]->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
                limit.f = *((float *)limit_data);
            } else if (inputs[1]->GetBlobDesc().data_type == DATA_TYPE_INT32) {
                limit.i = *((int *)limit_data);
            } else {
                return Status(TNNERR_PARAM_ERR, "RangeLayer has invalid limit data type");
            }
            layer_param->limit = limit;
        }
        
        //delta
        {
            auto delta_data = (void *)((char *)inputs[2]->GetHandle().base + inputs[2]->GetHandle().bytes_offset);
            auto delta = layer_param->delta;
            if (inputs[2]->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
                delta.f = *((float *)delta_data);
            } else if (inputs[2]->GetBlobDesc().data_type == DATA_TYPE_INT32) {
                delta.i = *((int *)delta_data);
            } else {
                return Status(TNNERR_PARAM_ERR, "RangeLayer has invalid delta data type");
            }
            layer_param->delta = delta;
        }
        
        //infer output shape
        Status status = TNN_OK;
        auto output_dims = DimsFunctionUtils::Range(layer_param->start, layer_param->limit,
                                                  layer_param->delta, layer_param->data_type, &status);
        RETURN_ON_NEQ(status, TNN_OK);
        
        outputs[0]->GetBlobDesc().dims = output_dims;
    }
    
    return TNN_OK;
}

Status CpuRagneLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto *layer_param = dynamic_cast<RangeLayerParam *>(param_);
    CHECK_PARAM_NULL(layer_param);

    const auto &output_blob = outputs[0];
    const auto &data_type   = output_blob->GetBlobDesc().data_type;
    int count               = DimsVectorUtils::Count(output_blob->GetBlobDesc().dims);
    if (data_type == DATA_TYPE_INT32) {
        auto output_data = static_cast<int32_t *>(output_blob->GetHandle().base);
        for (int i = 0; i < count; ++i) {
            output_data[i] = layer_param->start.i + i * layer_param->delta.i;
        }
    } else if (data_type == DATA_TYPE_FLOAT) {
        auto output_data = static_cast<float *>(output_blob->GetHandle().base);
        for (int i = 0; i < count; ++i) {
            output_data[i] = layer_param->start.f + i * layer_param->delta.f;
        }
    } else {
        LOGE("output blob of Shape Layer has wrong data type \n");
        return Status(TNNERR_COMMON_ERROR, "output blob has wrong data type");
    }
    return TNN_OK;
}

REGISTER_CPU_ACC(Ragne, LAYER_RANGE);
}  // namespace TNN_NS
