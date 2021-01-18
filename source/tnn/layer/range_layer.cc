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

#include "base_layer.h"
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS {
DECLARE_LAYER_WITH_FUNC(Range, LAYER_RANGE,
                        virtual Status FillLayerParamWithConstantResource(););

Status RangeLayer::InferOutputDataType() {
    auto status = BaseLayer::InferOutputDataType();
    if (runtime_model_ != RUNTIME_MODE_CONST_FOLD) {
        return status;
    }
    const auto& input_name = input_blobs_[0]->GetBlobDesc().name;
    const auto& const_res  = const_resource_;
    if (const_res != nullptr && const_res->find(input_name) != const_res->end()) {
        output_blobs_[0]->flag = output_blobs_[0]->flag | DATA_FLAG_ALLOCATE_IN_FORWARD;
    }
    return status;
}

Status RangeLayer::InferOutputShape(bool ignore_error) {
    //NOTE: This layer should not be excuted on device which is not NAIVE. see ConstantOfShapeLayer
    
    auto status = BaseLayer::InferOutputShape(ignore_error);
    RETURN_ON_NEQ(status, TNN_OK);
    
    auto *layer_param = dynamic_cast<RangeLayerParam *>(param_);
    CHECK_PARAM_NULL(layer_param);
    
    if (input_blobs_[0]->GetBlobDesc().device_type != DEVICE_NAIVE) {
        return Status(TNNERR_MODEL_ERR, "RangeLayer input blob has invalid device type");
    }
    
    auto output_dims = DimsVectorUtils::Range(layer_param->start, layer_param->limit,
                                              layer_param->delta, layer_param->type, &status);
    RETURN_ON_NEQ(status, TNN_OK);
    
    output_blobs_[0]->GetBlobDesc().dims = output_dims;
    
    return TNN_OK;
}

Status RangeLayer::FillLayerParamWithConstantResource() {
    Status status = TNN_OK;
    auto *layer_param = dynamic_cast<RangeLayerParam *>(param_);
    CHECK_PARAM_NULL(layer_param);
    
    if (input_blobs_.size() != 3) {
        return Status(TNNERR_PARAM_ERR, "RangeLayer has invalid layer param");
    }
    
    //start
    {
        const auto start_name = input_blobs_[0]->GetBlobDesc().name;
        if (const_resource_ != nullptr && const_resource_->find(start_name) != const_resource_->end()) {
            auto start_buffer = (*const_resource_)[start_name];
            layer_param->type = start_buffer->GetDataType();
            auto start_data   = start_buffer->force_to<float *>();
            auto start = layer_param->start;
            if (start_buffer->GetDataType() == DATA_TYPE_FLOAT) {
                start.f = *start_data;
            } else if (start_buffer->GetDataType() == DATA_TYPE_INT32) {
                start.i = *((int *)start_data);
            } else {
                return Status(TNNERR_PARAM_ERR, "RangeLayer has invalid start data type");
            }
            layer_param->start = start;
        }
    }
    
    //limit
    {
        const auto limit_name = input_blobs_[1]->GetBlobDesc().name;
        if (const_resource_ != nullptr && const_resource_->find(limit_name) != const_resource_->end()) {
            auto limit_buffer = (*const_resource_)[limit_name];
            layer_param->type = limit_buffer->GetDataType();
            auto limit_data   = limit_buffer->force_to<float *>();
            auto limit = layer_param->limit;
            if (limit_buffer->GetDataType() == DATA_TYPE_FLOAT) {
                limit.f = *limit_data;
            } else if (limit_buffer->GetDataType() == DATA_TYPE_INT32) {
                limit.i = *((int *)limit_data);
            } else {
                return Status(TNNERR_PARAM_ERR, "RangeLayer has invalid limit data type");
            }
            layer_param->limit = limit;
        }
    }
    
    //delta
    {
        const auto delta_name = input_blobs_[2]->GetBlobDesc().name;
        if (const_resource_ != nullptr && const_resource_->find(delta_name) != const_resource_->end()) {
            auto delta_buffer = (*const_resource_)[delta_name];
            layer_param->type = delta_buffer->GetDataType();
            auto delta_data   = delta_buffer->force_to<float *>();
            auto delta = layer_param->delta;
            if (delta_buffer->GetDataType() == DATA_TYPE_FLOAT) {
                delta.f = *delta_data;
            } else if (delta_buffer->GetDataType() == DATA_TYPE_INT32) {
                delta.i = *((int *)delta_data);
            } else {
                return Status(TNNERR_PARAM_ERR, "RangeLayer has invalid limit data type");
            }
            layer_param->delta = delta;
        }
    }
    return status;
}

REGISTER_LAYER(Range, LAYER_RANGE);

}  // namespace TNN_NS
